import os
import json
from math import sqrt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt


# =========================
# 설정
# =========================
FILE_PATH = r"csvPath"
SAVE_ROOT = r"Path"

FEATURES = ["Temperature_C", "Salinity_PSU", "pH", "DO_mgL", "Illuminance_Lux"]

# 시퀀스 길이
SEQ_LEN  = 432   # 입력: 72h (10분 간격 * 432)
PRED_LEN = 144   # 출력: 24h

# Embargo(예측일 기준 앞뒤 12시간)
EMBARGO_BEFORE = pd.Timedelta(hours=12)
EMBARGO_AFTER  = pd.Timedelta(hours=12)

# 학습 하이퍼파라미터 공통
BATCH_SIZE = 64
EPOCHS     = 6
LR         = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 2
PATIENCE   = 2     # EarlyStopping (val loss)

# 모델 선택
MODEL_TYPE = "DLINEAR"   # "LSTM" | "GRU" | "TCN" | "DLINEAR" | "MLP"

# 기타
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(SAVE_ROOT, exist_ok=True)

# =========================
# 유틸 함수
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def rmse_vec(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def smape_vec(y_true, y_pred):
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))

def compute_overall_and_pervar_metrics(y_true, y_pred, feature_names):
    overall = {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse_vec(y_true, y_pred)),
        "sMAPE": smape_vec(y_true, y_pred)
    }
    per_var = {}
    for j, name in enumerate(feature_names):
        t = y_true[:, j]
        p = y_pred[:, j]
        per_var[name] = {
            "MAE":  float(mean_absolute_error(t, p)),
            "RMSE": float(rmse_vec(t, p)),
            "sMAPE": smape_vec(t, p)
        }
    return overall, per_var


def df_slice_by_time(df, start_t, end_t):
    return df.loc[(df.index >= start_t) & (df.index <= end_t)]


def pick_target_days(index: pd.DatetimeIndex, year: int, per_month: int = 5):
    rng = np.random.default_rng(SEED)
    by_month = {m: [] for m in range(1, 13)}

    dt = index[1] - index[0]
    assert dt == pd.Timedelta(minutes=10), f"샘플 간격 10분 아님: {dt}"

    for m in range(1, 13):
        if m == 12:
            next_month_first = pd.Timestamp(year=year+1, month=1, day=1)
        else:
            next_month_first = pd.Timestamp(year=year, month=m+1, day=1)
        last = next_month_first - pd.Timedelta(days=1)

        start_day = 10 if m == 1 else 1
        cur = pd.Timestamp(year=year, month=m, day=start_day)
        candidates = []
        while cur <= last:
            start_t = pd.Timestamp(cur)
            end_t   = start_t + pd.Timedelta(hours=24) - dt
            hist_start = start_t - pd.Timedelta(hours=72)
            if (hist_start >= index[0]) and (end_t <= index[-1]):
                candidates.append(start_t)
            cur += pd.Timedelta(days=1)

        if candidates:
            k = min(per_month, len(candidates))
            chosen = sorted(list(rng.choice(candidates, size=k, replace=False)))
            by_month[m] = chosen

    return by_month

# =========================
# Dataset (훈련용)
# =========================
class IndexWindowDataset(Dataset):
    def __init__(self, data_scaled: np.ndarray, start_indices: list, seq_len: int, pred_len: int):
        self.data = data_scaled
        self.idxs = start_indices
        self.L = seq_len
        self.H = pred_len

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        i = self.idxs[k]
        x = self.data[i:i+self.L]
        y = self.data[i+self.L:i+self.L+self.H]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# =========================
# 모델군 구현
# =========================
class BaseHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)

class ModelLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = BaseHead(hidden_dim, output_dim * pred_len)
        self.pred_len = pred_len
        self.output_dim = output_dim
    def forward(self, x):
        out, _ = self.lstm(x)
        last_h = out[:, -1, :]
        y = self.head(last_h)
        return y.view(-1, self.pred_len, self.output_dim)

class ModelGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = BaseHead(hidden_dim, output_dim * pred_len)
        self.pred_len = pred_len
        self.output_dim = output_dim
    def forward(self, x):
        out, _ = self.gru(x)
        last_h = out[:, -1, :]
        y = self.head(last_h)
        return y.view(-1, self.pred_len, self.output_dim)

class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp
    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.relu1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.relu2(y)
        y = self.drop2(y)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(y + res)

class ModelTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_len, kernel_size=3, dropout=0.0):
        super().__init__()
        channels = [hidden_dim] * num_layers
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers += [TemporalBlock(in_ch, channels[i], kernel_size, stride=1, dilation=dilation, padding=padding, dropout=dropout)]
            in_ch = channels[i]
        self.tcn = nn.Sequential(*layers)
        self.head = BaseHead(in_ch, output_dim * pred_len)
        self.pred_len = pred_len
        self.output_dim = output_dim
    def forward(self, x):
        # x: (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # 시계열의 마지막 타임스텝 임베딩 사용
        last = y[:, :, -1]
        out = self.head(last)
        return out.view(-1, self.pred_len, self.output_dim)

class ModelDLinear(nn.Module):
    """ 간이 DLinear: 입력시퀀스(각 변수별)를 선형 변환으로 바로 H-steps 예측 """
    def __init__(self, input_dim, pred_len):
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len
        # 각 변수마다: SEQ_LEN -> PRED_LEN 선형층
        self.proj = nn.Parameter(torch.zeros(input_dim, SEQ_LEN, pred_len))
        nn.init.xavier_uniform_(self.proj)
    def forward(self, x):
        # x: (B,T,F)
        # 아인슈타인 표기: (b t f, f t h) -> (b h f)
        # PyTorch einsum: btf, fth -> bhf
        W = self.proj  # (F, T, H)
        y = torch.einsum('btf,fth->bhf', x, W)
        return y  # (B,H,F)

class ModelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pred_len, dropout=0.0):
        super().__init__()
        in_features = SEQ_LEN * input_dim
        out_features = pred_len * output_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )
        self.pred_len = pred_len
        self.output_dim = output_dim
    def forward(self, x):
        B = x.shape[0]
        y = self.net(x.reshape(B, -1))
        return y.view(B, self.pred_len, self.output_dim)


def build_model(model_type, input_dim, hidden_dim, num_layers, output_dim, pred_len):
    mt = model_type.upper()
    if mt == "LSTM":
        return ModelLSTM(input_dim, hidden_dim, num_layers, output_dim, pred_len)
    if mt == "GRU":
        return ModelGRU(input_dim, hidden_dim, num_layers, output_dim, pred_len)
    if mt == "TCN":
        return ModelTCN(input_dim, hidden_dim, num_layers, output_dim, pred_len, kernel_size=3, dropout=0.1)
    if mt == "DLINEAR":
        return ModelDLinear(input_dim, pred_len)
    if mt == "MLP":
        return ModelMLP(input_dim, hidden_dim, output_dim, pred_len, dropout=0.1)
    raise ValueError(f"Unknown MODEL_TYPE: {model_type}")

# =========================
# 윈도우 인덱스 유틸
# =========================

def merge_intervals(intervals):
    if not intervals:
        return []
    s = sorted(intervals, key=lambda x: x[0])
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1] + pd.Timedelta(0):
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(x[0], x[1]) for x in out]


def subtract_intervals(full_start, full_end, blocks):
    if full_start > full_end:
        return []
    blocks = merge_intervals(blocks)
    allowed = []
    cur = full_start
    for a, b in blocks:
        if b < full_start or a > full_end:
            continue
        a = max(a, full_start)
        b = min(b, full_end)
        if cur <= a - pd.Timedelta(minutes=10):
            allowed.append((cur, a - pd.Timedelta(minutes=10)))
        cur = b + pd.Timedelta(minutes=10)
    if cur <= full_end:
        allowed.append((cur, full_end))
    return allowed


def build_window_indices_multi(timestamps, allowed_ranges, seq_len, pred_len, stride=1):
    if not allowed_ranges:
        return []
    N = len(timestamps)
    L, H = seq_len, pred_len
    out = []
    last_start = N - (L + H)
    if last_start < 0:
        return out
    for i in range(0, last_start + 1, stride):
        x_start_t = timestamps[i]
        y_end_t   = timestamps[i + L + H - 1]
        for a, b in allowed_ranges:
            if (x_start_t >= a) and (y_end_t <= b):
                out.append(i)
                break
    return out

# =========================
# 데이터 로드 및 타겟일 선정
# =========================

df = pd.read_csv(FILE_PATH, parse_dates=["Timestamp"])  # Timestamp 필수
assert "Timestamp" in df.columns

df = df.set_index("Timestamp").sort_index()
timestamps = df.index
assert set(FEATURES).issubset(df.columns), "FEATURES가 CSV에 없음"

DT = (timestamps[1] - timestamps[0])
assert DT == pd.Timedelta(minutes=10), f"샘플 간격이 10분이 아님: {DT}"

df_feats = df[FEATURES].copy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

YEAR = timestamps[0].year

target_days_by_month = pick_target_days(df_feats.index, YEAR, per_month=5)
ensure_dir(SAVE_ROOT)
with open(os.path.join(SAVE_ROOT, "chosen_target_days.json"), "w", encoding="utf-8") as f:
    json.dump({f"{m:02d}": [str(d) for d in ds] for m, ds in target_days_by_month.items()}, f, indent=2, ensure_ascii=False)

# ===== 훈련 허용 구간 계산: 전체에서 테스트일 embargo 합집합 제외 =====
FULL_START, FULL_END = df_feats.index[0], df_feats.index[-1]
blocks = []
for m, days in target_days_by_month.items():
    for day_start in days:
        test_start = pd.Timestamp(day_start)
        test_end   = test_start + pd.Timedelta(hours=24) - DT
        a = test_start - EMBARGO_BEFORE
        b = test_end   + EMBARGO_AFTER
        blocks.append((a, b))

allowed_train_ranges = subtract_intervals(FULL_START, FULL_END, blocks)

# 스케일러 fit(훈련 허용 구간만)
df_train_for_scaler_list = [df_slice_by_time(df_feats, a, b) for (a, b) in allowed_train_ranges]
if not df_train_for_scaler_list:
    raise RuntimeError("훈련 허용 구간이 비어 있습니다. 데이터/설정을 확인하세요.")
df_train_for_scaler = pd.concat(df_train_for_scaler_list)
scaler = StandardScaler().fit(df_train_for_scaler.values)

data_scaled = pd.DataFrame(
    scaler.transform(df_feats.values),
    index=df_feats.index,
    columns=FEATURES
)

# 훈련 윈도우
train_idxs = build_window_indices_multi(
    df_feats.index,
    allowed_ranges=allowed_train_ranges,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    stride=1
)
if len(train_idxs) == 0:
    raise RuntimeError("훈련 윈도우가 0개")

train_ds = IndexWindowDataset(data_scaled.values, train_idxs, SEQ_LEN, PRED_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ===== 모델 구성 =====
model = build_model(MODEL_TYPE, input_dim=len(FEATURES), hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=len(FEATURES), pred_len=PRED_LEN).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_state = None
best_loss = np.inf
patience_left = PATIENCE

for ep in range(1, EPOCHS+1):
    model.train()
    tr_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
    tr_loss /= max(1, len(train_loader))

    if tr_loss < best_loss:
        best_loss = tr_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_left = PATIENCE
    else:
        patience_left -= 1
        if patience_left <= 0:
            pass

    print(f"[{MODEL_TYPE}] Ep {ep}/{EPOCHS} | train {tr_loss:.5f}")

if best_state is not None:
    model.load_state_dict(best_state)

# 저장
joblib.dump(scaler, os.path.join(SAVE_ROOT, f"scaler_final_{MODEL_TYPE}.bin"))
torch.save(model.state_dict(), os.path.join(SAVE_ROOT, f"model_final_{MODEL_TYPE}.pth"))

# =========================
# 테스트(기존 방식 유지)
# =========================
all_results = {}
ensure_dir(SAVE_ROOT)

lux_idx = FEATURES.index("Illuminance_Lux")

for m in range(1, 13):
    month_days = target_days_by_month.get(m, [])
    if not month_days:
        continue

    month_dir = os.path.join(SAVE_ROOT, f"month_{m:02d}_{MODEL_TYPE}")
    ensure_dir(month_dir)

    day_results = []
    for day_start in month_days:
        test_start = pd.Timestamp(day_start)
        test_end   = test_start + pd.Timedelta(hours=24) - DT
        hist_start = test_start - pd.Timedelta(hours=72)

        i0 = df_feats.index.get_indexer([hist_start])[0]
        assert df_feats.index[i0] == hist_start, "hist_start인덱스 오류."

        x_input = torch.tensor(
            data_scaled.values[i0:i0+SEQ_LEN], dtype=torch.float32
        ).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            yhat_scaled = model(x_input).squeeze(0).cpu().numpy()

        y_scaled = data_scaled.values[i0+SEQ_LEN:i0+SEQ_LEN+PRED_LEN]

        pred_inv = scaler.inverse_transform(yhat_scaled)
        y_inv    = scaler.inverse_transform(y_scaled)
        pred_inv[:, lux_idx] = np.clip(pred_inv[:, lux_idx], 0, None)

        hist_index = df_feats.index[i0:i0+SEQ_LEN]
        pred_index = df_feats.index[i0+SEQ_LEN:i0+SEQ_LEN+PRED_LEN]

        overall, per_var = compute_overall_and_pervar_metrics(y_inv, pred_inv, FEATURES)
        day_result = {
            "date": str(test_start.date()),
            "overall": overall,
            "per_variable": per_var
        }
        day_results.append(day_result)

        day_dir = os.path.join(month_dir, f"day_{test_start.strftime('%Y-%m-%d')}")
        ensure_dir(day_dir)

        with open(os.path.join(day_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(day_result, f, ensure_ascii=False, indent=2)

        df_hist = df_feats.iloc[i0:i0+SEQ_LEN].copy()
        df_pred = pd.DataFrame(pred_inv, index=pred_index, columns=FEATURES)
        df_truth= pd.DataFrame(y_inv,    index=pred_index, columns=FEATURES)
        df_hist.to_csv(os.path.join(day_dir, "history_72h.csv"), index_label="Timestamp")
        df_pred.to_csv(os.path.join(day_dir, "forecast_24h.csv"), index_label="Timestamp")
        df_truth.to_csv(os.path.join(day_dir, "truth_24h.csv"),    index_label="Timestamp")

        for col in FEATURES:
            plt.figure(figsize=(12, 4))
            plt.plot(df_hist.index, df_hist[col], label="History (Actual)")
            plt.plot(df_pred.index, df_pred[col], label="Forecast (Predicted)")
            plt.plot(df_truth.index, df_truth[col], label="Truth (Actual)", linestyle="--")
            plt.title(f"{col} | {test_start.date()} — History(72h) + Forecast vs Truth(24h)")
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(day_dir, f"{col}_history72h_forecast_truth24h.png"), dpi=150)
            plt.close()

        fig, axes = plt.subplots(len(FEATURES), 1, figsize=(14, 10), sharex=True)
        for j, col in enumerate(FEATURES):
            axes[j].plot(df_hist.index, df_hist[col], label="History (Actual)")
            axes[j].plot(df_pred.index, df_pred[col], label="Forecast (Predicted)")
            axes[j].plot(df_truth.index, df_truth[col], label="Truth (Actual)", linestyle="--")
            axes[j].set_ylabel(col)
            if j == 0:
                axes[j].legend(loc="upper left")
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"{MODEL_TYPE}: History(72h) + Forecast vs Truth(24h) | {test_start.date()}")
        fig.tight_layout()
        plt.savefig(os.path.join(month_dir, f"all_vars_{test_start.strftime('%Y-%m-%d')}.png"), dpi=150)
        plt.close()

    if day_results:
        vals = [r["overall"] for r in day_results]
        month_mean = {
            "MAE":  float(np.mean([v["MAE"] for v in vals])),
            "RMSE": float(np.mean([v["RMSE"] for v in vals])),
            "sMAPE":float(np.mean([v["sMAPE"] for v in vals]))
        }
        summary = {"days": day_results, "mean_overall": month_mean}
        with open(os.path.join(month_dir, "metrics_month.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

# 전체 요약
print("=== DONE v7 multi-architectures ===")
