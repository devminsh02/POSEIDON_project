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

# 학습 하이퍼파라미터
BATCH_SIZE = 64
EPOCHS     = 6
LR         = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 2
PATIENCE   = 2     # EarlyStopping (val loss)

# 검증(조기종료)은 시간순 블록(valid)은 쓰되, 여기선 선택
VAL_DAYS_FOR_EARLYSTOP = 7  # 사용하지 않음(전역 검증 구간 정의가 모호함). 필요시 확장 가능.

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
# 모델
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim * pred_len)
        self.pred_len   = pred_len
        self.output_dim = output_dim

    def forward(self, x):
        out, _ = self.lstm(x)
        last_h = out[:, -1, :]
        out    = self.fc(last_h)
        out    = out.view(-1, self.pred_len, self.output_dim)
        return out


# =========================
# 구간 연산 유틸
# =========================

def merge_intervals(intervals):
    """[(a,b), ...] → 겹치는 구간 병합, 정렬 반환"""
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
    """
    [full_start, full_end] 에서 blocks(차단 구간들의 리스트)를 빼고 남는 허용 구간들의 리스트 반환
    """
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
        if cur <= a - pd.Timedelta(minutes=10):  # 10분 격자 고려해 살짝 여유
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
        # 어느 허용 구간 하나에 완전히 포함되면 채택
        for a, b in allowed_ranges:
            if (x_start_t >= a) and (y_end_t <= b):
                out.append(i)
                break
    return out


# =========================
# 데이터 로드
# =========================

df = pd.read_csv(FILE_PATH, parse_dates=["Timestamp"])  # Timestamp 필수
assert "Timestamp" in df.columns, "CSV에 Timestamp 열이 필요"
df = df.set_index("Timestamp").sort_index()
timestamps = df.index
assert set(FEATURES).issubset(df.columns), "FEATURES가 CSV에 없음"

# 10분 간격 확인
DT = (timestamps[1] - timestamps[0])
assert DT == pd.Timedelta(minutes=10), f"샘플 간격이 10분이 아님: {DT}"

df_feats = df[FEATURES].copy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

YEAR = timestamps[0].year

# 테스트 타겟 일자 선정(월별 5일, 1월은 10일부터) — v5와 동일 정책
target_days_by_month = pick_target_days(df_feats.index, YEAR, per_month=5)
ensure_dir(SAVE_ROOT)
with open(os.path.join(SAVE_ROOT, "chosen_target_days.json"), "w", encoding="utf-8") as f:
    json.dump({f"{m:02d}": [str(d) for d in ds] for m, ds in target_days_by_month.items()}, f, indent=2, ensure_ascii=False)

# ===== 훈련 허용 구간 계산: 전체에서 테스트일 embargo 합집합을 제외 =====
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
    raise RuntimeError("훈련 허용 구간이 비어 있음. 데이터/설정을 확인필요.")
df_train_for_scaler = pd.concat(df_train_for_scaler_list)
scaler = StandardScaler().fit(df_train_for_scaler.values)

# 전체 변환
data_scaled = pd.DataFrame(
    scaler.transform(df_feats.values),
    index=df_feats.index,
    columns=FEATURES
)

# 훈련 윈도우 인덱스(여러 허용 구간)
train_idxs = build_window_indices_multi(
    df_feats.index,
    allowed_ranges=allowed_train_ranges,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    stride=1
)

if len(train_idxs) == 0:
    raise RuntimeError("훈련 윈도우가 0개. allowed_train_ranges확인필요")

# DataLoader
train_ds = IndexWindowDataset(data_scaled.values, train_idxs, SEQ_LEN, PRED_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 단일 모델 학습(전역 검증 생략; 필요시 별도 블록 검증을 설계하세요)
model = LSTMModel(len(FEATURES), HIDDEN_DIM, NUM_LAYERS, len(FEATURES), PRED_LEN).to(device)
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

    # 간이 early stopping: train loss 기준(보수적으로만 사용)
    if tr_loss < best_loss:
        best_loss = tr_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_left = PATIENCE
    else:
        patience_left -= 1
        if patience_left <= 0:
            pass

    print(f"[FINAL-FULLSPAN] Ep {ep}/{EPOCHS} | train {tr_loss:.5f}")

if best_state is not None:
    model.load_state_dict(best_state)

# 최종 모델/스케일러 저장
joblib.dump(scaler, os.path.join(SAVE_ROOT, "scaler_final.bin"))
torch.save(model.state_dict(), os.path.join(SAVE_ROOT, "model_final.pth"))

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

    month_dir = os.path.join(SAVE_ROOT, f"month_{m:02d}")
    ensure_dir(month_dir)

    day_results = []
    for day_start in month_days:
        test_start = pd.Timestamp(day_start)  # 자정
        test_end   = test_start + pd.Timedelta(hours=24) - DT
        hist_start = test_start - pd.Timedelta(hours=72)

        # 인덱스 정합성 확인(10분 격자 가정)
        i0 = df_feats.index.get_indexer([hist_start])[0]
        assert df_feats.index[i0] == hist_start, "hist_start가 인덱스에 없음."

        # 단일 윈도우 예측
        x_input = torch.tensor(
            data_scaled.values[i0:i0+SEQ_LEN], dtype=torch.float32
        ).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            yhat_scaled = model(x_input).squeeze(0).cpu().numpy()  # (H,F)

        # truth (스케일된 원본에서 같은 구간)
        y_scaled = data_scaled.values[i0+SEQ_LEN:i0+SEQ_LEN+PRED_LEN]

        # 역스케일
        pred_inv = scaler.inverse_transform(yhat_scaled)
        y_inv    = scaler.inverse_transform(y_scaled)
        pred_inv[:, lux_idx] = np.clip(pred_inv[:, lux_idx], 0, None)

        # 시간 인덱스
        hist_index = df_feats.index[i0:i0+SEQ_LEN]
        pred_index = df_feats.index[i0+SEQ_LEN:i0+SEQ_LEN+PRED_LEN]

        # 메트릭
        overall, per_var = compute_overall_and_pervar_metrics(y_inv, pred_inv, FEATURES)
        day_result = {
            "date": str(test_start.date()),
            "overall": overall,
            "per_variable": per_var
        }
        day_results.append(day_result)

        # 저장 디렉토리(예측일 단위)
        day_dir = os.path.join(month_dir, f"day_{test_start.strftime('%Y-%m-%d')}")
        ensure_dir(day_dir)

        # 아티팩트 저장(모델은 전역 하나만 저장)
        with open(os.path.join(day_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(day_result, f, ensure_ascii=False, indent=2)

        # CSV
        df_hist = df_feats.iloc[i0:i0+SEQ_LEN].copy()
        df_pred = pd.DataFrame(pred_inv, index=pred_index, columns=FEATURES)
        df_truth= pd.DataFrame(y_inv,    index=pred_index, columns=FEATURES)
        df_hist.to_csv(os.path.join(day_dir, "history_72h.csv"), index_label="Timestamp")
        df_pred.to_csv(os.path.join(day_dir, "forecast_24h.csv"), index_label="Timestamp")
        df_truth.to_csv(os.path.join(day_dir, "truth_24h.csv"),    index_label="Timestamp")

        # 시각화: 한 그래프에 History + Forecast vs Truth
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

        # 전체 변수 한 장(세로 스택)
        fig, axes = plt.subplots(len(FEATURES), 1, figsize=(14, 10), sharex=True)
        for j, col in enumerate(FEATURES):
            axes[j].plot(df_hist.index, df_hist[col], label="History (Actual)")
            axes[j].plot(df_pred.index, df_pred[col], label="Forecast (Predicted)")
            axes[j].plot(df_truth.index, df_truth[col], label="Truth (Actual)", linestyle="--")
            axes[j].set_ylabel(col)
            if j == 0:
                axes[j].legend(loc="upper left")
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"History(72h) + Forecast vs Truth(24h) | {test_start.date()}")
        fig.tight_layout()
        plt.savefig(os.path.join(day_dir, "all_vars_history72h_forecast_truth24h.png"), dpi=150)
        plt.close()

    # 월별 요약 저장
    if day_results:
        vals = [r["overall"] for r in day_results]
        month_mean = {
            "MAE":  float(np.mean([v["MAE"] for v in vals])),
            "RMSE": float(np.mean([v["RMSE"] for v in vals])),
            "sMAPE":float(np.mean([v["sMAPE"] for v in vals]))
        }
        all_results[f"month_{m:02d}"] = {"days": day_results, "mean_overall": month_mean}
        with open(os.path.join(month_dir, "metrics_month.json"), "w", encoding="utf-8") as f:
            json.dump(all_results[f"month_{m:02d}"], f, ensure_ascii=False, indent=2)

# 전체 요약 CSV/JSON
summary_rows = []
for mkey, v in all_results.items():
    mean_o = v["mean_overall"]
    summary_rows.append([mkey, mean_o["MAE"], mean_o["RMSE"], mean_o["sMAPE"]])

if summary_rows:
    df_summary = pd.DataFrame(summary_rows, columns=["month", "MAE", "RMSE", "sMAPE"])
    df_summary.to_csv(os.path.join(SAVE_ROOT, "metrics_v7_summary.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(SAVE_ROOT, "metrics_v7.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

print("=== DONE v7 (single final model, full-span training minus embargo blocks) ===")