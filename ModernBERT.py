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
SEQ_LEN  = 432   # 72h (10분 간격)
PRED_LEN = 144   # 24h

# Embargo(예측일 기준 앞뒤 12시간)
EMBARGO_BEFORE = pd.Timedelta(hours=12)
EMBARGO_AFTER  = pd.Timedelta(hours=12)

# 학습 하이퍼파라미터
BATCH_SIZE = 16
EPOCHS     = 6
LR         = 5e-5
PATIENCE   = 2

# 기타
VAL_DAYS_FOR_EARLYSTOP = 7  # (전역 valid 미사용)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hugging Face 모델
MODEL_NAME = "answerdotai/ModernBERT-base"  
FREEZE_BACKBONE = False
POOLING = "last"                            # "last" or "mean"

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
    """월별 예측일(자정 기준) 선택 (1월은 10일부터)
    look-back(72h) 및 예측(24h)이 전체 범위에 완전히 들어오는 날짜 중 seed=42로 랜덤 추출
    """
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
# Dataset
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
# ModernBERT 래퍼
# =========================
from transformers import AutoConfig, AutoModel

class ModernBERTTimeModel(nn.Module):
    def __init__(self, model_name: str, input_dim: int, pred_len: int, output_dim: int,
                 freeze_backbone: bool=False, pooling: str="last"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.pooling = pooling

        # ModernBERT는 trust_remote_code 불필요. SDPA 강제 지정.
        try:
            self.cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
            self.backbone = AutoModel.from_pretrained(
                model_name,
                config=self.cfg,
                trust_remote_code=False,
                attn_implementation="sdpa",
                torch_dtype=torch.float32  
            )
        except Exception as e:
            # 폴백: BERT
            print(f"[WARN] '{model_name}' 로드 실패: {e}\n"
                  f"       'bert-base-uncased'로 폴백합니다(SDPA).")
            model_name = "bert-base-uncased"
            self.cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
            self.backbone = AutoModel.from_pretrained(
                model_name,
                config=self.cfg,
                trust_remote_code=False,
                attn_implementation="sdpa",
                torch_dtype=torch.float32
            )

        self.hidden = self.cfg.hidden_size

        # 포지션 길이 체크(ModernBERT는 8192, BERT는 512가 일반적)
        max_pos = getattr(self.cfg, "max_position_embeddings", 512)
        if SEQ_LEN > max_pos:
            raise ValueError(f"SEQ_LEN {SEQ_LEN} > max_position_embeddings {max_pos} "
                             f"(백본: {model_name}). SEQ_LEN을 줄이기 or  긴 시퀀스 모델변경 필요.")

        # 수치 특징 → 히든 투영, pooled → (H×F) 회귀
        self.proj = nn.Linear(self.input_dim, self.hidden)
        self.head = nn.Linear(self.hidden, self.pred_len * self.output_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (B,T,F)
        B, T, F = x.shape
        embeds = self.proj(x)  # (B,T,H)
        attn_mask = torch.ones(B, T, dtype=torch.long, device=x.device)
        pos_ids = torch.arange(T, dtype=torch.long, device=x.device).unsqueeze(0)  # (1,T)
        out = self.backbone(inputs_embeds=embeds,
                            attention_mask=attn_mask,
                            position_ids=pos_ids)
        hs = out.last_hidden_state  # (B,T,H)
        if self.pooling == "mean":
            pooled = hs.mean(dim=1)
        else:
            pooled = hs[:, -1, :]
        y = self.head(pooled).view(B, self.pred_len, self.output_dim)
        return y

# =========================
# 윈도우/구간 유틸
# =========================
def merge_intervals(intervals):
    if not intervals: return []
    s = sorted(intervals, key=lambda x: x[0])
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1] + pd.Timedelta(0):
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(x[0], x[1]) for x in out]

def subtract_intervals(full_start, full_end, blocks):
    """[full_start, full_end]에서 blocks(금지 구간) 합집합을 뺀 허용 구간 리스트 반환"""
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
df = pd.read_csv(FILE_PATH, parse_dates=["Timestamp"])
assert "Timestamp" in df.columns, "CSV에 Timestamp 열 필요"

df = df.set_index("Timestamp").sort_index()
timestamps = df.index
assert set(FEATURES).issubset(df.columns), "FEATURES가 CSV에 없음"

DT = (timestamps[1] - timestamps[0])
assert DT == pd.Timedelta(minutes=10), f"샘플 간격 10분 아님: {DT}"

df_feats = df[FEATURES].copy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

YEAR = timestamps[0].year
target_days_by_month = pick_target_days(df_feats.index, YEAR, per_month=5)
ensure_dir(SAVE_ROOT)
with open(os.path.join(SAVE_ROOT, "chosen_target_days.json"), "w", encoding="utf-8") as f:
    json.dump({f"{m:02d}": [str(d) for d in ds] for m, ds in target_days_by_month.items()},
              f, indent=2, ensure_ascii=False)

# ===== 훈련 허용 구간: 전체에서 테스트일의 [start-12h, end+12h] 합집합을 제외 =====
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
    raise RuntimeError("훈련 허용 구간이 비어있음. 데이터/설정을 확인필요.")
df_train_for_scaler = pd.concat(df_train_for_scaler_list)
scaler = StandardScaler().fit(df_train_for_scaler.values)

# 전체 변환
data_scaled = pd.DataFrame(
    scaler.transform(df_feats.values),
    index=df_feats.index,
    columns=FEATURES
)

# 훈련 윈도우(여러 허용 구간)
train_idxs = build_window_indices_multi(
    df_feats.index,
    allowed_ranges=allowed_train_ranges,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    stride=1
)
if len(train_idxs) == 0:
    raise RuntimeError("훈련 윈도우가 0개. allowed_train_ranges 체크.")

train_ds = IndexWindowDataset(data_scaled.values, train_idxs, SEQ_LEN, PRED_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ===== 모델/최적화 =====
model = ModernBERTTimeModel(
    MODEL_NAME,
    input_dim=len(FEATURES),
    pred_len=PRED_LEN,
    output_dim=len(FEATURES),
    freeze_backbone=FREEZE_BACKBONE,
    pooling=POOLING
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  # AMP(optional)

best_state = None
best_loss = np.inf
patience_left = PATIENCE

for ep in range(1, EPOCHS+1):
    model.train()
    tr_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred = model(x)
            loss = criterion(pred, y)
        scaler_amp.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        tr_loss += loss.item()
    tr_loss /= max(1, len(train_loader))

    improved = tr_loss < best_loss
    if improved:
        best_loss = tr_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_left = PATIENCE
    else:
        patience_left -= 1
        if patience_left <= 0:
            pass

    print(f"[ModernBERT] Ep {ep}/{EPOCHS} | train {tr_loss:.6f} | best {best_loss:.6f}")

if best_state is not None:
    model.load_state_dict(best_state)

# 저장(최종 모델/스케일러)
joblib.dump(scaler, os.path.join(SAVE_ROOT, "scaler_final_modernbert.bin"))
torch.save(model.state_dict(), os.path.join(SAVE_ROOT, "model_final_modernbert.pth"))

# =========================
# 테스트(월별 5일 유지)
# =========================
all_results = {}
ensure_dir(SAVE_ROOT)

lux_idx = FEATURES.index("Illuminance_Lux")

for m in range(1, 13):
    days = target_days_by_month.get(m, [])
    if not days:
        continue

    month_dir = os.path.join(SAVE_ROOT, f"month_{m:02d}_ModernBERT")
    ensure_dir(month_dir)

    day_results = []
    for day_start in days:
        test_start = pd.Timestamp(day_start)
        test_end   = test_start + pd.Timedelta(hours=24) - DT
        hist_start = test_start - pd.Timedelta(hours=72)

        # 인덱스 정합성 확인(10분 격자 가정)
        i0 = df_feats.index.get_indexer([hist_start])[0]
        assert df_feats.index[i0] == hist_start, "hist_start 인덱스 오류."

        # 단일 윈도우 예측
        x_input = torch.tensor(
            data_scaled.values[i0:i0+SEQ_LEN], dtype=torch.float32
        ).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                yhat_scaled = model(x_input).squeeze(0).cpu().numpy()  # (H,F)

        # truth (스케일된 원본 같은 구간)
        y_scaled = data_scaled.values[i0+SEQ_LEN:i0+SEQ_LEN+PRED_LEN]

        # 역스케일
        pred_inv = scaler.inverse_transform(yhat_scaled)
        y_inv    = scaler.inverse_transform(y_scaled)
        # Lux 음수 하한 0
        pred_inv[:, lux_idx] = np.clip(pred_inv[:, lux_idx], 0, None)

        # 인덱스
        pred_index = df_feats.index[i0+SEQ_LEN:i0+SEQ_LEN+PRED_LEN]
        df_hist = df_feats.iloc[i0:i0+SEQ_LEN].copy()
        df_pred = pd.DataFrame(pred_inv, index=pred_index, columns=FEATURES)
        df_truth= pd.DataFrame(y_inv,    index=pred_index, columns=FEATURES)

        # 메트릭
        overall, per_var = compute_overall_and_pervar_metrics(y_inv, pred_inv, FEATURES)
        day_result = {"date": str(test_start.date()), "overall": overall, "per_variable": per_var}
        day_results.append(day_result)

        # 저장
        day_dir = os.path.join(month_dir, f"day_{test_start.strftime('%Y-%m-%d')}")
        ensure_dir(day_dir)
        with open(os.path.join(day_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(day_result, f, ensure_ascii=False, indent=2)
        df_hist.to_csv(os.path.join(day_dir, "history_72h.csv"), index_label="Timestamp")
        df_pred.to_csv(os.path.join(day_dir, "forecast_24h.csv"), index_label="Timestamp")
        df_truth.to_csv(os.path.join(day_dir, "truth_24h.csv"), index_label="Timestamp")

        # 시각화 (변수별)
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

        # 전체 변수 한 장
        fig, axes = plt.subplots(len(FEATURES), 1, figsize=(14, 10), sharex=True)
        for j, col in enumerate(FEATURES):
            axes[j].plot(df_hist.index, df_hist[col], label="History (Actual)")
            axes[j].plot(df_pred.index, df_pred[col], label="Forecast (Predicted)")
            axes[j].plot(df_truth.index, df_truth[col], label="Truth (Actual)", linestyle="--")
            axes[j].set_ylabel(col)
            if j == 0:
                axes[j].legend(loc="upper left")
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"ModernBERT: History(72h) + Forecast vs Truth(24h) | {test_start.date()}")
        fig.tight_layout()
        plt.savefig(os.path.join(month_dir, f"all_vars_{test_start.strftime('%Y-%m-%d')}.png"), dpi=150)
        plt.close()

    # 월별 요약
    if day_results:
        vals = [r["overall"] for r in day_results]
        month_mean = {
            "MAE":  float(np.mean([v["MAE"] for v in vals])),
            "RMSE": float(np.mean([v["RMSE"] for v in vals])),
            "sMAPE":float(np.mean([v["sMAPE"] for v in vals]))
        }
        with open(os.path.join(month_dir, "metrics_month.json"), "w", encoding="utf-8") as f:
            json.dump({"days": day_results, "mean_overall": month_mean}, f, ensure_ascii=False, indent=2)

print("=== DONE v7 (ModernBERT single final model, SDPA) ===")
