import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------
# 1. 데이터 로드 및 전처리
# -------------------------------
file_path = r"path"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"error: '{file_path}' path error.")
    raise SystemExit

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

features = ['Temperature_C', 'Salinity_PSU', 'pH', 'DO_mgL', 'Illuminance_Lux']
data = df[features].values

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# -------------------------------
# 2. 시퀀스 데이터 생성
# -------------------------------
def create_sequences(data, seq_length, pred_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + pred_length)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 96
PRED_LENGTH = 24
X, y = create_sequences(data_scaled, SEQ_LENGTH, PRED_LENGTH)

# -------------------------------
# 3. 데이터셋 및 데이터로더
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

BATCH_SIZE = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# 4. PatchTST 모델 정의 (수정 포함)
# -------------------------------
class PatchTST(nn.Module):
    def __init__(self, n_features, seq_len, pred_len, patch_len, stride, d_model, n_heads, n_layers):
        super(PatchTST, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        # 패치 수 계산: padding=stride로 가정(마지막에 stride만큼 패딩)
        self.padding = stride
        self.n_patches = int(((seq_len + self.padding) - patch_len) / stride + 1)

        # 1D 복제 패딩 (마지막 시간축에만 패딩)
        self.padding_patch = nn.ReplicationPad1d((0, self.padding))

        # 패치(길이 patch_len)를 d_model로 임베딩
        self.input_embedding = nn.Linear(patch_len, d_model)

        # 위치 인코딩 (n_patches 길이와 d_model 차원에 맞춤)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model))

        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 출력 레이어: 채널별로 (n_patches*d_model) -> pred_len
        self.output_layer = nn.Linear(d_model * self.n_patches, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        B = x.shape[0]

        # 채널 독립 처리를 위해 [B, C, T]
        x = x.permute(0, 2, 1)  # [B, C, T]

        # 패딩 후 패치 추출: [B, C, n_patches, patch_len]
        x = self.padding_patch(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # 패치 임베딩: [B, C, n_patches, d_model]
        x = self.input_embedding(x)

        # 트랜스포머 인코더 입력 형태로 변경: [B*C, n_patches, d_model]
        x = x.reshape(B * self.n_features, self.n_patches, -1)

        # 위치 인코딩 더하기
        x = x + self.positional_embedding

        # 인코딩
        x = self.transformer_encoder(x)  # [B*C, n_patches, d_model]

        # 채널 차원 복원 후 마지막 차원 펼치기: [B, C, n_patches*d_model]
        x = x.reshape(B, self.n_features, self.n_patches * x.shape[-1])

        # 채널별 선형 변환 → [B, C, pred_len]
        x = self.output_layer(x)

        # 최종 출력: [B, pred_len, C]
        out = x.permute(0, 2, 1)
        return out

# -------------------------------
# 5. 모델 학습
# -------------------------------
model = PatchTST(
    n_features=len(features),
    seq_len=SEQ_LENGTH,
    pred_len=PRED_LENGTH,
    patch_len=16,
    stride=8,
    d_model=128,
    n_heads=8,
    n_layers=3
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# (선택) 입출력 형태 빠른 체크
with torch.no_grad():
    xb, yb = next(iter(train_loader))
    yhat = model(xb)
    print(f"[Shape Check] X: {xb.shape} -> y_hat: {yhat.shape} (기대: [B, {PRED_LENGTH}, {len(features)}])")

EPOCHS = 10  # 실제 사용 시 더 큰 에폭 권장
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS}, "
        f"Train Loss: {train_loss/len(train_loader):.4f}, "
        f"Val Loss: {val_loss/len(val_loader):.4f}"
    )

print("\n--- 학습 완료 ---")

# -------------------------------
# 6. 모델 성능 평가 (지표)
# -------------------------------
model.eval()
all_predictions = []
all_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

# 배치 합치기
all_predictions = np.concatenate(all_predictions, axis=0)  # [N, pred_len, C]
all_targets = np.concatenate(all_targets, axis=0)          # [N, pred_len, C]

# (N * pred_len, C)로 변환
num_samples = all_predictions.shape[0]
all_predictions_reshaped = all_predictions.reshape(num_samples * PRED_LENGTH, len(features))
all_targets_reshaped = all_targets.reshape(num_samples * PRED_LENGTH, len(features))

print("\n--- 모델 성능 평가 ---")
for i, feature in enumerate(features):
    mse = mean_squared_error(all_targets_reshaped[:, i], all_predictions_reshaped[:, i])
    mae = mean_absolute_error(all_targets_reshaped[:, i], all_predictions_reshaped[:, i])
    r2 = r2_score(all_targets_reshaped[:, i], all_predictions_reshaped[:, i])
    print(f"[{feature}]")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}\n")

# -------------------------------
# 7. 모델 저장
# -------------------------------
MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "patchtst_model.pth")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"모델 '{MODEL_SAVE_PATH}' 에 저장.")

# -------------------------------
# 8. 저장된 모델 로드 및 사용 예시
# -------------------------------
loaded_model = PatchTST(
    n_features=len(features),
    seq_len=SEQ_LENGTH,
    pred_len=PRED_LENGTH,
    patch_len=16,
    stride=8,
    d_model=128,
    n_heads=8,
    n_layers=3
)
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model.eval()
print(f"\n '{MODEL_SAVE_PATH}' 모델 불러오기 완료.")

with torch.no_grad():
    sample_x, sample_y = next(iter(test_loader))
    prediction = loaded_model(sample_x)
    print(f"\n로드된 모델을 사용한 예측 결과 (첫 번째 샘플): \n{prediction[0].numpy().round(2)}")
    print(f"실제 값 (첫 번째 샘플): \n{sample_y[0].numpy().round(2)}")
