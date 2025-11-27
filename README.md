<h1>📦 Model Scripts Overview </h1>

본 프로젝트는 POSEIDON project 중 일부분 입니다.

<h3>목표</h3>
동일한 해양 환경 데이터(Temperature, Salinity, pH, DO, Illuminance)를 기반으로
72시간 입력 → 24시간 예측이라는 동일한 목표를 갖고 있습니다.
아래 네 개의 스크립트는 서로 다른 모델 구조를 사용하여 동일한 목적을 수행합니다.

<h3>🌊 1. NeoBERT.py — ModernBERT 기반 시계열 변환 모델</h3>


ModernBERT를 시계열 입력에 맞게 직접 변환하여 사용하는 Transformer 기반 모델입니다.

수치 시계열을 Linear Projection을 통해 BERT encoder에 입력하는 구조입니다.

Position IDs를 사용하여 시간 정보를 자연스럽게 반영합니다.

Backbone freeze 기능과 pooling 방식 선택(last / mean)을 지원합니다.

긴 시퀀스 처리에서 강점을 보이는 모델입니다.


<h3>⚡ 2. Transformer.py — PatchTST 기반 Pure Transformer 모델</h3>


PatchTST 구조를 구현한 Transformer 기반 예측 모델입니다.

시계열을 패치 단위로 분할하여 Transformer가 더 효율적으로 학습하도록 구성되어 있습니다.

Patch Embedding → Positional Embedding → Transformer Encoder 순으로 처리합니다.

입력 길이를 압축하여 성능과 효율성을 모두 확보한 구조입니다.


<h3>🌙 3. lstm.py — 전통적 LSTM 기반 Forecasting 모델</h3>

파일 특징입니다

가장 기본적인 시계열 모델인 LSTM 구조를 사용한 예측 모델입니다.

마지막 hidden state를 활용하여 24시간 예측을 수행합니다.

월별 타겟일 선정 및 ±12시간 embargo 등 시간 기반 데이터 분리 정책을 포함합니다.

가볍고 안정적이어서 baseline 비교에 적합합니다.


<h3>🧩 4. various.py — LSTM/GRU/TCN/DLinear/MLP 통합 모델 스크립트</h3>


여러 모델을 하나의 프레임워크에서 선택적으로 실행할 수 있는 통합 스크립트입니다.

지원 모델은 LSTM, GRU, TCN, DLinear, MLP입니다.

동일한 데이터/학습/평가 파이프라인을 공유하므로 모델 간 공정한 비교가 가능합니다.

다양한 시계열 모델을 실험하고 벤치마크할 수 있는 구조입니다.

## 👥 Contributors

- **민선홍 (Sun-hong Min)**
- **최문성 (Moonseong Choi)**
