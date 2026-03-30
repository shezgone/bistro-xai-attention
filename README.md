# BISTRO-XAI

**Explainable Macroeconomic Forecasting with BISTRO Attention Analysis**

BISTRO(BIS Time-series Regression Oracle) 파운데이션 모델의 Attention을 해부하여,
한국 CPI(소비자물가 전년동월비) 예측의 내부 작동 원리를 설명하는 XAI 파이프라인.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red" />
  <img src="https://img.shields.io/badge/BISTRO-91M%20params-green" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-lightgrey" />
</p>

---

## Overview

| 항목 | 내용 |
|------|------|
| **모델** | BISTRO (91M params, 12 layers, 12 heads, MOIRAI 기반 BIS 파인튜닝) |
| **학습 데이터** | BIS 4,925개 시계열, 63개국, 1970~2024 |
| **타겟** | 한국 CPI YoY (%) |
| **공변량** | 29개 거시경제 변수 (환율, 유가, 금리, 물가 등) |
| **예측 구간** | 2023-01 ~ 2023-12 (12개월, out-of-sample) |
| **Context** | ~120 patches (~10년) |

---

## Methodology

### 2-Stage Attention-Based Feature Selection

기존 PCA/Lasso 기반 변수 선택 대신, **모델 내부 Attention을 직접 활용**하는 방식.

```
Stage 1: 29개 변수 전체 투입 → Attention Matrix 추출 → 변수별 Attention 랭킹
    ↓ 상위 K개 선택 (Uniform Share 기준)
Stage 2: 선택된 10개 변수로 재추론 → 정밀 Attention Map + 예측
```

### Ablation Study (모델 불문 검증)

Attention 랭킹의 실제 기여도를 검증하기 위해, **변수 제거 실험**을 수행.

```
Full model (10 covariates) → Baseline RMSE
변수 하나씩 제거 → ΔRMSE = RMSE(removed) - RMSE(full)
ΔRMSE > 0: 제거 시 성능 악화 → 실제 기여 변수
ΔRMSE ≤ 0: 제거해도 무방 → 중복/노이즈
```

### 2x2 Diagnostic Framework

Attention Score와 Ablation Impact를 교차 분석하여 4가지 유형으로 분류.

```
                    Attention High          Attention Low
                ┌─────────────────────┬─────────────────────┐
  ΔRMSE > 0     │  Confirmed Driver   │  Hidden Contributor │
  (기여함)       │  핵심 예측 변수      │  대체 불가 독자 정보  │
                ├─────────────────────┼─────────────────────┤
  ΔRMSE ≤ 0     │  Spurious Attention │  Irrelevant         │
  (기여 안함)    │  허위 상관 / 중복    │  제거 후보           │
                └─────────────────────┴─────────────────────┘
```

> **핵심 인사이트**: Attention ≠ Importance. 높은 Attention이 반드시 예측 기여를 의미하지 않으며,
> 낮은 Attention의 변수가 오히려 핵심일 수 있다.

### Temporal Attention Analysis

모델이 **과거 어느 시점을 주목하는지** 분석하여 시차 구조를 파악.

- **Self-Attention (CPI→CPI)**: 최근 패치에 집중 → AR(자기회귀) 패턴
- **Cross-Attention (CPI→Covariate)**: 변수별로 다른 시차 구조
  - Recent Focus: 최근 12개월 집중 (동행/단기 선행 지표)
  - Distant Focus: 먼 과거 집중 (장기 수준 참조)
  - Diffuse: 균등 분포 (추세 정보 활용)

---

## Project Structure

```
bistro-xai/
├── app.py                      # Streamlit 대시보드 (8탭)
├── bistro_core.py              # BISTROConfig, AttentionAnalyzer 등 핵심 클래스
├── bistro_runner_30var.py      # 2-Stage 추론 파이프라인
├── ablation_study.py           # 변수 제거 + 순차 추가 실험
├── data_collector.py           # FRED API + BIS 데이터 수집
├── export_pdf.py               # 대시보드 결과 → PDF 리포트 생성
├── BISTRO_XAI_Walkthrough.ipynb  # 재현용 Jupyter Notebook
├── requirements.txt            # Streamlit 환경 의존성
└── data/
    ├── macro_panel_daily.csv       # 일별 패널 (29변수, 2003~2025)
    ├── stage1_screening.npz        # Stage 1 결과 (29변수 Attention 랭킹)
    ├── real_inference_results.npz  # Stage 2 결과 (예측 + Attention Map)
    └── ablation_results.npz        # Ablation 실험 결과
```

---

## Prerequisites

이 프로젝트는 **두 개의 Python 환경**을 사용합니다.

| 환경 | Python | 용도 | 주요 패키지 |
|------|--------|------|-------------|
| `.venv/` | 3.14 | Streamlit 대시보드, PDF 생성 | streamlit, plotly, fpdf2, kaleido |
| `.venv-bistro/` | 3.11 | BISTRO 모델 추론 | uni2ts, torch, gluonts |

> BISTRO 모델은 uni2ts(MOIRAI) 프레임워크 기반이며, Python 3.11이 필요합니다.
> 대시보드는 사전 계산된 .npz 결과만 로딩하므로 모델 없이도 실행 가능합니다.

### BISTRO 모델 설치

```bash
# 1. bistro 레포 클론
git lfs install
git clone https://github.com/bis-med-it/bistro.git /tmp/bistro-repo

# 2. Python 3.11 환경 생성
python3.11 -m venv .venv-bistro
.venv-bistro/bin/pip install -r /tmp/bistro-repo/requirements.txt
```

### 대시보드 환경 설치

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# PDF 생성을 사용하려면:
.venv/bin/pip install fpdf2 kaleido
```

---

## Usage

### 1. Streamlit Dashboard

```bash
.venv/bin/streamlit run app.py
# → http://localhost:8501
```

8개 탭으로 구성:

| 탭 | 내용 |
|----|------|
| Forecast Results | BISTRO 예측 vs 실제 CPI (2021~2025), AR(1) 벤치마크, 90% CI |
| Feature Selection | Stage 1 Attention 스크리닝, 변수별 원래 빈도(Daily/Monthly) |
| Cross-Variate Heatmap | 변수 간 Attention Matrix (Layer 12), zmax=60% |
| Variable Importance | CPI 타겟 행 Attention 분포 (자기참조 ~52%) |
| Temporal Patterns | 자기참조/공변량 시차 구조, 전체 heatmap, lag 유형 분류 |
| Layer Analysis | 12개 레이어별 Attention 변화 추적 |
| 2x2 Diagnostic | Attention vs Ablation 교차 진단 (4사분면) |
| Ablation & Incremental | 변수 제거 ΔRMSE + 순차 추가 RMSE 커브 |

### 2. Jupyter Notebook

대시보드와 동일한 분석을 코드 레벨에서 재현:

```bash
.venv/bin/jupyter notebook BISTRO_XAI_Walkthrough.ipynb
```

노트북 구조:
1. 데이터 확인 (29변수 패널)
2. Stage 1 결과 (Attention 스크리닝)
3. Stage 2 예측 결과 + Forecast 시각화
4. Cross-Variate Attention Heatmap (Layer 12)
5. Ablation Study (ΔRMSE)
6. 2x2 Diagnostic
7. Incremental Analysis
8. (Optional) 파이프라인 재실행 가이드

> 노트북은 사전 계산된 `.npz` 파일을 로딩합니다.
> 모델 재실행 없이 결과 탐색이 가능합니다.

### 3. PDF Report

```bash
.venv/bin/python3 export_pdf.py
# → BISTRO_XAI_Report.pdf (11페이지, Landscape A4)
```

### 4. Pipeline Re-execution (Optional)

모델을 직접 실행하여 결과를 재생성하려면:

```bash
# Step 1: 데이터 수집 (FRED API key 필요)
.venv-bistro/bin/python3 data_collector.py

# Step 2: 2-Stage 추론
.venv-bistro/bin/python3 bistro_runner_30var.py --top-k 10

# Step 3: Ablation Study
.venv-bistro/bin/python3 ablation_study.py

# Step 4: 대시보드 실행
.venv/bin/streamlit run app.py
```

---

## Key Findings

### Forecast Performance (2023, Out-of-Sample)

| 모델 | RMSE | 비고 |
|------|------|------|
| **BISTRO** | **1.19pp** | 방향성(하락) 포착, 급락 속도 미포착 |
| AR(1) | 1.59pp | 5%에 고정, 변화 없음 |

- 2023~24는 BIS가 학습에서 **의도적으로 제외한 OOS 구간**
- BISTRO는 AR(1) 대비 **25% 낮은 오차**
- 다만 2023년 급격한 디스인플레이션(5%→2.3%)의 속도는 미포착

### Attention Analysis

- **자기참조(Self-Attention) ~52%**: BinaryAttentionBias에 의한 구조적 특성
- **공변량 간 Attention 분포**: 나머지 ~48%가 10개 공변량에 비교적 균등 분배 (각 3~5%)
- **Temporal 패턴**: 자기참조는 최근 패치에 집중 (AR 패턴), 일부 공변량은 과거 패치에 집중 (장기 레벨 참조)

### Attention ≠ Importance

- CNY_USD: 높은 Attention + 낮은 ΔRMSE → **Spurious Attention** (장기 레벨 참조용)
- JP_Interbank3M: 낮은 Attention + 높은 ΔRMSE → **Hidden Contributor** (대체 불가 정보)

---

## Model Architecture

```
BISTRO (91M parameters)
├── Patch Embedding (patch_size=32, daily)
├── Transformer Encoder × 12 layers
│   ├── Multi-Head Attention (12 heads, d_model=768, head_dim=64)
│   │   ├── RoPE Positional Encoding
│   │   └── BinaryAttentionBias (same-var vs cross-var 구분)
│   ├── Residual Connection + LayerNorm
│   ├── SwiGLU Feed-Forward Network
│   └── Residual Connection + LayerNorm
└── Mixture Distribution Head (Student-t + LogNormal)
```

- **BinaryAttentionBias**: 같은 변수 토큰 vs 다른 변수 토큰을 구분하는 학습된 bias → 자기참조 ~52% 설명
- **RoPE**: 시간 순서 정보 인코딩 → Temporal Attention 패턴 형성
- **Residual Connection**: 최종 출력은 마지막 레이어(Layer 12) attention이 지배적

---

## References

- **BISTRO Paper**: Koyuncu, Kwon, Lombardi, Shin, Perez-Cruz. "BISTRO: A General-Purpose Time Series Model for Macroeconomic Forecasting." *BIS Quarterly Review*, March 2026.
- **BISTRO GitHub**: [github.com/bis-med-it/bistro](https://github.com/bis-med-it/bistro)
- **MOIRAI (Base Model)**: Woo et al. "Unified Training of Universal Time Series Forecasting Transformers." Salesforce, 2024.

---

## License

Apache 2.0
