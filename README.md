# BISTRO-XAI: Attention-Based Explainability for Macro Forecasting

BISTRO 파운데이션 모델(91M params, MOIRAI 아키텍처)의 **어텐션 맵**을 활용해
한국 CPI 예측에서 어떤 거시경제 변수가 실제로 기여하는지 식별하는 XAI 프레임워크.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red" />
  <img src="https://img.shields.io/badge/BISTRO-91M%20params-green" />
</p>

---

## 핵심 아이디어

트랜스포머 어텐션 가중치만으로는 변수 중요도를 판단할 수 없다.
어텐션이 높아도 예측력에 해로운 변수(Spurious Attention)가 존재하기 때문이다.
BISTRO-XAI는 **2x2 진단 프레임워크**로 이를 구분한다:

```
                    Attention High          Attention Low
                ┌─────────────────────┬─────────────────────┐
  dRMSE > 0     │  Confirmed Driver   │  Hidden Contributor │
  (제거 시 악화)  │  핵심 예측 변수      │  대체 불가 독자 정보  │
                ├─────────────────────┼─────────────────────┤
  dRMSE ≤ 0     │  Spurious Attention │  Irrelevant         │
  (제거해도 무방) │  허위 상관 / 중복    │  제거 후보           │
                └─────────────────────┴─────────────────────┘
```

> **핵심 인사이트**: Attention ≠ Importance.

---

## 변수 선택 파이프라인

```
Stage 0: 288개 FRED 변수 전수 스크리닝 (CTX=10, 어텐션 기준 랭킹)
    ↓ Top 25
Stage 1: CTX=120 풀 컨텍스트 재추론 + Leave-one-out ablation
    ↓ Harmful 2개 제거 (BR_CPI, BR_DiscountRate)
Incremental Addition: Greedy search by attention rank → 최적 N=18
```

Stage 0에서 단축 컨텍스트로 빠르게 전수 스크리닝한 뒤, Stage 1에서 풀 컨텍스트로 검증하는 2단계 구조.
Stage 0의 false positive(BR_CPI, BR_DiscountRate)를 Stage 1 ablation이 걸러낸다.

---

## 주요 결과

### Forecast Performance (Out-of-Sample)

| 구간 | BISTRO RMSE | AR(1) RMSE | 개선율 |
|------|-------------|------------|--------|
| 2023 | 1.16pp | 1.59pp | -27% |
| 2024 | 0.81pp | 0.92pp | -12% |

### Key Findings

- **AUD/USD**가 지배적 드라이버 — 12개 헤드 중 4개가 전담
- **변수가 많다고 좋지 않음** — 18개 최적 (18 > 23 > 288)
- **10년 컨텍스트 >> 3년** — CTX=120이 CTX=36보다 일관되게 우위
- **자기참조(Self-Attention) ~52%** — BinaryAttentionBias에 의한 구조적 특성

---

## 모델 사양

| 항목 | 값 |
|------|-----|
| 아키텍처 | MOIRAI (uni2ts), BIS 파인튜닝 |
| 파라미터 | 91M |
| 레이어/헤드 | 12 / 12 |
| d_model | 768 |
| 학습 데이터 | BIS 63개국, 4,925 시계열 (1970~2024) |
| 컨텍스트 | 120개월 (10년) |
| 예측 구간 | 12개월 |
| 패치 크기 | 32일 |
| 샘플 수 | 100 |

---

## 프로젝트 구조

```
bistro-xai/
│
│  # ── 대시보드 & 리포트 ──
├── app.py                       # Streamlit 대시보드 (어텐션 맵 탐색기)
├── bistro_core.py               # 공유 도메인 클래스 (PRESETS, TIER_LABELS, AttentionAnalyzer)
├── export_pdf_v2.py             # PDF 리포트 (파이프라인 결과)
├── export_pdf_v3.py             # PDF 리포트 확장 (2024 예측 포함)
│
│  # ── 데이터 수집 & 유틸리티 ──
├── data_collector.py            # FRED API + BIS 데이터 수집 → macro_panel.csv
├── inference_util.py            # AR(1) 베이스라인 유틸리티
├── preprocessing_util.py        # 일별→월별 집계, 신뢰구간 계산
│
│  # ── 변수 선택 파이프라인 ──
├── tournament_config.py         # 288개 FRED 시리즈 정의 (8개 카테고리)
├── tournament_runner.py         # 다단계 토너먼트 변수 선택 오케스트레이터
├── run_stage0_screening.py      # Stage 0: 전수 스크리닝 (CTX=10)
│
│  # ── 추론 실험 스크립트 ──
├── run_2024_inference.py        # 2024 예측 (최적 18변수)
├── run_9var_inference.py        # 9변수 서브셋 추론
├── run_tournament_winner.py     # 토너먼트 우승 변수 조합 추론
├── run_univariate_inference.py  # 단변량 베이스라인
├── run_ctx36_comparison.py      # 컨텍스트 윈도우 비교 (CTX=36 vs 120)
├── run_recent_attn_inference.py # 최근 어텐션 기반 변수 선택
│
├── mindmap_bistro_xai.md        # 프로젝트 아키텍처 Mermaid 마인드맵
└── data/                        # 패널 CSV + 추론 결과 (.npz, git 제외)
```

---

## 실행 방법

### 환경 설정

이 프로젝트는 **두 개의 Python 환경**을 사용합니다. 혼용 금지.

| 환경 | Python | 용도 |
|------|--------|------|
| `.venv/` | 3.14 | 대시보드, PDF, 시각화 |
| `.venv-bistro/` | 3.11 | BISTRO 추론 (uni2ts, torch) |

```bash
# 대시보드 환경
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# BISTRO 추론 환경 (Python 3.11 필요)
python3.11 -m venv .venv-bistro
git lfs install
git clone https://github.com/bis-med-it/bistro.git /tmp/bistro-repo
.venv-bistro/bin/pip install -r /tmp/bistro-repo/requirements.txt
```

### 대시보드

```bash
source .venv/bin/activate
streamlit run app.py
# → http://localhost:8501
```

### BISTRO 추론

```bash
source .venv-bistro/bin/activate

# Stage 0: 전수 스크리닝
python run_stage0_screening.py

# 최적 18변수 2024 예측
python run_2024_inference.py
```

### PDF 리포트

```bash
source .venv/bin/activate
python export_pdf_v3.py
```

---

## References

- **BISTRO Paper**: Koyuncu, Kwon, Lombardi, Shin, Perez-Cruz. "BISTRO: A General-Purpose Time Series Model for Macroeconomic Forecasting." *BIS Quarterly Review*, March 2026.
- **BISTRO GitHub**: [github.com/bis-med-it/bistro](https://github.com/bis-med-it/bistro)
- **MOIRAI (Base Model)**: Woo et al. "Unified Training of Universal Time Series Forecasting Transformers." Salesforce, 2024.
