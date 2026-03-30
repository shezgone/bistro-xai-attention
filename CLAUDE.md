# CLAUDE.md

## Rules

- 기술적 사실(모델 구조, 파라미터 수, 처리 방식, 성능 수치 등)은 추측하지 말 것. 불확실하면 코드/모델/문서를 먼저 확인한 뒤 답변할 것.
- "아마 ~일 것입니다"식 추측 금지. 모르면 "확인하겠습니다"라고 먼저 말하고 도구로 확인.

## Project Structure

- `.venv/` — Python 3.14, Streamlit 대시보드용
- `.venv-bistro/` — Python 3.11, uni2ts/torch (BISTRO 추론용)
- `app.py` — Streamlit 대시보드
- `bistro_runner_30var.py` — 2-Stage 추론 파이프라인
- `ablation_study.py` — 변수 제거 실험
- `data_collector.py` — FRED/BIS 데이터 수집
- `data/` — 패널 데이터 + 추론 결과 (.npz)
