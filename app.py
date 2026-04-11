"""
BISTRO-XAI | Attention Map Explorer
=====================================
Streamlit + Plotly 기반 인터랙티브 Attention 시각화 대시보드.
실제 BISTRO 추론 결과(data/real_inference_results.npz)가 있으면 자동으로 로딩.

실행:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# ============================================================
# Variable Annotations (한글 주석)
# ============================================================
VAR_LABEL = {
    # Target
    "CPI_KR_YoY":       "한국 CPI YoY",
    # Optimal 18
    "AUD_USD":           "호주달러/USD 환율",
    "CN_Interbank3M":    "중국 3개월 은행간금리",
    "US_UnempRate":      "미국 실업률",
    "JP_REER":           "일본 실질실효환율(BIS)",
    "JP_Interbank3M":    "일본 3개월 은행간금리",
    "JP_CoreCPI":        "일본 근원CPI",
    "KC_FSI":            "캔자스시티 금융스트레스지수",
    "KR_MfgProd":        "한국 제조업생산지수",
    "Pork":              "돼지고기 국제가격",
    "US_NFP":            "미국 비농업취업자수",
    "US_TradeTransEmp":  "미국 무역/운송업 고용",
    "THB_USD":           "태국 바트/USD 환율",
    "PPI_CopperNickel":  "구리/니켈 생산자물가",
    "CN_PPI":            "중국 생산자물가지수",
    "US_Mortgage15Y":    "미국 15년 모기지금리",
    "UK_10Y_Bond":       "영국 10년 국채금리",
    "US_ExportPI":       "미국 수출물가지수",
    "US_DepInstCredit":  "미국 예금기관 신용",
    # Stage 1 harmful / others
    "BR_CPI":            "브라질 CPI",
    "BR_DiscountRate":   "브라질 기준금리",
    "BRL_USD":           "브라질 헤알/USD",
    "PPI_DeepSeaFrt":    "원양해운 생산자물가",
    "KR_PPI_Energy":     "한국 에너지 PPI",
    "Energy_Idx":        "IMF 에너지가격지수",
    "PPI_Metals":        "금속 생산자물가",
    # Legacy 29-var
    "CNY_USD":           "위안/USD 환율",
    "US_M2":             "미국 M2 통화량 YoY",
    "US_ConsConf":       "미시간 소비자심리",
    "US_Unemp":          "미국 실업률(구)",
    "US_YieldSpread":    "미국 10Y-2Y 스프레드",
    "KR_Exports":        "한국 수출액 YoY",
    "VIX":               "S&P500 변동성지수",
    "China_CPI":         "중국 CPI YoY",
    "KR_Imports":        "한국 수입액 YoY",
    "Oil_Brent":         "브렌트유 가격",
    "NatGas_HH":         "천연가스(Henry Hub)",
    "DXY_Broad":         "달러 무역가중지수",
    "FedFunds":          "미국 연방기금금리",
    "Rate_KR":           "한국 기준금리",
    "Rate_ECB":          "ECB 기준금리",
    "Oil_WTI":           "WTI 원유 가격",
    "USD_KRW":           "원/달러 환율",
    "JPY_USD":           "엔/달러 환율",
    "Copper":            "구리 국제가격",
    "Wheat":             "밀 국제가격",
    "Corn":              "옥수수 국제가격",
    "US_CoreCPI_idx":    "미국 근원CPI YoY",
    "US_PPI":            "미국 PPI YoY",
    "KR_Interbank3M":    "한국 3개월 금리",
    "KR_LongRate":       "한국 장기금리",
    "CPI_US_YoY":        "미국 CPI YoY(BIS)",
    "CPI_XM_YoY":        "유로존 CPI YoY(BIS)",
    # Tournament extras
    "Moodys_BAA":        "무디스 BAA 회사채금리",
    "TED_Spread":        "TED 스프레드(은행간 리스크)",
    "US_MonBase":        "미국 본원통화",
    "US_UnempNSA":       "미국 실업률(비계절조정)",
    "US_MichSentiment":  "미시간 소비자심리지수",
    "HardLogs":          "원목 국제가격",
}


def vl(var_name: str) -> str:
    """변수명 + 한글 주석 반환. 예: 'AUD_USD (호주달러/USD 환율)'"""
    label = VAR_LABEL.get(var_name)
    if label:
        return f"{var_name} ({label})"
    return var_name


# 전역 테마: 흰 배경, 큰 폰트
pio.templates["bistro"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=14, color="#222222"),
        title=dict(font=dict(size=16)),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CCCCCC",
            borderwidth=1,
            font=dict(size=13),
        ),
        xaxis=dict(
            gridcolor="#EEEEEE", linecolor="#AAAAAA",
            tickfont=dict(size=13),
        ),
        yaxis=dict(
            gridcolor="#EEEEEE", linecolor="#AAAAAA",
            tickfont=dict(size=13),
        ),
    )
)
pio.templates.default = "bistro"

from bistro_core import (
    BISTROConfig,
    AttentionAnalyzer,
    TIER_LABELS,
    VARIABLE_FREQ,
    real_results_available,
    load_real_results,
    stage1_available,
    load_stage1_screening,
    ablation_available,
    load_ablation_results,
)

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="BISTRO-XAI | Attention Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; padding: 8px 16px; }
    .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; gap: 2px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helpers
# ============================================================

def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ensure_real_data():
    if st.session_state.get("real_loaded"):
        return (
            st.session_state["config"],
            st.session_state["hooks"],
            st.session_state["analyzer"],
            st.session_state["forecast"],
        )
    with st.spinner("Loading real BISTRO inference results..."):
        cfg, hooks, ana, forecast = load_real_results()
    st.session_state.update({
        "real_loaded": True,
        "attn_key": None,
        "config": cfg, "hooks": hooks, "analyzer": ana, "forecast": forecast,
    })
    return cfg, hooks, ana, forecast


# ============================================================
# Sidebar
# ============================================================

if not real_results_available():
    st.error("추론 결과가 없습니다. `bistro_runner_30var.py --daily`를 먼저 실행하세요.")
    st.stop()

with st.sidebar:
    st.title("🧠 BISTRO-XAI")
    st.caption("Attention Map Explorer")
    st.divider()
    st.markdown("### Foundation Model")
    st.markdown(
        "**BISTRO** (BIS Time-series Regression Oracle)\n\n"
        "- MOIRAI 아키텍처 기반 BIS 파인튜닝\n"
        "- **91M** parameters, 12 layers, 12 heads\n"
        "- **BIS 63개국 4,925개 시계열** (1970~2024)\n"
        "- Target: Korean CPI YoY (%)"
    )
    st.caption(
        "📌 **GitHub**: [bis-med-it/bistro]"
        "(https://github.com/bis-med-it/bistro)\n\n"
        "📄 Koyuncu, Kwon, Lombardi, Shin, Perez-Cruz. "
        "*BIS Quarterly Review*, March 2026."
    )


# ============================================================
# Load Data
# ============================================================

cfg, hooks, ana, forecast = ensure_real_data()

n = cfg.n_variates
variates_list = cfg.variates
active_variates = list(variates_list)  # Stage 2 결과 = 이미 선택된 변수만 포함

# ── Stage 1 스크리닝 데이터 로딩 (Feature Selection 탭용) ──
has_stage1 = stage1_available()
if has_stage1:
    if not st.session_state.get("s1_loaded"):
        s1_cfg, s1_hooks, s1_ana, s1_meta = load_stage1_screening()
        st.session_state.update({
            "s1_loaded": True,
            "s1_cfg": s1_cfg, "s1_hooks": s1_hooks,
            "s1_ana": s1_ana, "s1_meta": s1_meta,
        })
    s1_cfg  = st.session_state["s1_cfg"]
    s1_ana  = st.session_state["s1_ana"]
    s1_meta = st.session_state["s1_meta"]

with st.sidebar:
    # SOTA info (Stage 0→1 optimal 18)
    _sota_s0 = os.path.join(os.path.dirname(__file__), "data", "stage0", "stage0_ranking.npz")
    _sota_inc = os.path.join(os.path.dirname(__file__), "data", "stage0", "incremental_results.npz")
    _sota_f18 = os.path.join(os.path.dirname(__file__), "data", "forecast_optimal18.npz")
    if os.path.exists(_sota_s0) and os.path.exists(_sota_inc):
        _s0_d = np.load(_sota_s0, allow_pickle=True)
        _inc_d = np.load(_sota_inc, allow_pickle=True)
        _sota_k = int(_inc_d['best_k'])
        # Use actual forecast file RMSE if available (matches table)
        if os.path.exists(_sota_f18):
            _f18_side = np.load(_sota_f18, allow_pickle=True)
            _s2_side = np.load(os.path.join(os.path.dirname(__file__), "data", "real_inference_results.npz"), allow_pickle=True)
            _actual_side = _s2_side['forecast_actual']
            _valid_side = ~np.isnan(_actual_side)
            _f18_med = np.array(_f18_side['forecast_med'], dtype=float)
            _sota_rmse = float(np.sqrt(np.mean((_f18_med[_valid_side] - _actual_side[_valid_side]) ** 2)))
        else:
            _sota_rmse = float(_inc_d['best_rmse'])
        _sota_total = int(_s0_d['n_variates'])

        st.markdown("### SOTA Model")
        col1, col2 = st.columns(2)
        col1.metric("최적 변수", f"{_sota_k + 1}", help=f"1 target + {_sota_k} covariates")
        col2.metric("CTX Patches", 120)
        st.metric("RMSE", f"{_sota_rmse:.4f} pp")
        st.divider()
        st.markdown(f"**Pipeline**: 288 → 25 → **{_sota_k}** vars")
        st.caption(f"Stage 0 전수(288) → Stage 1 재추론(25) → 최적 {_sota_k}개")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Variables", n)
        col2.metric("CTX Patches", cfg.ctx_patches)

    st.divider()
    st.markdown("### Legacy Model")
    col1, col2 = st.columns(2)
    col1.metric("Variables", n, help="기존 29→11 파이프라인")
    col2.metric("Tokens", f"{cfg.n_tokens:,}")

# ── Ablation 데이터 로딩 ──
has_ablation = ablation_available()
if has_ablation:
    if not st.session_state.get("abl_loaded"):
        abl_data = load_ablation_results()
        st.session_state.update({"abl_loaded": True, "abl_data": abl_data})
    abl_data = st.session_state["abl_data"]

# ============================================================
# Header
# ============================================================

st.markdown("## BISTRO-XAI Attention Explorer")
if os.path.exists(os.path.join(os.path.dirname(__file__), "data", "stage0", "incremental_results.npz")):
    _hdr_inc = np.load(os.path.join(os.path.dirname(__file__), "data", "stage0", "incremental_results.npz"), allow_pickle=True)
    _hdr_k = int(_hdr_inc['best_k'])
    _hdr_vars = [str(v) for v in _hdr_inc['ranking'][:_hdr_k]]
    _hdr_f18 = os.path.join(os.path.dirname(__file__), "data", "forecast_optimal18.npz")
    if os.path.exists(_hdr_f18):
        _hdr_f18d = np.load(_hdr_f18, allow_pickle=True)
        _hdr_s2d = np.load(os.path.join(os.path.dirname(__file__), "data", "real_inference_results.npz"), allow_pickle=True)
        _hdr_actual = _hdr_s2d['forecast_actual']
        _hdr_valid = ~np.isnan(_hdr_actual)
        _hdr_rmse = float(np.sqrt(np.mean((np.array(_hdr_f18d['forecast_med'], dtype=float)[_hdr_valid] - _hdr_actual[_hdr_valid]) ** 2)))
    else:
        _hdr_rmse = float(_hdr_inc['best_rmse'])
    st.caption(
        f"**SOTA: {_hdr_k + 1}개 변수** (CPI_KR_YoY + {', '.join(_hdr_vars)}) | "
        f"288개 전수 스크리닝 → 최적 {_hdr_k} covariates | "
        f"RMSE **{_hdr_rmse:.4f}** | Forecast: 2023-01 ~ 2023-12"
    )
else:
    st.caption(
        f"**{n}개 변수** ({', '.join(variates_list)}) | "
        f"CTX {cfg.ctx_patches} patches/var | "
        f"Target: **{cfg.target_name}** | "
        f"Forecast: 2023-01 ~ 2023-12"
    )
st.divider()

# ============================================================
# Tabs
# ============================================================

tab_labels = [
    "🏦 Forecast Results",
    "🏆 Stage 0→1 Pipeline (288→18)",
    "🧠 Head Role Analysis",
    "📐 Layer Method Comparison",
    "⏱️ Temporal Patterns",
]
if forecast and forecast.get("cf_variates") is not None:
    tab_labels.append("🔀 CF Scenarios")

tabs = st.tabs(tab_labels)
tab_offset = 4  # tabs[0]=Forecast, tabs[1]=Stage0→1, tabs[2]=Head, tabs[3]=LayerMethod, tabs[4+]=remaining


# ----------------------------------------------------------
# Tab 0 (Real mode only): Forecast Results
# ----------------------------------------------------------

with tabs[0]:
    st.subheader("Korean CPI — BISTRO Forecast vs Actual")
    st.caption(
        "2023 forecast: context ~2013-01 ~ 2022-12 | "
        "2024 forecast: context ~2014-01 ~ 2023-12 | "
        "BISTRO는 BIS 63개국 4,925개 시계열로 파인튜닝. "
        "각 forecast 구간의 실제값은 추론 시 context에 미포함."
    )

    fc = forecast
    fc_dates = pd.to_datetime([d + "-01" for d in fc["date"]])
    actual_vals = fc["actual"]

    # Actual CPI — 패널에서 전체 월별 시계열 로딩 (2018~ 최신)
    import os
    _panel_path = os.path.join(os.path.dirname(__file__), "data", "macro_panel_daily.csv")
    if not os.path.exists(_panel_path):
        _panel_path = os.path.join(os.path.dirname(__file__), "data", "macro_panel.csv")
    _panel_cpi = pd.read_csv(_panel_path, index_col=0, parse_dates=True)["CPI_KR_YoY"]
    _cpi_monthly = _panel_cpi.resample("MS").last().dropna()
    _cpi_monthly = _cpi_monthly.loc["2021-01":"2025-12"]

    # 추론 기간 전후 분리
    _cpi_before = _cpi_monthly.loc[:"2022-12"]
    _cpi_forecast = _cpi_monthly.loc["2023-01":"2023-12"]
    _cpi_after = _cpi_monthly.loc["2024-01":]

    fig_fc = go.Figure()

    # Actual CPI — 추론 이전 (검정)
    fig_fc.add_trace(go.Scatter(
        x=_cpi_before.index, y=_cpi_before.values,
        mode="lines",
        line=dict(color="#333333", width=2.5),
        name="Actual CPI",
        hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
    ))

    # Actual CPI — 추론 구간 (빨간색 + 네모 마커)
    # 이전 구간 마지막 점과 연결
    _fc_x = list(_cpi_forecast.index)
    _fc_y = list(_cpi_forecast.values)
    if len(_cpi_before) > 0:
        _fc_x = [_cpi_before.index[-1]] + _fc_x
        _fc_y = [_cpi_before.values[-1]] + _fc_y
    fig_fc.add_trace(go.Scatter(
        x=_fc_x, y=_fc_y,
        mode="lines+markers",
        line=dict(color="#D62728", width=2.5),
        marker=dict(size=8, symbol="square"),
        name="Actual (forecast period)",
        hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
    ))

    # ── 2024 forecast 데이터 로딩 ──
    _f24_path = os.path.join(os.path.dirname(__file__), "data", "forecast_optimal18_2024.npz")
    _has_f24 = os.path.exists(_f24_path)
    if _has_f24:
        _f24 = np.load(_f24_path, allow_pickle=True)
        _f24_dates = pd.to_datetime([d + "-01" for d in _f24["forecast_date"]])

    # Actual CPI — 2023~2024 forecast 사이 (검정, 2023 추론 이후 ~ 2024 추론 이전)
    _cpi_between = _cpi_monthly.loc["2024-01":"2024-12"] if _has_f24 else _cpi_monthly.loc["2024-01":]
    _cpi_after_all = _cpi_monthly.loc["2025-01":] if _has_f24 else pd.Series(dtype=float)

    if len(_cpi_between) > 0 and not _has_f24:
        # No 2024 forecast — just show post-2023 as black line
        _af_x = [_cpi_forecast.index[-1]] + list(_cpi_between.index)
        _af_y = [_cpi_forecast.values[-1]] + list(_cpi_between.values)
        fig_fc.add_trace(go.Scatter(
            x=_af_x, y=_af_y,
            mode="lines",
            line=dict(color="#333333", width=2.5),
            name="Actual (post-forecast)",
            showlegend=False,
            hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
        ))
    elif _has_f24:
        # Connect 2023 forecast period end to 2024 forecast period with black line
        _bridge_x = [_cpi_forecast.index[-1]]
        _bridge_y = [_cpi_forecast.values[-1]]
        if len(_cpi_between) > 0:
            _bridge_x += list(_cpi_between.index)
            _bridge_y += list(_cpi_between.values)
        fig_fc.add_trace(go.Scatter(
            x=_bridge_x, y=_bridge_y,
            mode="lines+markers",
            line=dict(color="#E65100", width=2.5),
            marker=dict(size=8, symbol="square"),
            name="Actual (2024 forecast period)",
            hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
        ))
        # Post-2024 actual (검정)
        if len(_cpi_after_all) > 0 and len(_cpi_between) > 0:
            _af2_x = [_cpi_between.index[-1]] + list(_cpi_after_all.index)
            _af2_y = [_cpi_between.values[-1]] + list(_cpi_after_all.values)
            fig_fc.add_trace(go.Scatter(
                x=_af2_x, y=_af2_y,
                mode="lines",
                line=dict(color="#333333", width=2.5),
                name="Actual (post-forecast)",
                showlegend=False,
                hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
            ))

    # CI band
    ci_x = list(fc_dates) + list(fc_dates[::-1])
    ci_y = list(fc["ci_hi"]) + list(fc["ci_lo"][::-1])
    fig_fc.add_trace(go.Scatter(
        x=ci_x, y=ci_y,
        fill="toself",
        fillcolor="rgba(74,144,226,0.20)",
        line=dict(color="rgba(74,144,226,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="BISTRO 90% CI",
    ))

    # BISTRO median — 기존 29개 풀에서 attention top 10 선별
    fig_fc.add_trace(go.Scatter(
        x=fc_dates, y=fc["med"],
        mode="lines+markers",
        line=dict(color="#1A6FD4", width=3),
        marker=dict(size=9),
        name="29풀→11 (Attn top10)",
        hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
    ))

    # BISTRO (recent 9)
    _fr9_path = os.path.join(os.path.dirname(__file__), "data", "forecast_recent9.npz")
    if os.path.exists(_fr9_path):
        _fr9 = np.load(_fr9_path, allow_pickle=True)
        _fr9_dates = pd.to_datetime([d + "-01" for d in _fr9["forecast_date"]])
        fig_fc.add_trace(go.Scatter(
            x=_fr9_dates, y=_fr9["forecast_med"],
            mode="lines+markers",
            line=dict(color="#F59E0B", width=2.5),
            marker=dict(size=7, symbol="star"),
            name="29풀→9 (최근1년 Attn)",
            hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
        ))

    # BISTRO (18★) — Stage 0→1 optimal 18 from 288
    _f18_path = os.path.join(os.path.dirname(__file__), "data", "forecast_optimal18.npz")
    if os.path.exists(_f18_path):
        _f18 = np.load(_f18_path, allow_pickle=True)
        _f18_dates = pd.to_datetime([d + "-01" for d in _f18["forecast_date"]])
        fig_fc.add_trace(go.Scatter(
            x=_f18_dates, y=_f18["forecast_med"],
            mode="lines+markers",
            line=dict(color="#DC2626", width=3),
            marker=dict(size=8, symbol="hexagram"),
            name="★ 288풀→18 (최적)",
            hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
        ))

    # BISTRO (1) — univariate
    _f1_path = os.path.join(os.path.dirname(__file__), "data", "forecast_univariate.npz")
    if os.path.exists(_f1_path):
        _f1 = np.load(_f1_path, allow_pickle=True)
        _f1_dates = pd.to_datetime([d + "-01" for d in _f1["forecast_date"]])
        fig_fc.add_trace(go.Scatter(
            x=_f1_dates, y=_f1["forecast_med"],
            mode="lines+markers",
            line=dict(color="#000000", width=2, dash="dot"),
            marker=dict(size=6, symbol="x"),
            name="Univariate (공변량 없음)",
            hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
        ))

    # AR(1)
    if not all(np.isnan(fc["ar1"])):
        fig_fc.add_trace(go.Scatter(
            x=fc_dates, y=fc["ar1"],
            mode="lines+markers",
            line=dict(color="#888888", width=2, dash="dash"),
            marker=dict(size=7),
            name="AR(1) 통계 baseline",
            hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
        ))

    # ── 2024 forecast traces ──
    if _has_f24:
        # CI band (2024)
        ci24_x = list(_f24_dates) + list(_f24_dates[::-1])
        ci24_y = list(_f24["forecast_ci_hi"]) + list(_f24["forecast_ci_lo"][::-1])
        fig_fc.add_trace(go.Scatter(
            x=ci24_x, y=ci24_y,
            fill="toself",
            fillcolor="rgba(220,38,38,0.15)",
            line=dict(color="rgba(220,38,38,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="2024 BISTRO 90% CI",
        ))
        # Median (2024)
        fig_fc.add_trace(go.Scatter(
            x=_f24_dates, y=_f24["forecast_med"],
            mode="lines+markers",
            line=dict(color="#DC2626", width=3),
            marker=dict(size=8, symbol="hexagram"),
            name="★ 288풀→18 (2024)",
            hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
        ))
        # AR(1) 2024
        if "forecast_ar1" in _f24.files and not all(np.isnan(_f24["forecast_ar1"])):
            fig_fc.add_trace(go.Scatter(
                x=_f24_dates, y=_f24["forecast_ar1"],
                mode="lines+markers",
                line=dict(color="#888888", width=2, dash="dash"),
                marker=dict(size=5),
                name="AR(1) (2024)",
                showlegend=False,
                hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
            ))

    # Forecast start lines
    fig_fc.add_vline(
        x=fc_dates[0].timestamp() * 1000,
        line_dash="dash", line_color="rgba(100,100,100,0.6)",
        annotation_text="2023 forecast", annotation_position="top right",
        annotation_font_size=13,
    )
    if _has_f24:
        fig_fc.add_vline(
            x=_f24_dates[0].timestamp() * 1000,
            line_dash="dash", line_color="rgba(220,38,38,0.4)",
            annotation_text="2024 forecast", annotation_position="top right",
            annotation_font_size=13,
        )

    fig_fc.update_layout(
        yaxis_title="CPI YoY (%)",
        xaxis_title="",
        height=460,
        legend=dict(x=0.01, y=0.99, font=dict(size=13)),
        margin=dict(t=30, b=50),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # RMSE table
    valid = ~np.isnan(actual_vals)
    if valid.any():
        rmse_bistro = float(np.sqrt(np.mean((fc["med"][valid] - actual_vals[valid]) ** 2)))
        # BISTRO (recent 9) RMSE
        rmse_r9 = None
        if os.path.exists(_fr9_path):
            _fr9_rmse = np.load(_fr9_path, allow_pickle=True)
            fr9_med = np.array(_fr9_rmse["forecast_med"], dtype=float)
            rmse_r9 = float(np.sqrt(np.mean((fr9_med[valid] - actual_vals[valid]) ** 2)))

        # BISTRO (18★) RMSE
        rmse_18 = None
        if os.path.exists(_f18_path):
            _f18_rmse = np.load(_f18_path, allow_pickle=True)
            f18_med = np.array(_f18_rmse["forecast_med"], dtype=float)
            rmse_18 = float(np.sqrt(np.mean((f18_med[valid] - actual_vals[valid]) ** 2)))

        # BISTRO (1) RMSE
        rmse_1 = None
        if os.path.exists(_f1_path):
            _f1_rmse = np.load(_f1_path, allow_pickle=True)
            f1_med = np.array(_f1_rmse["forecast_med"], dtype=float)
            rmse_1 = float(np.sqrt(np.mean((f1_med[valid] - actual_vals[valid]) ** 2)))

        # BISTRO 18★ (2024) RMSE
        rmse_18_24 = None
        rmse_ar1_24 = None
        if _has_f24:
            f24_med = np.array(_f24["forecast_med"], dtype=float)
            f24_actual = np.array(_f24["forecast_actual"], dtype=float)
            f24_valid = ~np.isnan(f24_actual)
            if f24_valid.any():
                rmse_18_24 = float(np.sqrt(np.mean((f24_med[f24_valid] - f24_actual[f24_valid]) ** 2)))
                if "forecast_ar1" in _f24.files:
                    f24_ar1 = np.array(_f24["forecast_ar1"], dtype=float)
                    if not all(np.isnan(f24_ar1)):
                        rmse_ar1_24 = float(np.sqrt(np.mean((f24_ar1[f24_valid] - f24_actual[f24_valid]) ** 2)))

        if not all(np.isnan(fc["ar1"])):
            rmse_ar1 = float(np.sqrt(np.mean((fc["ar1"][valid] - actual_vals[valid]) ** 2)))
            r_rmse   = rmse_bistro / rmse_ar1

            st.markdown("**2023 Forecast RMSE**")
            rmse_items = []
            if rmse_1 is not None:
                rmse_items.append(("Univariate", rmse_1))
            rmse_items.append(("29풀→11", rmse_bistro))
            if rmse_18 is not None:
                rmse_items.append(("★ 288풀→18", rmse_18))
            rmse_items.append(("AR(1)", rmse_ar1))

            cols = st.columns(len(rmse_items))
            for i, (label, val) in enumerate(rmse_items):
                cols[i].metric(f"{label} RMSE", f"{val:.4f} pp")

        # 2024 RMSE
        if rmse_18_24 is not None:
            st.markdown("**2024 Forecast RMSE** (context에 2023 실제값 포함)")
            rmse_items_24 = [("★ 288풀→18", rmse_18_24)]
            if rmse_ar1_24 is not None:
                rmse_items_24.append(("AR(1)", rmse_ar1_24))
            cols24 = st.columns(len(rmse_items_24))
            for i, (label, val) in enumerate(rmse_items_24):
                cols24[i].metric(f"{label} RMSE", f"{val:.4f} pp")

        # ── 실험 설명 테이블 (RMSE 포함, 정렬) ──
        _exp_rows = []
        if rmse_18 is not None:
            _exp_rows.append(("★ 288풀→18 (최적)", 18, "288개 FRED",
                "Stage 0(CTX=10) 전수 attn → Stage 1(CTX=120) ablation → incremental",
                "288개를 단축 컨텍스트로 한 번에 스크리닝. 본 연구 최종 방법론.", rmse_18))
        _exp_rows.append(("29풀→11 (Attn top10)", 11, "29개 수작업",
            "Stage 1(29개) attn → top 10 → Stage 2 재추론",
            "도메인 지식 29개에서 attention top 10 선별. 29개 전체 대비 0.009pp 악화 — "
            "소규모 풀에서는 변수 축소의 이점이 미미.", rmse_bistro))
        if rmse_r9 is not None:
            _exp_rows.append(("29풀→9 (최근1년 Attn)", 10, "29개 수작업",
                "최근 12패치(≈1년) attention으로 top 9 선별",
                "단기 금리인상기 변수 급부상, 그러나 예측 개선 없음.", rmse_r9))
        if rmse_1 is not None:
            _exp_rows.append(("Univariate (공변량 없음)", 1, "-",
                "CPI_KR_YoY만 투입",
                "공변량 없이 AR(1)과 동등. 공변량 필요성 입증.", rmse_1))
        if not all(np.isnan(fc["ar1"])):
            _exp_rows.append(("AR(1) 통계 baseline", 0, "-",
                "직전 값 기반 자기회귀",
                "인플레이션 관성만 반영, 하향 전환 포착 불가.", rmse_ar1))

        _exp_rows.sort(key=lambda x: x[5])
        _exp_df = pd.DataFrame(_exp_rows, columns=["실험", "변수 수", "후보 풀", "방법", "설명", "RMSE (2023)"])
        _exp_df["RMSE (2023)"] = _exp_df["RMSE (2023)"].apply(lambda x: f"{x:.4f}")

        with st.expander("각 실험 방법 상세 설명 (2023 RMSE 순)"):
            st.dataframe(_exp_df, use_container_width=True, hide_index=True)

    # Monthly breakdown tables
    fc_df = pd.DataFrame({
        "Month":       fc["date"],
        "BISTRO Med":  [f"{v:.3f}%" for v in fc["med"]],
        "CI [5,95]":   [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(fc["ci_lo"], fc["ci_hi"])],
        "AR(1)":       [f"{v:.3f}%" if not np.isnan(v) else "—" for v in fc["ar1"]],
        "Actual":      [f"{v:.3f}%" if not np.isnan(v) else "—" for v in actual_vals],
    })
    with st.expander("📋 2023 월별 예측값 테이블"):
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

    if _has_f24:
        f24_df = pd.DataFrame({
            "Month":       _f24["forecast_date"],
            "BISTRO Med":  [f"{v:.3f}%" for v in _f24["forecast_med"]],
            "CI [5,95]":   [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in
                            zip(_f24["forecast_ci_lo"], _f24["forecast_ci_hi"])],
            "AR(1)":       [f"{v:.3f}%" if not np.isnan(v) else "—" for v in _f24["forecast_ar1"]],
            "Actual":      [f"{v:.3f}%" if not np.isnan(v) else "—" for v in _f24["forecast_actual"]],
        })
        with st.expander("📋 2024 월별 예측값 테이블 (context: ~2023-12)"):
            st.dataframe(f24_df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------
# Tab: Stage 0 → 1 Pipeline (288 → 18)
# ----------------------------------------------------------

_s0_path = os.path.join(os.path.dirname(__file__), "data", "stage0", "stage0_ranking.npz")
_s1r_path = os.path.join(os.path.dirname(__file__), "data", "stage0", "stage1_results.npz")
_inc_path = os.path.join(os.path.dirname(__file__), "data", "stage0", "incremental_results.npz")
_has_stage0 = os.path.exists(_s0_path) and os.path.exists(_s1r_path) and os.path.exists(_inc_path)

with tabs[1]:
    if not _has_stage0:
        st.warning("Stage 0→1 결과 파일이 없습니다. `run_stage0_screening.py`를 실행하세요.")
    else:
        _s0 = np.load(_s0_path, allow_pickle=True)
        _s1r = np.load(_s1r_path, allow_pickle=True)
        _inc = np.load(_inc_path, allow_pickle=True)

        st.subheader("Full-Variable Screening Pipeline (288 → 18)")

        st.markdown("""
**핵심 아이디어**: Transformer의 context window 제약(`max_seq_len=3,120`)으로 한 번에 26개 변수가 한계.
이를 **ctx_patches 축소**로 우회하여 288개 변수를 단일 추론에서 동시 평가한 뒤,
전체 컨텍스트로 재추론하는 2단계 파이프라인.

| 단계 | 설명 | 변수 수 |
|------|------|---------|
| **Stage 0** | CTX=10 (10개월)으로 288개 전수 attention 스크리닝 | 288 → 25 |
| **Stage 1** | CTX=120 (10년)으로 재추론 + Ablation으로 harmful 제거 | 25 → 23 |
| **Incremental** | Attention 순위대로 하나씩 추가, RMSE 최소점 탐색 | 23 → **18** |
""")

        st.info(
            "**왜 토너먼트(그룹 분할)가 아닌가?** "
            "26개씩 그룹으로 나누면 같은 그룹에 배정된 변수끼리만 경쟁하므로 **편성 편향**이 발생. "
            "CTX를 줄여 전체 변수를 한 번에 넣으면 **모든 변수가 동일 조건에서 공정하게 경쟁**."
        )

        # ── Funnel Metrics ──
        st.markdown("### Pipeline Summary")
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Stage 0 입력", f"{int(_s0['n_variates'])} vars", help="CTX=10, 전수 attention screening")
        fc2.metric("Stage 1 입력", f"{len(_s1r['top_vars'])} vars", help="CTX=120, full-context 재추론 + ablation")
        _n_harmful = len(_s1r['harmful_vars'])
        fc3.metric("Harmful 제거", f"-{_n_harmful}", help=f"제거: {', '.join(str(v) for v in _s1r['harmful_vars'])}")
        # Use actual forecast RMSE for consistency
        _pipe_f18 = os.path.join(os.path.dirname(__file__), "data", "forecast_optimal18.npz")
        if os.path.exists(_pipe_f18):
            _pf18 = np.load(_pipe_f18, allow_pickle=True)
            _ps2 = np.load(os.path.join(os.path.dirname(__file__), "data", "real_inference_results.npz"), allow_pickle=True)
            _pact = _ps2['forecast_actual']
            _pval = ~np.isnan(_pact)
            _pipe_rmse = float(np.sqrt(np.mean((np.array(_pf18['forecast_med'], dtype=float)[_pval] - _pact[_pval]) ** 2)))
        else:
            _pipe_rmse = float(_inc['best_rmse'])
        fc4.metric("최적 조합", f"{int(_inc['best_k'])} vars", delta=f"RMSE {_pipe_rmse:.4f}", delta_color="normal")

        st.divider()

        # ── Stage 0: Attention Ranking (288 vars) ──
        st.markdown("### Stage 0 — 전수 Attention Screening (288 vars, CTX=10)")
        st.markdown(f"""
**방법**: `max_seq_len=3,120` 이내에서 모든 변수를 넣기 위해 `ctx_patches`를 120 → **10**으로 축소.
288 × 10 = 2,880 토큰으로 한계 이내. 10패치 × 32일 = **약 10.5개월** 컨텍스트.

**목적**: 정확한 예측이 아니라 **변수 간 상대적 attention 순위**만 추출.
모든 변수가 동일 추론에서 경쟁하므로 그룹 편성 편향 없이 공정한 스크리닝.

- Self-attention: **{float(_s0['self_attn']):.1%}** (타겟이 자기 자신에게 주는 attention)
- 나머지 {float(1 - float(_s0['self_attn'])):.1%}가 287개 공변량에 분배
- 균등 배분 시 변수당 약 {(1 - float(_s0['self_attn'])) / 287:.4f} ({(1 - float(_s0['self_attn'])) / 287:.2%})
""")

        s0_vars = [str(v) for v in _s0['ranking_vars']]
        s0_attn = _s0['ranking_attn'].astype(float)

        # Top 30 bar chart
        n_show = min(30, len(s0_vars))
        fig_s0 = go.Figure()
        fig_s0.add_trace(go.Bar(
            x=[vl(v) for v in s0_vars[:n_show]], y=s0_attn[:n_show],
            marker_color=["#DC2626" if i < 25 else "#CCCCCC" for i in range(n_show)],
            hovertemplate="%{x}: <b>%{y:.4f}</b> (%{customdata:.2%})<extra></extra>",
            customdata=s0_attn[:n_show],
        ))
        fig_s0.add_hline(y=s0_attn[24] if len(s0_attn) > 24 else 0,
                         line_dash="dash", line_color="red",
                         annotation_text="Top 25 cutoff", annotation_position="top right")
        fig_s0.update_layout(
            title="Stage 0: Cross-Variate Attention (top 30 of 288)",
            xaxis=dict(title="Covariate", tickangle=-40, tickfont_size=10),
            yaxis=dict(title="Attention Fraction"),
            height=420, margin=dict(t=50, b=100),
        )
        st.plotly_chart(fig_s0, use_container_width=True)

        # Full ranking table (expandable)
        with st.expander(f"전체 288개 변수 attention 순위"):
            s0_df = pd.DataFrame({
                "Rank": range(1, len(s0_vars) + 1),
                "Variable": [vl(v) for v in s0_vars],
                "Attention": [f"{a:.4f}" for a in s0_attn],
                "Attention %": [f"{a:.2%}" for a in s0_attn],
                "Selected": ["Top 25" if i < 25 else "" for i in range(len(s0_vars))],
            })
            st.dataframe(s0_df, use_container_width=True, hide_index=True, height=400)

        st.divider()

        # ── Stage 1: Re-inference + Ablation ──
        st.markdown("### Stage 1 — 재추론 + Ablation (25 vars, CTX=120)")
        st.markdown(f"""
**방법**: Stage 0에서 선별된 top 25를 전체 컨텍스트(`CTX=120`, 약 10.5년)로 재추론.
Attention 순위가 Stage 0과 달라질 수 있음 — 단기(10개월)에서 중요해 보였던 변수가 장기에서는 아닐 수 있고, 그 반대도 가능.

**Ablation (Leave-One-Out)**: 각 변수를 하나씩 제거하고 재추론하여 RMSE 변화를 측정.
- `ΔRMSE > 0` → 제거하면 나빠짐 → **Helpful** (유지)
- `ΔRMSE ≤ 0` → 제거해도 같거나 좋아짐 → **Harmful** (제거)

- Self-attention: **{float(_s1r['self_attn']):.1%}** | Baseline RMSE: **{float(_s1r['baseline_rmse']):.4f}** pp
""")

        s1_rank_vars = [str(v) for v in _s1r['ranking_vars']]
        s1_rank_attn = _s1r['ranking_attn'].astype(float)
        s1_abl_vars = [str(v) for v in _s1r['ablation_vars']]
        s1_abl_delta = _s1r['ablation_delta'].astype(float)
        s1_harmful = [str(v) for v in _s1r['harmful_vars']]
        s1_final = [str(v) for v in _s1r['final_vars']]

        abl_map = dict(zip(s1_abl_vars, s1_abl_delta))

        # Attention + Ablation combined chart
        fig_s1 = go.Figure()

        # Attention bars
        bar_colors = []
        for v in s1_rank_vars:
            if v in s1_harmful:
                bar_colors.append("#EF4444")  # red = harmful
            elif v in s1_final:
                bar_colors.append("#10B981")  # green = helpful
            else:
                bar_colors.append("#9CA3AF")  # gray

        fig_s1.add_trace(go.Bar(
            x=[vl(v) for v in s1_rank_vars], y=s1_rank_attn,
            name="Attention",
            marker_color=bar_colors,
            hovertemplate="%{x}: attn=%{y:.4f}<extra></extra>",
        ))

        # Ablation delta as line overlay
        abl_y = [abl_map.get(v, 0) for v in s1_rank_vars]
        fig_s1.add_trace(go.Scatter(
            x=[vl(v) for v in s1_rank_vars], y=abl_y,
            mode="lines+markers",
            name="Ablation ΔRMSE",
            yaxis="y2",
            line=dict(color="#6366F1", width=2),
            marker=dict(size=8),
            hovertemplate="%{x}: ΔRMSE=%{y:+.4f}<extra></extra>",
        ))
        fig_s1.add_hline(y=0, line_dash="dash", line_color="red", line_width=1,
                         annotation_text="harmful threshold",
                         secondary_y=True if hasattr(fig_s1, 'secondary_y') else False)

        fig_s1.update_layout(
            title="Stage 1: Attention (bars) + Ablation ΔRMSE (line)",
            xaxis=dict(title="Covariate", tickangle=-40, tickfont_size=10),
            yaxis=dict(title="Attention Fraction", side="left"),
            yaxis2=dict(title="ΔRMSE (pp)", side="right", overlaying="y", showgrid=False),
            height=450, margin=dict(t=50, b=100),
            legend=dict(x=0.7, y=0.95),
        )
        st.plotly_chart(fig_s1, use_container_width=True)

        st.markdown(
            f"<span style='color:#10B981'>■</span> Helpful ({len(s1_final)}) &nbsp;&nbsp;"
            f"<span style='color:#EF4444'>■</span> Harmful ({len(s1_harmful)}: {', '.join(s1_harmful)})",
            unsafe_allow_html=True,
        )

        if len(s1_harmful) > 0:
            st.warning(
                f"**Harmful 변수 해석** ({', '.join(s1_harmful)}): "
                f"Stage 0(CTX=10)에서는 attention top 25에 진입했으나, "
                f"Stage 1(CTX=120) full context에서 재검증 시 예측에 기여하지 못함. "
                f"단기적 패턴 유사성으로 attention을 받았지만, 장기적으로는 타겟과 실질적 연결이 없는 것으로 판단. "
                f"**Stage 0 → Stage 1 2단계 검증이 이런 false positive를 걸러내는 역할.**"
            )

        # Stage 1 detail table
        with st.expander("Stage 1 상세 테이블"):
            s1_df = pd.DataFrame({
                "Rank": range(1, len(s1_rank_vars) + 1),
                "Variable": [vl(v) for v in s1_rank_vars],
                "Attention": [f"{a:.4f} ({a:.2%})" for a in s1_rank_attn],
                "ΔRMSE": [f"{abl_map.get(v, 0):+.4f}" for v in s1_rank_vars],
                "Status": ["Harmful" if v in s1_harmful else "Helpful" for v in s1_rank_vars],
            })
            st.dataframe(s1_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Incremental Addition Curve ──
        st.markdown("### Incremental Addition — 최적 변수 수 탐색")
        st.markdown("""
**방법**: Harmful 제거 후 남은 23개 변수를 **attention 순위대로 1개씩 추가**하며 매번 RMSE를 측정.
RMSE가 최소인 지점 = 최적 변수 수.

- Greedy 방식: 한 번 넣은 변수는 빼지 않음. 순서는 Stage 1 attention ranking 고정.
- 완전 탐색(C(23,N)개 조합)은 계산상 불가 → attention 순위 기반 근사.
- RMSE가 단조 감소하지 않음: 특정 변수 추가 시 오히려 악화 → **"더 많은 변수 = 더 좋은 예측"이 아님**을 실증.
""")

        inc_n = _inc['n_vars'].astype(int)
        inc_rmse = _inc['rmse'].astype(float)
        inc_added = [str(v) for v in _inc['added_var']]
        best_k = int(_inc['best_k'])
        best_rmse = _pipe_rmse if os.path.exists(_pipe_f18) else float(_inc['best_rmse'])

        fig_inc = go.Figure()
        inc_colors = ["#DC2626" if n == best_k else "#3B82F6" for n in inc_n]
        fig_inc.add_trace(go.Scatter(
            x=inc_n, y=inc_rmse,
            mode="lines+markers",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(size=9, color=inc_colors),
            customdata=inc_added,
            hovertemplate="N=%{x}: RMSE=%{y:.4f}<br>+%{customdata}<extra></extra>",
        ))
        # Best point
        fig_inc.add_trace(go.Scatter(
            x=[best_k], y=[best_rmse],
            mode="markers+text",
            marker=dict(size=16, color="#DC2626", symbol="star"),
            text=[f"N={best_k}, RMSE={best_rmse:.4f}"],
            textposition="top center",
            textfont=dict(size=13, color="#DC2626"),
            showlegend=False,
        ))
        # Reference lines
        fig_inc.add_hline(y=1.1895, line_dash="dot", line_color="#888",
                          annotation_text="기존 11변수 (1.1895)", annotation_position="bottom right")

        fig_inc.update_layout(
            title=f"Incremental Addition: Optimal N = {best_k} (RMSE {best_rmse:.4f})",
            xaxis=dict(title="Number of Covariates", dtick=1),
            yaxis=dict(title="RMSE (pp)"),
            height=420, margin=dict(t=50, b=50),
        )
        st.plotly_chart(fig_inc, use_container_width=True)

        # Incremental table
        with st.expander("Incremental Addition 상세"):
            inc_df = pd.DataFrame({
                "N": inc_n,
                "RMSE": [f"{r:.4f}" for r in inc_rmse],
                "Added Variable": inc_added,
                "Best": ["★" if n == best_k else "" for n in inc_n],
            })
            st.dataframe(inc_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── 2×2 Confirmed Driver Diagnostic ──
        st.markdown("### 2×2 Diagnostic — Confirmed Drivers (Stage 1, 25 vars)")
        st.markdown("""
**4사분면 해석** (X축: Attention Score, Y축: Ablation ΔRMSE):

| 사분면 | Attention | ΔRMSE | 의미 |
|--------|-----------|-------|------|
| **Confirmed Driver** | 높음 (≥ uniform) | > 0 (제거 시 악화) | 모델이 보고 있고, 실제로 기여하는 핵심 변수 |
| **Spurious Attention** | 높음 | ≤ 0 (제거해도 무방) | 모델이 보고 있지만 실제 기여 없음. 허위 상관 |
| **Hidden Contributor** | 낮음 (< uniform) | > 0 | 모델이 덜 보지만 빼면 악화. 대체 불가 독자 정보 |
| **Irrelevant** | 낮음 | ≤ 0 | 모델도 안 보고, 있어봤자 노이즈만 추가 |

경계선: Attention = uniform share (모든 공변량에 균등 배분 시 각 변수의 몫)
""")

        # Compute thresholds
        uniform_share = (1.0 - float(_s1r['self_attn'])) / len(s1_rank_vars)
        attn_threshold = uniform_share

        diag_x = [s1_rank_attn[s1_rank_vars.index(v)] if v in s1_rank_vars else 0 for v in s1_abl_vars]
        diag_y = [abl_map.get(v, 0) for v in s1_abl_vars]

        # Quadrant classification
        quadrants = []
        q_colors = []
        for v, ax, ay in zip(s1_abl_vars, diag_x, diag_y):
            if ax >= attn_threshold and ay > 0:
                quadrants.append("Confirmed Driver")
                q_colors.append("#10B981")
            elif ax >= attn_threshold and ay <= 0:
                quadrants.append("Spurious Attention")
                q_colors.append("#EF4444")
            elif ax < attn_threshold and ay > 0:
                quadrants.append("Hidden Contributor")
                q_colors.append("#F59E0B")
            else:
                quadrants.append("Irrelevant")
                q_colors.append("#9CA3AF")

        fig_diag = go.Figure()
        fig_diag.add_trace(go.Scatter(
            x=diag_x, y=diag_y,
            mode="markers+text",
            marker=dict(size=12, color=q_colors),
            text=s1_abl_vars,
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate="%{text}<br>Attn: %{x:.4f}<br>ΔRMSE: %{y:+.4f}<extra></extra>",
            showlegend=False,
        ))

        # Quadrant lines
        fig_diag.add_hline(y=0, line_dash="dash", line_color="rgba(100,100,100,0.5)")
        fig_diag.add_vline(x=attn_threshold, line_dash="dash", line_color="rgba(100,100,100,0.5)",
                           annotation_text=f"uniform share ({attn_threshold:.4f})")

        # Quadrant labels
        x_range = max(diag_x) - min(diag_x)
        y_range = max(diag_y) - min(diag_y)
        fig_diag.add_annotation(x=max(diag_x) - x_range*0.1, y=max(diag_y), text="Confirmed Driver",
                                showarrow=False, font=dict(color="#10B981", size=12))
        fig_diag.add_annotation(x=max(diag_x) - x_range*0.1, y=min(diag_y), text="Spurious Attention",
                                showarrow=False, font=dict(color="#EF4444", size=12))
        fig_diag.add_annotation(x=min(diag_x) + x_range*0.05, y=max(diag_y), text="Hidden Contributor",
                                showarrow=False, font=dict(color="#F59E0B", size=12))
        fig_diag.add_annotation(x=min(diag_x) + x_range*0.05, y=min(diag_y), text="Irrelevant",
                                showarrow=False, font=dict(color="#9CA3AF", size=12))

        fig_diag.update_layout(
            xaxis=dict(title="Attention Score"),
            yaxis=dict(title="Ablation ΔRMSE (pp)"),
            height=520, margin=dict(t=30, b=50),
        )
        st.plotly_chart(fig_diag, use_container_width=True)

        # Quadrant summary table
        diag_df = pd.DataFrame({
            "Variable": [vl(v) for v in s1_abl_vars],
            "Attention": [f"{x:.4f}" for x in diag_x],
            "ΔRMSE": [f"{y:+.4f}" for y in diag_y],
            "Quadrant": quadrants,
        })
        q_order = {"Confirmed Driver": 0, "Hidden Contributor": 1, "Spurious Attention": 2, "Irrelevant": 3}
        diag_df["_sort"] = diag_df["Quadrant"].map(q_order)
        diag_df = diag_df.sort_values(["_sort", "Variable"]).drop(columns=["_sort"])

        confirmed = diag_df[diag_df["Quadrant"] == "Confirmed Driver"]
        spurious = diag_df[diag_df["Quadrant"] == "Spurious Attention"]
        hidden = diag_df[diag_df["Quadrant"] == "Hidden Contributor"]

        irrelevant = diag_df[diag_df["Quadrant"] == "Irrelevant"]

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        col_d1.metric("Confirmed Drivers", len(confirmed))
        col_d2.metric("Hidden Contributors", len(hidden))
        col_d3.metric("Spurious Attention", len(spurious))
        col_d4.metric("Irrelevant", len(irrelevant))

        # Per-quadrant variable lists
        st.markdown("#### 사분면별 변수")

        if len(confirmed) > 0:
            st.success(
                f"**Confirmed Drivers** ({len(confirmed)}): "
                + ", ".join(confirmed["Variable"].tolist())
                + " — 모델이 주목하고, 실제 예측에 기여. 최종 조합의 핵심."
            )
        if len(hidden) > 0:
            st.warning(
                f"**Hidden Contributors** ({len(hidden)}): "
                + ", ".join(hidden["Variable"].tolist())
                + " — Attention은 낮지만 제거하면 RMSE 악화. 다른 변수와 보완적 정보를 제공하는 것으로 추정."
            )
        if len(spurious) > 0:
            st.error(
                f"**Spurious Attention** ({len(spurious)}): "
                + ", ".join(spurious["Variable"].tolist())
                + " — 모델이 주목하지만 실제 기여 없음. 패턴 유사성에 의한 허위 상관."
            )
        if len(irrelevant) > 0:
            st.info(
                f"**Irrelevant** ({len(irrelevant)}): "
                + ", ".join(irrelevant["Variable"].tolist())
                + " — Stage 0(CTX=10) 단기 스크리닝에서는 top 25에 진입했으나, "
                  "Stage 1(CTX=120) 장기 컨텍스트에서 attention도 낮고 예측 기여도 없음. "
                  "**2단계 검증이 false positive를 걸러낸 사례.**"
            )

        with st.expander("2×2 상세 분류"):
            st.dataframe(diag_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Final Selection Summary ──
        st.markdown("### 최종 결과")

        optimal_vars = [str(v) for v in _inc['ranking'][:best_k]]
        st.success(
            f"**288개 후보 → 최적 {best_k}개 조합 (RMSE {best_rmse:.4f})** "
            f"— 기존 11변수(1.1895) 대비 **{(1.1895 - best_rmse):.4f}pp ({(1.1895 - best_rmse)/1.1895*100:.1f}%) 개선**"
        )

        # Optimal variable table with attention + ablation
        opt_attn_map = dict(zip(s1_rank_vars, s1_rank_attn))
        opt_df = pd.DataFrame({
            "순위": range(1, best_k + 1),
            "변수": [vl(v) for v in optimal_vars],
            "Attention": [f"{opt_attn_map.get(v, 0):.4f} ({opt_attn_map.get(v, 0):.2%})" for v in optimal_vars],
            "Ablation ΔRMSE": [f"{abl_map.get(v, 0):+.4f}" for v in optimal_vars],
            "2×2 분류": [
                next((q for v2, q in zip(s1_abl_vars, quadrants) if str(v2) == v), "—")
                for v in optimal_vars
            ],
        })
        st.dataframe(opt_df, use_container_width=True, hide_index=True)

        st.markdown("""
**해석 요약:**
- 288개 FRED 거시경제 변수를 **순수 attention 기반**으로 스크리닝 (외부 통계 방법 미사용)
- Stage 0(단축 컨텍스트)에서 공정한 전수 평가 → Stage 1(전체 컨텍스트)에서 ablation 검증 → incremental로 최적 N 도출
- 기존 29개 수작업 선정 대비 **완전히 다른 변수 구성**으로 RMSE 개선 달성
- 2단계 검증으로 **false positive(Irrelevant)를 자동 필터링**하는 파이프라인 구조
""")


# ----------------------------------------------------------
# Tab: Head Role Analysis
# ----------------------------------------------------------

_head_path = os.path.join(os.path.dirname(__file__), "data", "stage0", "head_analysis.npz")
_has_head = os.path.exists(_head_path)

with tabs[2]:
    if not _has_head:
        st.warning("Head analysis 데이터가 없습니다. per-head attention 캡처를 실행하세요.")
    else:
        _hd = np.load(_head_path, allow_pickle=True)
        _h_variates = [str(v) for v in _hd['variates']]
        _h_nheads = int(_hd['n_heads'])
        _h_ctx = int(_hd['ctx_patches'])
        _h_attn = _hd['head_attention']  # [heads, T, T]
        _h_nvar = len(_h_variates)

        st.subheader(f"Attention Head Role Analysis (Layer 11, {_h_nheads} Heads)")
        st.markdown(f"""
Multi-head attention의 각 head가 **서로 다른 관점**으로 CPI 예측에 기여.
12개 head × 64차원 = 768차원. Head 평균이 아닌 **개별 head의 역할 분화**를 분석.
""")

        # Compute per-head importance
        _target_sl = slice(0, _h_ctx)
        _h_imp = {}  # head -> {var: importance}
        for hi in range(_h_nheads):
            a = _h_attn[hi]
            imp = {}
            for j, vn in enumerate(_h_variates):
                ks, ke = j * _h_ctx, (j + 1) * _h_ctx
                block = a[_target_sl, ks:ke]
                imp[vn] = float(block.sum() / _h_ctx)
            _h_imp[hi] = imp

        # ── 1. Overview Heatmap: Head × Variable ──
        st.markdown("### Head × Variable Attention Heatmap")
        st.caption("각 셀 = CPI target이 해당 변수에 부여한 attention (head별). 행합 ≈ 1.0")

        cov_names_raw = [v for v in _h_variates if v != "CPI_KR_YoY"]
        cov_names = [vl(v) for v in cov_names_raw]
        heatmap_data = []
        for hi in range(_h_nheads):
            row = [_h_imp[hi].get(v, 0) for v in cov_names_raw]
            heatmap_data.append(row)
        heatmap_arr = np.array(heatmap_data)

        # Add self-attention column
        self_attns = [_h_imp[hi]["CPI_KR_YoY"] for hi in range(_h_nheads)]

        fig_hm = go.Figure(data=go.Heatmap(
            z=heatmap_arr,
            x=cov_names,
            y=[f"Head {i}" for i in range(_h_nheads)],
            colorscale="YlOrRd",
            hovertemplate="Head %{y}<br>%{x}: %{z:.4f}<extra></extra>",
        ))
        fig_hm.update_layout(
            height=420, margin=dict(t=30, b=80),
            xaxis=dict(tickangle=-40, tickfont_size=10),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.divider()

        # ── 2. Head Classification ──
        st.markdown("### Head 유형 분류")
        st.markdown("""
| 유형 | 특징 | 해석 |
|------|------|------|
| **AUD_USD 전용** | Self-attn 80%+, 공변량 중 AUD_USD가 95%+ 독점 | 단일 변수의 환율 채널에 전담. AUD_USD attention이 압도적인 이유. |
| **균등 분산** | 공변량 고르게 분포 (HHI ≈ uniform) | 다변량 cross-variate 관계를 폭넓게 탐색. 실질적 multi-variate 분석 수행. |
| *주제 특화* (해당 없음) | 특정 카테고리 변수 집중 (HHI > uniform×1.3) | 현재 모델에서는 관측되지 않으나, 변수 구성이나 타겟이 바뀌면 특정 경제 채널(예: 금리, 원자재)에 특화된 head가 나타날 수 있음. |
""")

        # Classify heads
        head_classes = []
        for hi in range(_h_nheads):
            self_a = _h_imp[hi]["CPI_KR_YoY"]
            cov = {k: v for k, v in _h_imp[hi].items() if k != "CPI_KR_YoY"}
            total_cov = sum(cov.values())
            if total_cov == 0:
                head_classes.append(("Unknown", self_a, "", 0))
                continue
            shares = [v / total_cov for v in cov.values()]
            hhi = sum(s ** 2 for s in shares)
            top_var = max(cov, key=lambda v: cov[v])
            top_share = cov[top_var] / total_cov

            if top_share > 0.90:
                cls = f"AUD_USD 전용"
            elif hhi < (1.0 / len(shares)) * 1.3:
                cls = "균등 분산"
            else:
                cls = "주제 특화"
            head_classes.append((cls, self_a, top_var, top_share))

        # Metrics
        n_aud = sum(1 for c, _, _, _ in head_classes if "AUD" in c)
        n_dist = sum(1 for c, _, _, _ in head_classes if "균등" in c)
        mc1, mc2 = st.columns(2)
        mc1.metric("AUD_USD 전용", f"{n_aud} heads")
        mc2.metric("균등 분산", f"{n_dist} heads")

        # Per-head detail cards
        for hi in range(_h_nheads):
            cls, self_a, top_var, top_share = head_classes[hi]
            cov = {k: v for k, v in _h_imp[hi].items() if k != "CPI_KR_YoY"}
            top5 = sorted(cov, key=lambda v: -cov[v])[:5]
            top5_str = ", ".join([f"**{v}**({cov[v]:.2%})" for v in top5])

            if "AUD" in cls:
                icon = "🔴"
            elif "균등" in cls:
                icon = "🟢"
            else:
                icon = "🟡"

            with st.expander(f"{icon} Head {hi} — {cls} (self-attn {self_a:.1%})"):
                st.markdown(f"Top 5: {top5_str}")

                # Mini bar chart for this head
                fig_mini = go.Figure()
                sorted_cov = sorted(cov, key=lambda v: -cov[v])
                fig_mini.add_trace(go.Bar(
                    x=sorted_cov, y=[cov[v] for v in sorted_cov],
                    marker_color=["#DC2626" if v == "AUD_USD" else "#3B82F6" for v in sorted_cov],
                ))
                fig_mini.update_layout(height=250, margin=dict(t=10, b=60),
                                       xaxis=dict(tickangle=-40, tickfont_size=9),
                                       yaxis=dict(title="Attention"))
                st.plotly_chart(fig_mini, use_container_width=True)

        st.divider()

        # ── 3. Temporal Focus per Head ──
        st.markdown("### Temporal Focus — 시간적 초점")
        st.caption("각 head가 CPI 자기 자신의 과거 중 어느 시점에 집중하는지. Q1=가장 오래된 분기, Q4=가장 최근 분기.")

        temporal_data = []
        for hi in range(_h_nheads):
            a = _h_attn[hi]
            self_block = a[_target_sl, _target_sl]
            temporal = self_block.mean(axis=0)
            q = _h_ctx // 4
            quarters = [float(temporal[:q].mean()), float(temporal[q:2*q].mean()),
                        float(temporal[2*q:3*q].mean()), float(temporal[3*q:].mean())]
            peak = _h_ctx - int(np.argmax(temporal))

            if quarters[3] > quarters[0] * 3:
                pat = "Recent-focused"
            elif quarters[0] > quarters[3] * 3:
                pat = "Distant-focused"
            elif 10 <= peak <= 14:
                pat = "~12M lag"
            elif max(quarters) / (min(quarters) + 1e-10) < 1.5:
                pat = "Diffuse"
            else:
                pat = "Mixed"
            temporal_data.append((hi, quarters, peak, pat))

        fig_temp = go.Figure()
        for hi, quarters, peak, pat in temporal_data:
            fig_temp.add_trace(go.Bar(
                name=f"H{hi} ({pat})",
                x=["Q1 (oldest)", "Q2", "Q3", "Q4 (recent)"],
                y=quarters,
            ))
        fig_temp.update_layout(
            barmode="group", height=400, margin=dict(t=30, b=50),
            yaxis=dict(title="Mean Attention"),
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        temp_df = pd.DataFrame([{
            "Head": f"Head {hi}",
            "Q1 (oldest)": f"{q[0]:.4f}",
            "Q2": f"{q[1]:.4f}",
            "Q3": f"{q[2]:.4f}",
            "Q4 (recent)": f"{q[3]:.4f}",
            "Peak": f"{peak}M ago",
            "Pattern": pat,
        } for hi, q, peak, pat in temporal_data])
        with st.expander("Temporal Focus 상세"):
            st.dataframe(temp_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── 4. Head Correlation Matrix ──
        st.markdown("### Head 간 랭킹 상관 (Spearman)")
        st.caption("같은 유형끼리 높은 상관, 다른 유형끼리 낮거나 음의 상관 → 역할 분화 확인")

        from scipy.stats import spearmanr as _spearmanr
        _vlist = [v for v in _h_variates if v != "CPI_KR_YoY"]
        _hranks = {}
        for hi in range(_h_nheads):
            c = {k: _h_imp[hi][k] for k in _vlist}
            r = sorted(_vlist, key=lambda v: -c[v])
            _hranks[hi] = [r.index(v) for v in _vlist]

        corr_matrix = np.zeros((_h_nheads, _h_nheads))
        for hi in range(_h_nheads):
            for hj in range(_h_nheads):
                if hi == hj:
                    corr_matrix[hi][hj] = 1.0
                else:
                    sp, _ = _spearmanr(_hranks[hi], _hranks[hj])
                    corr_matrix[hi][hj] = sp

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=[f"H{i}" for i in range(_h_nheads)],
            y=[f"H{i}" for i in range(_h_nheads)],
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            hovertemplate="H%{y} ↔ H%{x}: ρ=%{z:.2f}<extra></extra>",
        ))
        fig_corr.update_layout(height=420, margin=dict(t=30, b=50))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.divider()

        # ── 5. Interpretation Summary ──
        st.markdown("### 해석 요약")
        st.markdown(f"""
**{_h_nheads}개 head 중 {n_aud}개가 AUD_USD에 전적으로 집중** — 이것이 AUD_USD의 attention이
다른 변수 대비 압도적인 이유. Head 평균 시 {n_aud}개 head의 AUD_USD attention이 합산됨.

**나머지 {n_dist}개 head가 실질적인 multi-variate 분석 수행:**
- 다양한 거시 변수를 고르게 참조 (HHI ≈ uniform)
- 개별 head의 top 변수를 보면 JP_REER, CN_Interbank3M, US_NFP, Pork, KR_MfgProd 등 카테고리가 다양
- 특정 주제에 명확히 특화된 head는 없으나, 각 head가 미세하게 다른 변수 조합에 주목

**시사점:**
- AUD_USD 전용 head가 다수(4개) 존재 → 모델이 AUD_USD를 특별히 중요한 신호로 취급
- 나머지 head는 균등 분산이지만 **head 간 랭킹 상관이 낮아** 서로 다른 관점을 제공
- 개별 head로 변수를 선택하면 편향 위험 → **head 평균이 안전한 선택**
""")


# ----------------------------------------------------------
# Tab: Layer Method Comparison
# ----------------------------------------------------------

with tabs[3]:
    st.subheader("Attention 탐색 방법론 비교")
    st.markdown("""
변수 선택 시 attention을 **어떤 방식으로 집계**하느냐에 따라 결과가 달라질 수 있다.
5가지 방법을 동일 조건(Stage 1, 25변수)에서 비교하여 최적 방법을 검증.
""")

    # ── 1. Methods Overview ──
    st.markdown("### 비교 대상 5가지 방법")

    st.markdown("""
| 방법 | 원리 | 수식/코드 |
|------|------|----------|
| **Last Layer** | 마지막 레이어(Layer 11) attention만 사용 | `importance = attn[Layer 11].mean(heads)` |
| **All Layers Average** | 12개 레이어 전체 평균 | `importance = mean(attn[Layer 0..11])` |
| **Attention Rollout** | 레이어 간 attention 행렬 곱 (정보 전파 추적) | `rollout = Π (0.5·A[i] + 0.5·I)` |
| **Variance-Weighted** | 레이어별 attention 분산으로 가중평균 (gradient 대리) | `importance = Σ var(attn[i]) · attn[i]` |
| **Focus-Weighted** | attention 집중도(1-entropy)로 가중 | `importance = attn · (1 - entropy/max_entropy)` |
""")

    st.divider()

    # ── 2. Method Explanations ──
    st.markdown("### 각 방법 상세")

    with st.expander("Last Layer (현재 방식)"):
        st.markdown("""
**마지막 레이어(Layer 11)의 attention만 사용.** 8개 head를 평균.

최종 출력에 가장 직접적으로 기여하는 attention. Transformer에서 마지막 레이어는
예측 head로 전달되기 직전의 표현을 생성하므로, "모델이 예측 시 실제로 주목한 곳"에 가장 가까운 지표.
""")

    with st.expander("All Layers Average"):
        st.markdown("""
**12개 레이어의 attention을 균등 평균.**

모든 깊이의 관계를 포착하지만, 초기 레이어(Layer 0~3)의 일반적 패턴(토큰 위치 유사성,
저수준 feature 매칭)이 섞여 변수 선택 정밀도가 오히려 하락.

실험에서 Layer 5~8에서 BR_DiscountRate, BR_CPI 같은 harmful 변수가 상위에 올라오는 것이 확인됨.
""")

    with st.expander("Attention Rollout (Abnar & Zuidema, 2020)"):
        st.markdown("""
**레이어별 attention 행렬을 순서대로 곱하여 정보 전파 경로 추적.**

```python
rollout = I  # 단위행렬
for layer in range(12):
    A_hat = 0.5 * A[layer] + 0.5 * I   # residual connection 반영
    A_hat = normalize_rows(A_hat)
    rollout = rollout @ A_hat           # 누적 곱
```

`rollout[i][j]` = "입력 토큰 j의 정보가 최종 토큰 i에 도달한 누적 비율"

Layer 11에서 CPI가 AUD_USD를 볼 때, AUD_USD 토큰은 이미 이전 레이어에서
다른 변수 정보가 섞인 상태. Rollout은 이 **간접 전파**를 추적.

**문제**: 중간 레이어의 harmful 변수(BR_CPI, BR_DiscountRate)에 대한 attention이
누적되어 최종 랭킹 오염.
""")

    with st.expander("Variance-Weighted (Gradient 대리)"):
        st.markdown("""
**각 레이어의 attention이 변수 간 얼마나 차별적인지(분산)를 가중치로 사용.**

```python
layer_weight[i] = var(importance_across_variables[Layer i])
importance = Σ layer_weight[i] × attention[Layer i]
```

Gradient(`∂output/∂attention`)의 실용적 대리 지표.
진짜 gradient를 구하려면 SDPA에서 attention weight를 중간 텐서로 반환해야 하나,
`F.scaled_dot_product_attention`이 이를 지원하지 않아 분산을 대리로 사용.

**문제**: Layer 0이 가중치의 51.6%를 차지 — AUD_USD에 대한 극단적 편중이 분산을 키운 것이지,
Layer 0이 중요한 것이 아님.
""")

    with st.expander("Focus-Weighted (Entropy 기반)"):
        st.markdown("""
**attention이 분산되지 않고 집중된 query 토큰에 더 높은 가중치.**

```python
focus = 1 - entropy(attention_row) / max_entropy  # 0=uniform, 1=focused
importance = attention × focus
```

"확신 있게 본 것"만 중시하는 접근. 진짜 gradient-weighted attention
(`importance = attention × |∂output/∂attention|`)의 대리.

**한계**: GluonTS predictor가 `torch.no_grad()` 내에서 실행되어 gradient chain 단절.
Entropy를 대리로 사용했으나 Last Layer 랭킹과 거의 동일하여 추가 정보 없음.
""")

    st.divider()

    # ── 3. RMSE Results ──
    st.markdown("### Incremental Addition RMSE 비교")
    st.caption("각 방법의 랭킹 순서대로 변수를 1개씩 추가하며 RMSE 측정. 최적 N에서의 RMSE로 비교.")

    # Results from experiments (verified values from same-run comparison)
    method_results = [
        ("Last Layer", "~1.145", 18, "일관되게 최선. 최종 예측 직전의 attention이 변수 선택에 가장 적합.", "#10B981"),
        ("All Layers Average", "~1.161", 20, "초기 레이어의 일반적 패턴(위치 유사성)이 노이즈로 작용.", "#6366F1"),
        ("Attention Rollout", "~1.160", 22, "중간 레이어의 harmful 변수 attention이 누적되어 오염.", "#F59E0B"),
        ("Variance-Weighted", "~1.162", 10, "Layer 0의 AUD_USD 편중이 분산을 지배, 가중치 왜곡.", "#EF4444"),
        ("Focus-Weighted", "~1.169", 21, "gradient 추출 불가로 entropy 대리 사용. 랭킹 거의 동일.", "#9CA3AF"),
    ]

    # Bar chart
    fig_methods = go.Figure()
    fig_methods.add_trace(go.Bar(
        x=[m[0] for m in method_results],
        y=[float(m[1].replace("~", "")) for m in method_results],
        marker_color=[m[4] for m in method_results],
        text=[f"N={m[2]}" for m in method_results],
        textposition="auto",
        hovertemplate="%{x}<br>RMSE: %{y:.3f}<br>최적 N: %{text}<extra></extra>",
    ))
    fig_methods.add_hline(y=1.1895, line_dash="dot", line_color="#888",
                          annotation_text="기존 11변수 (1.1895)", annotation_position="bottom right")
    fig_methods.update_layout(
        yaxis=dict(title="Optimal RMSE (pp)", range=[1.13, 1.18]),
        height=400, margin=dict(t=30, b=50),
    )
    st.plotly_chart(fig_methods, use_container_width=True)

    # Detail table
    meth_df = pd.DataFrame([{
        "방법": m[0],
        "최적 RMSE": m[1],
        "최적 N": m[2],
        "비고": m[3],
    } for m in method_results])
    st.dataframe(meth_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── 4. Layer-wise Top 5 ──
    st.markdown("### 레이어별 Top 5 변수")
    st.caption("중간 레이어(5~8)에서 harmful 변수(BR_CPI, BR_DiscountRate)가 상위 진입 → All Average, Rollout에서 노이즈 유발")

    layer_top5 = {
        0: ["AUD_USD", "US_DepInstCredit", "US_TradeTransEmp", "JP_Interbank3M", "US_ExportPI"],
        1: ["AUD_USD", "Energy_Idx", "PPI_Metals", "BRL_USD", "US_DepInstCredit"],
        2: ["AUD_USD", "US_UnempRate", "Pork", "BR_CPI ❌", "CN_Interbank3M"],
        3: ["AUD_USD", "US_DepInstCredit", "Pork", "THB_USD", "US_TradeTransEmp"],
        4: ["AUD_USD", "KR_MfgProd", "Pork", "BRL_USD", "US_ExportPI"],
        5: ["BRL_USD", "BR_DiscountRate ❌", "BR_CPI ❌", "Pork", "PPI_CopperNickel"],
        6: ["US_ExportPI", "BR_DiscountRate ❌", "BR_CPI ❌", "KR_PPI_Energy", "AUD_USD"],
        7: ["BR_DiscountRate ❌", "BR_CPI ❌", "US_ExportPI", "KR_PPI_Energy", "UK_10Y_Bond"],
        8: ["AUD_USD", "BR_DiscountRate ❌", "UK_10Y_Bond", "BRL_USD", "PPI_DeepSeaFrt"],
        9: ["AUD_USD", "US_DepInstCredit", "BR_DiscountRate ❌", "US_Mortgage15Y", "BRL_USD"],
        10: ["AUD_USD", "US_Mortgage15Y", "KC_FSI", "BRL_USD", "BR_CPI ❌"],
        11: ["AUD_USD", "CN_Interbank3M", "US_UnempRate", "JP_REER", "JP_Interbank3M"],
    }

    fig_layer = go.Figure()
    for rank in range(5):
        vars_at_rank = [layer_top5[li][rank] for li in range(12)]
        colors = ["#EF4444" if "❌" in v else "#3B82F6" for v in vars_at_rank]
        fig_layer.add_trace(go.Scatter(
            x=[f"L{li}" for li in range(12)],
            y=[rank] * 12,
            mode="markers+text",
            marker=dict(size=10, color=colors),
            text=vars_at_rank,
            textposition="top center",
            textfont=dict(size=8),
            showlegend=False,
            hovertemplate="Layer %{x}, Rank %{y}<br>%{text}<extra></extra>",
        ))

    fig_layer.update_layout(
        yaxis=dict(title="Attention Rank", autorange="reversed", dtick=1),
        xaxis=dict(title="Layer"),
        height=350, margin=dict(t=30, b=50),
    )
    st.plotly_chart(fig_layer, use_container_width=True)

    st.markdown(
        "<span style='color:#EF4444'>●</span> Harmful 변수 (BR_CPI, BR_DiscountRate) — "
        "Layer 5~8에서 상위 진입, Layer 11에서는 탈락",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 5. Ranking Correlation ──
    st.markdown("### 방법 간 Spearman 순위 상관")
    st.caption("Last Layer와 다른 방법들 간 상관이 낮음 → 방법 선택이 결과에 큰 영향")

    corr_data = {
        "": ["Last Layer", "All Avg", "Rollout"],
        "Last Layer": [1.00, -0.335, 0.352],
        "All Avg": [-0.335, 1.00, 0.373],
        "Rollout": [0.352, 0.373, 1.00],
    }
    corr_df = pd.DataFrame(corr_data)

    fig_rc = go.Figure(data=go.Heatmap(
        z=[[1.00, -0.335, 0.352],
           [-0.335, 1.00, 0.373],
           [0.352, 0.373, 1.00]],
        x=["Last Layer", "All Avg", "Rollout"],
        y=["Last Layer", "All Avg", "Rollout"],
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in
              [[1.00, -0.335, 0.352], [-0.335, 1.00, 0.373], [0.352, 0.373, 1.00]]],
        texttemplate="%{text}",
        hovertemplate="%{y} ↔ %{x}: ρ=%{z:.3f}<extra></extra>",
    ))
    fig_rc.update_layout(height=350, margin=dict(t=30, b=50))
    st.plotly_chart(fig_rc, use_container_width=True)

    st.markdown("""
**Last Layer vs All Average: ρ = −0.335** (음의 상관!) → 두 방법이 **정반대 방향**으로 변수를 평가.
All Average가 높게 본 변수를 Last Layer는 낮게 보고, 그 반대도 성립.
""")

    st.divider()

    # ── 6. Conclusion ──
    st.markdown("### 결론")
    st.success("""
**마지막 레이어(Last Layer) attention이 변수 선택에 최적.**

- 최종 예측 직전의 attention → "모델이 실제로 뭘 보고 예측했는가"에 가장 가까운 지표
- 초기/중간 레이어는 harmful 변수의 허위 상관이 포함 → Rollout, All Average에서 노이즈 유발
- Gradient-weighted는 이론적으로 우수하나, SDPA 구현상 gradient 추출 불가
- **5가지 방법 비교로 Last Layer 선택의 정당성을 실증**
""")

    st.info("""
**Residual Vector 분석은?** 마지막 레이어의 768차원 출력 벡터는 변수별 기여도 분해가 어려워
변수 선택 목적으로는 부적합. Attention(어디를 봤는가) + Ablation(실제 기여) 조합이면 충분.
Residual 분석은 mechanistic interpretability(SAE, probing) 방향 — 별개의 연구 주제.
""")


# ----------------------------------------------------------
# (Cross-Variate Heatmap tab removed — content in Stage 0→1 Pipeline tab)
# ----------------------------------------------------------

if False:  # Cross-Variate REMOVED
    st.subheader("Cross-Variate Attention Matrix")
    st.caption(
        "**행(Row) = Query** (주목하는 변수)  |  "
        "**열(Col) = Key** (주목받는 변수)  |  "
        "값 = Query 변수의 전체 attention 중 Key로 향하는 **비율** (행 합계 ≈ 1.0)"
    )

    cross_df_full = ana.cross_variate_matrix()
    # active_variates 필터링
    cross_df = cross_df_full.loc[active_variates, active_variates]
    n_hm = len(active_variates)

    # 퍼센트 레이블 텍스트
    z_pct = [[f"{v*100:.1f}%" for v in row] for row in cross_df.values]

    fig1 = go.Figure(data=go.Heatmap(
        z=cross_df.values,
        x=cross_df.columns.tolist(),
        y=cross_df.index.tolist(),
        colorscale="YlOrRd",
        zmin=0, zmax=0.6,
        text=z_pct,
        texttemplate="%{text}",
        hovertemplate=(
            "Query: <b>%{y}</b><br>"
            "Key:   <b>%{x}</b><br>"
            "Attention 비율: <b>%{z:.1%}</b>"
            "<extra></extra>"
        ),
        colorbar=dict(title="비율", thickness=14, tickformat=".0%", dtick=0.1),
        xgap=1, ygap=1,
    ))

    # 타겟 행 강조 (active_variates 기준 인덱스)
    t_hm = active_variates.index(cfg.target_name) if cfg.target_name in active_variates else 0
    fig1.add_shape(
        type="rect",
        x0=-0.5, x1=n_hm - 0.5,
        y0=t_hm - 0.5, y1=t_hm + 0.5,
        line=dict(color="#3A7BD5", width=2.5),
        fillcolor="rgba(58,123,213,0.06)",
    )

    fig1.update_layout(
        xaxis=dict(title="Key (attended-to)", title_font_size=14,
                   tickangle=-35, tickfont_size=13),
        yaxis=dict(title="Query (attending)", title_font_size=14,
                   autorange="reversed", tickfont_size=13),
        height=max(440, 65 * n_hm),
        margin=dict(l=140, r=80, t=40, b=130),
    )
    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("📋 수치 테이블"):
        st.dataframe(cross_df.round(5), use_container_width=True)


# ----------------------------------------------------------
# (Variable Importance tab removed — content in Stage 0→1 Pipeline tab)
# ----------------------------------------------------------

if False:  # Variable Importance REMOVED
    st.subheader(f"Variable Importance — {cfg.target_name} 예측 기준")
    st.caption(
        "타겟 변수의 attention 중 각 변수로 향하는 **비율**. "
        "자기 참조(self-attention)를 포함한 전체 분포를 보여줍니다."
    )

    imp_full = ana.target_importance()
    imp = imp_full[imp_full.index.isin(active_variates)].sort_values(ascending=True)
    n_imp = len(imp)

    colors = ["#E24B4A" if v == cfg.target_name else "#4C9BE8" for v in imp.index]
    max_val = float(imp.values.max())

    fig2 = go.Figure(go.Bar(
        x=imp.values,
        y=imp.index.tolist(),
        orientation="h",
        marker_color=colors,
        marker_line=dict(width=0),
        hovertemplate="%{y}: <b>%{x:.1%}</b><extra></extra>",
        text=[f"{v:.1%}" for v in imp.values],
        textposition="outside",
        textfont=dict(size=13),
    ))
    fig2.add_annotation(
        x=0.99, y=0.02, xref="paper", yref="paper",
        text="<span style='color:#D62728'>■</span> Target &nbsp;<span style='color:#1A6FD4'>■</span> Covariate",
        showarrow=False, font=dict(size=13), align="right",
    )
    fig2.update_layout(
        xaxis=dict(title="Attention 비율", title_font_size=14,
                   range=[0, min(max_val * 1.25, 1.0)],
                   tickformat=".0%", tickfont_size=13),
        yaxis=dict(title="Variable", title_font_size=14, tickfont_size=14),
        height=max(380, 55 * n_imp),
        showlegend=False,
        margin=dict(l=150, r=130, t=40, b=60),
    )
    st.plotly_chart(fig2, use_container_width=True)

    rank_df = (
        imp.sort_values(ascending=False)
        .rename("Attention Weight")
        .reset_index()
        .rename(columns={"index": "Variable"})
    )
    rank_df["Tier"] = rank_df["Variable"].map(lambda v: TIER_LABELS.get(v, "—"))
    rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))
    rank_df["Attention Weight"] = rank_df["Attention Weight"].map(lambda v: f"{v:.1%}")
    st.dataframe(rank_df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------
# Tab: Temporal Patterns
# ----------------------------------------------------------

with tabs[4]:
    st.subheader("Temporal Attention Patterns — 최적 18변수 (SOTA)")
    st.caption(
        "Stage 0→1 최적 18변수 모델 (Last Layer, head 평균) 기준. "
        "타겟 CPI가 각 변수의 **과거 어느 시점**을 주목하는지 분석."
    )

    # Load from head_analysis.npz (최적 18변수 + target, 12 heads)
    if not _has_head:
        st.warning("head_analysis.npz가 없습니다.")
    else:
        _tp_attn = _hd['head_attention']  # [12 heads, T, T]
        _tp_vars = [str(v) for v in _hd['variates']]
        _tp_ctx = int(_hd['ctx_patches'])
        _tp_nvar = len(_tp_vars)
        _tp_covs = [v for v in _tp_vars if v != "CPI_KR_YoY"]

        # Head-averaged attention (last layer)
        _tp_avg = _tp_attn.mean(axis=0)  # [T, T]
        _tp_target_sl = slice(0, _tp_ctx)

        # Temporal attention per variable: CPI query의 마지막 12패치가 각 변수의 과거 패치에 주는 attention
        _tp_temporal = {}
        _tp_peak_lags = {}
        for j, vn in enumerate(_tp_vars):
            ks, ke = j * _tp_ctx, (j + 1) * _tp_ctx
            # 마지막 12 query 패치의 평균
            q_start = max(0, _tp_ctx - 12)
            block = _tp_avg[q_start:_tp_ctx, ks:ke]
            t_arr = block.mean(axis=0)
            _tp_temporal[vn] = t_arr
            peak_idx = int(np.argmax(t_arr))
            _tp_peak_lags[vn] = _tp_ctx - 1 - peak_idx

        # ── 1. Self-attention temporal ──
        st.markdown("### 1. CPI → CPI 자기 참조 vs 최상위 공변량")

        # Importance for sorting
        _tp_imp = {}
        for j, vn in enumerate(_tp_vars):
            ks, ke = j * _tp_ctx, (j + 1) * _tp_ctx
            block = _tp_avg[_tp_target_sl, ks:ke]
            _tp_imp[vn] = float(block.sum() / _tp_ctx)
        top_cov_tp = max(_tp_covs, key=lambda v: _tp_imp[v])

        t_self = _tp_temporal["CPI_KR_YoY"]
        t_top = _tp_temporal[top_cov_tp]
        recent_cut = max(0, _tp_ctx - 12)

        fig_tp1 = make_subplots(rows=1, cols=2,
            subplot_titles=[
                "CPI_KR_YoY → CPI_KR_YoY (자기 참조)",
                f"CPI_KR_YoY → {top_cov_tp} (최상위 공변량)",
            ], horizontal_spacing=0.10)

        for ci, (series, color) in enumerate([(t_self, "#E24B4A"), (t_top, "#3B82F6")], start=1):
            x_vals = list(range(len(series)))
            bar_clrs = [color if i >= recent_cut else "rgba(200,200,200,0.3)" for i in x_vals]
            fig_tp1.add_trace(go.Bar(x=x_vals, y=series, marker_color=bar_clrs,
                hovertemplate="Patch %{x}: %{y:.5f}<extra></extra>", showlegend=False),
                row=1, col=ci)
            fig_tp1.add_vrect(x0=recent_cut-0.5, x1=len(series)-0.5,
                fillcolor="rgba(255,215,0,0.12)", line_width=0, row=1, col=ci)

        fig_tp1.update_xaxes(title_text="Past Patch Index")
        fig_tp1.update_yaxes(title_text="Mean Attention", col=1)
        fig_tp1.update_layout(height=380, margin=dict(t=70, b=50))
        st.plotly_chart(fig_tp1, use_container_width=True)

        self_peak = _tp_peak_lags["CPI_KR_YoY"]
        top_peak = _tp_peak_lags[top_cov_tp]
        st.caption(
            f"CPI 자기 참조 peak: **-{self_peak}M ago** | "
            f"{top_cov_tp} peak: **-{top_peak}M ago** | 황금 영역 = 최근 12개월"
        )

        # Auto-interpretation
        with st.expander("📖 패턴 해석", expanded=True):
            _self_recent = self_peak <= 12
            _top_distant = top_peak >= 60

            st.markdown(f"""
**CPI 자기 참조** (peak: -{self_peak}M):
{"- 최근 값에 집중 → 자연스러운 AR(자기회귀) 패턴. '직전 CPI가 다음 CPI를 가장 잘 예측'" if _self_recent else f"- {self_peak}개월 전에 집중 — 기저효과 또는 과거 인플레이션 구조 참조"}

**{top_cov_tp}** (peak: -{top_peak}M):
""")
            if _top_distant:
                # Map peak to approximate date
                _ctx_start = pd.Timestamp("2023-01-01") - pd.Timedelta(days=120*32)
                _peak_date = (_ctx_start + pd.Timedelta(days=(120-top_peak)*32)).strftime('%Y년 %m월')
                st.markdown(f"""
- 과거 초기({_peak_date} 부근)에 attention이 집중되고, **최근에는 거의 보지 않음**
- 이 시기는 {top_cov_tp}의 **구조적 전환점** 또는 **극단값**이 있었던 시기일 가능성
- 모델이 {top_cov_tp}의 **과거 수준(level)**을 기준점으로 참조 — "과거 대비 현재가 어느 위치인가"를 파악하는 용도
- **최근 값을 안 보는 것은 {top_cov_tp}의 단기 변동이 아니라 장기 수준 정보가 예측에 기여한다는 의미**
- 이는 Attention이 높지만 **최근 값 변화에 둔감한** 이유를 설명 — Instance Normalization이 최근 수준 이동을 상쇄할 수 있음
""")
            elif top_peak <= 12:
                st.markdown(f"""
- 최근 12개월에 집중 → **{top_cov_tp}의 단기 변동이 CPI에 직접 전달**
- 가격 전달 경로(환율, 원자재 등)의 전형적 패턴
""")
            else:
                st.markdown(f"""
- 중기({top_peak}개월 전) 집중 → **경기 사이클 또는 통화정책 시차 효과** 반영 가능
""")

        st.divider()

        # ── 2. Heatmap: 전 공변량 × 시간 ──
        st.markdown("### 2. 공변량 × 시간 Attention Heatmap")
        st.caption("최적 18변수의 temporal attention. Attention 순위순 정렬. 최근 36개월 확대.")

        sorted_covs_tp = sorted(_tp_covs, key=lambda v: -_tp_imp[v])
        sorted_covs_tp_label = [vl(v) for v in sorted_covs_tp]
        heat_matrix_tp = np.array([_tp_temporal[cov] for cov in sorted_covs_tp])
        show_recent_tp = min(36, _tp_ctx)
        heat_recent_tp = heat_matrix_tp[:, -show_recent_tp:]

        fig_heat_tp = go.Figure(data=go.Heatmap(
            z=heat_recent_tp, x=list(range(show_recent_tp)), y=sorted_covs_tp_label,
            colorscale="YlOrRd",
            hovertemplate="<b>%{y}</b><br>-%{customdata}M ago<br>Attn: %{z:.5f}<extra></extra>",
            customdata=np.tile(np.arange(show_recent_tp, 0, -1), (len(sorted_covs_tp_label), 1)),
        ))
        fig_heat_tp.update_layout(
            xaxis=dict(title=f"Past Context (최근 {show_recent_tp}개월)",
                tickvals=[i for i in range(show_recent_tp) if (show_recent_tp-i)%6==0],
                ticktext=[f"-{show_recent_tp-i}M" for i in range(show_recent_tp) if (show_recent_tp-i)%6==0]),
            yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
            height=max(350, len(sorted_covs_tp)*30+100), margin=dict(t=30, b=60, l=140),
        )
        st.plotly_chart(fig_heat_tp, use_container_width=True)

        st.divider()

        # ── 3. Peak Lag 비교 ──
        st.markdown("### 3. 변수별 Peak Attention Lag")
        st.caption("각 변수에서 attention이 가장 높은 시점이 현재 기준 몇 개월 전인지.")

        lag_df_tp = pd.DataFrame({
            "Variable": sorted_covs_tp_label,
            "Peak Lag": [_tp_peak_lags[v] for v in sorted_covs_tp],
            "Attn Score": [f"{_tp_imp[v]:.2%}" for v in sorted_covs_tp],
        }).sort_values("Peak Lag")

        max_lag_tp = max(_tp_peak_lags[v] for v in _tp_covs) or 1
        lag_colors_tp = [
            f"rgb({min(255, int(row['Peak Lag']/max_lag_tp*255))}, "
            f"{min(255, int((1-row['Peak Lag']/max_lag_tp)*200))}, 80)"
            for _, row in lag_df_tp.iterrows()
        ]

        fig_lag_tp = go.Figure(go.Bar(
            x=lag_df_tp["Variable"], y=lag_df_tp["Peak Lag"],
            marker_color=lag_colors_tp,
            text=[f"{lag}M" for lag in lag_df_tp["Peak Lag"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Peak: -%{y}M ago<extra></extra>",
        ))
        fig_lag_tp.update_layout(
            yaxis=dict(title="Peak Lag (months ago)"),
            height=400, margin=dict(t=30, b=60),
        )
        st.plotly_chart(fig_lag_tp, use_container_width=True)

        # Auto-interpretation for Peak Lag chart
        _n_distant = sum(1 for v in _tp_covs if _tp_peak_lags[v] >= 60)
        _n_recent = sum(1 for v in _tp_covs if _tp_peak_lags[v] <= 12)
        _recent_vars = [v for v in _tp_covs if _tp_peak_lags[v] <= 12]
        _distant_vars = [v for v in _tp_covs if _tp_peak_lags[v] >= 100]

        with st.expander("📖 Peak Lag 해석", expanded=True):
            st.markdown(f"""
**전체 분포**: {len(_tp_covs)}개 변수 중 **{_n_recent}개가 최근(≤12M)**, **{_n_distant}개가 100M+ 과거**에 peak.

**최근 집중 변수 (경제적 의미 명확)**:
{chr(10).join(f'- **{v}** (-{_tp_peak_lags[v]}M): 최근 값이 CPI에 직접 전달되는 경로' for v in _recent_vars)}

**과거 집중 변수 ({len(_distant_vars)}개, 100M+)**:
{', '.join(f'{v}(-{_tp_peak_lags[v]}M)' for v in _distant_vars)}

이들이 8~10년 전 값에 peak를 보이는 것은 두 가지 가능성:
1. **수준(level) 참조**: 모델이 해당 변수의 "과거 기준 수준"을 앵커로 사용하여 현재 위치를 가늠.
   예: AUD_USD가 2013년(패리티 근처)의 수준을 참조 → "그때 대비 지금 얼마나 떨어졌는가"
2. **Artifact**: 컨텍스트 초기 구간에 특이 패턴이 있어 모델이 과적합.
   Instance Normalization이 최근 수준 이동을 상쇄하면서 상대적으로 과거 극단값에 attention이 몰릴 수 있음.

**결론**: Peak Lag만으로 변수의 가치를 판단하면 안 됨. **Ablation ΔRMSE**(실제 제거 시 RMSE 변화)와 교차 확인 필수.
Distant Focus이지만 Ablation에서 helpful인 변수(예: AUD_USD ΔRMSE=+0.017)는 수준 정보가 실제로 기여하는 것.
""")

        st.divider()

        # ── 4. Lag 유형 분류 ──
        st.markdown("### 4. Temporal Lag 유형 분류")

        RECENT_TH = 12
        DISTANT_TH = 60
        _tp_lag_cats = {}
        for cov in _tp_covs:
            t_arr = _tp_temporal[cov]
            n_pts = len(t_arr)
            recent_mass = float(np.sum(t_arr[-RECENT_TH:])) / (float(np.sum(t_arr)) + 1e-12)
            lag = _tp_peak_lags[cov]
            t_norm = t_arr / (np.sum(t_arr) + 1e-12)
            entropy = -float(np.sum(t_norm * np.log(t_norm + 1e-12)))
            concentration = 1 - entropy / np.log(n_pts)

            if lag <= RECENT_TH and recent_mass > 0.3:
                cat = "Recent Focus"
            elif lag >= DISTANT_TH:
                cat = "Distant Focus"
            elif concentration < 0.05:
                cat = "Diffuse"
            else:
                cat = "Mid-range"
            _tp_lag_cats[cov] = {"cat": cat, "lag": lag, "recent_mass": recent_mass, "conc": concentration}

        _cat_colors = {"Recent Focus": "#2CA02C", "Mid-range": "#EF9F27", "Distant Focus": "#D62728", "Diffuse": "#999999"}
        _cat_descs = {
            "Recent Focus": "최근 12개월 집중 — 가격 전달, 동행 지표",
            "Mid-range": "1~5년 전 집중 — 경기 사이클, 통화정책 시차",
            "Distant Focus": "5년+ 과거 — 장기 구조 또는 artifact",
            "Diffuse": "전 구간 균등 — 수준/추세 정보",
        }

        _cat_cols = st.columns(4)
        for col, cn in zip(_cat_cols, ["Recent Focus", "Mid-range", "Distant Focus", "Diffuse"]):
            members = [v for v, info in _tp_lag_cats.items() if info["cat"] == cn]
            col.markdown(
                f"<span style='color:{_cat_colors[cn]}'>■</span> **{cn}** ({len(members)})<br>"
                f"<small>{_cat_descs[cn]}</small>",
                unsafe_allow_html=True)
            for m in members:
                col.caption(f"  · {m} (-{_tp_lag_cats[m]['lag']}M)")

        # Summary table
        st.divider()
        _tp_lag_df = pd.DataFrame([{
            "Variable": cov,
            "Peak Lag": f"-{info['lag']}M",
            "Recent 12M": f"{info['recent_mass']:.1%}",
            "Concentration": f"{info['conc']:.3f}",
            "Type": info["cat"],
        } for cov, info in sorted(_tp_lag_cats.items(), key=lambda x: x[1]["lag"])])

        st.dataframe(_tp_lag_df, use_container_width=True, hide_index=True)

        with st.expander("📝 해석 가이드"):
            st.markdown("""
**이 분석은 최적 18변수 모델(Stage 0→1 SOTA)** 기준입니다.

**Peak Lag**: 모델이 해당 변수의 몇 개월 전 값에 가장 주목하는지.
- **짧은 lag (Recent)**: 최근 변동이 CPI에 직접 전달. 환율, 원자재 가격 등.
- **긴 lag (Distant)**: 먼 과거 참조. 장기 구조적 관계 또는 학습 데이터의 특이 패턴(artifact).

**Recent 12M Mass**: 전체 temporal attention 중 최근 12개월에 집중된 비율.

**Concentration**: 0=전 구간 균등, 1=특정 시점 집중.

**경제적 해석 예시**:
- AUD_USD(Recent Focus): 호주달러의 최근 움직임이 원자재 가격/글로벌 리스크를 즉시 반영
- JP_Interbank3M(Mid-range): 일본 금리의 과거 추세가 아시아 금융 환경에 시차를 두고 영향
- Diffuse 변수: 수준(level) 자체가 정보 — 특정 시점이 아니라 "현재가 과거 대비 어느 위치인가"
""")


# ----------------------------------------------------------
# (Layer Analysis tab removed — content in Layer Method Comparison tab)
# ----------------------------------------------------------

if False:  # Layer Analysis REMOVED
    st.subheader("Layer-by-Layer Attention Evolution")
    st.caption(
        "각 Transformer 레이어에서 cross-variate attention 변화. "
        "초기 레이어 = 로컬 시간 패턴, 후반 레이어 = 글로벌 변수 관계 경향."
    )

    layer_vars = active_variates
    col_q, col_k = st.columns(2)
    with col_q:
        q_idx = layer_vars.index(cfg.target_name) if cfg.target_name in layer_vars else 0
        query_var = st.selectbox("Query Variable", layer_vars, index=q_idx, key="l4_query")
    with col_k:
        key_var   = st.selectbox("Key Variable",   layer_vars, index=0,    key="l4_key")

    layer_df = ana.layer_comparison(query_var, key_var)
    layer_df["layer_num"] = (
        layer_df["layer"].str.extract(r"layers\.(\d+)").astype(int)
    )
    layer_df = layer_df.sort_values("layer_num")
    avg_val  = float(layer_df["attention"].mean())

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=layer_df["layer_num"], y=layer_df["attention"],
        mode="lines+markers",
        marker=dict(size=12, color="#D62728", line=dict(width=2, color="white")),
        line=dict(width=3, color="#D62728"),
        hovertemplate="Layer %{x}: <b>%{y:.1%}</b><extra></extra>",
        name=f"{query_var} → {key_var}",
    ))
    fig4.add_hline(
        y=avg_val, line_dash="dot", line_color="#888888", line_width=2,
        annotation_text=f"평균: {avg_val:.1%}",
        annotation_position="bottom right",
        annotation_font_size=13,
    )
    fig4.update_layout(
        xaxis=dict(
            title="Layer", title_font_size=14,
            tickmode="array",
            tickvals=layer_df["layer_num"].tolist(),
            ticktext=[f"L{n}" for n in layer_df["layer_num"].tolist()],
            tickfont_size=13,
        ),
        yaxis=dict(title="Attention 비율", tickformat=".0%", title_font_size=14),
        height=400, margin=dict(t=40, b=60),
        legend=dict(x=0.01, y=0.99, font=dict(size=13)),
    )
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("🗺️ 선택 레이어 전체 N×N 히트맵"):
        layer_names = hooks.get_layer_names()
        sel_layer = st.selectbox("Layer 선택", layer_names, key="l4_layer_sel",
                                 index=len(layer_names) - 1)
        cross_layer_full = ana.cross_variate_matrix(sel_layer)
        cross_layer = cross_layer_full.loc[active_variates, active_variates]
        fig4b = px.imshow(
            cross_layer, color_continuous_scale="YlOrRd",
            labels=dict(x="Key", y="Query", color="Attention"),
            title=sel_layer, aspect="equal",
        )
        fig4b.update_layout(height=max(350, 60 * len(active_variates)), margin=dict(t=60, b=80))
        st.plotly_chart(fig4b, use_container_width=True)


# ----------------------------------------------------------
# (2×2 Diagnostic tab removed — content in Stage 0→1 Pipeline tab)
# ----------------------------------------------------------

QUAD_COLORS = {
    "Confirmed Driver":      "#E24B4A",
    "Spurious Attention":    "#EF9F27",
    "Hidden Contributor":    "#4C9BE8",
    "Irrelevant":            "#AAAAAA",
}
QUAD_DESC = {
    "Confirmed Driver":      "Attention 높음 + 제거 시 성능 저하 → 핵심 예측 변수",
    "Spurious Attention":    "Attention 높음 + 제거해도 무방 → 중복/허위 상관 변수",
    "Hidden Contributor":    "Attention 낮음 + 제거 시 성능 저하 → 대체 불가 독자 정보",
    "Irrelevant":            "Attention·기여 모두 낮음 → 제거 후보",
}


def get_quadrant(a, delta_rmse, attn_thresh, delta_thresh):
    if a >= attn_thresh and delta_rmse >= delta_thresh:
        return "Confirmed Driver"
    elif a >= attn_thresh and delta_rmse < delta_thresh:
        return "Spurious Attention"
    elif a < attn_thresh and delta_rmse >= delta_thresh:
        return "Hidden Contributor"
    return "Irrelevant"


if False:  # 2x2 REMOVED
    st.subheader("2×2 Diagnostic: Attention × Ablation Impact")

    imp_all = ana.target_importance()

    # ── Attention threshold: 자기 참조 제외 후 공변량 간 균등 배분 ──
    self_attn_share = float(imp_all.iloc[cfg.target_idx])
    n_covariates    = cfg.n_variates - 1
    attn_thresh     = (1.0 - self_attn_share) / n_covariates if n_covariates > 0 else 0.5

    # ── Ablation ΔRMSE threshold: 0 (제거 시 성능 저하 여부) ──
    # ΔRMSE > 0: 제거하면 성능 악화 = 기여하는 변수
    # ΔRMSE ≤ 0: 제거해도 성능 유지/개선 = 중복 또는 노이즈
    delta_thresh = 0.0

    # ── 데이터 구성 ────────────────────────────────────────────
    if has_ablation:
        ab = abl_data
        abl_var_list = list(ab["abl_vars"])
        abl_delta_list = list(ab["abl_delta_rmse"])

        diag_vars = []
        diag_attn = []
        diag_delta = []
        for vi, vname in enumerate(variates_list):
            if vname == cfg.target_name:
                continue
            diag_vars.append(vname)
            diag_attn.append(float(imp_all.iloc[vi]))
            if vname in abl_var_list:
                diag_delta.append(float(abl_delta_list[abl_var_list.index(vname)]))
            else:
                diag_delta.append(0.0)

        diag_attn = np.array(diag_attn)
        diag_delta = np.array(diag_delta)
        x_label = "Cross-Variate Attention (%)"
        y_label = "Ablation ΔRMSE (pp)"
    else:
        diag_vars = [v for v in active_variates if v != cfg.target_name]
        diag_attn = np.array([float(imp_all[v]) for v in diag_vars])
        rng_abl = np.random.default_rng(seed=sum(ord(c) for c in "".join(variates_list)))
        diag_delta = rng_abl.uniform(-0.05, 0.10, size=len(diag_vars))
        x_label = "Cross-Variate Attention (%)"
        y_label = "Ablation ΔRMSE (mock, pp)"

    # ── 축 설명 ────────────────────────────────────────────────
    st.caption(
        "**X축**: 타겟(CPI) → 공변량 Attention 비율  |  "
        "**Y축**: 해당 변수 제거 시 RMSE 변화량 (ΔRMSE > 0 = 제거 시 악화)  |  "
        "타겟(CPI) 자신은 제외"
    )

    # ── Threshold 설정 근거 설명 ────────────────────────────────
    with st.expander("📏 Threshold 설정 근거", expanded=True):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(f"""
**Attention Threshold: `{attn_thresh:.1%}`**

- 타겟(CPI) 자기 참조 = **{self_attn_share:.1%}** (Transformer 보편적 특성)
- 외부 attention budget = {1-self_attn_share:.1%}
- 공변량 {n_covariates}개 균등 배분 시 각 **(1 − {self_attn_share:.1%}) / {n_covariates} = {attn_thresh:.1%}**
- 이 기준 **초과** = "균등 배분 이상으로 주목"
""")
        with col_t2:
            st.markdown(f"""
**Ablation Threshold: `ΔRMSE = 0`**

- ΔRMSE > 0: 변수를 **제거하면 RMSE 상승** → 예측에 기여
- ΔRMSE ≤ 0: 변수를 **제거해도 RMSE 유지/하락** → 중복 또는 노이즈
- 0 기준은 자연스러운 분기점: 성능에 도움이 되는가 아닌가
- Attention과 결합하여 "보기만 하는 변수" vs "실제 핵심 변수" 구분
""")

    if has_ablation:
        st.info(
            f"Baseline (10 covariates) RMSE = **{ab['baseline_rmse']:.4f} pp**  |  "
            "각 ΔRMSE는 해당 변수 1개를 제거한 후 재추론한 결과입니다.",
            icon="📐",
        )

    # ── 사분면 분류 ─────────────────────────────────────────────
    diag_df = pd.DataFrame({
        "Variable":  diag_vars,
        "Attention": diag_attn,
        "ΔRMSE":     diag_delta,
        "Quadrant":  [get_quadrant(a, d, attn_thresh, delta_thresh) for a, d in zip(diag_attn, diag_delta)],
        "Tier":      [TIER_LABELS.get(v, "—") for v in diag_vars],
    })

    fig5 = go.Figure()
    for quad, color in QUAD_COLORS.items():
        sub = diag_df[diag_df["Quadrant"] == quad]
        if sub.empty:
            continue
        fig5.add_trace(go.Scatter(
            x=sub["Attention"], y=sub["ΔRMSE"],
            mode="markers+text",
            marker=dict(size=18, color=color, line=dict(width=2, color="white")),
            text=sub["Variable"], textposition="top center",
            textfont=dict(size=14),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Attention: %{x:.1%}<br>"
                "ΔRMSE: %{y:+.4f} pp<br>"
                f"Quadrant: <b>{quad}</b><extra></extra>"
            ),
            name=quad,
        ))

    # Threshold 선
    fig5.add_vline(x=attn_thresh, line_dash="dash", line_color="rgba(80,80,80,0.5)", line_width=1.5,
                   annotation_text=f"(1−self)/(N−1) = {attn_thresh:.1%}",
                   annotation_position="top", annotation_font_size=12)
    fig5.add_hline(y=delta_thresh, line_dash="dash", line_color="rgba(80,80,80,0.5)", line_width=1.5,
                   annotation_text="ΔRMSE = 0",
                   annotation_position="right", annotation_font_size=12)

    # 축 범위
    all_attn_vals = list(diag_attn) + [attn_thresh]
    all_delta_vals = list(diag_delta) + [delta_thresh]
    attn_lo = max(0, min(all_attn_vals) * 0.6)
    attn_hi = max(all_attn_vals) * 1.4
    delta_lo = min(all_delta_vals) * 1.5 if min(all_delta_vals) < 0 else -0.02
    delta_hi = max(all_delta_vals) * 1.5

    # 사분면 라벨
    for rx, ry, label, lcolor in [
        (attn_thresh + (attn_hi - attn_thresh) * 0.5, delta_thresh + (delta_hi - delta_thresh) * 0.5,
         "Confirmed\nDriver", "#E24B4A"),
        (attn_thresh + (attn_hi - attn_thresh) * 0.5, delta_lo + (delta_thresh - delta_lo) * 0.5,
         "Spurious\nAttention", "#CC8800"),
        (attn_lo + (attn_thresh - attn_lo) * 0.5, delta_thresh + (delta_hi - delta_thresh) * 0.5,
         "Hidden\nContributor", "#3A7BD5"),
        (attn_lo + (attn_thresh - attn_lo) * 0.5, delta_lo + (delta_thresh - delta_lo) * 0.5,
         "Irrelevant", "#888888"),
    ]:
        fig5.add_annotation(
            x=rx, y=ry, xref="x", yref="y",
            text=label, showarrow=False,
            font=dict(size=13, color=lcolor), opacity=0.25,
        )

    fig5.update_layout(
        xaxis=dict(title=x_label, range=[attn_lo, attn_hi],
                   tickformat=".1%", title_font_size=14),
        yaxis=dict(title=y_label, range=[delta_lo, delta_hi],
                   title_font_size=14),
        height=520, legend=dict(title="Quadrant", x=1.01, y=1, font=dict(size=13)),
        margin=dict(t=40, r=200, b=60),
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ── 사분면 해석 카드 ────────────────────────────────────────
    cols = st.columns(4)
    for col, (quad, desc) in zip(cols, QUAD_DESC.items()):
        with col:
            color = QUAD_COLORS[quad]
            cnt   = int((diag_df["Quadrant"] == quad).sum())
            st.markdown(
                f"<span style='color:{color}; font-size:1.1rem;'>■</span> "
                f"**{quad}** ({cnt}개)<br>"
                f"<small style='color:#666'>{desc}</small>",
                unsafe_allow_html=True,
            )

    # ── 다중공선성 해석 노트 ────────────────────────────────────
    st.divider()
    with st.expander("📝 해석 시 주의사항: Attention ≠ 예측 기여도", expanded=False):
        st.markdown("""
**Attention이 높지만 ΔRMSE ≤ 0인 변수 (Spurious Attention)**

이 사분면의 변수는 "무관한 변수"가 아닐 수 있습니다. 세 가지 가능성을 고려해야 합니다:

1. **확률적 변동 (Stochastic Noise)**: BISTRO는 `num_samples=100`으로 샘플링하므로
   |ΔRMSE| < ~0.02pp 수준의 차이는 실행마다 달라질 수 있는 노이즈 범위입니다.
   예를 들어 CNY_USD(ΔRMSE=-0.005pp)는 통계적으로 유의하지 않을 가능성이 높습니다.

2. **정보 중복 (Redundancy)**: 해당 변수의 정보가 다른 공변량 조합에 의해 간접적으로
   이미 포착되고 있을 수 있습니다. 직접적으로 같은 지표가 아니더라도, 여러 변수의 조합이
   해당 변수의 신호를 대체할 수 있습니다.

3. **허위 상관 (Spurious Correlation)**: 모델이 우연한 패턴 매칭을 학습했을 수 있습니다.
   특히 Temporal Attention이 최근이 아닌 과거 구간에 집중하거나,
   특정 레이어(L0)에서만 높은 attention을 보이면 artifact일 가능성이 큽니다.

**핵심**: ΔRMSE가 0에 가까운 변수는 "해롭다"보다 **"있어도 없어도 차이가 없다"**로 해석하는 것이 정확합니다.

**Hidden Contributor 변수**

Attention은 낮지만 제거하면 성능이 떨어지는 변수는 **독자적 정보(orthogonal signal)**를
제공합니다. 다른 변수로 대체할 수 없는 고유 정보를 담고 있으므로 반드시 유지해야 합니다.
이는 "Attention is not Explanation" (Jain & Wallace, 2019)의 실증 사례입니다.
""")

    # ── 분류 테이블 ─────────────────────────────────────────────
    st.divider()
    st.caption("**전체 변수 분류 테이블**")
    disp = (
        diag_df[["Variable", "Attention", "ΔRMSE", "Quadrant", "Tier"]]
        .sort_values("Attention", ascending=False)
        .reset_index(drop=True)
    )
    disp["Attention"] = disp["Attention"].map(lambda v: f"{v:.2%}")
    disp["ΔRMSE"]     = disp["ΔRMSE"].map(lambda v: f"{v:+.4f}")
    disp.columns = ["Variable", "Attention (%)", "ΔRMSE (pp)", "Quadrant", "Tier"]
    disp.index = disp.index + 1

    def _color_quad(val):
        return f"color: {QUAD_COLORS.get(val, 'black')}"

    st.dataframe(
        disp.style.applymap(_color_quad, subset=["Quadrant"]),
        use_container_width=True,
    )


# ----------------------------------------------------------
# (Ablation & Incremental tab removed — content in Stage 0→1 Pipeline tab)
# ----------------------------------------------------------

if False:  # REMOVED
    st.subheader("Ablation & Incremental Addition Study")

    if not has_ablation:
        st.info(
            "Ablation 결과가 없습니다. `ablation_study.py`를 실행하세요.",
            icon="ℹ️",
        )
    else:
        ab = abl_data
        st.caption(
            f"**Baseline (10 covariates)**: RMSE = {ab['baseline_rmse']:.4f} pp, "
            f"MAE = {ab['baseline_mae']:.4f} pp  |  "
            "변수를 하나씩 제거(ablation)하거나 attention 순서대로 추가(incremental)하여 "
            "각 공변량의 실질적 기여도를 측정합니다."
        )

        # ── 1. Ablation: ΔRMSE bar chart ──────────────────────
        st.markdown("### 1. Ablation — 변수 제거 시 RMSE 변화 (ΔRMSE)")
        st.caption(
            "ΔRMSE > 0 → 제거하면 성능이 악화 (해당 변수가 예측에 기여)  |  "
            "ΔRMSE < 0 → 제거하면 오히려 성능 향상 (노이즈 또는 과적합 변수)"
        )

        # Sort by delta_rmse descending
        abl_sorted_idx = np.argsort(ab["abl_delta_rmse"])[::-1]
        abl_sorted_vars = [ab["abl_vars"][i] for i in abl_sorted_idx]
        abl_sorted_delta = ab["abl_delta_rmse"][abl_sorted_idx]

        colors_abl = ["#D62728" if d > 0 else "#2CA02C" for d in abl_sorted_delta]

        fig_abl = go.Figure()
        fig_abl.add_trace(go.Bar(
            x=abl_sorted_vars,
            y=abl_sorted_delta,
            marker_color=colors_abl,
            text=[f"{d:+.4f}" for d in abl_sorted_delta],
            textposition="outside",
            textfont=dict(size=13),
            hovertemplate="<b>%{x}</b><br>ΔRMSE: %{y:+.4f} pp<extra></extra>",
        ))
        fig_abl.add_hline(y=0, line_dash="dot", line_color="#999", line_width=1.5)
        fig_abl.update_layout(
            xaxis_title="Covariate (removed)",
            yaxis_title="ΔRMSE (pp)",
            height=420,
            margin=dict(t=40, b=60),
            xaxis=dict(tickfont=dict(size=13)),
        )
        st.plotly_chart(fig_abl, use_container_width=True)

        # ── 2. Incremental: RMSE curve ────────────────────────
        st.divider()
        st.markdown("### 2. Incremental Addition — Attention 순서대로 변수 추가")
        st.caption(
            "Attention ranking 순서대로 공변량을 하나씩 추가하며 RMSE 변화를 추적합니다. "
            "추가 변수가 성능에 미치는 한계 효과를 볼 수 있습니다."
        )

        inc_x = list(range(1, len(ab["inc_rmse"]) + 1))
        inc_labels_clean = [l.replace("+", "") for l in ab["inc_labels"]]

        fig_inc = go.Figure()
        fig_inc.add_trace(go.Scatter(
            x=inc_x,
            y=ab["inc_rmse"],
            mode="lines+markers+text",
            line=dict(color="#1A6FD4", width=3),
            marker=dict(size=10, color="#1A6FD4"),
            text=[f"{r:.3f}" for r in ab["inc_rmse"]],
            textposition="top center",
            textfont=dict(size=12),
            hovertemplate="<b>%{customdata}</b><br>%{x} covariates<br>RMSE: %{y:.4f} pp<extra></extra>",
            customdata=inc_labels_clean,
        ))

        # Baseline reference line
        fig_inc.add_hline(
            y=ab["baseline_rmse"], line_dash="dash", line_color="#E24B4A", line_width=2,
            annotation_text=f"Full (10 cov) = {ab['baseline_rmse']:.4f}",
            annotation_position="right",
            annotation_font=dict(size=12, color="#E24B4A"),
        )

        # Best point highlight
        best_idx = int(np.argmin(ab["inc_rmse"]))
        fig_inc.add_trace(go.Scatter(
            x=[inc_x[best_idx]],
            y=[ab["inc_rmse"][best_idx]],
            mode="markers",
            marker=dict(size=16, color="#2CA02C", symbol="star", line=dict(width=2, color="white")),
            name=f"Best: {inc_labels_clean[best_idx]}",
            hovertemplate=f"Best: {inc_labels_clean[best_idx]}<br>RMSE: {ab['inc_rmse'][best_idx]:.4f}<extra></extra>",
        ))

        fig_inc.update_layout(
            xaxis=dict(
                title="Number of Covariates",
                tickvals=inc_x,
                ticktext=[f"{x}<br><small>{inc_labels_clean[i]}</small>" for i, x in enumerate(inc_x)],
                tickfont=dict(size=11),
            ),
            yaxis_title="RMSE (pp)",
            height=450,
            margin=dict(t=40, b=80),
            showlegend=True,
            legend=dict(x=0.7, y=0.95),
        )
        st.plotly_chart(fig_inc, use_container_width=True)

        # ── 3. Ranking comparison table ───────────────────────
        st.divider()
        st.markdown("### 3. Attention Ranking vs Ablation Ranking")
        st.caption(
            "Attention이 높다고 해서 반드시 예측 성능에 기여하는 것은 아닙니다. "
            "두 랭킹의 차이가 큰 변수는 모델이 '보지만 의존하지 않는' 패턴 매칭 대상일 수 있습니다."
        )

        # Build comparison dataframe
        attn_rank_order = list(ab["attn_ranking"])
        abl_rank_order = [ab["abl_vars"][i] for i in np.argsort(ab["abl_delta_rmse"])[::-1]]

        comp_rows = []
        for v in attn_rank_order:
            a_rank = attn_rank_order.index(v) + 1
            b_rank = abl_rank_order.index(v) + 1 if v in abl_rank_order else None
            a_val = float(ab["attn_values"][attn_rank_order.index(v)])
            b_delta = float(ab["abl_delta_rmse"][list(ab["abl_vars"]).index(v)]) if v in ab["abl_vars"] else None
            comp_rows.append({
                "Variable": v,
                "Attn Rank": a_rank,
                "Attn Score": f"{a_val:.2%}",
                "Abl Rank": b_rank,
                "ΔRMSE": f"{b_delta:+.4f}" if b_delta is not None else "—",
                "Rank Δ": a_rank - b_rank if b_rank else "—",
            })

        comp_df = pd.DataFrame(comp_rows)

        def _color_rank_delta(val):
            if val == "—" or val == 0:
                return ""
            v = int(val) if isinstance(val, (int, float)) else 0
            if v > 0:
                return "color: #D62728"  # attention overestimates
            elif v < 0:
                return "color: #2CA02C"  # attention underestimates
            return ""

        st.dataframe(
            comp_df.style.applymap(_color_rank_delta, subset=["Rank Δ"]),
            use_container_width=True,
            hide_index=True,
        )

        # ── 4. Key findings ───────────────────────────────────
        st.divider()
        st.markdown("### Key Findings")

        # Find variables where attention and ablation rankings diverge most
        biggest_gap_vars = sorted(comp_rows, key=lambda r: abs(r["Rank Δ"]) if isinstance(r["Rank Δ"], (int, float)) else 0, reverse=True)
        top_abl = abl_rank_order[0] if abl_rank_order else "—"
        top_attn = attn_rank_order[0] if attn_rank_order else "—"

        cols_find = st.columns(3)
        with cols_find[0]:
            st.metric("Attention #1", top_attn)
            st.caption("가장 높은 cross-variate attention")
        with cols_find[1]:
            st.metric("Ablation #1", top_abl)
            st.caption("제거 시 가장 큰 성능 저하")
        with cols_find[2]:
            n_harmful = int(np.sum(ab["abl_delta_rmse"] <= 0))
            optimal_n = len(ab["abl_vars"]) - n_harmful
            st.metric("Optimal # Covariates", f"{optimal_n}")
            st.caption(f"전체 {len(ab['abl_vars'])}개 − ΔRMSE≤0 {n_harmful}개")

        if top_attn != top_abl:
            st.warning(
                f"**Attention ≠ Functional importance**: "
                f"Attention이 가장 높은 **{top_attn}**과 "
                f"ablation에서 가장 중요한 **{top_abl}**이 다릅니다. "
                f"Attention은 패턴 매칭 강도를, ablation은 예측 기여도를 반영합니다.",
                icon="⚠️",
            )

        # ── ΔRMSE ≤ 0 변수 해석 노트 ──────────────────────────
        neg_vars = [ab["abl_vars"][i] for i in range(len(ab["abl_vars"])) if ab["abl_delta_rmse"][i] <= 0]
        if neg_vars:
            st.divider()
            st.markdown("### Redundancy Note")
            st.info(
                f"**ΔRMSE ≤ 0 변수** ({len(neg_vars)}개): {', '.join(neg_vars)}\n\n"
                "이 변수들은 제거해도 RMSE가 유지되거나 오히려 개선됩니다. "
                "그러나 **경제적으로 무관한 변수라는 의미는 아닙니다.**\n\n"
                "- **확률적 변동**: BISTRO는 `num_samples=100`으로 샘플링하므로 "
                "|ΔRMSE| < ~0.02pp 수준의 차이는 실행마다 달라질 수 있는 노이즈 범위입니다. "
                "예를 들어 CNY_USD(ΔRMSE=-0.005pp)는 통계적으로 유의하지 않을 가능성이 높습니다.\n"
                "- **정보 중복**: 해당 변수의 신호가 다른 공변량 조합에 의해 간접적으로 "
                "이미 포착되고 있을 수 있습니다.\n\n"
                "**해석**: \"있으면 해롭다\"가 아니라 **\"있어도 없어도 차이가 없다\"**에 가깝습니다. "
                "경제적으로 의미 있는 변수라면, 향후 변수 조합이 바뀔 때 기여도가 달라질 수 있으므로 "
                "섣불리 제거하기보다는 모니터링 대상으로 유지하는 것이 적절합니다.",
                icon="📌",
            )


# ----------------------------------------------------------
# Tab: CF Scenarios (Real mode + CF data only)
# ----------------------------------------------------------

if forecast and forecast.get("cf_variates") is not None:
    with tabs[-1]:
        st.subheader("Context Sensitivity — ±1σ per Covariate")
        st.caption(
            "Attention 상위 공변량의 **과거 문맥(120개월)**을 ±1σ 수준 이동한 후 재추론한 결과입니다. "
            "\"과거에 해당 변수의 수준이 1σ만큼 달랐더라면 모델이 CPI를 어떻게 예측했을까?\"를 측정하며, "
            "미래 시나리오 분석이 아닌 **모델의 입력 민감도(input sensitivity)** 측정입니다."
        )

        fc        = forecast
        cf_vars   = fc["cf_variates"]
        cf_plus   = fc["cf_preds_plus"]    # (n_cov, 12)
        cf_minus  = fc["cf_preds_minus"]   # (n_cov, 12)
        cf_sigmas = fc["cf_sigmas"]
        cf_imps   = fc["cf_impacts"]
        n_cov     = len(cf_vars)

        fc_dates  = pd.to_datetime([d + "-01" for d in fc["date"]])
        actual    = fc["actual"]

        fig_cf = make_subplots(
            rows=1, cols=n_cov,
            subplot_titles=[
                f"{v}  (σ={cf_sigmas[i]:.3f}, impact={cf_imps[i]:.4f} pp)"
                for i, v in enumerate(cf_vars)
            ],
            horizontal_spacing=min(0.12, 0.8 / max(n_cov, 2)),
        )

        for ci, cov_name in enumerate(cf_vars):
            col = ci + 1

            # baseline
            fig_cf.add_trace(go.Scatter(
                x=fc_dates, y=fc["med"],
                mode="lines+markers",
                line=dict(color="#1A6FD4", width=3),
                marker=dict(size=8),
                name="Baseline" if ci == 0 else None,
                showlegend=(ci == 0),
                hovertemplate="%{x|%Y-%m}: <b>%{y:.3f}%</b> (baseline)<extra></extra>",
            ), row=1, col=col)

            # +1σ
            fig_cf.add_trace(go.Scatter(
                x=fc_dates, y=cf_plus[ci],
                mode="lines+markers",
                line=dict(color="#D62728", width=2.5, dash="dash"),
                marker=dict(size=7, symbol="triangle-up"),
                name=f"+1σ {cov_name}" if ci == 0 else None,
                showlegend=(ci == 0),
                hovertemplate=f"%{{x|%Y-%m}}: <b>%{{y:.3f}}%</b> (+1σ {cov_name})<extra></extra>",
            ), row=1, col=col)

            # -1σ
            fig_cf.add_trace(go.Scatter(
                x=fc_dates, y=cf_minus[ci],
                mode="lines+markers",
                line=dict(color="#2CA02C", width=2.5, dash="dash"),
                marker=dict(size=7, symbol="triangle-down"),
                name=f"-1σ {cov_name}" if ci == 0 else None,
                showlegend=(ci == 0),
                hovertemplate=f"%{{x|%Y-%m}}: <b>%{{y:.3f}}%</b> (-1σ {cov_name})<extra></extra>",
            ), row=1, col=col)

            # actual
            if not all(np.isnan(actual)):
                fig_cf.add_trace(go.Scatter(
                    x=fc_dates, y=actual,
                    mode="lines+markers",
                    line=dict(color="#333333", width=3),
                    marker=dict(size=9, symbol="square"),
                    name="Actual" if ci == 0 else None,
                    showlegend=(ci == 0),
                    hovertemplate="%{x|%Y-%m}: <b>%{y:.3f}%</b> (actual)<extra></extra>",
                ), row=1, col=col)

        fig_cf.update_yaxes(title_text="CPI YoY (%)", col=1, title_font_size=14)
        fig_cf.update_xaxes(tickfont_size=13)
        fig_cf.update_layout(
            height=500,
            legend=dict(
                x=0.01, y=1.18, orientation="h",
                font=dict(size=14),
                bgcolor="white", bordercolor="#CCCCCC", borderwidth=1,
            ),
            margin=dict(t=100, b=60),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cf, use_container_width=True)

        # ── Δ(차이) 그래프: baseline 대비 예측 변화량 ──────────
        st.divider()
        st.subheader("Baseline 대비 예측 변화량 (Δ)")
        st.caption(
            "위 그래프에서 선이 겹쳐 보이는 이유는 변화량이 최대 ~0.12pp 수준이기 때문입니다. "
            "아래는 **baseline 대비 차이(Δ)**만 확대하여 공변량 교란의 실제 효과를 보여줍니다."
        )

        fig_delta = make_subplots(
            rows=1, cols=n_cov,
            subplot_titles=[
                f"Δ {v}  (σ={cf_sigmas[i]:.3f})"
                for i, v in enumerate(cf_vars)
            ],
            horizontal_spacing=min(0.12, 0.8 / max(n_cov, 2)),
        )

        for ci, cov_name in enumerate(cf_vars):
            col = ci + 1
            delta_plus  = cf_plus[ci] - fc["med"]
            delta_minus = cf_minus[ci] - fc["med"]

            fig_delta.add_trace(go.Scatter(
                x=fc_dates, y=delta_plus,
                mode="lines+markers",
                line=dict(color="#D62728", width=3),
                marker=dict(size=8, symbol="triangle-up"),
                name=f"+1σ {cov_name}" if ci == 0 else None,
                showlegend=(ci == 0),
                hovertemplate=f"%{{x|%Y-%m}}: <b>%{{y:+.4f}} pp</b> (+1σ {cov_name})<extra></extra>",
            ), row=1, col=col)

            fig_delta.add_trace(go.Scatter(
                x=fc_dates, y=delta_minus,
                mode="lines+markers",
                line=dict(color="#2CA02C", width=3),
                marker=dict(size=8, symbol="triangle-down"),
                name=f"-1σ {cov_name}" if ci == 0 else None,
                showlegend=(ci == 0),
                hovertemplate=f"%{{x|%Y-%m}}: <b>%{{y:+.4f}} pp</b> (-1σ {cov_name})<extra></extra>",
            ), row=1, col=col)

            # 0 기준선
            fig_delta.add_hline(
                y=0, line_dash="dot", line_color="#999999", line_width=1.5,
                row=1, col=col,
            )

        fig_delta.update_yaxes(title_text="Δ CPI forecast (pp)", col=1, title_font_size=14)
        fig_delta.update_xaxes(tickfont_size=13)
        fig_delta.update_layout(
            height=420,
            legend=dict(
                x=0.01, y=1.18, orientation="h",
                font=dict(size=14),
                bgcolor="white", bordercolor="#CCCCCC", borderwidth=1,
            ),
            margin=dict(t=80, b=60),
            hovermode="x unified",
        )
        st.plotly_chart(fig_delta, use_container_width=True)

        # σ 요약 테이블
        st.divider()
        st.caption("**공변량별 Context Sensitivity 요약**")
        cf_summary = pd.DataFrame({
            "Variable":       cf_vars,
            "σ (context)":    [f"{s:.4f}" for s in cf_sigmas],
            "Mean |Δforecast| (pp)": [f"{v:.4f}" for v in cf_imps],
            "Max +1σ shift (pp)":   [f"{float(np.max(np.abs(cf_plus[i] - fc['med']))):.4f}" for i in range(n_cov)],
            "Max -1σ shift (pp)":   [f"{float(np.max(np.abs(cf_minus[i] - fc['med']))):.4f}" for i in range(n_cov)],
        })
        st.dataframe(cf_summary, use_container_width=True, hide_index=True)
