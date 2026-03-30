"""
BISTRO-XAI | Attention Map Explorer
=====================================
Streamlit + Plotly 기반 인터랙티브 Attention 시각화 대시보드.
실제 BISTRO 추론 결과(data/real_inference_results.npz)가 있으면 자동으로 로딩.

실행:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

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
    st.caption(
        "📌 **GitHub**: `bis-med-it/bistro`\n\n"
        "Korean CPI (KR YoY) + 10 covariates\n\n"
        "Forecast start: 2023-01"
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
    col1, col2 = st.columns(2)
    col1.metric("Variables", n)
    col2.metric("CTX Patches", cfg.ctx_patches, help="변수당 과거 문맥 패치 수 (≈ 월 수)")
    st.metric("Layers", len(hooks.get_layer_names()))
    st.metric("Total Tokens", f"{cfg.n_tokens:,}", help=f"{n} vars × {cfg.ctx_patches} patches")

# ── Ablation 데이터 로딩 ──
has_ablation = ablation_available()
if has_ablation:
    if not st.session_state.get("abl_loaded"):
        abl_data = load_ablation_results()
        st.session_state.update({"abl_loaded": True, "abl_data": abl_data})
    abl_data = st.session_state["abl_data"]

# Sidebar: Stage 정보
if has_stage1:
    with st.sidebar:
        st.divider()
        s1_total = s1_meta.get("s1_n_total") or (s1_cfg.n_variates - 1)
        if isinstance(s1_total, np.ndarray):
            s1_total = int(s1_total)
        st.markdown(f"**Pipeline**: {1 + s1_total} → **{n}** vars")
        st.caption(f"Stage 1 스크리닝 {1 + s1_total}개 → Stage 2 선택 {n}개")

# ============================================================
# Header
# ============================================================

st.markdown("## BISTRO-XAI Attention Explorer")
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
    "🔬 Feature Selection",
    "📊 Cross-Variate Heatmap",
    "📈 Variable Importance",
    "⏱️ Temporal Patterns",
    "🔍 Layer Analysis",
    "🎯 2×2 Diagnostic",
    "🧪 Ablation & Incremental",
]
if forecast and forecast.get("cf_variates") is not None:
    tab_labels.append("🔀 CF Scenarios")

tabs = st.tabs(tab_labels)
tab_offset = 1


# ----------------------------------------------------------
# Tab 0 (Real mode only): Forecast Results
# ----------------------------------------------------------

with tabs[0]:
    st.subheader("Korean CPI — BISTRO Forecast vs Actual (2023)")
    st.caption(
        "Context window: ~2013-01 ~ 2022-12 | Forecast: 2023-01 ~ 2023-12 | "
        "BISTRO는 BIS 63개국 4,925개 시계열로 파인튜닝 (한국 CPI 포함 가능성 높음). "
        "2023년 데이터는 추론 시 context에 미포함."
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

    # Actual CPI — 추론 이후 (검정)
    if len(_cpi_after) > 0:
        _af_x = [_cpi_forecast.index[-1]] + list(_cpi_after.index)
        _af_y = [_cpi_forecast.values[-1]] + list(_cpi_after.values)
        fig_fc.add_trace(go.Scatter(
            x=_af_x, y=_af_y,
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

    # BISTRO median
    fig_fc.add_trace(go.Scatter(
        x=fc_dates, y=fc["med"],
        mode="lines+markers",
        line=dict(color="#1A6FD4", width=3),
        marker=dict(size=9),
        name="BISTRO (median)",
        hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
    ))

    # AR(1)
    if not all(np.isnan(fc["ar1"])):
        fig_fc.add_trace(go.Scatter(
            x=fc_dates, y=fc["ar1"],
            mode="lines+markers",
            line=dict(color="#888888", width=2, dash="dash"),
            marker=dict(size=7),
            name="AR(1) baseline",
            hovertemplate="%{x|%Y-%m}: <b>%{y:.2f}%</b><extra></extra>",
        ))

    # Forecast start line
    fig_fc.add_vline(
        x=fc_dates[0].timestamp() * 1000,
        line_dash="dash", line_color="rgba(100,100,100,0.6)",
        annotation_text="Forecast start", annotation_position="top right",
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
        st.markdown(f"**RMSE (BISTRO):** `{rmse_bistro:.4f}` pp")
        if not all(np.isnan(fc["ar1"])):
            rmse_ar1 = float(np.sqrt(np.mean((fc["ar1"][valid] - actual_vals[valid]) ** 2)))
            r_rmse   = rmse_bistro / rmse_ar1
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("BISTRO RMSE", f"{rmse_bistro:.4f} pp")
            col_b.metric("AR(1) RMSE",  f"{rmse_ar1:.4f} pp")
            color = "normal" if r_rmse < 1 else "inverse"
            col_c.metric("R-RMSE (BISTRO/AR1)", f"{r_rmse:.3f}",
                         delta=f"{'Better' if r_rmse < 1 else 'Worse'} than AR(1)",
                         delta_color=color)

    # Monthly breakdown table
    fc_df = pd.DataFrame({
        "Month":       fc["date"],
        "BISTRO Med":  [f"{v:.3f}%" for v in fc["med"]],
        "CI [5,95]":   [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(fc["ci_lo"], fc["ci_hi"])],
        "AR(1)":       [f"{v:.3f}%" if not np.isnan(v) else "—" for v in fc["ar1"]],
        "Actual":      [f"{v:.3f}%" if not np.isnan(v) else "—" for v in actual_vals],
    })
    with st.expander("📋 월별 예측값 테이블"):
        st.dataframe(fc_df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------
# Tab: Feature Selection
# ----------------------------------------------------------

with tabs[tab_offset + 0]:
    st.subheader("Attention-Based Feature Selection")

    if has_stage1:
        # ── Stage 1 전체 변수 랭킹 (29개) ─────────────────────
        st.caption(
            f"**Stage 1**: 전체 {s1_cfg.n_variates}개 변수로 BISTRO 추론 → "
            f"Attention 기반 공변량 랭킹. "
            f"균등 배분 기준(uniform share) 이상인 변수를 선택하여 Stage 2에서 재추론."
        )

        s1_imp = s1_ana.target_importance()
        s1_self = float(s1_imp.iloc[s1_cfg.target_idx])
        s1_cov_imp = s1_imp.drop(s1_cfg.target_name).sort_values(ascending=False)
        n_s1_cov = len(s1_cov_imp)
        s1_uniform = (1.0 - s1_self) / n_s1_cov if n_s1_cov > 0 else 0

        # Stage 2에서 선택된 변수 (현재 대시보드에 로딩된 변수)
        s2_covariates = [v for v in variates_list if v != cfg.target_name]

        # ── 랭킹 차트: 전체 28개, 선택된 것 파란색 / 미선택 회색 ──
        cumsum = s1_cov_imp.cumsum()
        fig_fs = make_subplots(specs=[[{"secondary_y": True}]])

        bar_colors = [
            "#1A6FD4" if v in s2_covariates else "#CCCCCC"
            for v in s1_cov_imp.index
        ]
        fig_fs.add_trace(
            go.Bar(
                x=s1_cov_imp.index.tolist(), y=s1_cov_imp.values,
                marker_color=bar_colors,
                text=[f"{v:.1%}" for v in s1_cov_imp.values],
                textposition="outside", textfont=dict(size=11),
                name="Attention (%)",
                hovertemplate="<b>%{x}</b>: %{y:.2%}<extra></extra>",
            ),
            secondary_y=False,
        )

        fig_fs.add_trace(
            go.Scatter(
                x=cumsum.index.tolist(), y=cumsum.values,
                mode="lines+markers",
                line=dict(color="#D62728", width=2.5),
                marker=dict(size=6),
                name="Cumulative",
                hovertemplate="Top-%{pointNumber+1}: cumulative %{y:.1%}<extra></extra>",
            ),
            secondary_y=True,
        )

        fig_fs.add_hline(
            y=s1_uniform, line_dash="dot", line_color="#999999", line_width=1.5,
            annotation_text=f"균등 배분: {s1_uniform:.2%}",
            annotation_position="right", annotation_font_size=11,
            secondary_y=False,
        )

        fig_fs.update_layout(
            xaxis=dict(title="Covariate (Stage 1, all)", tickangle=-40, tickfont_size=11, title_font_size=14),
            yaxis=dict(title="Attention (%)", tickformat=".1%", title_font_size=14),
            yaxis2=dict(title="Cumulative (%)", tickformat=".0%", title_font_size=14, range=[0, 1.05]),
            height=max(450, 30 * n_s1_cov),
            legend=dict(x=0.01, y=0.99, font=dict(size=13)),
            margin=dict(t=40, b=120),
        )
        st.plotly_chart(fig_fs, use_container_width=True)

        # ── 메트릭 ────────────────────────────────────────────
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        col_i1.metric("Self-Attention", f"{s1_self:.1%}")
        col_i2.metric("Cross-Attn Budget", f"{1-s1_self:.1%}")
        col_i3.metric("전체 공변량", n_s1_cov)
        col_i4.metric("선택됨 (Stage 2)", len(s2_covariates))

        # 주기별 분포
        n_daily_all = sum(1 for v in s1_cov_imp.index if VARIABLE_FREQ.get(v) == "daily")
        n_monthly_all = sum(1 for v in s1_cov_imp.index if VARIABLE_FREQ.get(v) == "monthly")
        n_daily_sel = sum(1 for v in s2_covariates if VARIABLE_FREQ.get(v) == "daily")
        n_monthly_sel = sum(1 for v in s2_covariates if VARIABLE_FREQ.get(v) == "monthly")
        col_f1, col_f2 = st.columns(2)
        col_f1.metric("📅 Daily 변수", f"{n_daily_sel}/{n_daily_all}", help="선택/전체")
        col_f2.metric("📆 Monthly 변수", f"{n_monthly_sel}/{n_monthly_all}", help="선택/전체")

        st.divider()

        # ── 선택 기준 설명 ─────────────────────────────────────
        st.markdown("### Selection Criteria")
        n_above = int((s1_cov_imp >= s1_uniform).sum())
        coverage = float(s1_cov_imp[s1_cov_imp.index.isin(s2_covariates)].sum())

        st.markdown(f"""
- **균등 배분 기준**: `(1 − self_attn) / N = (1 − {s1_self:.1%}) / {n_s1_cov} = **{s1_uniform:.2%}**`
- 기준 이상 변수: **{n_above}개** / {n_s1_cov}개
- 최종 선택: **{len(s2_covariates)}개** (cross-attention의 **{coverage:.1%}** 커버)
- 선택된 변수로 **Stage 2 재추론** 완료 → 이후 탭은 모두 Stage 2 결과 기준
""")

        # ── 전체 변수 테이블 (선택 여부 표시) ──────────────────
        rows = []
        for rank, (v, a) in enumerate(s1_cov_imp.items(), 1):
            freq = VARIABLE_FREQ.get(v, "—")
            freq_label = "📅 Daily" if freq == "daily" else ("📆 Monthly" if freq == "monthly" else "—")
            rows.append({
                "Rank": rank,
                "Variable": v,
                "Freq": freq_label,
                "Stage1 Attention": f"{a:.2%}",
                "Tier": TIER_LABELS.get(v, "—"),
                "vs Uniform": f"{a / s1_uniform:.1f}×" if s1_uniform > 0 else "—",
                "Status": "✅ Selected" if v in s2_covariates else "❌ Excluded",
            })
        sel_df = pd.DataFrame(rows)

        def _color_status(val):
            if "Selected" in str(val):
                return "color: #1a6b3c; font-weight: bold"
            return "color: #999999"

        st.dataframe(
            sel_df.style.map(_color_status, subset=["Status"]),
            use_container_width=True, hide_index=True,
        )

        # ── Tier 분포 ─────────────────────────────────────────
        st.divider()
        st.markdown("### Tier별 분포")
        tier_data = pd.DataFrame({
            "Variable": s1_cov_imp.index,
            "Attention": s1_cov_imp.values,
            "Tier": [TIER_LABELS.get(v, "Other") for v in s1_cov_imp.index],
            "Selected": [v in s2_covariates for v in s1_cov_imp.index],
        })
        tier_agg = tier_data.groupby("Tier").agg(
            Total_Attn=("Attention", "sum"),
            Count=("Variable", "count"),
            Selected=("Selected", "sum"),
        ).reset_index().sort_values("Total_Attn", ascending=False)

        tier_colors = {"T1": "#D62728", "T2": "#FF7F0E", "T3": "#2CA02C", "T4": "#1A6FD4", "Other": "#999999"}
        fig_tier = go.Figure(go.Bar(
            x=tier_agg["Tier"],
            y=tier_agg["Total_Attn"],
            text=[f"{v:.1%} ({int(s)}/{int(c)})" for v, c, s in
                  zip(tier_agg["Total_Attn"], tier_agg["Count"], tier_agg["Selected"])],
            textposition="outside",
            marker_color=[tier_colors.get(t, "#999") for t in tier_agg["Tier"]],
        ))
        fig_tier.update_layout(
            yaxis=dict(title="Total Attention", tickformat=".0%"),
            xaxis=dict(title="Tier"),
            height=300, margin=dict(t=30, b=50),
        )
        st.plotly_chart(fig_tier, use_container_width=True)
        st.caption("괄호 안: (선택/전체) 변수 수")

    else:
        # Stage 1 없을 때 — Stage 2 데이터로 간단 표시
        st.caption("Stage 1 스크리닝 데이터 없음. 현재 로딩된 변수의 attention을 표시합니다.")
        imp_fs = ana.target_importance()
        cov_imp_fs = imp_fs.drop(cfg.target_name).sort_values(ascending=False)
        fig_fs = go.Figure(go.Bar(
            x=cov_imp_fs.index.tolist(), y=cov_imp_fs.values,
            marker_color="#1A6FD4",
            text=[f"{v:.1%}" for v in cov_imp_fs.values],
            textposition="outside",
        ))
        fig_fs.update_layout(
            yaxis=dict(title="Attention (%)", tickformat=".1%"),
            xaxis=dict(title="Covariate", tickangle=-35),
            height=400, margin=dict(t=40, b=100),
        )
        st.plotly_chart(fig_fs, use_container_width=True)


# ----------------------------------------------------------
# Tab: Cross-Variate Heatmap
# ----------------------------------------------------------

with tabs[tab_offset + 1]:
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
# Tab: Variable Importance
# ----------------------------------------------------------

with tabs[tab_offset + 2]:
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

with tabs[tab_offset + 3]:
    st.subheader("Temporal Attention Patterns")
    st.caption(
        "타겟 변수의 예측 패치가 **과거 어느 시점을 주목**하는지. "
        "황금색 = 최근 12 patches. spike 위치로 기저효과·전달 시차 분석."
    )

    imp_temp = ana.target_importance()
    active_covs_temp = [v for v in active_variates if v != cfg.target_name]
    top_cov = imp_temp[active_covs_temp].idxmax() if active_covs_temp else imp_temp.drop(cfg.target_name).idxmax()
    t_self  = ana.temporal_attention(cfg.target_name, cfg.target_name)
    t_cov   = ana.temporal_attention(cfg.target_name, top_cov)

    n_p            = len(t_self)
    recent_cutoff  = max(0, n_p - 12)

    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"{cfg.target_name} → {cfg.target_name}  (자기 참조)",
            f"{cfg.target_name} → {top_cov}  (최상위 공변량)",
        ],
        horizontal_spacing=0.10,
    )

    for col_idx, (series, color) in enumerate(
        [(t_self, "#E24B4A"), (t_cov, "#EF9F27")], start=1
    ):
        x_vals    = list(range(len(series)))
        bar_clrs  = [
            color if i >= recent_cutoff
            else f"rgba({','.join(str(c) for c in _hex_to_rgb(color))},0.30)"
            for i in x_vals
        ]
        fig3.add_trace(
            go.Bar(x=x_vals, y=series, marker_color=bar_clrs,
                   hovertemplate="Patch %{x}: <b>%{y:.5f}</b><extra></extra>",
                   showlegend=False),
            row=1, col=col_idx,
        )
        fig3.add_vrect(
            x0=recent_cutoff - 0.5, x1=len(series) - 0.5,
            fillcolor="rgba(255,215,0,0.14)", line_width=0,
            row=1, col=col_idx,
        )
        if len(series) > 0:
            fig3.add_annotation(
                x=recent_cutoff + (len(series) - recent_cutoff) / 2,
                y=float(np.max(series)) * 0.92,
                text="최근 12", showarrow=False,
                font=dict(size=9, color="goldenrod"),
                row=1, col=col_idx,
            )

    fig3.update_xaxes(title_text="Past Patch Index", title_font_size=14, tickfont_size=13)
    fig3.update_yaxes(title_text="Mean Attention Weight", col=1, title_font_size=14, tickfont_size=13)
    fig3.update_layout(height=420, margin=dict(t=80, b=60))
    st.plotly_chart(fig3, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        s5 = pd.Series(t_self).nlargest(5).reset_index()
        s5.columns = ["Patch Index", "Attention"]
        s5["Attention"] = s5["Attention"].map(lambda v: f"{v:.5f}")
        st.caption(f"**자기 참조 Top 5 패치** ({cfg.target_name})")
        st.dataframe(s5, use_container_width=True, hide_index=True)
    with col_b:
        c5 = pd.Series(t_cov).nlargest(5).reset_index()
        c5.columns = ["Patch Index", "Attention"]
        c5["Attention"] = c5["Attention"].map(lambda v: f"{v:.5f}")
        st.caption(f"**{top_cov} Top 5 패치**")
        st.dataframe(c5, use_container_width=True, hide_index=True)

    # 해석 설명
    self_peak = int(np.argmax(t_self))
    cov_peak = int(np.argmax(t_cov))
    self_recent = "최근" if self_peak >= recent_cutoff else "과거"
    cov_recent = "최근" if cov_peak >= recent_cutoff else "과거"

    with st.expander("📖 해석 가이드", expanded=True):
        st.markdown(f"""
**X축**: Past Patch Index (0 = 가장 오래된 과거 ≈ {cfg.ctx_patches}개월 전, {cfg.ctx_patches-1} = 가장 최근)

**자기 참조 ({cfg.target_name} → {cfg.target_name})**:
- Peak가 **{self_recent} (patch {self_peak})**에 위치 → 모델이 {cfg.target_name}의 최근 값을 가장 많이 참조
- 자연스러운 AR(자기회귀) 패턴: "최근 CPI가 미래 CPI를 예측하는 데 가장 중요"

**최상위 공변량 ({cfg.target_name} → {top_cov})**:
- Peak가 **{cov_recent} (patch {cov_peak})**에 위치
{"- 모델이 " + top_cov + "의 **장기 기준 수준(level)**을 참조 → '과거 대비 현재 위치'를 파악하는 용도" if cov_peak < recent_cutoff else "- 모델이 " + top_cov + "의 최근 변동을 직접 참조"}
{"- 이는 Attention은 높지만 Ablation 기여가 낮은 이유와 연결됨: 장기 레벨 참조는 패턴 유사성(attention)은 높이지만, 예측 정확도(ablation)에는 실질적으로 기여하지 않을 수 있음" if cov_peak < recent_cutoff else ""}
""")

    # ── Temporal Lag Structure: 전체 공변량 심층 분석 ──────────
    st.divider()
    st.markdown("### Temporal Lag Structure — 변수별 시차 분석")
    st.caption(
        "CPI 예측 시 모델이 **각 공변량의 몇 개월 전 값**을 가장 주목하는지 분석합니다. "
        "Patch ≈ 1개월 (patch_size=32, daily). Patch 0 = 가장 오래된 과거, "
        f"Patch {cfg.ctx_patches-1} = 가장 최근."
    )

    all_covs = [v for v in variates_list if v != cfg.target_name]

    # 각 공변량의 temporal attention 수집
    temporal_data = {}
    peak_lags = {}
    for cov in all_covs:
        t_arr = ana.temporal_attention(cfg.target_name, cov)
        temporal_data[cov] = t_arr
        peak_idx = int(np.argmax(t_arr))
        # lag = 최근으로부터 몇 개월 전 (0 = 가장 최근)
        peak_lags[cov] = cfg.ctx_patches - 1 - peak_idx

    # ── 1. Heatmap: 전 공변량 × 시간 ──────────────────────
    st.markdown("#### 1. 공변량 × 시간 Attention Heatmap")

    # attention ranking 순서로 정렬
    imp_for_sort = ana.target_importance()
    sorted_covs = sorted(all_covs, key=lambda v: float(imp_for_sort[v]), reverse=True)

    heat_matrix = np.array([temporal_data[cov] for cov in sorted_covs])

    # 최근 36개월만 확대 표시 (전체가 너무 넓으면)
    show_recent = min(36, cfg.ctx_patches)
    heat_recent = heat_matrix[:, -show_recent:]

    # x축 라벨: 개월 전
    x_labels = [f"-{show_recent - i}M" if (show_recent - i) % 6 == 0 else ""
                for i in range(show_recent)]

    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_recent,
        x=list(range(show_recent)),
        y=sorted_covs,
        colorscale="YlOrRd",
        hovertemplate="<b>%{y}</b><br>Patch (-%{customdata}M ago)<br>Attention: %{z:.5f}<extra></extra>",
        customdata=np.tile(np.arange(show_recent, 0, -1), (len(sorted_covs), 1)),
    ))

    fig_heat.update_layout(
        xaxis=dict(
            title=f"Past Context (최근 {show_recent}개월)",
            tickvals=[i for i in range(show_recent) if (show_recent - i) % 6 == 0],
            ticktext=[f"-{show_recent - i}M" for i in range(show_recent) if (show_recent - i) % 6 == 0],
            tickfont=dict(size=12),
        ),
        yaxis=dict(title="", tickfont=dict(size=13), autorange="reversed"),
        height=max(350, len(sorted_covs) * 35 + 100),
        margin=dict(t=40, b=60, l=140),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── 2. Peak Lag 비교 차트 ─────────────────────────────
    st.markdown("#### 2. 변수별 Peak Attention Lag (최근 기준)")
    st.caption(
        "각 공변량에서 attention이 가장 높은 시점이 현재로부터 **몇 개월 전**인지. "
        "짧은 lag = 최근 값에 집중 (동행/후행 지표), 긴 lag = 과거 값에 집중 (선행 지표 또는 artifact)."
    )

    lag_df = pd.DataFrame({
        "Variable": sorted_covs,
        "Peak Lag (months)": [peak_lags[v] for v in sorted_covs],
        "Peak Attention": [float(np.max(temporal_data[v])) for v in sorted_covs],
        "Attn Score": [f"{float(imp_for_sort[v]):.2%}" for v in sorted_covs],
    })
    lag_df = lag_df.sort_values("Peak Lag (months)", ascending=True)

    # 색상: 최근(초록) ~ 과거(빨강)
    max_lag = max(peak_lags.values()) if peak_lags else 1
    lag_colors = [
        f"rgb({min(255, int(lag / max(max_lag, 1) * 255))}, "
        f"{min(255, int((1 - lag / max(max_lag, 1)) * 200))}, 80)"
        for lag in lag_df["Peak Lag (months)"]
    ]

    fig_lag = go.Figure()
    fig_lag.add_trace(go.Bar(
        x=lag_df["Variable"],
        y=lag_df["Peak Lag (months)"],
        marker_color=lag_colors,
        text=[f"{lag}M" for lag in lag_df["Peak Lag (months)"]],
        textposition="outside",
        textfont=dict(size=13),
        hovertemplate="<b>%{x}</b><br>Peak lag: %{y}개월 전<extra></extra>",
    ))
    fig_lag.update_layout(
        xaxis=dict(title="", tickfont=dict(size=13)),
        yaxis=dict(title="Peak Lag (months ago)", tickfont=dict(size=13)),
        height=400,
        margin=dict(t=40, b=60),
    )
    st.plotly_chart(fig_lag, use_container_width=True)

    # ── 3. 시차 구조 유형 분류 ────────────────────────────
    st.markdown("#### 3. Lag Structure 유형 분류")

    RECENT_THRESHOLD = 12   # 12개월 이내 = 최근
    DISTANT_THRESHOLD = 60  # 60개월 이상 = 원거리

    lag_categories = {}
    for cov in all_covs:
        t_arr = temporal_data[cov]
        n_pts = len(t_arr)
        recent_mass = float(np.sum(t_arr[-RECENT_THRESHOLD:])) / float(np.sum(t_arr)) if np.sum(t_arr) > 0 else 0

        lag = peak_lags[cov]
        # 분포의 엔트로피로 집중도 판단
        t_norm = t_arr / (np.sum(t_arr) + 1e-12)
        entropy = -float(np.sum(t_norm * np.log(t_norm + 1e-12)))
        max_entropy = np.log(n_pts)
        concentration = 1 - entropy / max_entropy  # 0=균등, 1=집중

        if lag <= RECENT_THRESHOLD and recent_mass > 0.3:
            cat = "Recent Focus"
            desc = "최근 값에 집중 — 동행/단기 선행 지표"
            color = "#2CA02C"
        elif lag >= DISTANT_THRESHOLD:
            cat = "Distant Focus"
            desc = "먼 과거에 집중 — 장기 구조 참조 또는 artifact"
            color = "#D62728"
        elif concentration < 0.05:
            cat = "Diffuse"
            desc = "전 구간 균등 참조 — 추세/수준 정보 활용"
            color = "#999999"
        else:
            cat = "Mid-range"
            desc = "중기(1~5년 전) 집중 — 경기 사이클 반영 가능"
            color = "#EF9F27"

        lag_categories[cov] = {"category": cat, "description": desc, "color": color,
                                "lag": lag, "recent_mass": recent_mass, "concentration": concentration}

    cat_cols = st.columns(4)
    cat_names = ["Recent Focus", "Mid-range", "Distant Focus", "Diffuse"]
    cat_colors = {"Recent Focus": "#2CA02C", "Mid-range": "#EF9F27", "Distant Focus": "#D62728", "Diffuse": "#999999"}
    cat_descs = {
        "Recent Focus": "최근 12개월에 집중 — 동행/단기 선행",
        "Mid-range": "1~5년 전 집중 — 경기 사이클",
        "Distant Focus": "5년+ 과거 집중 — 장기 구조/artifact",
        "Diffuse": "전 구간 균등 — 추세/수준 정보",
    }

    for col, cat_name in zip(cat_cols, cat_names):
        with col:
            members = [v for v, info in lag_categories.items() if info["category"] == cat_name]
            c = cat_colors[cat_name]
            st.markdown(
                f"<span style='color:{c}; font-size:1.1rem;'>■</span> "
                f"**{cat_name}** ({len(members)}개)<br>"
                f"<small style='color:#666'>{cat_descs[cat_name]}</small>",
                unsafe_allow_html=True,
            )
            if members:
                for m in members:
                    st.caption(f"  · {m} (peak: -{lag_categories[m]['lag']}M)")

    # ── 4. 해석 테이블 ────────────────────────────────────
    st.divider()
    st.caption("**전체 변수 Temporal Lag 요약**")
    lag_summary = pd.DataFrame([
        {
            "Variable": cov,
            "Peak Lag": f"-{info['lag']}M",
            "Recent 12M Mass": f"{info['recent_mass']:.1%}",
            "Concentration": f"{info['concentration']:.3f}",
            "Type": info["category"],
        }
        for cov, info in sorted(lag_categories.items(), key=lambda x: x[1]["lag"])
    ])

    def _color_lag_type(val):
        return f"color: {cat_colors.get(val, 'black')}"

    st.dataframe(
        lag_summary.style.applymap(_color_lag_type, subset=["Type"]),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("📝 해석 가이드", expanded=False):
        st.markdown("""
**Peak Lag**: CPI 예측 시 해당 변수의 attention이 가장 높은 시점 (현재 기준 몇 개월 전)

**Recent 12M Mass**: 전체 temporal attention 중 최근 12개월에 집중된 비율.
높으면 모델이 최근 값을 중시, 낮으면 과거 구간을 더 참조.

**Concentration**: 0에 가까우면 전 구간 균등 참조, 1에 가까우면 특정 시점에 집중.

**유형별 경제적 해석**:
- **Recent Focus**: 해당 변수의 최근 움직임이 CPI에 직접 반영. 수입물가, 환율 등 가격 전달 경로.
- **Mid-range**: 1~5년 전 값을 참조. 경기 사이클, 통화정책 시차 효과 포착.
- **Distant Focus**: 5년+ 과거에 집중. 장기 구조적 관계일 수도 있으나, 표본 초기의 특이 패턴에 대한 과적합(artifact) 가능성도 있음. Ablation ΔRMSE와 교차 확인 필요.
- **Diffuse**: 특정 시점 없이 전체 추세를 참조. 수준(level) 정보를 활용하는 변수.
""")


# ----------------------------------------------------------
# Tab: Layer Analysis
# ----------------------------------------------------------

with tabs[tab_offset + 4]:
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
# Tab: 2×2 Diagnostic
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


with tabs[tab_offset + 5]:
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
# Tab: Ablation & Incremental
# ----------------------------------------------------------

with tabs[tab_offset + 6]:
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
