"""
BISTRO-XAI 결과 PDF 리포트 생성
================================
사전 계산된 .npz 결과를 로딩하여 PDF 리포트로 출력.

실행:
    .venv/bin/python3 export_pdf.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "BISTRO_XAI_Report.pdf")
IMG_DIR = os.path.join(os.path.dirname(__file__), "_report_imgs")
os.makedirs(IMG_DIR, exist_ok=True)

# ── 데이터 로딩 ──────────────────────────────────────────────
s1 = np.load(f"{DATA_DIR}/stage1_screening.npz", allow_pickle=True)
s2 = np.load(f"{DATA_DIR}/real_inference_results.npz", allow_pickle=True)
ab = np.load(f"{DATA_DIR}/ablation_results.npz", allow_pickle=True)

# 패널에서 전체 CPI 로딩
panel_path = os.path.join(DATA_DIR, "macro_panel_daily.csv")
if not os.path.exists(panel_path):
    panel_path = os.path.join(DATA_DIR, "macro_panel.csv")
panel_cpi = pd.read_csv(panel_path, index_col=0, parse_dates=True)["CPI_KR_YoY"]
cpi_monthly = panel_cpi.resample("MS").last().dropna().loc["2021-01":"2025-12"]

# Plotly 공통 설정
LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Arial", size=13, color="#222"),
    margin=dict(l=80, r=40, t=60, b=60),
)


def save_fig(fig, name, width=1200, height=500):
    path = os.path.join(IMG_DIR, f"{name}.png")
    fig.write_image(path, width=width, height=height, scale=2)
    return path


# ============================================================
# 1. Forecast Results
# ============================================================
fc_dates = pd.to_datetime([d + "-01" for d in s2["forecast_date"]])
cpi_before = cpi_monthly.loc[:"2022-12"]
cpi_fc = cpi_monthly.loc["2023-01":"2023-12"]
cpi_after = cpi_monthly.loc["2024-01":]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=cpi_before.index, y=cpi_before.values,
    mode="lines", line=dict(color="#333", width=2.5), name="Actual CPI"))

fc_x = [cpi_before.index[-1]] + list(cpi_fc.index)
fc_y = [cpi_before.values[-1]] + list(cpi_fc.values)
fig1.add_trace(go.Scatter(x=fc_x, y=fc_y, mode="lines+markers",
    line=dict(color="#D62728", width=2.5), marker=dict(size=7, symbol="square"),
    name="Actual (forecast period)"))

if len(cpi_after) > 0:
    af_x = [cpi_fc.index[-1]] + list(cpi_after.index)
    af_y = [cpi_fc.values[-1]] + list(cpi_after.values)
    fig1.add_trace(go.Scatter(x=af_x, y=af_y, mode="lines",
        line=dict(color="#333", width=2.5), showlegend=False))

ci_x = list(fc_dates) + list(fc_dates[::-1])
ci_y = list(s2["forecast_ci_hi"]) + list(s2["forecast_ci_lo"][::-1])
fig1.add_trace(go.Scatter(x=ci_x, y=ci_y, fill="toself",
    fillcolor="rgba(74,144,226,0.2)", line=dict(color="rgba(74,144,226,0)"),
    name="BISTRO 90% CI"))
fig1.add_trace(go.Scatter(x=fc_dates, y=s2["forecast_med"], mode="lines+markers",
    line=dict(color="#1A6FD4", width=3), marker=dict(size=8), name="BISTRO (median)"))
if not all(np.isnan(s2["forecast_ar1"])):
    fig1.add_trace(go.Scatter(x=fc_dates, y=s2["forecast_ar1"], mode="lines+markers",
        line=dict(color="#888", width=2, dash="dash"), marker=dict(size=6), name="AR(1)"))
fig1.add_vline(x=fc_dates[0].timestamp()*1000, line_dash="dash", line_color="rgba(100,100,100,0.5)")
fig1.update_layout(title="Korean CPI — BISTRO Forecast vs Actual (2023)",
    yaxis_title="CPI YoY (%)", height=450, **LAYOUT)
img1 = save_fig(fig1, "01_forecast")

# ============================================================
# 2. Feature Selection (Stage 1)
# ============================================================
s1_vars = s1["s1_ranking_vars"].tolist()
s1_attn = s1["s1_ranking_attn"]
s1_uniform = float(s1["s1_uniform_share"])
s2_selected = s2["s2_selected_vars"].tolist()
colors_fs = ["#1A6FD4" if v in s2_selected else "#CCCCCC" for v in s1_vars]
cumsum = np.cumsum(s1_attn)

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Bar(x=s1_vars, y=s1_attn, marker_color=colors_fs,
    text=[f"{a:.1%}" for a in s1_attn], textposition="outside", name="Attention"),
    secondary_y=False)
fig2.add_trace(go.Scatter(x=s1_vars, y=cumsum, mode="lines+markers",
    line=dict(color="#D62728", width=2.5), name="Cumulative"), secondary_y=True)
fig2.add_hline(y=s1_uniform, line_dash="dot", line_color="#999",
    annotation_text=f"Uniform: {s1_uniform:.2%}", secondary_y=False)
fig2.update_layout(title="Stage 1: Attention-Based Feature Screening (28 covariates)",
    xaxis=dict(tickangle=-40), yaxis=dict(tickformat=".1%"),
    yaxis2=dict(tickformat=".0%", range=[0, 1.05]), height=480, **LAYOUT)
img2 = save_fig(fig2, "02_feature_selection")

# ============================================================
# 3. Cross-Variate Heatmap (Stage 2, last layer)
# ============================================================
variates = s2["variates"].tolist()
n_var = int(s2["n_variates"])
ctx_p = int(s2["ctx_patches"])
last_attn = s2["attn_arrays"][-1]

cross = np.zeros((n_var, n_var))
for qi in range(n_var):
    qs, qe = qi * ctx_p, (qi + 1) * ctx_p
    row_sum = 0.0
    for ki in range(n_var):
        ks, ke = ki * ctx_p, (ki + 1) * ctx_p
        cross[qi, ki] = last_attn[qs:qe, ks:ke].mean()
        row_sum += cross[qi, ki]
    if row_sum > 0:
        cross[qi] /= row_sum

z_pct = [[f"{v*100:.1f}%" for v in row] for row in cross]
fig3 = go.Figure(data=go.Heatmap(
    z=cross, x=variates, y=variates, colorscale="YlOrRd", zmin=0, zmax=0.6,
    text=z_pct, texttemplate="%{text}",
    colorbar=dict(title="Attention", tickformat=".0%", dtick=0.1), xgap=1, ygap=1))
fig3.update_layout(title="Cross-Variate Attention Matrix (Stage 2, Layer 12)",
    xaxis=dict(title="Key", tickangle=-35), yaxis=dict(title="Query", autorange="reversed"),
    height=550, **LAYOUT)
img3 = save_fig(fig3, "03_cross_variate", width=1000, height=600)

# ============================================================
# 4. Variable Importance
# ============================================================
cpi_row = cross[0]
imp_order = np.argsort(cpi_row)
fig4 = go.Figure(go.Bar(
    x=cpi_row[imp_order], y=[variates[i] for i in imp_order], orientation="h",
    marker_color=["#E24B4A" if variates[i] == "CPI_KR_YoY" else "#4C9BE8" for i in imp_order],
    text=[f"{v:.1%}" for v in cpi_row[imp_order]], textposition="outside"))
fig4.update_layout(title="Variable Importance — CPI_KR_YoY Attention Distribution",
    xaxis=dict(title="Attention (%)", tickformat=".1%"), height=420, **LAYOUT)
img4 = save_fig(fig4, "04_variable_importance")

# ============================================================
# 5. Ablation dRMSE
# ============================================================
abl_vars = ab["abl_vars"].tolist()
abl_delta = ab["abl_delta_rmse"]
sort_idx = np.argsort(abl_delta)
sorted_vars = [abl_vars[i] for i in sort_idx]
sorted_delta = abl_delta[sort_idx]
colors_ab = ["#E24B4A" if d > 0 else "#4C9BE8" for d in sorted_delta]

fig5 = go.Figure(go.Bar(x=sorted_delta, y=sorted_vars, orientation="h",
    marker_color=colors_ab, text=[f"{d:+.4f}" for d in sorted_delta], textposition="outside"))
fig5.add_vline(x=0, line_color="#333", line_width=1.5)
fig5.update_layout(title="Ablation dRMSE (removal impact)",
    xaxis_title="dRMSE (pp)", height=420, **LAYOUT)
img5 = save_fig(fig5, "05_ablation")

# ============================================================
# 6. 2×2 Diagnostic
# ============================================================
attn_ranking = ab["attn_ranking"].tolist()
attn_values = ab["attn_values"]

# app.py와 동일한 로직: raw attention % + 균등 배분 threshold
# cross[0] = CPI 타겟 행 attention 분포 (이미 위에서 계산됨)
self_attn_share = cross[0, 0]
n_covariates = n_var - 1
attn_thresh = (1.0 - self_attn_share) / n_covariates if n_covariates > 0 else 0.05
delta_thresh = 0.0

QUAD_COLORS = {"Confirmed Driver": "#E24B4A", "Spurious Attention": "#EF9F27",
               "Hidden Contributor": "#4C9BE8", "Irrelevant": "#AAAAAA"}

diag_rows = []
for v in abl_vars:
    # raw attention % from cross-variate matrix (CPI row)
    if v in variates:
        vi = variates.index(v)
        a_raw = cross[0, vi]
    else:
        a_raw = 0.0
    d = abl_delta[abl_vars.index(v)]
    if a_raw >= attn_thresh and d >= delta_thresh: q = "Confirmed Driver"
    elif a_raw >= attn_thresh: q = "Spurious Attention"
    elif d >= delta_thresh: q = "Hidden Contributor"
    else: q = "Irrelevant"
    diag_rows.append({"var": v, "attn": a_raw, "delta": d, "quad": q})

fig6 = go.Figure()
for quad, color in QUAD_COLORS.items():
    pts = [r for r in diag_rows if r["quad"] == quad]
    if not pts: continue
    fig6.add_trace(go.Scatter(
        x=[r["attn"] for r in pts], y=[r["delta"] for r in pts],
        mode="markers+text", name=quad,
        marker=dict(size=14, color=color, line=dict(width=1, color="#333")),
        text=[r["var"] for r in pts], textposition="top center", textfont=dict(size=10)))
fig6.add_vline(x=attn_thresh, line_dash="dash", line_color="#666",
    annotation_text=f"Uniform: {attn_thresh:.2%}")
fig6.add_hline(y=0.0, line_dash="dash", line_color="#666")
fig6.update_layout(title="2x2 Diagnostic: Attention vs Ablation",
    xaxis_title="Cross-Variate Attention (%)", yaxis_title="Ablation dRMSE (pp)",
    xaxis=dict(tickformat=".1%"), height=500, **LAYOUT)
img6 = save_fig(fig6, "06_diagnostic_2x2")

# ============================================================
# 7. Temporal Attention Patterns
# ============================================================
# temporal_attention: forecast patches가 과거 각 패치에 주는 attention 평균
forecast_patches = 12
target_idx = 0  # CPI_KR_YoY

# 자기 참조 temporal
qs_t, qe_t = target_idx * ctx_p, (target_idx + 1) * ctx_p
q_fc = max(qs_t, qe_t - forecast_patches)
t_self = last_attn[q_fc:qe_t, qs_t:qe_t].mean(axis=0)

# 최상위 공변량 temporal
top_cov_idx = 1 + np.argmax(cross[0, 1:])
top_cov_name = variates[top_cov_idx]
ks_t, ke_t = top_cov_idx * ctx_p, (top_cov_idx + 1) * ctx_p
t_cov = last_attn[q_fc:qe_t, ks_t:ke_t].mean(axis=0)

n_p = len(t_self)
recent_cutoff = max(0, n_p - 12)

fig_temp = make_subplots(rows=1, cols=2, subplot_titles=[
    f"{variates[0]} -> {variates[0]} (Self-Attention)",
    f"{variates[0]} -> {top_cov_name} (Top Covariate)"])

for col_idx, (series, color) in enumerate([(t_self, "#E24B4A"), (t_cov, "#EF9F27")], start=1):
    x_vals = list(range(len(series)))
    bar_clrs = [color if i >= recent_cutoff else f"rgba(200,200,200,0.5)" for i in x_vals]
    fig_temp.add_trace(go.Bar(x=x_vals, y=series, marker_color=bar_clrs, showlegend=False),
        row=1, col=col_idx)
    fig_temp.add_vrect(x0=recent_cutoff-0.5, x1=len(series)-0.5,
        fillcolor="rgba(255,215,0,0.14)", line_width=0, row=1, col=col_idx)

fig_temp.update_xaxes(title_text="Past Patch Index")
fig_temp.update_yaxes(title_text="Mean Attention Weight", col=1)
fig_temp.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Arial", size=13, color="#222"), margin=dict(l=80, r=40, t=80, b=60))
img_temp = save_fig(fig_temp, "07_temporal_patterns")

# Temporal: 전 공변량 heatmap + peak lag
all_covs = [v for v in variates if v != variates[0]]
temporal_data = {}
peak_lags = {}
for cov in all_covs:
    ci = variates.index(cov)
    ks_c, ke_c = ci * ctx_p, (ci + 1) * ctx_p
    t_arr = last_attn[q_fc:qe_t, ks_c:ke_c].mean(axis=0)
    temporal_data[cov] = t_arr
    peak_idx = int(np.argmax(t_arr))
    peak_lags[cov] = ctx_p - 1 - peak_idx

# heatmap: attention ranking 순서
sorted_covs = sorted(all_covs, key=lambda v: cross[0, variates.index(v)], reverse=True)
show_recent = min(36, ctx_p)
heat_matrix = np.array([temporal_data[cov][-show_recent:] for cov in sorted_covs])

fig_heat = go.Figure(data=go.Heatmap(z=heat_matrix, x=list(range(show_recent)),
    y=sorted_covs, colorscale="YlOrRd"))
fig_heat.update_layout(
    xaxis=dict(title=f"Past Context (recent {show_recent} patches)",
        tickvals=[i for i in range(show_recent) if (show_recent-i) % 6 == 0],
        ticktext=[f"-{show_recent-i}M" for i in range(show_recent) if (show_recent-i) % 6 == 0]),
    yaxis=dict(title="", autorange="reversed"),
    height=max(350, len(sorted_covs)*35+100), margin=dict(t=40, b=60, l=140),
    paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Arial", size=13, color="#222"))
img_heat = save_fig(fig_heat, "07b_temporal_heatmap", width=1200, height=max(400, len(sorted_covs)*30+100))

# peak lag bar chart
lag_df = pd.DataFrame({
    "Variable": sorted_covs,
    "Peak Lag": [peak_lags[v] for v in sorted_covs]}).sort_values("Peak Lag")
max_lag = max(peak_lags.values()) if peak_lags else 1
lag_colors = [f"rgb({min(255,int(lag/max(max_lag,1)*255))},{min(255,int((1-lag/max(max_lag,1))*200))},80)"
    for lag in lag_df["Peak Lag"]]
fig_lag = go.Figure(go.Bar(x=lag_df["Variable"], y=lag_df["Peak Lag"], marker_color=lag_colors,
    text=[f"{lag}M" for lag in lag_df["Peak Lag"]], textposition="outside"))
fig_lag.update_layout(xaxis=dict(tickangle=-40), yaxis_title="Peak Lag (months ago)",
    height=400, margin=dict(t=40, b=80),
    paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Arial", size=13, color="#222"))
img_lag = save_fig(fig_lag, "07c_peak_lag")

# temporal 해석 텍스트
self_peak = int(np.argmax(t_self))
cov_peak = int(np.argmax(t_cov))
self_recent_str = "recent" if self_peak >= recent_cutoff else "distant past"
cov_recent_str = "recent" if cov_peak >= recent_cutoff else "distant past"

temporal_desc = (
    f"Self-attention ({variates[0]}->{variates[0]}): peak at {self_recent_str} (patch {self_peak}) "
    f"-- natural AR pattern, recent CPI values drive prediction.\n"
    f"Top covariate ({variates[0]}->{top_cov_name}): peak at {cov_recent_str} (patch {cov_peak})."
)
if cov_peak < recent_cutoff:
    temporal_desc += (
        f"\n{top_cov_name} references long-term level (~{ctx_p-1-cov_peak} months ago). "
        "This explains the Attention-Ablation discrepancy: long-term level reference "
        "raises pattern similarity (attention) but may not contribute to forecast accuracy (ablation)."
    )

# Lag structure 유형 분류
RECENT_THRESHOLD = 12
DISTANT_THRESHOLD = 60
lag_categories = {}
for cov in all_covs:
    t_arr = temporal_data[cov]
    recent_mass = float(np.sum(t_arr[-RECENT_THRESHOLD:])) / float(np.sum(t_arr)) if np.sum(t_arr) > 0 else 0
    lag = peak_lags[cov]
    t_norm = t_arr / (np.sum(t_arr) + 1e-12)
    entropy = -float(np.sum(t_norm * np.log(t_norm + 1e-12)))
    max_entropy = np.log(len(t_arr))
    concentration = 1 - entropy / max_entropy
    if lag <= RECENT_THRESHOLD and recent_mass > 0.3:
        cat = "Recent Focus"
    elif lag >= DISTANT_THRESHOLD:
        cat = "Distant Focus"
    elif concentration < 0.05:
        cat = "Diffuse"
    else:
        cat = "Mid-range"
    lag_categories[cov] = {"category": cat, "lag": lag, "recent_mass": recent_mass, "concentration": concentration}

lag_type_counts = {}
for info in lag_categories.values():
    lag_type_counts[info["category"]] = lag_type_counts.get(info["category"], 0) + 1
lag_type_summary = ", ".join(f"{k}: {v}" for k, v in sorted(lag_type_counts.items()))

# ============================================================
# 8. Incremental RMSE
# ============================================================
inc_rmse = ab["inc_rmse"]
inc_n = ab["inc_n_vars"]
inc_labels = ab["inc_labels"].tolist()
baseline_rmse = float(ab["baseline_rmse"])

fig7 = go.Figure()
fig7.add_trace(go.Scatter(x=inc_n, y=inc_rmse, mode="lines+markers",
    line=dict(color="#1A6FD4", width=2.5), marker=dict(size=8),
    text=inc_labels, hovertemplate="%{text}<br>RMSE: %{y:.4f}"))
fig7.add_hline(y=baseline_rmse, line_dash="dot", line_color="#E24B4A",
    annotation_text=f"Full: {baseline_rmse:.4f}")
fig7.update_layout(title="Incremental RMSE: Adding variables by attention rank",
    xaxis_title="# Covariates", yaxis_title="RMSE (pp)", height=400, **LAYOUT)
img7 = save_fig(fig7, "07_incremental")

# ============================================================
# PDF 생성
# ============================================================
print("Generating PDF...")

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "BISTRO-XAI Attention Analysis Report", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

pdf = PDF(orientation="L", unit="mm", format="A4")
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ── Title page ──
pdf.add_page()
pdf.ln(30)
pdf.set_font("Helvetica", "B", 28)
pdf.cell(0, 15, "BISTRO-XAI", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 16)
pdf.cell(0, 10, "Attention Analysis & Feature Selection Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 8, "Korean CPI (YoY) Forecasting with BISTRO Foundation Model", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(100, 100, 100)

info_lines = [
    f"Model: BISTRO (91M params, 12 layers, 12 heads, fine-tuned on BIS 4,925 series)",
    f"Target: CPI_KR_YoY | Covariates: {len(s2_selected)} selected from {int(s1['n_variates'])-1} candidates",
    f"Context: ~120 patches (~10 years) | Forecast: 2023-01 ~ 2023-12",
    f"Data mode: {s1['data_mode']}",
]
for line in info_lines:
    pdf.cell(0, 6, line, align="C", new_x="LMARGIN", new_y="NEXT")

# ── Charts ──
sections = [
    ("1. Forecast Results", img1,
     "BISTRO median forecast vs Actual CPI (2021-2025). "
     "Red squares = actual CPI during forecast period (2023). "
     "The 2023 disinflation trend was not captured by the model -- "
     "BISTRO predicted persistent inflation while actual CPI declined steadily."),
    ("2. Feature Selection (Stage 1)", img2,
     f"Attention-based screening: {len(s2_selected)} variables selected from {len(s1_vars)} covariates. "
     f"Uniform share threshold: {s1_uniform:.2%}. "
     "Blue bars = selected for Stage 2, gray = filtered out. "
     "Variables above the uniform line receive above-average attention from CPI."),
    ("3. Cross-Variate Attention Matrix", img3,
     f"Stage 2 (Layer 12, last layer). Self-attention (CPI->CPI): {cross[0,0]:.1%}. "
     f"Top covariate: {variates[1+np.argmax(cross[0,1:])]} ({cross[0,1+np.argmax(cross[0,1:])]:.1%}). "
     "Diagonal dominance reflects BinaryAttentionBias -- a learned bias that distinguishes "
     "same-variable vs cross-variable tokens. Colorscale capped at 60%."),
    ("4. Variable Importance", img4,
     f"CPI target row from the cross-variate matrix. Self-attention dominates (~{cross[0,0]:.0%}). "
     "Among covariates, attention is distributed relatively evenly (3-5% each), "
     "reflecting the model's multivariate information aggregation."),
    ("5. Ablation Study", img5,
     f"Each variable removed one at a time; dRMSE = RMSE(removed) - RMSE(full). "
     f"dRMSE > 0: removal hurts forecast (variable contributes). "
     f"Baseline RMSE: {baseline_rmse:.4f}pp. "
     f"Variables with positive dRMSE: {int(np.sum(abl_delta > 0))}/{len(abl_delta)}."),
    ("6. 2x2 Diagnostic: Attention vs Ablation", img6,
     "Attention =/= Causal Importance. Vertical line = uniform attention threshold. "
     "Confirmed Driver: high attention + positive dRMSE (genuine predictors). "
     "Spurious Attention: high attention + low dRMSE (correlated but redundant). "
     "Hidden Contributor: low attention + positive dRMSE (unique info not captured by attention)."),
]

# Temporal sections (7a, 7b, 7c)
sections += [
    ("7a. Temporal Attention: Self vs Top Covariate", img_temp, temporal_desc),
    ("7b. Temporal Attention Heatmap (all covariates)", img_heat,
     f"Each row = one covariate, showing attention weight across the recent {show_recent} patches. "
     "Brighter = higher attention. Variables sorted by cross-variate attention rank. "
     f"Lag structure types: {lag_type_summary}."),
    ("7c. Peak Attention Lag by Variable", img_lag,
     "Bar height = number of months ago where attention peaks for each variable. "
     "Short lag (green) = recent-focused (coincident/short-leading indicators). "
     "Long lag (red) = distant-focused (long-term structure reference or artifact)."),
    ("8. Incremental Analysis", img7,
     f"Adding variables one by one in attention-rank order. "
     f"Best RMSE at {int(inc_n[np.argmin(inc_rmse)])} covariates ({float(inc_rmse[np.argmin(inc_rmse)]):.4f}pp). "
     f"Full model ({int(inc_n[-1])} covariates): {baseline_rmse:.4f}pp. "
     "Diminishing returns after the top variables, suggesting attention ranking "
     "effectively identifies the most informative covariates."),
]

for title, img_path, desc in sections:
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, desc)
    pdf.ln(3)
    # 이미지를 페이지 너비에 맞춤
    img_w = 257  # A4 landscape - margins
    pdf.image(img_path, x=15, w=img_w)

# ── Summary table page ──
pdf.add_page()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "9. Summary Table - Attention vs Ablation", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

pdf.set_font("Helvetica", "B", 9)
col_widths = [8, 35, 20, 20, 20, 20, 45]
headers = ["#", "Variable", "Attn Rank", "Attn %", "dRMSE", "Abl Rank", "Quadrant"]
for w, h in zip(col_widths, headers):
    pdf.cell(w, 7, h, border=1, align="C")
pdf.ln()

abl_rank_order = np.argsort(-abl_delta)
pdf.set_font("Helvetica", "", 9)
for rank, i in enumerate(abl_rank_order, 1):
    v = abl_vars[i]
    a_rank = attn_ranking.index(v) + 1 if v in attn_ranking else 0
    a_val = cross[0, variates.index(v)] if v in variates else 0.0
    d = abl_delta[i]
    if a_val >= attn_thresh and d >= delta_thresh: q = "Confirmed Driver"
    elif a_val >= attn_thresh: q = "Spurious Attention"
    elif d >= delta_thresh: q = "Hidden Contributor"
    else: q = "Irrelevant"

    row = [str(rank), v, str(a_rank), f"{a_val:.2%}", f"{d:+.4f}", str(rank), q]
    for w, val in zip(col_widths, row):
        pdf.cell(w, 6, val, border=1, align="C")
    pdf.ln()

# ── Save ──
pdf.output(OUTPUT_PDF)
print(f"\nSaved: {OUTPUT_PDF}")

# Cleanup temp images
import shutil
shutil.rmtree(IMG_DIR, ignore_errors=True)
print("Done.")
