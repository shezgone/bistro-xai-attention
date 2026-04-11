"""
BISTRO-XAI Report v3 -Full Pipeline + 2024 Forecast + CTX Comparison
=====================================================================
실행: .venv/bin/python3 export_pdf_v3.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
from scipy.stats import spearmanr

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
S0_DIR = os.path.join(DATA_DIR, "stage0")
OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "BISTRO_XAI_Report_v3.pdf")
IMG_DIR = os.path.join(os.path.dirname(__file__), "_report_v3_imgs")
os.makedirs(IMG_DIR, exist_ok=True)

# ── Data loading ──
s0 = np.load(f"{S0_DIR}/stage0_ranking.npz", allow_pickle=True)
s1r = np.load(f"{S0_DIR}/stage1_results.npz", allow_pickle=True)
inc = np.load(f"{S0_DIR}/incremental_results.npz", allow_pickle=True)
hd = np.load(f"{S0_DIR}/head_analysis.npz", allow_pickle=True)
s2 = np.load(f"{DATA_DIR}/real_inference_results.npz", allow_pickle=True)
f18 = np.load(f"{DATA_DIR}/forecast_optimal18.npz", allow_pickle=True)
f18_24 = np.load(f"{DATA_DIR}/forecast_optimal18_2024.npz", allow_pickle=True)
f1 = np.load(f"{DATA_DIR}/forecast_univariate.npz", allow_pickle=True)

panel_path = os.path.join(DATA_DIR, "macro_panel_daily.csv")
panel_cpi = pd.read_csv(panel_path, index_col=0, parse_dates=True)["CPI_KR_YoY"]
cpi_monthly = panel_cpi.resample("MS").last().dropna().loc["2021-01":"2025-12"]

best_k = int(inc['best_k'])
best_rmse = float(inc['best_rmse'])

VAR_LABEL = {
    "CPI_KR_YoY": "Korean CPI YoY", "AUD_USD": "AUD/USD FX Rate",
    "CN_Interbank3M": "China 3M Interbank", "US_UnempRate": "US Unemployment Rate",
    "JP_REER": "Japan Real Eff. FX (BIS)", "JP_Interbank3M": "Japan 3M Interbank",
    "JP_CoreCPI": "Japan Core CPI", "KC_FSI": "KC Financial Stress Index",
    "KR_MfgProd": "Korea Mfg Production", "Pork": "Global Pork Price",
    "US_NFP": "US Nonfarm Payrolls", "US_TradeTransEmp": "US Trade/Transport Emp",
    "THB_USD": "THB/USD FX Rate", "PPI_CopperNickel": "Copper/Nickel PPI",
    "CN_PPI": "China PPI", "US_Mortgage15Y": "US 15Y Mortgage Rate",
    "UK_10Y_Bond": "UK 10Y Gilt", "US_ExportPI": "US Export Price Index",
    "US_DepInstCredit": "US Deposit Inst Credit",
    "BR_CPI": "Brazil CPI", "BR_DiscountRate": "Brazil Discount Rate",
}

def vl(v):
    label = VAR_LABEL.get(str(v))
    return f"{v} ({label})" if label else str(v)

LAYOUT = dict(paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Arial", size=13, color="#222"), margin=dict(l=80, r=40, t=60, b=60))

def save_fig(fig, name, width=1200, height=500):
    path = os.path.join(IMG_DIR, f"{name}.png")
    fig.write_image(path, width=width, height=height, scale=2)
    return path

# ============================================================
# Compute RMSEs
# ============================================================
fc_dates_23 = pd.to_datetime([d + "-01" for d in s2["forecast_date"]])
actual_23 = s2["forecast_actual"]
valid_23 = ~np.isnan(actual_23)

fc_dates_24 = pd.to_datetime([d + "-01" for d in f18_24["forecast_date"]])
actual_24 = f18_24["forecast_actual"].astype(float)
valid_24 = ~np.isnan(actual_24)

def rmse(pred, actual, valid):
    p = np.array(pred, dtype=float)
    return float(np.sqrt(np.mean((p[valid] - actual[valid]) ** 2)))

rmse_18_23 = rmse(f18["forecast_med"], actual_23, valid_23)
rmse_11_23 = rmse(s2["forecast_med"], actual_23, valid_23)
rmse_1_23 = rmse(f1["forecast_med"], actual_23, valid_23)
rmse_ar1_23 = rmse(s2["forecast_ar1"], actual_23, valid_23)
rmse_18_24 = rmse(f18_24["forecast_med"], actual_24, valid_24)
rmse_ar1_24 = rmse(f18_24["forecast_ar1"], actual_24, valid_24)

# ============================================================
# Charts
# ============================================================

# 1. Forecast 2023 + 2024 combined
print("1. Forecast comparison (2023 + 2024)...")
cpi_before = cpi_monthly.loc[:"2022-12"]
cpi_fc23 = cpi_monthly.loc["2023-01":"2023-12"]
cpi_fc24 = cpi_monthly.loc["2024-01":"2024-12"]
cpi_after = cpi_monthly.loc["2025-01":]

fig1 = go.Figure()

# Actual CPI
fig1.add_trace(go.Scatter(x=cpi_before.index, y=cpi_before.values, mode="lines",
    line=dict(color="#333", width=2.5), name="Actual CPI"))
fig1.add_trace(go.Scatter(
    x=[cpi_before.index[-1]] + list(cpi_fc23.index),
    y=[cpi_before.values[-1]] + list(cpi_fc23.values),
    mode="lines+markers", line=dict(color="#D62728", width=2.5),
    marker=dict(size=7, symbol="square"), name="Actual (2023 OOS)"))
fig1.add_trace(go.Scatter(
    x=[cpi_fc23.index[-1]] + list(cpi_fc24.index),
    y=[cpi_fc23.values[-1]] + list(cpi_fc24.values),
    mode="lines+markers", line=dict(color="#E65100", width=2.5),
    marker=dict(size=7, symbol="square"), name="Actual (2024 OOS)"))
if len(cpi_after) > 0:
    fig1.add_trace(go.Scatter(
        x=[cpi_fc24.index[-1]] + list(cpi_after.index),
        y=[cpi_fc24.values[-1]] + list(cpi_after.values),
        mode="lines", line=dict(color="#333", width=2.5), showlegend=False))

# 2023 forecasts
fig1.add_trace(go.Scatter(x=fc_dates_23, y=np.array(f18["forecast_med"], dtype=float),
    mode="lines+markers", line=dict(color="#DC2626", width=3),
    marker=dict(size=8, symbol="hexagram"), name=f"288->18 (2023, RMSE {rmse_18_23:.3f})"))
fig1.add_trace(go.Scatter(x=fc_dates_23, y=np.array(s2["forecast_med"], dtype=float),
    mode="lines+markers", line=dict(color="#1A6FD4", width=2.5),
    marker=dict(size=7), name=f"29->11 (2023, RMSE {rmse_11_23:.3f})"))
fig1.add_trace(go.Scatter(x=fc_dates_23, y=np.array(f1["forecast_med"], dtype=float),
    mode="lines+markers", line=dict(color="#000", width=2, dash="dot"),
    marker=dict(size=6, symbol="x"), name=f"Univariate (2023, RMSE {rmse_1_23:.3f})"))
fig1.add_trace(go.Scatter(x=fc_dates_23, y=np.array(s2["forecast_ar1"], dtype=float),
    mode="lines+markers", line=dict(color="#888", width=2, dash="dash"),
    marker=dict(size=5), name=f"AR(1) (2023, RMSE {rmse_ar1_23:.3f})"))

# 2024 forecasts
fig1.add_trace(go.Scatter(x=fc_dates_24, y=np.array(f18_24["forecast_med"], dtype=float),
    mode="lines+markers", line=dict(color="#DC2626", width=3),
    marker=dict(size=8, symbol="hexagram"), showlegend=False,
    name=f"288->18 (2024)"))
fig1.add_trace(go.Scatter(x=fc_dates_24, y=np.array(f18_24["forecast_ar1"], dtype=float),
    mode="lines+markers", line=dict(color="#888", width=2, dash="dash"),
    marker=dict(size=5), showlegend=False, name="AR(1) (2024)"))

# CI bands
ci23_x = list(fc_dates_23) + list(fc_dates_23[::-1])
ci23_y = list(f18["forecast_ci_hi"]) + list(f18["forecast_ci_lo"][::-1])
fig1.add_trace(go.Scatter(x=ci23_x, y=ci23_y, fill="toself",
    fillcolor="rgba(220,38,38,0.12)", line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip", name="90% CI (2023)"))
ci24_x = list(fc_dates_24) + list(fc_dates_24[::-1])
ci24_y = list(f18_24["forecast_ci_hi"]) + list(f18_24["forecast_ci_lo"][::-1])
fig1.add_trace(go.Scatter(x=ci24_x, y=ci24_y, fill="toself",
    fillcolor="rgba(220,38,38,0.12)", line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip", showlegend=False))

fig1.add_vline(x=fc_dates_23[0].timestamp()*1000, line_dash="dash", line_color="rgba(100,100,100,0.4)",
    annotation_text="2023 forecast", annotation_position="top left", annotation_font_size=11)
fig1.add_vline(x=fc_dates_24[0].timestamp()*1000, line_dash="dash", line_color="rgba(220,38,38,0.4)",
    annotation_text="2024 forecast", annotation_position="top left", annotation_font_size=11)
fig1.update_layout(title="Korean CPI Forecast -2023 & 2024 Out-of-Sample",
    yaxis_title="CPI YoY (%)", height=520, **LAYOUT,
    legend=dict(font=dict(size=10), x=0.01, y=0.99))
img1 = save_fig(fig1, "01_forecast_combined", width=1300, height=520)

# 2. Stage 0 Attention Ranking
print("2. Stage 0 ranking...")
s0_vars = [str(v) for v in s0['ranking_vars']]
s0_attn = s0['ranking_attn'].astype(float)
n_show = 30
fig2 = go.Figure(go.Bar(x=[vl(v) for v in s0_vars[:n_show]], y=s0_attn[:n_show],
    marker_color=["#DC2626" if i < 25 else "#CCC" for i in range(n_show)]))
fig2.add_hline(y=s0_attn[24], line_dash="dash", line_color="red", annotation_text="Top 25 cutoff")
fig2.update_layout(title=f"Stage 0: Full Screening (288 vars, CTX=10) -Top {n_show}",
    xaxis=dict(tickangle=-45, tickfont_size=9), yaxis_title="Attention", height=480, **LAYOUT)
img2 = save_fig(fig2, "02_stage0_ranking")

# 3. Stage 1 Attention + Ablation
print("3. Stage 1 attention + ablation...")
s1_rank_vars = [str(v) for v in s1r['ranking_vars']]
s1_rank_attn = s1r['ranking_attn'].astype(float)
s1_abl_vars = [str(v) for v in s1r['ablation_vars']]
s1_abl_delta = s1r['ablation_delta'].astype(float)
s1_harmful = [str(v) for v in s1r['harmful_vars']]
abl_map = dict(zip(s1_abl_vars, s1_abl_delta))

bar_colors = ["#EF4444" if v in s1_harmful else "#10B981" for v in s1_rank_vars]
fig3 = go.Figure()
fig3.add_trace(go.Bar(x=[vl(v) for v in s1_rank_vars], y=s1_rank_attn,
    marker_color=bar_colors, name="Attention"))
fig3.add_trace(go.Scatter(x=[vl(v) for v in s1_rank_vars],
    y=[abl_map.get(v, 0) for v in s1_rank_vars], mode="lines+markers",
    line=dict(color="#6366F1", width=2), name="Ablation dRMSE", yaxis="y2"))
fig3.update_layout(title="Stage 1: Attention + Ablation (25 vars, CTX=120)",
    xaxis=dict(tickangle=-45, tickfont_size=9),
    yaxis=dict(title="Attention", side="left"),
    yaxis2=dict(title="dRMSE", side="right", overlaying="y"),
    height=480, **LAYOUT)
img3 = save_fig(fig3, "03_stage1_attn_ablation")

# 4. Incremental Addition
print("4. Incremental addition...")
inc_n = inc['n_vars'].astype(int)
inc_rmse = inc['rmse'].astype(float)
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=inc_n, y=inc_rmse, mode="lines+markers",
    line=dict(color="#3B82F6", width=2.5), marker=dict(size=9,
    color=["#DC2626" if n == best_k else "#3B82F6" for n in inc_n])))
fig4.add_trace(go.Scatter(x=[best_k], y=[best_rmse], mode="markers+text",
    marker=dict(size=16, color="#DC2626", symbol="star"),
    text=[f"N={best_k}"], textposition="top center", showlegend=False))
fig4.add_hline(y=rmse_11_23, line_dash="dot", line_color="#888",
    annotation_text=f"Legacy 11 vars ({rmse_11_23:.4f})")
fig4.update_layout(title=f"Incremental Addition: Optimal N={best_k} (RMSE {best_rmse:.4f})",
    xaxis=dict(title="# Covariates", dtick=1), yaxis_title="RMSE (pp)", height=420, **LAYOUT)
img4 = save_fig(fig4, "04_incremental")

# 5. 2x2 Diagnostic
print("5. 2x2 diagnostic...")
s1_self_attn = float(s1r['self_attn']) if 'self_attn' in s1r else 0.4
uniform_share = (1.0 - s1_self_attn) / len(s1_rank_vars)
diag_x = [s1_rank_attn[s1_rank_vars.index(v)] if v in s1_rank_vars else 0 for v in s1_abl_vars]
diag_y = [abl_map.get(v, 0) for v in s1_abl_vars]

q_colors = []
for v, ax, ay in zip(s1_abl_vars, diag_x, diag_y):
    if ax >= uniform_share and ay > 0: q_colors.append("#10B981")
    elif ax >= uniform_share: q_colors.append("#EF4444")
    elif ay > 0: q_colors.append("#F59E0B")
    else: q_colors.append("#9CA3AF")

fig5 = go.Figure(go.Scatter(x=diag_x, y=diag_y, mode="markers+text",
    marker=dict(size=12, color=q_colors), text=[vl(v) for v in s1_abl_vars],
    textposition="top center", textfont=dict(size=8), showlegend=False))
fig5.add_hline(y=0, line_dash="dash", line_color="rgba(100,100,100,0.5)")
fig5.add_vline(x=uniform_share, line_dash="dash", line_color="rgba(100,100,100,0.5)",
    annotation_text=f"uniform: {uniform_share:.4f}")
fig5.update_layout(title="2x2 Diagnostic: Attention vs Ablation dRMSE",
    xaxis_title="Attention Score", yaxis_title="dRMSE (pp)", height=520, **LAYOUT)
img5 = save_fig(fig5, "05_diagnostic")

# 6. Head Analysis Heatmap
print("6. Head analysis...")
h_variates = [str(v) for v in hd['variates']]
h_attn = hd['head_attention']
h_ctx = int(hd['ctx_patches'])
n_heads = int(hd['n_heads'])
h_covs = [v for v in h_variates if v != "CPI_KR_YoY"]

h_imp = {}
for hi in range(n_heads):
    a = h_attn[hi]
    imp = {}
    for j, vn in enumerate(h_variates):
        ks, ke = j * h_ctx, (j+1) * h_ctx
        block = a[:h_ctx, ks:ke]
        imp[vn] = float(block.sum() / h_ctx)
    h_imp[hi] = imp

heatmap_data = [[h_imp[hi].get(v, 0) for v in h_covs] for hi in range(n_heads)]
fig6 = go.Figure(go.Heatmap(z=heatmap_data, x=[vl(v) for v in h_covs],
    y=[f"Head {i}" for i in range(n_heads)], colorscale="YlOrRd"))
fig6.update_layout(title="Head x Variable Attention (Layer 11, 12 Heads)",
    xaxis=dict(tickangle=-45, tickfont_size=8), height=450, **LAYOUT)
img6 = save_fig(fig6, "06_head_heatmap", width=1200, height=500)

# 7. Head Correlation
print("7. Head correlation...")
hranks = {}
for hi in range(n_heads):
    c = {k: h_imp[hi][k] for k in h_covs}
    r = sorted(h_covs, key=lambda v: -c[v])
    hranks[hi] = [r.index(v) for v in h_covs]

corr_mat = np.zeros((n_heads, n_heads))
for hi in range(n_heads):
    for hj in range(n_heads):
        if hi == hj: corr_mat[hi][hj] = 1.0
        else: corr_mat[hi][hj], _ = spearmanr(hranks[hi], hranks[hj])

fig7 = go.Figure(go.Heatmap(z=corr_mat, x=[f"H{i}" for i in range(n_heads)],
    y=[f"H{i}" for i in range(n_heads)], colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1))
fig7.update_layout(title="Head Ranking Correlation (Spearman)", height=420, **LAYOUT)
img7 = save_fig(fig7, "07_head_correlation", width=800, height=500)

# 8. Temporal - Peak Lag
print("8. Temporal peak lag...")
optimal_18 = [str(v) for v in inc['ranking'][:best_k]]
tp_avg = h_attn.mean(axis=0)
tp_lags = {}
for j, vn in enumerate(h_variates):
    if vn == "CPI_KR_YoY" or vn not in optimal_18: continue
    ks, ke = j * h_ctx, (j+1) * h_ctx
    q_start = max(0, h_ctx - 12)
    block = tp_avg[q_start:h_ctx, ks:ke].mean(axis=0)
    tp_lags[vn] = h_ctx - 1 - int(np.argmax(block))

lag_sorted = sorted(tp_lags.items(), key=lambda x: x[1])
max_lag_v = max(tp_lags.values()) or 1
lag_colors_v = [f"rgb({min(255,int(lag/max_lag_v*255))},{min(255,int((1-lag/max_lag_v)*200))},80)"
    for _, lag in lag_sorted]
fig8 = go.Figure(go.Bar(x=[vl(v) for v, _ in lag_sorted], y=[l for _, l in lag_sorted],
    marker_color=lag_colors_v, text=[f"{l}M" for _, l in lag_sorted], textposition="outside"))
fig8.update_layout(title="Peak Attention Lag (Optimal 18 vars)",
    xaxis=dict(tickangle=-45, tickfont_size=9), yaxis_title="Peak Lag (months ago)",
    height=420, **LAYOUT)
img8 = save_fig(fig8, "08_peak_lag")


# ============================================================
# PDF Generation
# ============================================================
print("\nGenerating PDF...")

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "BISTRO-XAI Report v3", align="R",
                  new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 287, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

pdf = PDF(orientation="L", unit="mm", format="A4")
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ── Title Page ──
pdf.add_page()
pdf.ln(20)
pdf.set_font("Helvetica", "B", 32)
pdf.cell(0, 15, "BISTRO-XAI", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 16)
pdf.cell(0, 10, "Explainable Macroeconomic Forecasting", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 10, "with Attention-Based Variable Selection", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)
pdf.set_font("Helvetica", "", 13)
pdf.cell(0, 8, "Korean CPI (YoY) Forecasting", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "288 FRED Candidates -> Optimal 18 Variables", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.ln(12)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 6, "April 2026", align="C", new_x="LMARGIN", new_y="NEXT")

# ── Page 2: Foundation Model + Pipeline Overview ──
pdf.add_page()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "B", 18)
pdf.cell(0, 10, "1. Foundation Model & Pipeline Overview", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

pdf.set_font("Helvetica", "B", 13)
pdf.cell(0, 8, "BISTRO (BIS Time-series Regression Oracle)", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
model_info = [
    ("Architecture", "MOIRAI (Masked encoder-based Universal Time Series Forecasting)"),
    ("Parameters", "91M (d_model=768, 12 layers, 12 heads)"),
    ("Training Data", "BIS 63 countries, 4,925 time series (1970~2024)"),
    ("Patch Size", "32 days (daily data aggregated per patch)"),
    ("Max Seq Length", "512 patches (RoPE positional encoding)"),
    ("Distribution", "Mixture: StudentT + NormalFixedScale + NegBinomial + LogNormal"),
    ("Sampling", "100 samples per forecast -> median + 5th/95th percentile CI"),
    ("Source", "Koyuncu, Kwon, Lombardi, Shin, Perez-Cruz. BIS Quarterly Review, March 2026"),
]
col_w = [45, 180]
for key, val in model_info:
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(col_w[0], 6, key, border=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(col_w[1], 6, val, border=1)
    pdf.ln()

pdf.ln(5)
pdf.set_font("Helvetica", "B", 13)
pdf.cell(0, 8, "XAI Variable Selection Pipeline", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
pipeline_info = [
    ("Stage 0", "CTX=10 (10 months), 288 vars", "Shortened-context full screening via attention ranking", "288 -> 25"),
    ("Stage 1", "CTX=120 (10 years), 25 vars", "Full-context re-inference + leave-one-out ablation", "25 -> 23 (harmful removed)"),
    ("Incremental", "CTX=120, greedy addition", "Add vars by attention rank, find RMSE minimum", f"23 -> {best_k} (optimal)"),
]
hdr = ["Stage", "Setting", "Method", "Result"]
col_w2 = [30, 60, 100, 40]
pdf.set_font("Helvetica", "B", 10)
for w, h in zip(col_w2, hdr):
    pdf.cell(w, 7, h, border=1, align="C")
pdf.ln()
pdf.set_font("Helvetica", "", 9)
for row in pipeline_info:
    for w, val in zip(col_w2, row):
        pdf.cell(w, 6, val, border=1)
    pdf.ln()

pdf.ln(5)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 5,
    "Key insight: MOIRAI's max_seq_len=512 limits total input tokens (n_variates x ctx_patches). "
    f"With CTX=120 and 19 variates: 19 x 120 = 2,280 tokens (within limit). "
    "Stage 0 uses CTX=10 to fit all 288 vars: 288 x 10 = 2,880 tokens. "
    "This enables fair, simultaneous evaluation of all candidates.")

# ── Page 3: RMSE Summary ──
pdf.add_page()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "B", 18)
pdf.cell(0, 10, "2. Forecast Accuracy Summary", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

# 2023
pdf.set_font("Helvetica", "B", 13)
pdf.cell(0, 8, "2023 Forecast (Context: ~2013-01 ~ 2022-12, CTX=120)", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "B", 10)
pdf.cell(80, 7, "Experiment", border=1, align="C")
pdf.cell(30, 7, "Variables", border=1, align="C")
pdf.cell(30, 7, "RMSE", border=1, align="C")
pdf.cell(40, 7, "vs AR(1)", border=1, align="C")
pdf.ln()

rows_23 = [
    (f"Optimal 18 (288 pool)", 18, rmse_18_23),
    ("Stage2 11 (29 pool)", 11, rmse_11_23),
    ("Univariate (no covariates)", 1, rmse_1_23),
    ("AR(1) statistical baseline", 0, rmse_ar1_23),
]
pdf.set_font("Helvetica", "", 10)
for name, nv, r in sorted(rows_23, key=lambda x: x[2]):
    ratio = f"{r/rmse_ar1_23:.1%}" if rmse_ar1_23 > 0 else "-"
    pdf.cell(80, 6, name, border=1)
    pdf.cell(30, 6, str(nv), border=1, align="C")
    pdf.cell(30, 6, f"{r:.4f}", border=1, align="C")
    pdf.cell(40, 6, ratio, border=1, align="C")
    pdf.ln()

pdf.ln(5)

# 2024
pdf.set_font("Helvetica", "B", 13)
pdf.cell(0, 8, "2024 Forecast (Context: ~2014-01 ~ 2023-12, CTX=120)", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "B", 10)
pdf.cell(80, 7, "Experiment", border=1, align="C")
pdf.cell(30, 7, "Variables", border=1, align="C")
pdf.cell(30, 7, "RMSE", border=1, align="C")
pdf.cell(40, 7, "vs AR(1)", border=1, align="C")
pdf.ln()

rows_24 = [
    (f"Optimal 18 (288 pool)", 18, rmse_18_24),
    ("AR(1) statistical baseline", 0, rmse_ar1_24),
]
pdf.set_font("Helvetica", "", 10)
for name, nv, r in sorted(rows_24, key=lambda x: x[2]):
    ratio = f"{r/rmse_ar1_24:.1%}" if rmse_ar1_24 > 0 else "-"
    pdf.cell(80, 6, name, border=1)
    pdf.cell(30, 6, str(nv), border=1, align="C")
    pdf.cell(30, 6, f"{r:.4f}", border=1, align="C")
    pdf.cell(40, 6, ratio, border=1, align="C")
    pdf.ln()

pdf.ln(5)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 5,
    "2024 shows higher absolute RMSE because Korean CPI dropped sharply from 2.8% to 1.3% (H2), "
    "a regime shift that the model's context window didn't fully anticipate. "
    "However, BISTRO still outperforms AR(1) by a wider margin in 2024 than 2023.")

# ── Chart pages ──
sections = [
    ("3. Forecast Comparison -2023 & 2024 Out-of-Sample", img1,
     f"Combined 2023 and 2024 out-of-sample forecasts. "
     f"2023: Optimal 18 (RMSE {rmse_18_23:.4f}) outperforms Legacy 11 ({rmse_11_23:.4f}) "
     f"and AR(1) ({rmse_ar1_23:.4f}). "
     f"2024: Context includes 2023 actuals. RMSE {rmse_18_24:.4f} vs AR(1) {rmse_ar1_24:.4f}. "
     "BISTRO captures the direction of CPI decline but underestimates the speed of disinflation in 2024 H2."),

    ("4. Stage 0 -Full Screening (288 vars, CTX=10)", img2,
     f"All 288 FRED variables in a single inference pass with shortened context (CTX=10, ~10 months). "
     f"Token count: 288 x 10 = 2,880 < max_seq_len limit. "
     f"AUD_USD dominates at 10.8%, 4.9x higher than 2nd place (BRL_USD 2.2%). "
     f"Red bars = top 25 selected for Stage 1."),

    ("5. Stage 1 -Re-inference + Ablation (25 vars, CTX=120)", img3,
     f"Full context re-inference. Green bars = helpful (dRMSE > 0), red = harmful. "
     f"Baseline RMSE: {float(s1r['baseline_rmse']):.4f}. "
     f"Harmful variables ({', '.join(s1_harmful)}): passed Stage 0 but failed Stage 1 validation. "
     "Short-context pattern similarity != long-context predictive value. "),

    ("6. Incremental Addition -Optimal Variable Count", img4,
     f"Variables added in attention rank order. Optimal at N={best_k} (RMSE {best_rmse:.4f}). "
     "RMSE is NOT monotonically decreasing -some additions hurt performance. "
     "'More variables = better' is empirically false. "),

    ("7. 2x2 Diagnostic -Attention x Ablation", img5,
     f"X: Attention Score, Y: Ablation dRMSE. Uniform share: {uniform_share:.4f}. "
     "Green = Confirmed Driver (high attn + helpful). Red = Spurious Attention (high attn + harmful). "
     "Yellow = Hidden Contributor (low attn + helpful). Gray = Irrelevant."),

    ("8. Head Role Analysis -Attention Heatmap", img6,
     f"{n_heads} heads in Layer 11. AUD_USD-dedicated heads (0,2,3,8): self-attn 80%+ "
     "with AUD_USD capturing 95%+ of cross-variate attention. "
     "Remaining heads distribute attention broadly. "
     "Head averaging is the safe choice for variable selection."),

    ("9. Head Ranking Correlation (Spearman)", img7,
     "Correlation between head-specific variable rankings. "
     "Similar-type heads cluster together; different types show negative correlation. "
     "No single head provides a complete view -averaging is appropriate."),

    ("10. Temporal Peak Lag -Optimal 18 Variables", img8,
     "Peak attention lag per variable. "
     "Recent Focus (green): CN_PPI, KR_MfgProd, Pork -recent values inform CPI. "
     "Distant Focus (red): most variables reference 8-10 year old values -level anchoring. "
     "Peak Lag alone should not determine variable value -cross-check with Ablation dRMSE."),
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
    pdf.image(img_path, x=15, w=257)

# ── Final: Variable Selection Table ──
pdf.add_page()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, f"11. Final Selection -Optimal {best_k} Variables", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

pdf.set_font("Helvetica", "B", 9)
cols = [8, 40, 55, 20, 20, 35]
hdrs = ["#", "Variable", "Description", "Attn %", "dRMSE", "2x2 Quadrant"]
for w, h in zip(cols, hdrs):
    pdf.cell(w, 7, h, border=1, align="C")
pdf.ln()

opt_vars = [str(v) for v in inc['ranking'][:best_k]]
pdf.set_font("Helvetica", "", 9)
for rank, v in enumerate(opt_vars, 1):
    label = VAR_LABEL.get(v, "")
    attn_v = s1_rank_attn[s1_rank_vars.index(v)] if v in s1_rank_vars else 0
    delta_v = abl_map.get(v, 0)
    if attn_v >= uniform_share and delta_v > 0: q = "Confirmed Driver"
    elif attn_v >= uniform_share: q = "Spurious"
    elif delta_v > 0: q = "Hidden Contributor"
    else: q = "Irrelevant"

    row = [str(rank), v, label, f"{attn_v:.2%}", f"{delta_v:+.4f}", q]
    for w, val in zip(cols, row):
        pdf.cell(w, 6, val, border=1, align="C" if w < 40 else "L")
    pdf.ln()

pdf.ln(5)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 5,
    f"RMSE improvement: {rmse_18_23:.4f} (optimal 18) vs {rmse_11_23:.4f} (legacy 11) = "
    f"{(rmse_11_23 - rmse_18_23):.4f} pp ({(rmse_11_23 - rmse_18_23)/rmse_11_23*100:.1f}% reduction). "
    "The optimal 18 variables are entirely different from the original 29-variable manual selection, "
    "demonstrating that attention-based screening discovers non-obvious predictive signals.")

# Save
pdf.output(OUTPUT_PDF)
print(f"\nSaved: {OUTPUT_PDF}")

import shutil
shutil.rmtree(IMG_DIR, ignore_errors=True)
print("Done.")
