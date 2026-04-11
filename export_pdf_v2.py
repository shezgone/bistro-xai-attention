"""
BISTRO-XAI Report v2 - Full-Variable Screening Pipeline Results
================================================================
Stage 0→1 최적 18변수 모델 + 전체 실험 비교 + Head/Layer 분석.

실행: .venv/bin/python3 export_pdf_v2.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
S0_DIR = os.path.join(DATA_DIR, "stage0")
OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "BISTRO_XAI_Report_v2.pdf")
IMG_DIR = os.path.join(os.path.dirname(__file__), "_report_v2_imgs")
os.makedirs(IMG_DIR, exist_ok=True)

# ── Data loading ──
s0 = np.load(f"{S0_DIR}/stage0_ranking.npz", allow_pickle=True)
s1r = np.load(f"{S0_DIR}/stage1_results.npz", allow_pickle=True)
inc = np.load(f"{S0_DIR}/incremental_results.npz", allow_pickle=True)
hd = np.load(f"{S0_DIR}/head_analysis.npz", allow_pickle=True)
s2 = np.load(f"{DATA_DIR}/real_inference_results.npz", allow_pickle=True)
f18 = np.load(f"{DATA_DIR}/forecast_optimal18.npz", allow_pickle=True)
f1 = np.load(f"{DATA_DIR}/forecast_univariate.npz", allow_pickle=True)
s1_old = np.load(f"{DATA_DIR}/stage1_screening.npz", allow_pickle=True)

panel_path = os.path.join(DATA_DIR, "macro_panel_daily.csv")
panel_cpi = pd.read_csv(panel_path, index_col=0, parse_dates=True)["CPI_KR_YoY"]
cpi_monthly = panel_cpi.resample("MS").last().dropna().loc["2021-01":"2025-12"]

# Variable labels
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
# Charts
# ============================================================

# 1. Forecast Comparison (all experiments)
print("1. Forecast comparison...")
fc_dates = pd.to_datetime([d + "-01" for d in s2["forecast_date"]])
actual = s2["forecast_actual"]
valid = ~np.isnan(actual)
cpi_before = cpi_monthly.loc[:"2022-12"]
cpi_fc = cpi_monthly.loc["2023-01":"2023-12"]
cpi_after = cpi_monthly.loc["2024-01":]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=cpi_before.index, y=cpi_before.values, mode="lines",
    line=dict(color="#333", width=2.5), name="Actual CPI"))
fig1.add_trace(go.Scatter(x=[cpi_before.index[-1]]+list(cpi_fc.index),
    y=[cpi_before.values[-1]]+list(cpi_fc.values), mode="lines+markers",
    line=dict(color="#D62728", width=2.5), marker=dict(size=7, symbol="square"), name="Actual (forecast)"))
if len(cpi_after) > 0:
    fig1.add_trace(go.Scatter(x=[cpi_fc.index[-1]]+list(cpi_after.index),
        y=[cpi_fc.values[-1]]+list(cpi_after.values), mode="lines",
        line=dict(color="#333", width=2.5), showlegend=False))

# All experiment lines
experiments = [
    (f18["forecast_med"], "#DC2626", 3, "hexagram", "Optimal 18 (288 pool)"),
    (s2["forecast_med"], "#1A6FD4", 2.5, "circle", "Stage2 11 (29 pool)"),
    (s1_old["forecast_med"], "#8B5CF6", 2, "diamond", "Stage1 29 (all)"),
    (f1["forecast_med"], "#000000", 2, "x", "Univariate (no cov)"),
]
if not all(np.isnan(s2["forecast_ar1"])):
    experiments.append((s2["forecast_ar1"], "#888", 2, "circle", "AR(1) baseline"))

for med, color, width, symbol, name in experiments:
    dash = "dash" if "AR" in name else ("dot" if "Uni" in name else None)
    fig1.add_trace(go.Scatter(x=fc_dates, y=np.array(med, dtype=float), mode="lines+markers",
        line=dict(color=color, width=width, dash=dash), marker=dict(size=7, symbol=symbol), name=name))

fig1.add_vline(x=fc_dates[0].timestamp()*1000, line_dash="dash", line_color="rgba(100,100,100,0.5)")
fig1.update_layout(title="Korean CPI Forecast - Multi-Experiment Comparison",
    yaxis_title="CPI YoY (%)", height=480, **LAYOUT)
img1 = save_fig(fig1, "01_forecast_comparison")

# RMSE table data
rmse_items = []
for med, name in [(f18["forecast_med"], "Optimal 18 (288->18)"),
    (s2["forecast_med"], "Stage2 11 (29->11)"), (s1_old["forecast_med"], "Stage1 29 (all)"),
    (f1["forecast_med"], "Univariate")]:
    m = np.array(med, dtype=float)
    rmse_items.append((name, float(np.sqrt(np.mean((m[valid]-actual[valid])**2)))))
if not all(np.isnan(s2["forecast_ar1"])):
    ar1 = s2["forecast_ar1"]
    rmse_items.append(("AR(1)", float(np.sqrt(np.mean((ar1[valid]-actual[valid])**2)))))

# 2. Stage 0 Attention Ranking
print("2. Stage 0 ranking...")
s0_vars = [str(v) for v in s0['ranking_vars']]
s0_attn = s0['ranking_attn'].astype(float)
n_show = 30
fig2 = go.Figure(go.Bar(x=[vl(v) for v in s0_vars[:n_show]], y=s0_attn[:n_show],
    marker_color=["#DC2626" if i < 25 else "#CCC" for i in range(n_show)]))
fig2.add_hline(y=s0_attn[24], line_dash="dash", line_color="red",
    annotation_text="Top 25 cutoff")
fig2.update_layout(title=f"Stage 0: Full Screening (288 vars, CTX=10) - Top {n_show}",
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
best_k = int(inc['best_k'])
best_rmse = float(inc['best_rmse'])

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=inc_n, y=inc_rmse, mode="lines+markers",
    line=dict(color="#3B82F6", width=2.5), marker=dict(size=9,
    color=["#DC2626" if n == best_k else "#3B82F6" for n in inc_n])))
fig4.add_trace(go.Scatter(x=[best_k], y=[best_rmse], mode="markers+text",
    marker=dict(size=16, color="#DC2626", symbol="star"),
    text=[f"N={best_k}"], textposition="top center", showlegend=False))
fig4.add_hline(y=1.1895, line_dash="dot", line_color="#888", annotation_text="Legacy 11 vars (1.1895)")
fig4.update_layout(title=f"Incremental Addition: Optimal N={best_k} (RMSE {best_rmse:.4f})",
    xaxis=dict(title="# Covariates", dtick=1), yaxis_title="RMSE (pp)", height=420, **LAYOUT)
img4 = save_fig(fig4, "04_incremental")

# 5. 2x2 Diagnostic
print("5. 2x2 diagnostic...")
uniform_share = (1.0 - float(s1r['self_attn'])) / len(s1_rank_vars)
diag_x = [s1_rank_attn[s1_rank_vars.index(v)] if v in s1_rank_vars else 0 for v in s1_abl_vars]
diag_y = [abl_map.get(v, 0) for v in s1_abl_vars]

q_colors = []
q_names = []
for v, ax, ay in zip(s1_abl_vars, diag_x, diag_y):
    if ax >= uniform_share and ay > 0: q_colors.append("#10B981"); q_names.append("Confirmed Driver")
    elif ax >= uniform_share: q_colors.append("#EF4444"); q_names.append("Spurious Attention")
    elif ay > 0: q_colors.append("#F59E0B"); q_names.append("Hidden Contributor")
    else: q_colors.append("#9CA3AF"); q_names.append("Irrelevant")

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
from scipy.stats import spearmanr
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

# 8. Layer Method Comparison
print("8. Layer method comparison...")
methods = [("Last Layer", 1.145, "#10B981"), ("All Avg", 1.161, "#6366F1"),
    ("Rollout", 1.160, "#F59E0B"), ("Var-Weighted", 1.162, "#EF4444"),
    ("Focus-Wtd", 1.169, "#9CA3AF")]
fig8 = go.Figure(go.Bar(x=[m[0] for m in methods], y=[m[1] for m in methods],
    marker_color=[m[2] for m in methods], text=[f"{m[1]:.3f}" for m in methods], textposition="auto"))
fig8.add_hline(y=1.1895, line_dash="dot", line_color="#888", annotation_text="Legacy 11 (1.1895)")
fig8.update_layout(title="Attention Method Comparison (Optimal RMSE)",
    yaxis=dict(title="RMSE", range=[1.13, 1.18]), height=400, **LAYOUT)
img8 = save_fig(fig8, "08_layer_methods", width=900, height=400)

# 9. Temporal - Peak Lag
print("9. Temporal peak lag...")
optimal_18 = [str(v) for v in inc['ranking'][:best_k]]
tp_avg = h_attn.mean(axis=0)
tp_lags = {}
for j, vn in enumerate(h_variates):
    if vn == "CPI_KR_YoY": continue
    if vn not in optimal_18: continue
    ks, ke = j * h_ctx, (j+1) * h_ctx
    q_start = max(0, h_ctx - 12)
    block = tp_avg[q_start:h_ctx, ks:ke].mean(axis=0)
    tp_lags[vn] = h_ctx - 1 - int(np.argmax(block))

lag_sorted = sorted(tp_lags.items(), key=lambda x: x[1])
max_lag_v = max(tp_lags.values()) or 1
lag_colors_v = [f"rgb({min(255,int(lag/max_lag_v*255))},{min(255,int((1-lag/max_lag_v)*200))},80)"
    for _, lag in lag_sorted]
fig9 = go.Figure(go.Bar(x=[vl(v) for v, _ in lag_sorted], y=[l for _, l in lag_sorted],
    marker_color=lag_colors_v, text=[f"{l}M" for _, l in lag_sorted], textposition="outside"))
fig9.update_layout(title="Peak Attention Lag (Optimal 18 vars)",
    xaxis=dict(tickangle=-45, tickfont_size=9), yaxis_title="Peak Lag (months ago)",
    height=420, **LAYOUT)
img9 = save_fig(fig9, "09_peak_lag")

# ============================================================
# PDF Generation
# ============================================================
print("\nGenerating PDF...")

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "BISTRO-XAI Report v2 - Full-Variable Screening Pipeline", align="R",
                  new_x="LMARGIN", new_y="NEXT")
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

# Title
pdf.add_page()
pdf.ln(25)
pdf.set_font("Helvetica", "B", 28)
pdf.cell(0, 15, "BISTRO-XAI v2", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 16)
pdf.cell(0, 10, "Full-Variable Screening & Optimal Covariate Selection", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 8, "Korean CPI (YoY) Forecasting - 288 Candidates -> Optimal 18", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(100, 100, 100)

rmse_18 = rmse_items[0][1]
info = [
    "Model: BISTRO/Moirai 1.0 (311M params, 12 layers, 12 heads, d_model=768)",
    f"Pipeline: Stage 0 (CTX=10, 288 vars) -> Stage 1 (CTX=120, 25 vars) -> Optimal {best_k} vars",
    f"SOTA RMSE: {rmse_18:.4f} pp (vs Legacy 11-var: 1.1895, improvement: {(1.1895-rmse_18):.4f} pp)",
    "Forecast: 2023-01 ~ 2023-12 | Context: ~10.5 years | max_seq_len: 3,120 tokens",
]
for line in info:
    pdf.cell(0, 6, line, align="C", new_x="LMARGIN", new_y="NEXT")

# RMSE Summary page
pdf.add_page()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "RMSE Summary - All Experiments", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

pdf.set_font("Helvetica", "B", 10)
col_w = [80, 30]
pdf.cell(col_w[0], 7, "Experiment", border=1, align="C")
pdf.cell(col_w[1], 7, "RMSE", border=1, align="C")
pdf.ln()

pdf.set_font("Helvetica", "", 10)
for name, rmse in sorted(rmse_items, key=lambda x: x[1]):
    pdf.cell(col_w[0], 6, name, border=1)
    pdf.cell(col_w[1], 6, f"{rmse:.4f}", border=1, align="C")
    pdf.ln()

pdf.ln(5)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 5,
    "Note: RMSE values may vary by ~0.005 pp between runs due to stochastic sampling (num_samples=100). "
    "All comparisons use the same actual values for consistency.")

# Chart pages
sections = [
    ("1. Forecast Comparison - All Experiments", img1,
     f"Multi-experiment overlay. SOTA Optimal 18 (RMSE {rmse_18:.4f}) vs Legacy 11 (1.1895) vs "
     f"Stage1 29 (1.1805) vs Univariate ({rmse_items[3][1]:.4f}) vs AR(1). "
     "Covariates clearly improve forecasting - univariate BISTRO equals AR(1). "
     "Wider candidate pool (288) yields better variable combinations than manual 29."),

    ("2. Stage 0 - Full Screening (288 vars, CTX=10)", img2,
     f"All 288 FRED variables in a single inference pass with shortened context (CTX=10, ~10 months). "
     f"Token count: 288 x 10 = 2,880 < max_seq_len 3,120. "
     f"Purpose: relative attention ranking only (RMSE not meaningful with short context). "
     f"AUD_USD dominates at 10.8%, 4.9x higher than 2nd place (BRL_USD 2.2%). "
     f"Red bars = top 25 selected for Stage 1."),

    ("3. Stage 1 - Re-inference + Ablation (25 vars, CTX=120)", img3,
     f"Full context re-inference. Green bars = helpful (dRMSE > 0), red = harmful. "
     f"Baseline RMSE: {float(s1r['baseline_rmse']):.4f}. "
     f"Harmful variables ({', '.join(s1_harmful)}): passed Stage 0 but failed Stage 1 validation. "
     "Short-context pattern similarity != long-context predictive value. "
     "Two-stage verification filters false positives automatically."),

    ("4. Incremental Addition - Optimal Variable Count", img4,
     f"Variables added one by one in attention rank order. "
     f"Optimal at N={best_k} (RMSE {best_rmse:.4f}). "
     "RMSE is NOT monotonically decreasing - some additions hurt performance. "
     "'More variables = better' is empirically false. "
     "Greedy approach (attention order); exhaustive search infeasible (C(23,18) = 33,649)."),

    ("5. 2x2 Diagnostic - Confirmed Drivers", img5,
     f"X: Attention Score, Y: Ablation dRMSE. Uniform share threshold: {uniform_share:.4f}. "
     "Confirmed Driver (green): high attn + helpful. Spurious Attention (red): high attn + not helpful. "
     "Hidden Contributor (yellow): low attn + helpful. Irrelevant (gray): low attn + not helpful. "
     f"Irrelevant vars ({', '.join(s1_harmful)}): Stage 0 false positives caught by Stage 1."),

    ("6. Head Role Analysis - Attention Heatmap", img6,
     f"{n_heads} heads in Layer 11. 4 heads (0,2,3,8) are AUD_USD-dedicated (self-attn 80%+ "
     "with AUD_USD capturing 95%+ of cross-variate attention). "
     f"Remaining 8 heads distribute attention broadly across multiple variables. "
     "This explains AUD_USD's dominant overall attention - 4/12 heads are specialized for it. "
     "Head averaging remains the safe choice for variable selection (individual heads are biased)."),

    ("7. Head Ranking Correlation (Spearman)", img7,
     "Correlation between head-specific variable rankings. "
     "Similar-type heads show high correlation (H1-H11: 0.79), different types show low or negative "
     "(H6-H11: -0.64). Confirms functional differentiation across heads. "
     "No single head provides a complete view - averaging across heads is appropriate."),

    ("8. Layer Method Comparison", img8,
     "5 attention aggregation methods compared via incremental RMSE. "
     "Last Layer (~1.145) consistently best. All Layers Average (~1.161) and Attention Rollout (~1.160) "
     "worse due to harmful variable noise from mid-layers (5-8). "
     "Variance-Weighted (~1.162) distorted by Layer 0's extreme AUD_USD concentration. "
     "Focus-Weighted (~1.169) nearly identical ranking to Last Layer. "
     "Conclusion: Last Layer attention is optimal for variable selection in this model."),

    ("9. Temporal Peak Lag - Optimal 18 Variables", img9,
     "Peak attention lag per variable (months before forecast). "
     "Recent Focus (green, <12M): CN_PPI, KR_MfgProd, Pork - recent values directly inform CPI. "
     "Distant Focus (red, >60M): most variables reference 8-10 year old values. "
     "Two possible interpretations: (1) level anchoring - model uses historical reference points, "
     "or (2) artifact from Instance Normalization neutralizing recent level shifts. "
     "Peak Lag alone should not determine variable value - cross-check with Ablation dRMSE."),
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

# Final selection table
pdf.add_page()
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, f"10. Final Selection - Optimal {best_k} Variables", new_x="LMARGIN", new_y="NEXT")
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
    ax = attn_v
    if ax >= uniform_share and delta_v > 0: q = "Confirmed Driver"
    elif ax >= uniform_share: q = "Spurious"
    elif delta_v > 0: q = "Hidden Contributor"
    else: q = "Irrelevant"

    row = [str(rank), v, label, f"{attn_v:.2%}", f"{delta_v:+.4f}", q]
    for w, val in zip(cols, row):
        pdf.cell(w, 6, val, border=1, align="C" if w < 40 else "L")
    pdf.ln()

pdf.output(OUTPUT_PDF)
print(f"\nSaved: {OUTPUT_PDF}")

import shutil
shutil.rmtree(IMG_DIR, ignore_errors=True)
print("Done.")
