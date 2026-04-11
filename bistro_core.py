"""
BISTRO-XAI Core — Domain Classes
=================================
기존 bistro_attention_extractor.py 에서 matplotlib/seaborn 의존성을 제거하고
Streamlit 앱에서 재활용할 수 있도록 정리한 도메인 계층.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Variable Presets (Korean CPI XAI pipeline)
# ============================================================

PRESETS: Dict[str, List[str]] = {
    "Korean CPI 6-var": [
        "CPI", "Oil_Dubai", "USD_KRW", "Import_Price", "PPI", "Agri_Price"
    ],
    "Korean CPI 10-var": [
        "CPI", "Oil_Dubai", "USD_KRW", "Import_Price", "PPI", "Agri_Price",
        "GDP", "Unemployment", "CSI", "Retail_Sales"
    ],
    "Full 23-var": [
        # Tier 1: 직접 가격 전달
        "CPI", "Oil_Dubai", "USD_KRW", "Import_Price", "PPI", "Agri_Price", "LNG",
        # Tier 2: 수요 측
        "GDP", "Unemployment", "CSI", "Retail_Sales", "HouseDebt", "Wage",
        # Tier 3: 통화/금융
        "Base_Rate", "M2", "CD91", "Gov_Bond3Y", "Inflation_Exp",
        # Tier 4: 글로벌
        "BDI", "Supply_Chain", "China_PPI", "US_CPI", "KOSPI",
    ],
    "Custom": [],
}

TIER_LABELS: Dict[str, str] = {
    # Tier 1: 직접 가격 전달 (환율, 원자재, 물가지수)
    "CPI_KR_YoY": "T1",
    "Oil_WTI": "T1", "Oil_Brent": "T1", "NatGas_HH": "T1",
    "USD_KRW": "T1", "CNY_USD": "T1", "JPY_USD": "T1", "DXY_Broad": "T1",
    "Copper": "T1", "Wheat": "T1", "Corn": "T1", "Gold": "T1",
    # (legacy names)
    "CPI": "T1", "Oil_Dubai": "T1", "Import_Price": "T1", "PPI": "T1",
    "Agri_Price": "T1", "LNG": "T1",
    # Tier 2: 수요 측 / 실물경제
    "KR_Exports": "T2", "KR_Imports": "T2", "KR_Unemp": "T2",
    "US_Unemp": "T2", "US_ConsConf": "T2",
    # (legacy names)
    "GDP": "T2", "Unemployment": "T2", "CSI": "T2",
    "Retail_Sales": "T2", "HouseDebt": "T2", "Wage": "T2",
    # Tier 3: 통화/금융
    "Rate_KR": "T3", "Rate_ECB": "T3", "FedFunds": "T3",
    "KR_Interbank3M": "T3", "KR_LongRate": "T3", "JP_Interbank3M": "T3",
    "US_M2": "T3", "US_YieldSpread": "T3", "VIX": "T3",
    # (legacy names)
    "Base_Rate": "T3", "M2": "T3", "CD91": "T3",
    "Gov_Bond3Y": "T3", "Inflation_Exp": "T3",
    # Tier 4: 글로벌 물가/교역
    "CPI_US_YoY": "T4", "CPI_XM_YoY": "T4", "China_CPI": "T4",
    "US_CoreCPI_idx": "T4", "US_PPI": "T4",
    # (legacy names)
    "BDI": "T4", "Supply_Chain": "T4", "China_PPI": "T4",
    "US_CPI": "T4", "KOSPI": "T4",
}

# 변수별 원본 데이터 주기 (daily / monthly)
VARIABLE_FREQ: Dict[str, str] = {
    # Daily: 환율, 원자재 가격, 금융시장 지표
    "USD_KRW": "daily", "JPY_USD": "daily", "CNY_USD": "daily", "DXY_Broad": "daily",
    "Oil_Brent": "daily", "NatGas_HH": "daily", "Gold": "daily",
    "US_YieldSpread": "daily", "VIX": "daily",
    "Oil_WTI": "daily",  # BIS 원본도 daily
    # Monthly: 거시경제 지표
    "CPI_KR_YoY": "monthly", "Rate_KR": "monthly", "Rate_ECB": "monthly",
    "CPI_US_YoY": "monthly", "CPI_XM_YoY": "monthly",
    "Copper": "monthly", "Wheat": "monthly", "Corn": "monthly",
    "FedFunds": "monthly", "US_CPI_idx": "monthly", "US_CoreCPI_idx": "monthly",
    "US_PPI": "monthly", "US_Unemp": "monthly", "US_ConsConf": "monthly",
    "US_M2": "monthly", "GSCPI": "monthly",
    "KR_Interbank3M": "monthly", "KR_Unemp": "monthly", "KR_LongRate": "monthly",
    "JP_Interbank3M": "monthly", "China_CPI": "monthly",
    "KR_Imports": "monthly", "KR_Exports": "monthly",
}


# ============================================================
# BISTROConfig
# ============================================================

class BISTROConfig:
    """
    BISTRO 모델 실행 설정.

    Parameters
    ----------
    variates   : 변수 이름 리스트 (순서 = 토큰 순서)
    target_idx : 예측 타겟 인덱스 (default 0)
    ctx_patches: Context 패치 수 (default 228 ≈ 20년)
    pdt_patches: Prediction 패치 수 (default 12 ≈ 12개월)
    patch_size : 패치 크기 (일 수, BISTRO 고정값 32)
    """

    def __init__(
        self,
        variates: List[str],
        target_idx: int = 0,
        ctx_patches: int = 228,
        pdt_patches: int = 12,
        patch_size: int = 32,
    ):
        self.variates = variates
        self.target_idx = target_idx
        self.ctx_patches = ctx_patches
        self.pdt_patches = pdt_patches
        self.patch_size = patch_size

    @property
    def n_variates(self) -> int:
        return len(self.variates)

    @property
    def n_tokens(self) -> int:
        return self.n_variates * self.ctx_patches

    @property
    def target_name(self) -> str:
        return self.variates[self.target_idx]

    def variate_slice(self, idx: int) -> Tuple[int, int]:
        return (idx * self.ctx_patches, (idx + 1) * self.ctx_patches)

    def variate_slice_by_name(self, name: str) -> Tuple[int, int]:
        return self.variate_slice(self.variates.index(name))

    def ctx_years(self) -> float:
        return (self.ctx_patches * self.patch_size) / 365

    def pdt_months(self) -> int:
        return round((self.pdt_patches * self.patch_size) / 30)


# ============================================================
# AttentionHookManager
# ============================================================

class AttentionHookManager:
    """
    PyTorch forward hook으로 attention weights를 캡처.
    실제 BISTRO 모델 연동 시 사용; 합성 데모에서는 데이터를 직접 주입.
    """

    def __init__(self):
        self.attention_maps: Dict[str, np.ndarray] = {}
        self._hooks = []

    def _make_hook(self, layer_name: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
            elif hasattr(output, 'attentions'):
                attn_weights = output.attentions
            else:
                attn_weights = output
            if hasattr(attn_weights, 'detach'):
                self.attention_maps[layer_name] = attn_weights.detach().cpu().numpy()
            else:
                self.attention_maps[layer_name] = np.array(attn_weights)
        return hook_fn

    def auto_find_attention_modules(self, model) -> List[str]:
        found = []
        for name, module in model.named_modules():
            mtype = type(module).__name__.lower()
            if any(kw in mtype for kw in ['attention', 'self_attn', 'mha']):
                if not any(kw in mtype for kw in ['norm', 'dropout', 'linear']):
                    found.append(name)
        return found

    def register_hooks(self, model, module_names: Optional[List[str]] = None):
        self.clear()
        if module_names is None:
            module_names = self.auto_find_attention_modules(model)
        for name in module_names:
            parts = name.split('.')
            mod = model
            try:
                for p in parts:
                    mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
                hook = mod.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
            except Exception:
                pass

    def register_hooks_by_pattern(self, model, pattern: str = "self_attn"):
        self.clear()
        for name, module in model.named_modules():
            if pattern in name:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def clear(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.attention_maps.clear()

    def get_layer_names(self) -> List[str]:
        """레이어 이름을 숫자 순서로 정렬하여 반환 (예: layers.9 < layers.10 < layers.11)."""
        import re
        def _numeric_key(name: str):
            return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', name)]
        return sorted(self.attention_maps.keys(), key=_numeric_key)


# ============================================================
# AttentionAnalyzer
# ============================================================

class AttentionAnalyzer:
    """
    캡처된 attention maps에서 cross-variate, temporal, layer 분석 수행.
    """

    def __init__(self, config: BISTROConfig, hook_manager: AttentionHookManager):
        self.config = config
        self.hooks = hook_manager

    def _avg_attention(self, layer_name: Optional[str] = None) -> np.ndarray:
        if layer_name is None:
            layer_name = self.hooks.get_layer_names()[-1]
        attn = self.hooks.attention_maps[layer_name]
        if attn.ndim == 4:
            attn = attn[0].mean(axis=0)   # [batch, heads, T, T] → [T, T]
        elif attn.ndim == 3:
            attn = attn.mean(axis=0)       # [heads, T, T] → [T, T]
        return attn

    def cross_variate_matrix(self, layer_name: Optional[str] = None) -> pd.DataFrame:
        """
        N×N cross-variate attention fraction.
        각 셀 = query 변수 i의 전체 attention 중 key 변수 j로 향하는 비율.
        각 행의 합 ≈ 1.0 (확률 해석 가능).

        계산: frac[i,j] = sum(block_ij) / n_tokens_i
        """
        attn = self._avg_attention(layer_name)
        cfg = self.config
        n = cfg.n_variates
        fracs = np.zeros((n, n))
        for i in range(n):
            qs, qe = cfg.variate_slice(i)
            qe = min(qe, attn.shape[0])
            n_i = max(qe - qs, 1)
            for j in range(n):
                ks, ke = cfg.variate_slice(j)
                ke = min(ke, attn.shape[1])
                block = attn[qs:qe, ks:ke]
                fracs[i, j] = block.sum() / n_i if block.size > 0 else 0.0
        return pd.DataFrame(fracs, index=cfg.variates, columns=cfg.variates)

    def target_importance(self, layer_name: Optional[str] = None) -> pd.Series:
        """타겟 변수의 각 변수에 대한 attention 비율 (cross-variate matrix의 타겟 행)."""
        cross = self.cross_variate_matrix(layer_name)
        return cross.iloc[self.config.target_idx]

    def temporal_attention(
        self,
        query_var: str,
        key_var: str,
        layer_name: Optional[str] = None,
        forecast_patches: int = 12,
    ) -> np.ndarray:
        """
        query_var의 마지막 forecast_patches 쿼리 토큰이
        key_var의 각 과거 패치에 주는 attention 평균.
        """
        attn = self._avg_attention(layer_name)
        cfg = self.config
        qs, qe = cfg.variate_slice_by_name(query_var)
        ks, ke = cfg.variate_slice_by_name(key_var)
        qe = min(qe, attn.shape[0])
        ke = min(ke, attn.shape[1])
        q_fc = max(qs, qe - forecast_patches)
        block = attn[q_fc:qe, ks:ke]
        # 각 forecast 쿼리 토큰의 key 토큰별 평균 attention → 과거 위치별 집중도
        return block.mean(axis=0)

    def layer_comparison(self, query_var: str, key_var: str) -> pd.DataFrame:
        """레이어별 query_var→key_var cross-variate attention 값."""
        rows = []
        for ln in self.hooks.get_layer_names():
            c = self.cross_variate_matrix(ln)
            rows.append({'layer': ln, 'attention': c.loc[query_var, key_var]})
        return pd.DataFrame(rows)

    def all_layers_avg(self) -> pd.DataFrame:
        """전체 레이어 평균 cross-variate matrix."""
        mats = [
            self.cross_variate_matrix(ln).values
            for ln in self.hooks.get_layer_names()
        ]
        return pd.DataFrame(
            np.mean(mats, axis=0),
            index=self.config.variates,
            columns=self.config.variates,
        )


# ============================================================
# Synthetic Attention Generator
# ============================================================

def create_synthetic_attention(
    config: BISTROConfig,
    n_layers: int = 6,
    seed: int = 42,
) -> AttentionHookManager:
    """
    실제 BISTRO 모델 없이 현실적인 attention 패턴을 합성.

    특징:
    - 대각선 블록(자기 참조) 강함
    - 변수 0→1 cross-attention 약간 추가 (CPI→Oil 상관)
    - 12개월 전 패치에 계절성 spike
    """
    hook_mgr = AttentionHookManager()
    n_tok = config.n_tokens
    rng = np.random.default_rng(seed)

    for li in range(n_layers):
        attn = rng.dirichlet(np.ones(n_tok), size=n_tok) * 0.3

        # 자기 참조 강화
        for v in range(config.n_variates):
            s, e = config.variate_slice(v)
            attn[s:e, s:e] += rng.uniform(0.3, 0.6, (e - s, e - s))

        # 변수 0→1 교차 attention (CPI→Oil 유형)
        if config.n_variates > 1:
            ts, te = config.variate_slice(0)
            os_, oe = config.variate_slice(1)
            attn[ts:te, os_:oe] += rng.uniform(0.15, 0.35, (te - ts, oe - os_))

        # 12개월 전 spike (기저효과)
        for v in range(config.n_variates):
            s, e = config.variate_slice(v)
            lag = int(365 / config.patch_size)
            if e - lag > s:
                attn[e - 12:e, e - lag: e - lag + 5] += 0.25

        # Row-normalize
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-9)

        # 8 head 시뮬레이션
        nh = 8
        mh = np.stack([
            attn + rng.normal(0, 0.02, attn.shape)
            for _ in range(nh)
        ])
        mh = np.clip(mh, 0, None)
        mh = mh / (mh.sum(axis=-1, keepdims=True) + 1e-9)

        hook_mgr.attention_maps[f"encoder.layers.{li}.self_attn"] = mh[np.newaxis, ...]

    return hook_mgr


# ============================================================
# Real Inference Results Loader
# ============================================================

REAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "real_inference_results.npz")


def real_results_available() -> bool:
    return os.path.exists(REAL_RESULTS_PATH)


def load_real_results():
    """
    bistro_runner.py 가 저장한 실제 추론 결과를 로딩.

    Returns
    -------
    config   : BISTROConfig
    hooks    : AttentionHookManager  (실제 attention 포함)
    analyzer : AttentionAnalyzer
    forecast : dict  (date, med, ci_lo, ci_hi, ar1, actual, history_date, history_cpi)
    """
    data = np.load(REAL_RESULTS_PATH, allow_pickle=True)

    variates    = [str(v) for v in data["variates"]]
    n_variates  = int(data["n_variates"])
    attn_arrays = data["attn_arrays"]   # (n_layers, q_len, kv_len)
    # ctx_patches: npz에 저장된 값 우선, 없으면 토큰 수에서 역산
    if "ctx_patches" in data:
        ctx_patches = int(data["ctx_patches"])
    else:
        ctx_patches = int(attn_arrays.shape[-1] // n_variates)
    layer_names = [str(s) for s in data["layer_names"]]

    cfg = BISTROConfig(
        variates=variates,
        target_idx=0,
        ctx_patches=ctx_patches,
        pdt_patches=12,
        patch_size=32,
    )

    hook_mgr = AttentionHookManager()
    for i, name in enumerate(layer_names):
        # 저장 형식: (q_len, kv_len) → app 이 기대하는 (1, 1, q, k) 형태로 변환
        hook_mgr.attention_maps[name] = attn_arrays[i][np.newaxis, np.newaxis, ...]

    analyzer = AttentionAnalyzer(cfg, hook_mgr)

    # CF keys — backward-compatible (None if not present)
    def _safe(key):
        return data[key] if key in data else None

    forecast = {
        "date":   [str(d) for d in data["forecast_date"]],
        "med":    data["forecast_med"],
        "ci_lo":  data["forecast_ci_lo"],
        "ci_hi":  data["forecast_ci_hi"],
        "ar1":    data["forecast_ar1"],
        "actual": data["forecast_actual"],
        "history_date": [str(d) for d in data["history_date"]],
        "history_cpi":  data["history_cpi"],
        # counterfactual (None if npz was generated without CF)
        "cf_variates":    [str(v) for v in _safe("cf_variates")] if _safe("cf_variates") is not None else None,
        "cf_impacts":     _safe("cf_impacts"),
        "cf_sigmas":      _safe("cf_sigmas"),
        "cf_preds_plus":  _safe("cf_preds_plus"),
        "cf_preds_minus": _safe("cf_preds_minus"),
    }

    return cfg, hook_mgr, analyzer, forecast


# ============================================================
# Stage 1 Screening Results Loader
# ============================================================

STAGE1_PATH = os.path.join(os.path.dirname(__file__), "data", "stage1_screening.npz")


def stage1_available() -> bool:
    return os.path.exists(STAGE1_PATH)


def load_stage1_screening():
    """
    Stage 1 (전체 변수 스크리닝) 결과 로딩.

    Returns
    -------
    s1_cfg      : BISTROConfig  (전체 변수 기준)
    s1_hooks    : AttentionHookManager
    s1_analyzer : AttentionAnalyzer
    s1_meta     : dict  (ranking_vars, ranking_attn, self_attn, uniform_share)
    """
    data = np.load(STAGE1_PATH, allow_pickle=True)

    variates    = [str(v) for v in data["variates"]]
    n_variates  = int(data["n_variates"])
    attn_arrays = data["attn_arrays"]
    if "ctx_patches" in data:
        ctx_patches = int(data["ctx_patches"])
    else:
        ctx_patches = int(attn_arrays.shape[-1] // n_variates)
    layer_names = [str(s) for s in data["layer_names"]]

    cfg = BISTROConfig(
        variates=variates, target_idx=0,
        ctx_patches=ctx_patches, pdt_patches=12, patch_size=32,
    )

    hook_mgr = AttentionHookManager()
    for i, name in enumerate(layer_names):
        hook_mgr.attention_maps[name] = attn_arrays[i][np.newaxis, np.newaxis, ...]

    analyzer = AttentionAnalyzer(cfg, hook_mgr)

    def _safe(key):
        return data[key] if key in data else None

    meta = {
        "ranking_vars":  [str(v) for v in data["s1_ranking_vars"]] if "s1_ranking_vars" in data else None,
        "ranking_attn":  data["s1_ranking_attn"] if "s1_ranking_attn" in data else None,
        "self_attn":     float(data["s1_self_attn"]) if "s1_self_attn" in data else None,
        "uniform_share": float(data["s1_uniform_share"]) if "s1_uniform_share" in data else None,
        "s1_n_total":    int(data["n_variates"]) - 1 if "n_variates" in data else None,
        "forecast_date": [str(d) for d in data["forecast_date"]] if "forecast_date" in data else None,
        "forecast_med":  data["forecast_med"] if "forecast_med" in data else None,
    }

    return cfg, hook_mgr, analyzer, meta


# ============================================================
# Ablation Results Loader
# ============================================================

ABLATION_PATH = os.path.join(os.path.dirname(__file__), "data", "ablation_results.npz")


def ablation_available() -> bool:
    return os.path.exists(ABLATION_PATH)


def load_ablation_results() -> dict:
    """ablation_study.py 결과 로딩."""
    data = np.load(ABLATION_PATH, allow_pickle=True)
    return {
        "baseline_rmse":    float(data["baseline_rmse"]),
        "baseline_mae":     float(data["baseline_mae"]),
        "attn_ranking":     [str(v) for v in data["attn_ranking"]],
        "attn_values":      data["attn_values"].astype(float),
        "abl_vars":         [str(v) for v in data["abl_vars"]],
        "abl_rmse":         data["abl_rmse"].astype(float),
        "abl_mae":          data["abl_mae"].astype(float),
        "abl_delta_rmse":   data["abl_delta_rmse"].astype(float),
        "inc_labels":       [str(v) for v in data["inc_labels"]],
        "inc_n_vars":       data["inc_n_vars"].astype(int),
        "inc_rmse":         data["inc_rmse"].astype(float),
        "inc_mae":          data["inc_mae"].astype(float),
    }
