"""
Tournament Attention + RMSE Variable Selection Pipeline
=======================================================
300개 후보 변수에서 다단계 토너먼트로 최적 ~10개 공변량을 도출.

실행: .venv-bistro/bin/python3 tournament_runner.py [--top-k 10] [--repeats 2] [--resume 0]
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

BISTRO_REPO = "/tmp/bistro-repo"
MODEL_PATH  = f"{BISTRO_REPO}/bistro-finetuned"
sys.path.insert(0, f"{BISTRO_REPO}/src")
sys.path.insert(0, os.path.dirname(__file__))

# Shared constants (keep in sync with bistro_runner_30var.py, ablation_study.py)
CTX  = 120
PDT  = 12
PSZ  = 32
BSZ  = 32
TARGET_COL = "CPI_KR_YoY"
FORECAST_START_DATE = "2023-01-01"
FREQ = "M"

# Tournament constants
MAX_VARS_PER_GROUP = 25   # +1 target = 26 total = max_seq_len 3120
TOP_K_PER_GROUP    = 10
HARMFUL_THRESHOLD  = 0.0  # ΔRMSE <= this → harmful, remove
MAX_ROUNDS         = 5

TOURNAMENT_DIR = os.path.join(os.path.dirname(__file__), "data", "tournament")
TOURNAMENT_PANEL = os.path.join(os.path.dirname(__file__), "data", "macro_panel_tournament_daily.csv")

DEFAULT_ANCHORS = ["JP_Interbank3M", "China_CPI", "US_ConsConf"]


# ============================================================
# Data Structures
# ============================================================

@dataclass
class GroupResult:
    group_id: str
    covariates: List[str]
    attention_scores: Dict[str, float]
    top_k_vars: List[str]
    ablation_delta: Dict[str, float]
    survivors: List[str]
    baseline_rmse: float
    forecast_med: Optional[List[float]] = None

@dataclass
class RoundResult:
    round_num: int
    groups: List[GroupResult]
    all_survivors: List[str]
    scored_variables: Dict[str, float]
    elapsed_seconds: float = 0.0

@dataclass
class TournamentState:
    all_candidates: List[str]
    anchor_vars: List[str]
    rounds: List[RoundResult] = field(default_factory=list)
    final_selection: Optional[List[str]] = None
    config: Dict = field(default_factory=dict)


# ============================================================
# Panel Loading
# ============================================================

_panel_cache = None

def load_tournament_panel():
    """토너먼트 패널 로딩 (캐시). inf 제거 + 극단값 클리핑."""
    global _panel_cache
    if _panel_cache is None:
        print(f"Loading tournament panel: {TOURNAMENT_PANEL}")
        _panel_cache = pd.read_csv(TOURNAMENT_PANEL, index_col=0, parse_dates=True)

        # inf → NaN → 보간
        _panel_cache = _panel_cache.replace([np.inf, -np.inf], np.nan)
        _panel_cache = _panel_cache.ffill().bfill()

        # inf 포함 컬럼 제거
        inf_cols = [c for c in _panel_cache.columns if _panel_cache[c].isnull().any()]
        if inf_cols:
            _panel_cache = _panel_cache.drop(columns=inf_cols)
            print(f"  Dropped {len(inf_cols)} columns with residual NaN/inf: {inf_cols[:10]}")

        print(f"  Shape: {_panel_cache.shape}")
    return _panel_cache


def get_panel_subset(covariates: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """패널에서 target + 지정 공변량만 추출. 없는 변수는 건너뜀."""
    panel = load_tournament_panel()
    available = [c for c in covariates if c in panel.columns]
    missing = [c for c in covariates if c not in panel.columns]
    if missing:
        print(f"  Warning: {len(missing)} vars not in panel: {missing[:5]}...")
    subset = panel[[TARGET_COL] + available].copy()
    return subset, available


# ============================================================
# Group Formation
# ============================================================

def form_groups(
    candidates: List[str],
    anchors: List[str],
    max_per_group: int = MAX_VARS_PER_GROUP,
    seed: int = 42,
) -> List[List[str]]:
    """
    후보 변수를 그룹으로 분할.
    각 그룹에 anchor 변수 포함, 나머지는 랜덤 셔플 후 분배.
    """
    anchor_set = [a for a in anchors if a in candidates]
    non_anchor = [c for c in candidates if c not in anchor_set]

    rng = np.random.default_rng(seed)
    rng.shuffle(non_anchor)

    slots = max_per_group - len(anchor_set)
    groups = []
    for i in range(0, len(non_anchor), slots):
        chunk = non_anchor[i:i + slots]
        groups.append(anchor_set + chunk)

    return groups


# ============================================================
# Per-Group Pipeline
# ============================================================

def run_group_inference(
    covariates: List[str],
    module,
    n_layers: int,
) -> Tuple[pd.DataFrame, Dict, List[str], np.ndarray]:
    """그룹 추론 + attention 캡처."""
    from bistro_runner_30var import run_bistro_inference_daily

    panel, available_covs = get_panel_subset(covariates)

    forecast_df, preds_bl, prep, pred, captured, layers = \
        run_bistro_inference_daily(panel, available_covs, module, n_layers,
                                   capture_attention=True)

    return forecast_df, captured, layers, available_covs


def compute_group_attention(
    captured: Dict,
    layer_names: List[str],
    covariates: List[str],
) -> Tuple[Dict[str, float], float]:
    """attention 랭킹 계산."""
    from bistro_runner_30var import compute_attention_ranking

    variates = [TARGET_COL] + covariates
    n_var = len(variates)
    cov_imp, self_attn, _ = compute_attention_ranking(
        captured, layer_names, variates, n_var
    )
    scores = {v: float(a) for v, a in cov_imp.items()}
    return scores, float(self_attn)


def _extract_rmse(forecast_df) -> float:
    """forecast_df에서 RMSE 추출."""
    if "actual" not in forecast_df.columns:
        return float("nan")
    actual = forecast_df["actual"].values.astype(float)
    preds = forecast_df["bistro_med"].values.astype(float)
    valid = ~np.isnan(actual)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean((preds[valid] - actual[valid]) ** 2)))


def run_group_ablation(
    covariates: List[str],
    top_k_vars: List[str],
    module,
    n_layers: int,
) -> Tuple[Dict[str, float], float]:
    """top-K 변수에 대한 leave-one-out ablation. daily 패널 직접 사용."""
    from bistro_runner_30var import run_bistro_inference_daily

    # Baseline: 전체 공변량
    panel_bl, available_covs = get_panel_subset(covariates)
    forecast_bl, _, _, _, _, _ = run_bistro_inference_daily(
        panel_bl, available_covs, module, n_layers, capture_attention=False
    )
    baseline_rmse = _extract_rmse(forecast_bl)

    # Leave-one-out
    delta = {}
    for var in top_k_vars:
        if var not in available_covs:
            continue
        remaining = [c for c in available_covs if c != var]
        if not remaining:
            delta[var] = 0.0
            continue
        try:
            panel_lo, covs_lo = get_panel_subset(remaining)
            forecast_lo, _, _, _, _, _ = run_bistro_inference_daily(
                panel_lo, covs_lo, module, n_layers, capture_attention=False
            )
            rmse_without = _extract_rmse(forecast_lo)
            delta[var] = rmse_without - baseline_rmse
        except Exception as e:
            print(f"    Ablation failed for {var}: {e}")
            delta[var] = 0.0

    return delta, baseline_rmse


def run_group_pipeline(
    covariates: List[str],
    group_id: str,
    module,
    n_layers: int,
    top_k: int = TOP_K_PER_GROUP,
    run_ablation: bool = True,
) -> GroupResult:
    """단일 그룹 전체 파이프라인."""
    print(f"\n{'─' * 50}")
    print(f"Group {group_id}: {len(covariates)} covariates")
    print(f"{'─' * 50}")

    # Step 1: Inference + attention
    t0 = time.time()
    forecast_df, captured, layers, available_covs = \
        run_group_inference(covariates, module, n_layers)
    t_infer = time.time() - t0
    print(f"  Inference: {t_infer:.0f}s")

    # Step 2: Attention ranking
    attn_scores, self_attn = compute_group_attention(
        captured, layers, available_covs
    )
    sorted_vars = sorted(attn_scores, key=lambda v: -attn_scores[v])
    top_k_vars = sorted_vars[:top_k]

    print(f"  Self-attention: {self_attn:.1%}")
    print(f"  Top {top_k}: {top_k_vars}")

    # Step 3: Ablation (optional)
    ablation_delta = {}
    baseline_rmse = 0.0
    if run_ablation:
        t0 = time.time()
        ablation_delta, baseline_rmse = run_group_ablation(
            available_covs, top_k_vars, module, n_layers
        )
        t_abl = time.time() - t0
        print(f"  Ablation: {t_abl:.0f}s, baseline RMSE: {baseline_rmse:.4f}")
        for v in top_k_vars:
            d = ablation_delta.get(v, 0.0)
            status = "harmful" if d <= HARMFUL_THRESHOLD else "OK"
            print(f"    {v:<25s} ΔRMSE={d:+.4f} [{status}]")

    # Step 4: Survivors
    if run_ablation and ablation_delta:
        survivors = [v for v in top_k_vars
                     if ablation_delta.get(v, 0.0) > HARMFUL_THRESHOLD]
    else:
        # attention-only mode: top K 전부 생존
        survivors = top_k_vars

    print(f"  Survivors: {len(survivors)} — {survivors}")

    # Forecast median for later comparison
    fc_med = None
    if "bistro_med" in forecast_df.columns:
        fc_med = forecast_df["bistro_med"].tolist()

    # 메모리 절약: captured attention 해제
    del captured

    return GroupResult(
        group_id=group_id,
        covariates=available_covs,
        attention_scores=attn_scores,
        top_k_vars=top_k_vars,
        ablation_delta=ablation_delta,
        survivors=survivors,
        baseline_rmse=baseline_rmse,
        forecast_med=fc_med,
    )


# ============================================================
# Cross-Group Aggregation
# ============================================================

def aggregate_scores(
    group_results: List[GroupResult],
    use_ablation: bool = True,
) -> Dict[str, float]:
    """
    그룹 간 점수 집계.
    score = 0.4 * normalized_attention + 0.6 * normalized_delta_rmse
    ablation 없으면 attention만 사용.
    """
    var_scores: Dict[str, List[float]] = {}

    for gr in group_results:
        # Normalize attention within group
        attn_vals = list(gr.attention_scores.values())
        attn_max = max(attn_vals) if attn_vals else 1.0

        for var in gr.attention_scores:
            norm_attn = gr.attention_scores[var] / attn_max if attn_max > 0 else 0

            if use_ablation and gr.ablation_delta:
                delta = gr.ablation_delta.get(var, None)
                if delta is not None:
                    # Normalize delta RMSE within group
                    deltas = [d for d in gr.ablation_delta.values() if d is not None]
                    delta_max = max(abs(d) for d in deltas) if deltas else 1.0
                    norm_delta = delta / delta_max if delta_max > 0 else 0
                    score = 0.4 * norm_attn + 0.6 * max(norm_delta, 0)
                else:
                    score = norm_attn
            else:
                score = norm_attn

            if var not in var_scores:
                var_scores[var] = []
            var_scores[var].append(score)

    # 변수별 최고 점수 채택
    return {v: max(scores) for v, scores in var_scores.items()}


def select_survivors(
    scored_vars: Dict[str, float],
    anchor_vars: List[str],
    group_results: List[GroupResult],
) -> List[str]:
    """
    생존자 선택.
    - 어떤 그룹에서든 survivor로 선정된 변수는 생존
    - Anchor 변수는 자동 생존
    """
    # 모든 그룹의 survivors 합집합
    all_survivors = set()
    for gr in group_results:
        all_survivors.update(gr.survivors)

    # Anchor 자동 포함
    for a in anchor_vars:
        if a in scored_vars:
            all_survivors.add(a)

    # 점수 순 정렬
    sorted_survivors = sorted(all_survivors, key=lambda v: -scored_vars.get(v, 0))

    return sorted_survivors


# ============================================================
# State Persistence
# ============================================================

def save_state(state: TournamentState, path: str):
    """상태 저장 (JSON)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def to_serializable(obj):
        if isinstance(obj, (GroupResult, RoundResult, TournamentState)):
            return asdict(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(path, "w") as f:
        json.dump(asdict(state), f, default=to_serializable, indent=2, ensure_ascii=False)


def save_round_results(round_result: RoundResult, round_dir: str):
    """라운드별 그룹 결과 저장."""
    os.makedirs(round_dir, exist_ok=True)

    for gr in round_result.groups:
        path = os.path.join(round_dir, f"{gr.group_id}.npz")
        np.savez(path,
            group_id=gr.group_id,
            covariates=np.array(gr.covariates),
            attention_scores_vars=np.array(list(gr.attention_scores.keys())),
            attention_scores_vals=np.array(list(gr.attention_scores.values())),
            top_k_vars=np.array(gr.top_k_vars),
            survivors=np.array(gr.survivors),
            baseline_rmse=np.array(gr.baseline_rmse),
        )

    summary = {
        "round": round_result.round_num,
        "n_groups": len(round_result.groups),
        "n_survivors": len(round_result.all_survivors),
        "survivors": round_result.all_survivors,
        "elapsed_seconds": round_result.elapsed_seconds,
        "top_scored": sorted(
            round_result.scored_variables.items(),
            key=lambda x: -x[1]
        )[:20],
    }
    with open(os.path.join(round_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_tournament_summary(state: TournamentState):
    """최종 리더보드 저장."""
    summary_path = os.path.join(TOURNAMENT_DIR, "tournament_summary.json")

    # 전 라운드 통합 — 변수별 최고 점수 + 생존 라운드
    variable_history = {}
    for rr in state.rounds:
        for var, score in rr.scored_variables.items():
            if var not in variable_history:
                variable_history[var] = {"best_score": 0, "rounds_survived": 0}
            variable_history[var]["best_score"] = max(
                variable_history[var]["best_score"], score
            )
            if var in rr.all_survivors:
                variable_history[var]["rounds_survived"] += 1

    leaderboard = sorted(
        [{"variable": v, **h} for v, h in variable_history.items()],
        key=lambda x: -x["best_score"]
    )

    summary = {
        "final_selection": state.final_selection,
        "total_rounds": len(state.rounds),
        "initial_candidates": len(state.all_candidates),
        "leaderboard": leaderboard[:50],
        "config": state.config,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved tournament summary: {summary_path}")


# ============================================================
# Main Tournament Loop
# ============================================================

def main(
    top_k: int = TOP_K_PER_GROUP,
    n_shuffle_repeats: int = 2,
    resume_round: int = 0,
    anchors: Optional[List[str]] = None,
):
    from uni2ts.model.moirai import MoiraiModule

    if anchors is None:
        anchors = DEFAULT_ANCHORS

    os.makedirs(TOURNAMENT_DIR, exist_ok=True)

    # ── 모델 로딩 (한 번만) ────────────────────────────────
    print("Loading BISTRO model...")
    module = MoiraiModule.from_pretrained(MODEL_PATH)
    module.eval()
    n_layers = len(list(module.encoder.layers))
    print(f"  {n_layers} transformer layers.\n")

    # ── 후보 변수 목록 (패널에 있는 것만) ────────────────────
    panel = load_tournament_panel()
    all_candidates = [c for c in panel.columns if c != TARGET_COL]
    print(f"Total candidates: {len(all_candidates)}")
    print(f"Anchors: {anchors}")

    # ── 토너먼트 상태 ────────────────────────────────────────
    state = TournamentState(
        all_candidates=all_candidates,
        anchor_vars=anchors,
        config={
            "max_vars_per_group": MAX_VARS_PER_GROUP,
            "top_k_per_group": top_k,
            "n_shuffle_repeats": n_shuffle_repeats,
            "harmful_threshold": HARMFUL_THRESHOLD,
            "anchors": anchors,
        },
    )

    candidates = all_candidates[:]

    for round_num in range(resume_round + 1, MAX_ROUNDS + 1):
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_num}: {len(candidates)} candidates")
        print(f"{'=' * 60}")

        # 하이브리드: Round 1~2 attention only, 3+ ablation
        use_ablation = round_num >= 3
        mode = "Attention + Ablation" if use_ablation else "Attention only"
        print(f"  Mode: {mode}")

        t_round_start = time.time()
        all_group_results: List[GroupResult] = []

        for shuffle_idx in range(n_shuffle_repeats):
            seed = round_num * 100 + shuffle_idx
            groups = form_groups(candidates, anchors, MAX_VARS_PER_GROUP, seed=seed)
            print(f"\n  Shuffle {shuffle_idx + 1}/{n_shuffle_repeats}: "
                  f"{len(groups)} groups (seed={seed})")

            for gi, group_covs in enumerate(groups):
                gid = f"R{round_num}_G{gi + 1:02d}_S{shuffle_idx + 1}"
                try:
                    result = run_group_pipeline(
                        group_covs, gid, module, n_layers,
                        top_k=top_k,
                        run_ablation=use_ablation,
                    )
                    all_group_results.append(result)
                except Exception as e:
                    print(f"\n  ⚠ Group {gid} FAILED: {e}")
                    print(f"    Skipping this group and continuing...")
                    continue

        # ── 집계 ─────────────────────────────────────────────
        scored = aggregate_scores(all_group_results, use_ablation=use_ablation)
        survivors = select_survivors(scored, anchors, all_group_results)
        elapsed = time.time() - t_round_start

        round_result = RoundResult(
            round_num=round_num,
            groups=all_group_results,
            all_survivors=survivors,
            scored_variables=scored,
            elapsed_seconds=elapsed,
        )
        state.rounds.append(round_result)

        # ── 저장 ─────────────────────────────────────────────
        round_dir = os.path.join(TOURNAMENT_DIR, f"round_{round_num:02d}")
        save_round_results(round_result, round_dir)
        save_state(state, os.path.join(TOURNAMENT_DIR, "tournament_state.json"))

        print(f"\n  Round {round_num} summary:")
        print(f"    Survivors: {len(survivors)} / {len(candidates)}")
        print(f"    Elapsed: {elapsed / 60:.1f} min")
        print(f"    Top 10: {survivors[:10]}")

        # ── 수렴 체크 ────────────────────────────────────────
        if len(state.rounds) >= 2:
            prev_survivors = set(state.rounds[-2].all_survivors)
            curr_survivors = set(survivors)
            if prev_survivors == curr_survivors:
                print(f"\n  ✓ Converged! Survivor set unchanged.")
                break

        if len(survivors) <= MAX_VARS_PER_GROUP:
            print(f"\n  ✓ Survivors fit in single group ({len(survivors)} ≤ {MAX_VARS_PER_GROUP})")
            if round_num < MAX_ROUNDS:
                # 최종 라운드: ablation 포함해서 한 번 더
                print(f"  Running final validation round with ablation...")
                gid_final = f"R{round_num + 1}_FINAL"
                final_result = run_group_pipeline(
                    survivors, gid_final, module, n_layers,
                    top_k=min(top_k, len(survivors)),
                    run_ablation=True,
                )
                final_scored = aggregate_scores([final_result], use_ablation=True)
                final_survivors = select_survivors(
                    final_scored, anchors, [final_result]
                )

                final_round = RoundResult(
                    round_num=round_num + 1,
                    groups=[final_result],
                    all_survivors=final_survivors,
                    scored_variables=final_scored,
                )
                state.rounds.append(final_round)
                survivors = final_survivors

                final_dir = os.path.join(TOURNAMENT_DIR, "final")
                save_round_results(final_round, final_dir)

                print(f"\n  Final survivors: {len(final_survivors)} — {final_survivors}")
            break

        candidates = survivors

    # ── 최종 결과 ─────────────────────────────────────────────
    state.final_selection = survivors
    save_state(state, os.path.join(TOURNAMENT_DIR, "tournament_state.json"))
    save_tournament_summary(state)

    print(f"\n{'=' * 60}")
    print(f"TOURNAMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Rounds: {len(state.rounds)}")
    print(f"  Initial: {len(state.all_candidates)} candidates")
    print(f"  Final: {len(survivors)} selected — {survivors}")
    print(f"  Results: {TOURNAMENT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tournament Variable Selection")
    parser.add_argument("--top-k", type=int, default=TOP_K_PER_GROUP)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--anchors", type=str, default=None,
                        help="Comma-separated anchor variables")
    args = parser.parse_args()

    anchor_list = args.anchors.split(",") if args.anchors else None
    main(
        top_k=args.top_k,
        n_shuffle_repeats=args.repeats,
        resume_round=args.resume,
        anchors=anchor_list,
    )
