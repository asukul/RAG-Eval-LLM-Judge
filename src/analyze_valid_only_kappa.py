"""Recompute pairwise κ on valid-only intersection.

For each judge pair, compute quadratic-weighted Cohen's κ using ONLY rows
where both judges have non-None scores. Compare with the published κ matrix
(which treats None as 0).

Predicts: Gemini 2.5 Pro and Gemini 3.1 Pro Preview pair-κ values will RISE
because the None-as-0 effect inflated their apparent disagreement with peers.

Usage:
    py -3 papers/P4_llm_as_judge/analyze_valid_only_kappa.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "backend" / "data" / "eval"
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_JSON = ROOT / "judge_valid_only_kappa.json"

JUDGE_ORDER = [
    "Claude Opus 4.7 (OpenRouter)",
    "Claude Sonnet 4.6 (OpenRouter)",
    "GPT-5.5 (reasoning=low)",
    "GPT-4o (chat)",
    "Gemini 3.1 Pro Preview (OpenRouter)",
    "Gemini 2.5 Pro (OpenRouter)",
    "DeepSeek V4 Pro (OpenRouter)",
    "Qwen 3.6 Plus (OpenRouter)",
    "Gemma 4 26B (OpenRouter)",
]
SHORT = {
    "Claude Opus 4.7 (OpenRouter)": "Opus 4.7",
    "Claude Sonnet 4.6 (OpenRouter)": "Sonnet 4.6",
    "GPT-5.5 (reasoning=low)": "GPT-5.5 low",
    "GPT-4o (chat)": "GPT-4o",
    "Gemini 3.1 Pro Preview (OpenRouter)": "Gem 3.1 Prev",
    "Gemini 2.5 Pro (OpenRouter)": "Gem 2.5 Pro",
    "DeepSeek V4 Pro (OpenRouter)": "DSV4 Pro",
    "Qwen 3.6 Plus (OpenRouter)": "Qwen 3.6+",
    "Gemma 4 26B (OpenRouter)": "Gemma 4 26B",
}

PER_JUDGE_FILES = {
    "Claude Opus 4.7 (OpenRouter)": "results_dspace_fulltext_vertex_Claude_Opus_4.7_OpenRouter__20260425_034626.json",
    "Claude Sonnet 4.6 (OpenRouter)": "results_dspace_fulltext_vertex_Claude_Sonnet_4.6_OpenRouter__20260425_034626.json",
    "GPT-5.5 (reasoning=low)": "results_dspace_fulltext_vertex_GPT-5.5_reasoning_low__20260425_034626.json",
    "GPT-4o (chat)": "results_dspace_fulltext_vertex_GPT-4o_chat__20260425_034626.json",
    "Gemini 3.1 Pro Preview (OpenRouter)": "results_dspace_fulltext_vertex_Gemini_3.1_Pro_Preview_OpenRouter__20260425_034626.json",
    "Gemini 2.5 Pro (OpenRouter)": "results_dspace_fulltext_vertex_Gemini_2.5_Pro_OpenRouter__20260425_034626.json",
    "DeepSeek V4 Pro (OpenRouter)": "results_dspace_fulltext_vertex_DeepSeek_V4_Pro_OpenRouter__20260425_034626.json",
    "Qwen 3.6 Plus (OpenRouter)": "results_dspace_fulltext_vertex_Qwen_3.6_Plus_OpenRouter__20260425_102101.json",
    "Gemma 4 26B (OpenRouter)": "results_dspace_fulltext_vertex_Gemma_4_26B_OpenRouter__20260425_102101.json",
}
MERGED_JSON = "results_dspace_fulltext_vertex_multijudge_9judge_20260425.json"


def load_scores(label: str) -> np.ndarray:
    """Load 570-element score vector with None as -1 sentinel."""
    path = EVAL_DIR / PER_JUDGE_FILES[label]
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for q in data["queries"]:
        for r in q.get("retrieved", []):
            s = r.get("judge_score")
            out.append(-1 if s is None else int(s))
    return np.array(out, dtype=int)


def cohen_kappa_quadratic(a: np.ndarray, b: np.ndarray) -> float | None:
    """Quadratic-weighted Cohen's kappa. Both arrays same length, integer scores 0..N.

    Returns None if either array has zero variance OR if there are fewer than 30
    valid pairs (statistical floor for ordinal κ).
    """
    if len(a) != len(b) or len(a) == 0:
        return None
    n = len(a)
    if n < 30:
        return None

    n_classes = max(int(a.max()), int(b.max())) + 1
    if n_classes < 2:
        return None

    # Confusion matrix
    O = np.zeros((n_classes, n_classes), dtype=float)
    for x, y in zip(a, b):
        O[x, y] += 1
    O /= n

    # Marginals
    p_a = O.sum(axis=1)
    p_b = O.sum(axis=0)
    E = np.outer(p_a, p_b)

    # Quadratic weights
    indices = np.arange(n_classes)
    W = ((indices[:, None] - indices[None, :]) ** 2) / ((n_classes - 1) ** 2)

    obs_disagree = (W * O).sum()
    exp_disagree = (W * E).sum()
    if exp_disagree == 0:
        return None
    return float(1 - obs_disagree / exp_disagree)


def main() -> int:
    print("Loading per-judge scores...")
    scores = {l: load_scores(l) for l in JUDGE_ORDER}

    n_judges = len(JUDGE_ORDER)
    valid_kappa = np.zeros((n_judges, n_judges))
    valid_n = np.zeros((n_judges, n_judges), dtype=int)

    for i, li in enumerate(JUDGE_ORDER):
        for j, lj in enumerate(JUDGE_ORDER):
            if i == j:
                valid_kappa[i, j] = 1.0
                valid_n[i, j] = int((scores[li] >= 0).sum())
            else:
                a = scores[li]
                b = scores[lj]
                mask = (a >= 0) & (b >= 0)
                a_v = a[mask]
                b_v = b[mask]
                k = cohen_kappa_quadratic(a_v, b_v)
                valid_kappa[i, j] = k if k is not None else float("nan")
                valid_n[i, j] = int(mask.sum())

    # Load published κ for comparison
    merged = json.loads((EVAL_DIR / MERGED_JSON).read_text(encoding="utf-8"))
    pub_kappa_dict = merged["kappa_matrix"]
    pub_kappa = np.zeros((n_judges, n_judges))
    for i, li in enumerate(JUDGE_ORDER):
        for j, lj in enumerate(JUDGE_ORDER):
            pub_kappa[i, j] = float(pub_kappa_dict[li][lj])

    # Per-judge: how many pairs include this judge with reduced n_valid?
    print("\nPer-pair valid-row counts (n where both judges scored):\n")
    print(f"{'Pair':<32s}  {'n_valid':>8s}  {'κ_valid':>8s}  {'κ_None=0':>10s}  {'Δκ':>7s}")
    deltas = []
    for i, li in enumerate(JUDGE_ORDER):
        for j, lj in enumerate(JUDGE_ORDER):
            if i < j:
                k_v = valid_kappa[i, j]
                k_p = pub_kappa[i, j]
                n_v = valid_n[i, j]
                delta = k_v - k_p
                deltas.append((SHORT[li] + " / " + SHORT[lj], n_v, k_v, k_p, delta))
    # Sort by absolute delta
    deltas.sort(key=lambda x: abs(x[4]), reverse=True)
    for label, n, kv, kp, d in deltas:
        print(f"{label:<32s}  {n:>8d}  {kv:>8.4f}  {kp:>10.4f}  {d:>+7.4f}")

    # Summary stats: which pairs shifted UP, which DOWN
    deltas_arr = np.array([d[4] for d in deltas])
    pairs_up = int((deltas_arr > 0).sum())
    pairs_down = int((deltas_arr < 0).sum())
    print(f"\nPairs where κ rose under valid-only: {pairs_up}/{len(deltas)}")
    print(f"Pairs where κ fell under valid-only: {pairs_down}/{len(deltas)}")
    print(f"Mean |Δκ|: {np.mean(np.abs(deltas_arr)):.4f}")
    print(f"Max  |Δκ|: {np.max(np.abs(deltas_arr)):.4f}")

    # Per-judge mean Δκ (positive = judge κ-with-peers improved under valid-only)
    print("\nPer-judge mean Δκ (positive = peers' agreement with this judge improved with valid-only):")
    judge_deltas = {}
    for i, li in enumerate(JUDGE_ORDER):
        d_list = []
        for j, lj in enumerate(JUDGE_ORDER):
            if i != j and not np.isnan(valid_kappa[i, j]):
                d_list.append(valid_kappa[i, j] - pub_kappa[i, j])
        mean_d = float(np.mean(d_list))
        judge_deltas[li] = mean_d
        print(f"  {SHORT[li]:<14s}  Δ̄κ = {mean_d:+.4f}")

    out = {
        "config": {"n_judges": n_judges},
        "valid_kappa_matrix": {
            li: {lj: float(valid_kappa[i, j]) if not np.isnan(valid_kappa[i, j]) else None
                 for j, lj in enumerate(JUDGE_ORDER)}
            for i, li in enumerate(JUDGE_ORDER)
        },
        "valid_n_matrix": {
            li: {lj: int(valid_n[i, j]) for j, lj in enumerate(JUDGE_ORDER)}
            for i, li in enumerate(JUDGE_ORDER)
        },
        "published_kappa_matrix": {
            li: {lj: float(pub_kappa[i, j]) for j, lj in enumerate(JUDGE_ORDER)}
            for i, li in enumerate(JUDGE_ORDER)
        },
        "delta_kappa_per_judge": judge_deltas,
        "summary_stats": {
            "n_pairs_kappa_rose": pairs_up,
            "n_pairs_kappa_fell": pairs_down,
            "mean_abs_delta": float(np.mean(np.abs(deltas_arr))),
            "max_abs_delta": float(np.max(np.abs(deltas_arr))),
        },
        "interpretation": (
            f"Recomputing κ on valid-only intersection (vs None=0): "
            f"{pairs_up}/{len(deltas)} pairs rose, {pairs_down}/{len(deltas)} fell. "
            f"Mean shift {np.mean(np.abs(deltas_arr)):.3f}."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
