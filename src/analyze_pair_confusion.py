"""Higher-order calibration analysis: pair-level confusion matrices.

Tests whether off-diagonal asymmetries (judge A's score → judge B's score
flow patterns) predict residual κ variance after KL divergence is controlled.

For each pair, computes:
  - 4×4 confusion matrix C[i,j] = P(judge A says i, judge B says j)
  - diagonal_density: trace(C) — fraction of exact-agreement pairs
  - asymmetry_index: mean |C[i,j] - C[j,i]| over off-diagonal — measures whether
    one judge systematically scores higher than the other
  - dispersion: weighted disagreement = sum_{i!=j} C[i,j] * |i-j|^2 — penalizes
    far-apart disagreements (proxy for "catastrophic miscalibration")

Then adds these to the OLS regression κ ~ KL + ... and reports incremental R².

Usage:
    py -3 papers/P4_llm_as_judge/analyze_pair_confusion.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "backend" / "data" / "eval"
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_JSON = ROOT / "judge_pair_confusion.json"

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
    path = EVAL_DIR / PER_JUDGE_FILES[label]
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for q in data["queries"]:
        for r in q.get("retrieved", []):
            s = r.get("judge_score")
            out.append(-1 if s is None else int(s))
    return np.array(out, dtype=int)


def confusion_matrix(a: np.ndarray, b: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """4×4 normalized confusion matrix on valid pairs only."""
    mask = (a >= 0) & (b >= 0)
    a_v = a[mask]
    b_v = b[mask]
    C = np.zeros((n_classes, n_classes), dtype=float)
    for x, y in zip(a_v, b_v):
        C[x, y] += 1
    if C.sum() > 0:
        C /= C.sum()
    return C


def features_from_confusion(C: np.ndarray) -> dict[str, float]:
    """Compute higher-order calibration features from a 4×4 confusion matrix."""
    n = C.shape[0]
    diag = float(np.trace(C))

    # Asymmetry: how much does C[i,j] differ from C[j,i] off-diagonal?
    asym_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if (C[i, j] + C[j, i]) > 1e-9:
                asym = abs(C[i, j] - C[j, i])
                asym_pairs.append(asym)
    asymmetry = float(np.mean(asym_pairs)) if asym_pairs else 0.0

    # Signed asymmetry: positive = judge A scores systematically lower (more mass below diagonal)
    below_diag = sum(C[i, j] for i in range(n) for j in range(n) if i > j)
    above_diag = sum(C[i, j] for i in range(n) for j in range(n) if i < j)
    signed_asym = float(below_diag - above_diag)  # in [-1, 1]

    # Quadratic dispersion: penalizes far-apart disagreements
    indices = np.arange(n)
    distance_sq = (indices[:, None] - indices[None, :]) ** 2
    dispersion = float((C * distance_sq).sum())

    # Effective rank of confusion matrix (a higher-order joint structure measure)
    sv = np.linalg.svd(C, compute_uv=False)
    sv_normalized = sv / sv.sum() if sv.sum() > 0 else sv
    effective_rank = float(np.exp(-(sv_normalized * np.log(sv_normalized + 1e-12)).sum()))

    return {
        "diagonal_density": diag,
        "asymmetry": asymmetry,
        "signed_asymmetry": signed_asym,
        "dispersion": dispersion,
        "effective_rank": effective_rank,
    }


def main() -> int:
    print("Loading per-judge scores...")
    scores = {l: load_scores(l) for l in JUDGE_ORDER}

    # Load published κ matrix
    merged = json.loads((EVAL_DIR / MERGED_JSON).read_text(encoding="utf-8"))
    kappa = merged["kappa_matrix"]

    # Load existing KL features
    kl_data = json.loads((ROOT / "judge_kl_kappa_analysis.json").read_text(encoding="utf-8"))
    kl_lookup = {p["pair"]: p["kl"] for p in kl_data["pair_table"]}

    rows = []
    confusion_data = {}
    for i, li in enumerate(JUDGE_ORDER):
        for j, lj in enumerate(JUDGE_ORDER):
            if i < j:
                C = confusion_matrix(scores[li], scores[lj])
                feats = features_from_confusion(C)
                pair = f"{SHORT[li]} / {SHORT[lj]}"
                confusion_data[pair] = {"C": C.tolist(), **feats}
                rows.append({
                    "pair": pair,
                    "kappa": float(kappa[li][lj]),
                    "kl": kl_lookup[pair],
                    **feats,
                })

    df = pd.DataFrame(rows)
    print(f"Pairs: {len(df)}")
    print()

    # Print correlation of each feature with κ
    print("Bivariate Pearson r with κ:")
    print(f"{'feature':<22s}  {'r':>8s}  {'p':>10s}")
    from scipy.stats import pearsonr
    for col in ["kl", "diagonal_density", "asymmetry", "signed_asymmetry", "dispersion", "effective_rank"]:
        r, p = pearsonr(df[col], df["kappa"])
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
        print(f"  {col:<22s}  {r:+8.4f}  {p:>10.3g} {sig}")
    print()

    # Nested regression: κ ~ KL, then add each higher-order feature one at a time
    print("Adding higher-order calibration features to regression (vs M1 = KL only):")
    print(f"{'Model':<28s}  {'R²':>7s}  {'adj R²':>8s}  {'ΔR² vs M1':>11s}  {'F':>7s}  {'p':>10s}")

    y = df["kappa"]
    M1_X = sm.add_constant(df[["kl"]])
    M1 = sm.OLS(y, M1_X).fit()

    feature_sets = {
        "M1: KL": ["kl"],
        "M5a: KL + diagonal_density": ["kl", "diagonal_density"],
        "M5b: KL + asymmetry": ["kl", "asymmetry"],
        "M5c: KL + signed_asymmetry": ["kl", "signed_asymmetry"],
        "M5d: KL + dispersion": ["kl", "dispersion"],
        "M5e: KL + effective_rank": ["kl", "effective_rank"],
        "M6:  KL + ALL higher-order": ["kl", "diagonal_density", "asymmetry", "signed_asymmetry", "dispersion", "effective_rank"],
    }

    fits = {}
    for name, preds in feature_sets.items():
        X = sm.add_constant(df[preds])
        res = sm.OLS(y, X).fit()
        fits[name] = res
        if name == "M1: KL":
            d_r2 = 0.0
            f_stat = float("nan")
            p_val = float("nan")
        else:
            df_diff = M1.df_resid - res.df_resid
            f_stat = ((M1.ssr - res.ssr) / df_diff) / (res.ssr / res.df_resid) if df_diff > 0 else float("nan")
            p_val = float(1.0 - f_dist.cdf(f_stat, df_diff, res.df_resid)) if not np.isnan(f_stat) else float("nan")
            d_r2 = res.rsquared - M1.rsquared
        print(f"  {name:<28s}  {res.rsquared:.4f}  {res.rsquared_adj:.4f}  {d_r2:+11.4f}  {f_stat:>7.2f}  {p_val:>10.3g}")
    print()

    # Final compound model M6 details
    print("M6 (KL + all higher-order) full coefficient table:")
    m6 = fits["M6:  KL + ALL higher-order"]
    print(m6.summary())
    print()

    # Generate confusion-matrix grid figure (top 4 highest-κ + bottom 4 lowest-κ pairs)
    df_sorted = df.sort_values("kappa", ascending=False)
    top4 = df_sorted.head(4).to_dict("records")
    bot4 = df_sorted.tail(4).to_dict("records")
    selected = top4 + bot4

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, row in zip(axs.flat, selected):
        pair = row["pair"]
        C = np.array(confusion_data[pair]["C"])
        im = ax.imshow(C, cmap="Blues", vmin=0, vmax=C.max())
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if C[i, j] > C.max() / 2 else "black")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(["0", "1", "2", "3"])
        ax.set_yticklabels(["0", "1", "2", "3"])
        side = "TOP" if row["kappa"] > 0.7 else "BOTTOM"
        ax.set_title(f"{pair}\nκ={row['kappa']:.3f} [{side}]")
        ax.set_xlabel("Judge B score")
        ax.set_ylabel("Judge A score")

    plt.tight_layout()
    fig_path = FIG_DIR / "judge_pair_confusion_matrices.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig_path}")

    # Save summary
    out = {
        "config": {"n_pairs": len(df), "n_classes": 4},
        "pair_features": confusion_data,
        "regression_summary": {
            name: {
                "r_squared": float(res.rsquared),
                "adj_r_squared": float(res.rsquared_adj),
                "n_predictors": int(res.df_model),
                "coefs": {
                    p: {"beta": float(res.params[p]), "p": float(res.pvalues[p])}
                    for p in res.params.index
                },
            } for name, res in fits.items()
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved data: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
