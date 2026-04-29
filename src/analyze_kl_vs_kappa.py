"""Per-judge score distribution analysis + pairwise KL divergence vs κ.

Tests H5 (marginal-distribution similarity) vs H1 (calibration philosophy as separate effect).

Loads the 9 per-judge JSONs, extracts the flat 570-pair score vector per judge,
computes:
  - Per-judge marginal P(score = k) for k in {0, 1, 2, 3}
  - Pairwise symmetric KL divergence on score marginals (Jeffreys divergence)
  - Pearson and Spearman correlation between pairwise (KL, κ)
  - 3-panel figure: histograms, KL heatmap, KL-vs-κ scatter

Usage:
    py -3 papers/P4_llm_as_judge/analyze_kl_vs_kappa.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Standalone-repo paths. Within-corpus 9-judge JSON ships in results/
# under the dspace_fulltext_vertex_multijudge_*_9judge.json filename.
REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "results"
FIG_DIR = REPO / "figures"
OUT_DIR = REPO / "results"

# Order judges in the same family-block ordering used elsewhere
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
SHORT_LABELS = {
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
# Cluster assignment (from FINDINGS_9judge.md §9, two emergent clusters)
CLUSTER = {
    "Claude Sonnet 4.6 (OpenRouter)": "reasoning-generous",
    "GPT-5.5 (reasoning=low)": "reasoning-generous",
    "DeepSeek V4 Pro (OpenRouter)": "reasoning-generous",
    "Claude Opus 4.7 (OpenRouter)": "strict-mid",
    "GPT-4o (chat)": "strict-mid",
    "Qwen 3.6 Plus (OpenRouter)": "strict-mid",
    "Gemma 4 26B (OpenRouter)": "strict-mid",
    "Gemini 2.5 Pro (OpenRouter)": "strict-outlier",
    "Gemini 3.1 Pro Preview (OpenRouter)": "strict-outlier",
}

# Per-judge JSON file mapping
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
# Source for κ matrix (already computed)
MERGED_JSON = "results_dspace_fulltext_vertex_multijudge_9judge_20260425.json"

# Smoothing: avoid log(0) in KL by adding pseudo-count to zero bins
SMOOTHING = 1e-6


def load_scores(label: str) -> np.ndarray:
    """Load flat 570-pair score vector for a given judge.

    None scores (Opus 4.7's q56-q57 outage gap) are kept as -1 sentinel and
    excluded from histogram + KL computation.
    """
    path = EVAL_DIR / PER_JUDGE_FILES[label]
    data = json.loads(path.read_text(encoding="utf-8"))
    scores: List[int] = []
    for q in data["queries"]:
        for r in q.get("retrieved", []):
            s = r.get("judge_score")
            if s is None:
                scores.append(-1)
            else:
                scores.append(int(s))
    return np.array(scores, dtype=int)


def histogram(scores: np.ndarray) -> np.ndarray:
    """Marginal P(score = k) for k in {0, 1, 2, 3}, ignoring -1 sentinels."""
    valid = scores[scores >= 0]
    counts = np.bincount(valid, minlength=4).astype(float)
    return counts / counts.sum()


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """Standard KL(p || q) with smoothing."""
    p_smooth = p + SMOOTHING
    q_smooth = q + SMOOTHING
    p_smooth /= p_smooth.sum()
    q_smooth /= q_smooth.sum()
    return float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))


def jeffreys(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric Jeffreys divergence = KL(p||q) + KL(q||p)."""
    return kl_div(p, q) + kl_div(q, p)


def main() -> None:
    print("Loading per-judge scores...")
    scores: Dict[str, np.ndarray] = {l: load_scores(l) for l in JUDGE_ORDER}
    for l in JUDGE_ORDER:
        s = scores[l]
        n_valid = int((s >= 0).sum())
        n_missing = int((s < 0).sum())
        print(f"  {SHORT_LABELS[l]:<14s}  n_valid={n_valid:4d}  n_missing={n_missing:3d}")

    print("\nComputing per-judge marginal histograms...")
    hists: Dict[str, np.ndarray] = {l: histogram(scores[l]) for l in JUDGE_ORDER}
    print(f"\n{'Judge':<16s}  {'P(0)':>6s} {'P(1)':>6s} {'P(2)':>6s} {'P(3)':>6s}  mean   entropy")
    for l in JUDGE_ORDER:
        h = hists[l]
        mean_score = float(np.dot(h, np.arange(4)))
        ent = -float(np.sum(h[h > 0] * np.log2(h[h > 0])))
        print(f"  {SHORT_LABELS[l]:<14s}  {h[0]:.3f}  {h[1]:.3f}  {h[2]:.3f}  {h[3]:.3f}  {mean_score:.3f}  {ent:.3f}")

    print("\nComputing pairwise Jeffreys (symmetric KL) divergence...")
    n = len(JUDGE_ORDER)
    kl_mat = np.zeros((n, n))
    for i, li in enumerate(JUDGE_ORDER):
        for j, lj in enumerate(JUDGE_ORDER):
            if i == j:
                kl_mat[i, j] = 0.0
            else:
                kl_mat[i, j] = jeffreys(hists[li], hists[lj])

    print("\nLoading κ matrix from merged JSON...")
    merged = json.loads((EVAL_DIR / MERGED_JSON).read_text(encoding="utf-8"))
    kappa = merged["kappa_matrix"]

    # Extract upper-triangular pairs
    kl_pairs: List[float] = []
    kappa_pairs: List[float] = []
    pair_labels: List[str] = []
    same_cluster: List[bool] = []
    for i, li in enumerate(JUDGE_ORDER):
        for j, lj in enumerate(JUDGE_ORDER):
            if i < j:
                kl_pairs.append(kl_mat[i, j])
                kappa_pairs.append(float(kappa[li][lj]))
                pair_labels.append(f"{SHORT_LABELS[li]} / {SHORT_LABELS[lj]}")
                same_cluster.append(CLUSTER[li] == CLUSTER[lj])

    print(f"\n{len(kl_pairs)} unique pairs (9 choose 2 = 36).")
    print("\nKL divergence vs κ correlation:")
    p_corr, p_pval = pearsonr(kl_pairs, kappa_pairs)
    s_corr, s_pval = spearmanr(kl_pairs, kappa_pairs)
    print(f"  Pearson  r = {p_corr:+.4f}  (p = {p_pval:.4g})")
    print(f"  Spearman r = {s_corr:+.4f}  (p = {s_pval:.4g})")
    print(f"  R² (Pearson)  = {p_corr ** 2:.4f}  i.e. KL explains {100 * p_corr ** 2:.1f}% of κ variance")

    # Within-cluster vs cross-cluster decomposition
    in_kl = [kl_pairs[i] for i in range(len(kl_pairs)) if same_cluster[i]]
    in_k = [kappa_pairs[i] for i in range(len(kl_pairs)) if same_cluster[i]]
    out_kl = [kl_pairs[i] for i in range(len(kl_pairs)) if not same_cluster[i]]
    out_k = [kappa_pairs[i] for i in range(len(kl_pairs)) if not same_cluster[i]]
    print(f"\nWithin-cluster pairs (n={len(in_kl)}):  mean κ = {np.mean(in_k):.4f}, mean KL = {np.mean(in_kl):.4f}")
    print(f"Cross-cluster pairs (n={len(out_kl)}):  mean κ = {np.mean(out_k):.4f}, mean KL = {np.mean(out_kl):.4f}")

    # 3-panel figure
    print("\nGenerating 3-panel figure...")
    fig = plt.figure(figsize=(18, 6))

    # Panel A: per-judge score histograms (grouped bars)
    ax1 = fig.add_subplot(1, 3, 1)
    bar_width = 0.09
    x_pos = np.arange(4)
    cluster_colors = {
        "reasoning-generous": "#2E7D32",
        "strict-mid": "#1565C0",
        "strict-outlier": "#C62828",
    }
    for k, l in enumerate(JUDGE_ORDER):
        h = hists[l]
        col = cluster_colors[CLUSTER[l]]
        ax1.bar(x_pos + k * bar_width, h, bar_width,
                label=SHORT_LABELS[l], color=col, alpha=0.7,
                edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x_pos + 4 * bar_width)
    ax1.set_xticklabels(["0", "1", "2", "3"])
    ax1.set_xlabel("Judge score (0-3 ordinal rubric)")
    ax1.set_ylabel("P(score = k) marginal")
    ax1.set_title("A. Per-judge marginal score distributions\n(green=reasoning-generous, blue=strict-mid, red=strict-outlier)")
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: KL divergence heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    short = [SHORT_LABELS[l] for l in JUDGE_ORDER]
    im = ax2.imshow(kl_mat, cmap="viridis", vmin=0, vmax=kl_mat.max())
    ax2.set_xticks(np.arange(n))
    ax2.set_yticks(np.arange(n))
    ax2.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(short, fontsize=8)
    ax2.set_title("B. Pairwise Jeffreys divergence\n(symmetric KL on score marginals)")
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{kl_mat[i, j]:.2f}", ha="center", va="center",
                     color="white" if kl_mat[i, j] > kl_mat.max() / 2 else "black", fontsize=7)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Panel C: scatter KL vs κ
    ax3 = fig.add_subplot(1, 3, 3)
    in_color = "#1B5E20"  # dark green for within-cluster
    out_color = "#B71C1C"  # dark red for cross-cluster
    ax3.scatter(in_kl, in_k, c=in_color, label=f"Within-cluster (n={len(in_kl)})",
                s=70, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax3.scatter(out_kl, out_k, c=out_color, label=f"Cross-cluster (n={len(out_kl)})",
                s=70, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Linear fit
    z = np.polyfit(kl_pairs, kappa_pairs, 1)
    pfit = np.poly1d(z)
    xline = np.linspace(min(kl_pairs), max(kl_pairs), 100)
    ax3.plot(xline, pfit(xline), "--", color="black", alpha=0.5, linewidth=1)
    ax3.text(0.04, 0.04,
             f"Pearson r = {p_corr:+.3f}  (p = {p_pval:.2g})\n"
             f"Spearman ρ = {s_corr:+.3f}  (p = {s_pval:.2g})\n"
             f"R² = {p_corr ** 2:.3f}\n"
             f"KL explains {100 * p_corr ** 2:.1f}% of κ variance",
             transform=ax3.transAxes, fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))
    ax3.set_xlabel("Jeffreys (symmetric KL) divergence on score marginals")
    ax3.set_ylabel("Pairwise quadratic-weighted Cohen's κ")
    ax3.set_title("C. KL divergence vs κ\n(does marginal-distribution similarity predict κ?)")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(left=-0.05)

    plt.tight_layout()
    out_fig = FIG_DIR / "judge_calibration_mechanism.png"
    plt.savefig(out_fig, dpi=140, bbox_inches="tight")
    print(f"  saved: {out_fig}")
    plt.close()

    # Save numeric results to JSON for paper / future use
    results = {
        "config": {
            "n_judges": n,
            "n_pairs": len(kl_pairs),
            "smoothing": SMOOTHING,
        },
        "per_judge": {
            l: {
                "P_score": hists[l].tolist(),
                "mean_score": float(np.dot(hists[l], np.arange(4))),
                "entropy_bits": float(-np.sum(hists[l][hists[l] > 0] * np.log2(hists[l][hists[l] > 0]))),
                "cluster": CLUSTER[l],
                "n_valid": int((scores[l] >= 0).sum()),
                "n_missing": int((scores[l] < 0).sum()),
            }
            for l in JUDGE_ORDER
        },
        "kl_matrix": {
            li: {lj: float(kl_mat[i, j]) for j, lj in enumerate(JUDGE_ORDER)}
            for i, li in enumerate(JUDGE_ORDER)
        },
        "correlations": {
            "pearson_r": p_corr,
            "pearson_p": p_pval,
            "spearman_r": s_corr,
            "spearman_p": s_pval,
            "r_squared": p_corr ** 2,
            "kl_explains_pct": 100 * p_corr ** 2,
        },
        "within_vs_cross_cluster": {
            "within_n": len(in_kl),
            "within_mean_kappa": float(np.mean(in_k)),
            "within_mean_kl": float(np.mean(in_kl)),
            "cross_n": len(out_kl),
            "cross_mean_kappa": float(np.mean(out_k)),
            "cross_mean_kl": float(np.mean(out_kl)),
        },
        "pair_table": [
            {"pair": pair_labels[i], "kl": kl_pairs[i], "kappa": kappa_pairs[i],
             "same_cluster": bool(same_cluster[i])}
            for i in range(len(kl_pairs))
        ],
    }
    out_json = OUT_DIR / "judge_kl_kappa_analysis.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  saved: {out_json}")

    print("\n=== SUMMARY ===")
    print(f"  KL divergence explains {100 * p_corr ** 2:.1f}% of pairwise κ variance.")
    if p_corr ** 2 > 0.5:
        print("  -> H5 (calibration distribution) is the dominant explanatory factor.")
    elif p_corr ** 2 > 0.3:
        print("  -> H5 is a major component but not dominant; H1/H2 contribute.")
    else:
        print("  -> H5 is a minor factor; H1/H2 dominate.")


if __name__ == "__main__":
    main()
