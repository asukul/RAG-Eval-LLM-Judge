"""Generate κ-matrix heatmap for P4 §6.4.2 from the ablation JSON.

Reads the combined multi-judge results file and writes a PNG heatmap of the
pairwise quadratic-weighted Cohen's κ. Run once after any multi-judge eval.

Usage:
    py -3 src/make_kappa_heatmap.py \
        [--input results/<your_multijudge_json>.json]

Defaults to the newest multijudge JSON if --input is omitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Standalone-repo paths.
REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "results"
FIG_DIR = REPO / "figures"


# Short labels for readability in the figure (map from full label prefix).
# Single-line versions for clean 45-degree rotated x-axis labels.
SHORT_LABELS = {
    "Claude Opus 4.7": "Claude Opus 4.7",
    "Claude Sonnet 4.6": "Claude Sonnet 4.6",
    "Gemini 2.5 Pro": "Gemini 2.5 Pro",
    "Gemini 3.1 Pro Preview": "Gemini 3.1 Pro Preview",
    "GPT-5.5 (reasoning=low)": "GPT-5.5 (reason=low)",
    "GPT-4o (chat)": "GPT-4o",
    "DeepSeek V4 Pro": "DeepSeek V4 Pro",
    "DeepSeek V4 Flash": "DeepSeek V4 Flash",
}


def short(label: str) -> str:
    for prefix, short_form in SHORT_LABELS.items():
        if label.startswith(prefix):
            return short_form
    return label


def landis_koch_band(k: float) -> str:
    if k < 0.0:   return "poor"
    if k < 0.20:  return "slight"
    if k < 0.40:  return "fair"
    if k < 0.60:  return "moderate"
    if k < 0.80:  return "substantial"
    return "almost perfect"


def make_heatmap(input_path: Path, output_path: Path) -> None:
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    labels = data["config"]["judge_labels"]
    n = len(labels)
    mat = np.zeros((n, n))
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            v = data["kappa_matrix"][l1][l2]
            mat[i, j] = v if v is not None else np.nan

    short_labels = [short(l) for l in labels]

    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=150)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isnan(v):
                txt = "n/a"
                color = "gray"
            else:
                txt = f"{v:.2f}"
                # White text on dark cells (low κ on red) and dark on light
                color = "white" if v < 0.35 or v > 0.85 else "black"
            # Add Landis-Koch band in small text underneath (skip diagonal)
            if i == j:
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
            else:
                band = landis_koch_band(v) if not np.isnan(v) else ""
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
                ax.text(j, i + 0.28, band, ha="center", va="center",
                        fontsize=6.5, color=color, style="italic")

    # Colorbar with Landis-Koch bands annotated
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, aspect=24)
    cbar.set_label("Quadratic-weighted Cohen's κ", fontsize=10)
    band_edges = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00]
    band_labels = ["slight", "fair", "moderate", "substantial", "almost perfect"]
    for edge, label in zip(band_edges[:-1], band_labels):
        cbar.ax.axhline(edge, color="black", linewidth=0.3, alpha=0.3)
    # Secondary tick labels
    cbar.set_ticks(band_edges)
    cbar.ax.tick_params(labelsize=8)

    config = data.get("config", {})
    n_q = config.get("n_queries", "?")
    n_pairs = config.get("n_retrieved_pairs", "?")
    coll = config.get("collection", "?")
    ts = config.get("timestamp", "?")

    ax.set_title(
        f"LLM judge agreement on ISU DSpace retrieval — "
        f"{n_q} queries × top-{int(n_pairs)//int(n_q) if n_q else '?'} = {n_pairs} pairs\n"
        f"Collection: {coll}   ·   Run: {ts}",
        fontsize=10, pad=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}  ({output_path.stat().st_size} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=str, default=None,
                    help="Path to results_*_multijudge_*.json (default: newest)")
    ap.add_argument("--output", type=str, default=None,
                    help="Path to output PNG (default: figures/kappa_matrix_5judge.png)")
    args = ap.parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        candidates = sorted(EVAL_DIR.glob("results_*_multijudge_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("ERROR: no multijudge JSON found in", EVAL_DIR)
            return 1
        input_path = candidates[0]
    print(f"input: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        # Dynamic name based on judge count in the source JSON
        with input_path.open(encoding="utf-8") as f:
            n_judges = len(json.load(f).get("config", {}).get("judge_labels", []))
        output_path = FIG_DIR / f"kappa_matrix_{n_judges}judge.png"
    make_heatmap(input_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
