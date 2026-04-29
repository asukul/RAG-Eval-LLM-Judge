"""Generate figures/disclosure_template.png — a labeled mockup of the
proposed community-disclosure norm for LLM-judged retrieval.

The norm: every nDCG@10 (or any LLM-judged retrieval metric) report should
disclose the judging configuration explicitly:

    nDCG@10 = X.XX  via  [family]/[model]/[reasoning-config], N pairs, DATE

This figure renders three example "good" disclosures and contrasts with one
"bad" undisclosed report, so the norm is visually self-explanatory.

Run from repo root:
    py -3 src/make_disclosure_template.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "disclosure_template.png"


COLOR_BAD = "#c8102e"
COLOR_GOOD = "#0e4a86"
COLOR_GREY = "#57606a"
COLOR_BG_BAD = "#ffeaea"
COLOR_BG_GOOD = "#eaf3fb"
COLOR_BORDER = "#d0d7de"


def main():
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=180)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    ax.text(
        5, 4.7,
        "Proposed disclosure norm for LLM-judged retrieval metrics",
        fontsize=14, fontweight="bold", ha="center", color=COLOR_GOOD,
    )
    ax.text(
        5, 4.35,
        "Every nDCG@10 / P@5 / MRR / kappa report should explicitly disclose the judging configuration",
        fontsize=10, ha="center", color=COLOR_GREY, style="italic",
    )

    # Bad example
    bad_box = patches.FancyBboxPatch(
        (0.3, 3.05), 9.4, 0.95,
        boxstyle="round,pad=0.04",
        linewidth=1.2, edgecolor=COLOR_BAD, facecolor=COLOR_BG_BAD,
    )
    ax.add_patch(bad_box)
    ax.text(0.55, 3.75, "BAD (current practice)",
            fontsize=10, fontweight="bold", color=COLOR_BAD, va="top")
    ax.text(0.55, 3.40, '"Our retriever achieves nDCG@10 = 0.86."',
            fontsize=12, color=COLOR_BAD, family="monospace", va="top")
    ax.text(9.55, 3.40,
            "Which judge? Which reasoning mode?\n"
            "How many pairs? When?  Not reproducible.",
            fontsize=8, color=COLOR_BAD, ha="right", va="top", style="italic")

    # Good — schema
    schema_box = patches.FancyBboxPatch(
        (0.3, 1.85), 9.4, 0.95,
        boxstyle="round,pad=0.04",
        linewidth=1.2, edgecolor=COLOR_GOOD, facecolor=COLOR_BG_GOOD,
    )
    ax.add_patch(schema_box)
    ax.text(0.55, 2.55, "GOOD (proposed norm) - schema",
            fontsize=10, fontweight="bold", color=COLOR_GOOD, va="top")
    ax.text(0.55, 2.20,
            "nDCG@10 = X.XX  via  [family] / [model] / [reasoning-config], N pairs, DATE",
            fontsize=12, color=COLOR_GOOD, family="monospace", va="top",
            fontweight="bold")

    # Good — three concrete examples
    examples = [
        "nDCG@10 = 0.86  via  Anthropic/claude-sonnet-4.6/reasoning, 570 pairs, 2026-04-25",
        "nDCG@10 = 0.84  via  OpenAI/gpt-5.5/reasoning=low,        570 pairs, 2026-04-25",
        "nDCG@10 = 0.45  via  Google/gemini-3.1-pro-prev/thinking,  570 pairs, 2026-04-25",
    ]
    ex_box = patches.FancyBboxPatch(
        (0.3, 0.15), 9.4, 1.55,
        boxstyle="round,pad=0.04",
        linewidth=1.2, edgecolor=COLOR_GOOD, facecolor=COLOR_BG_GOOD,
    )
    ax.add_patch(ex_box)
    ax.text(0.55, 1.55, "GOOD (proposed norm) - three concrete examples",
            fontsize=10, fontweight="bold", color=COLOR_GOOD, va="top")
    for i, ex in enumerate(examples):
        ax.text(0.55, 1.20 - i * 0.32, ex,
                fontsize=10, color=COLOR_GOOD, family="monospace", va="top")
    ax.text(9.55, 0.30,
            "Same 570 docs.  Different judges.  1.9x nDCG@10 spread (0.45 - 0.86).",
            fontsize=8, color=COLOR_GREY, ha="right", va="top", style="italic",
            fontweight="bold")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
