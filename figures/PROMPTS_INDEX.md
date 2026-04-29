# Figure prompts index — for AI image-generation comparison

This directory contains 11 paper figures (PNG + SVG where applicable) plus a `.txt` prompt file alongside each PNG. The `.txt` files are detailed image-generation prompts written for **GPT Image 2.0** and **Gemini Nano Banana 2** (or any other modern AI image generator) to recreate the figure for visual A/B comparison.

## Why this exists

The figures shipped here are matplotlib- and Mermaid-rendered. AI-generated alternatives may produce more polished or more reviewer-friendly versions for paper submission. Use the `.txt` prompts to regenerate each figure on multiple platforms, then compare visual quality, clarity, and information density.

## Figure inventory

### Conceptual diagrams (Mermaid-rendered originals)

| File | Prompt file | Description |
|---|---|---|
| `diagram_1_standard_rag_flow.png` | `diagram_1_standard_rag_flow.txt` | Standard RAG application flow (left-to-right pipeline) |
| `diagram_2_standard_eval.png` | `diagram_2_standard_eval.txt` | Standard single-judge LLM-as-Judge evaluation (with weak-point callout) |
| `diagram_3_our_multijudge_fanout.png` | `diagram_3_our_multijudge_fanout.txt` | Our 9-judge ThreadPool fan-out architecture |
| `diagram_4_norm_vs_ours.png` | `diagram_4_norm_vs_ours.txt` | Side-by-side: norm vs ours, with weak/strong callouts |
| `diagram_5_calibration_topology.png` | `diagram_5_calibration_topology.txt` | Cluster topology diagram (3 clusters with intra/cross edges and κ labels) |

### Data visualizations (matplotlib-rendered originals)

| File | Prompt file | Description |
|---|---|---|
| `kappa_matrix_9judge.png` | `kappa_matrix_9judge.txt` | **Figure 3 (canonical)**: 9×9 κ heatmap |
| `kappa_matrix_7judge.png` | `kappa_matrix_7judge.txt` | Figure 3a (appendix): 7-judge intermediate |
| `kappa_matrix_5judge.png` | `kappa_matrix_5judge.txt` | Figure 3b (appendix): 5-judge preliminary |
| `judge_calibration_mechanism.png` | `judge_calibration_mechanism.txt` | 3-panel: histograms + KL heatmap + KL-vs-κ scatter (Figure 6) |
| `judge_kappa_regression_decomposition.png` | `judge_kappa_regression_decomposition.txt` | 4-panel regression decomposition |
| `judge_pair_confusion_matrices.png` | `judge_pair_confusion_matrices.txt` | 8-panel grid: top-4 vs bottom-4 confusion matrices |

## Prompt structure (consistent across all `.txt` files)

Each `.txt` file follows this template:

```
# Image-generation prompt — <figure-name>

## One-line summary
[The headline of what the figure shows]

## Style guidance
[Visual style: colors, fonts, aspect ratio, no shadows/3D]

## Detailed scene description (or Plot specification)
[Element-by-element layout for diagrams; type-and-data spec for plots]

## Data (where applicable)
[Full data tables or values needed to reproduce the figure]

## Special annotation
[Highlights, callouts, emphasis instructions]

## Caption (place below figure)
[The actual paper caption to use under the figure]
```

## How to use

### For GPT Image 2.0
1. Copy the contents of one `.txt` file
2. Paste into GPT Image 2.0's prompt box
3. Generate; download result
4. Optionally iterate with refinement prompts

### For Gemini Nano Banana 2
1. Copy the contents of one `.txt` file
2. Paste into Gemini's image generation interface
3. Generate; download result

### For comparison
- Generate the same figure on both platforms
- Compare against the original (the `.png` next to the `.txt`)
- Pick the version that best fits the paper's visual identity

## Recommended evaluation criteria

When comparing AI-generated alternatives to the originals:
1. **Information density** — does the figure communicate the same data without omission?
2. **Label readability** — are tick labels, axis labels, and annotations clear?
3. **Visual hierarchy** — is the most important element (e.g., the matrix-highest κ cell, the weak-point callout) emphasized?
4. **Print quality** — does it scale to PDF print without aliasing?
5. **Color accessibility** — does the color choice work in greyscale and for color-blind readers?
6. **Caption-figure coherence** — does the figure support the caption claims without redundant prose?

## Notes on accuracy

- The data tables in the prompts (κ matrices, KL values, regression coefficients) are exact — derived from the actual experimental results in this directory's JSON files
- The cluster colors (green / blue / red for reasoning-generous / strict-mid / strict-outlier) are consistent across all diagrams
- Cluster assignment uses the **published** mean scores (with None=0 in aggregates) — see §6 of the paper for the valid-only mean reframing of Gemini judges
- All figures derive from the same 9-judge ablation: 5 model families, 570 (query, doc) pairs, 5,130 score records, USD 18.30 spend, 2026-04-25 run
