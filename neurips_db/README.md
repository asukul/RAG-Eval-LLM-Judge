# NeurIPS 2026 Datasets & Benchmarks Track — Submission Bundle

This folder contains the **anonymized** LaTeX submission bundle for the NeurIPS 2026 D&B Track. Source material lives in `papers/neurips_db.md` (markdown master); this folder is the anonymized + LaTeX-compiled version.

```
neurips_db/
├── main.tex                  ← anonymized LaTeX source
├── references.bib            ← 25 BibTeX entries (incl. gwet2008 added for AC2)
├── neurips_data_2026.sty     ← PLACEHOLDER style (replace with official when CFP drops)
├── _verify_citations.py      ← citation verifier (currently 23/23 cited keys resolve)
├── figures/
│   ├── kappa_matrix_9judge.png        ← Fig. 1 (9×9 κ matrix)
│   ├── disclosure_template.png        ← Fig. 2 (community disclosure-template proposal)
│   ├── bias_diagnostics_panel.png     ← Fig. 3 (3-panel bias diagnostics)
│   ├── judge_calibration_mechanism.png  (held in reserve for §6.2 mechanism if needed)
│   └── judge_kappa_regression_decomposition.png  (held in reserve)
└── README.md                 ← this file
```

## ⚠️ PLACEHOLDER STYLE FILE NOTICE

`neurips_data_2026.sty` is a **placeholder**. The official NeurIPS 2026 D&B style file will be published at https://nips.cc/Conferences/2026/CallForDatasetsBenchmarks once the CFP drops (typically March-April for a June deadline). The placeholder reproduces the layout / spacing of the NeurIPS 2024 + 2025 D&B style files, so the compiled PDF approximates the final layout closely enough for **page-count and figure-fit verification ahead of the official release**, but the final submission MUST use the official style file.

**When the official style drops:**
1. Download `neurips_data_2026.sty` (or whatever the year-specific name is) from the conference website.
2. Replace this placeholder.
3. Re-run `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
4. Re-verify page count is under the limit (NeurIPS 2025 D&B was 9 pp main + 4 pp checklist + unlimited appendix; 2026 likely similar).

## Anonymization status

✅ **Author name** "Adisak Sukul" — removed from title block (`\textit{[anonymized for double-blind review]}`).
✅ **Affiliation** "Iowa State University" — replaced with "a US-public-university institutional repository" in main text.
✅ **Email** `asukul@iastate.edu` — removed.
✅ **Repository URL** `github.com/asukul/RAG-Eval-LLM-Judge` — replaced with `<ANONYMOUS REPO URL>` in `\section{Reproducibility}`. **TODO before submission**: replace with actual `anonymous.4open.science` snapshot URL.
✅ **Acknowledgments section** — `[redacted for double-blind review; will be restored at camera-ready]`.
✅ **Citation file** (`CITATION.cff`) — not part of this bundle; do NOT upload to OpenReview supplementary as it contains author identity.

**Re-audit before submitting:**
```bash
grep -nE "Adisak|Sukul|Iowa State|asukul@|github\.com/asukul|isu-research" main.tex
# Expected: no hits
```

## Build (local)

If you have `pdflatex` + `bibtex` installed (TeX Live, MiKTeX, etc.):

```bash
cd neurips_db/
pdflatex main
bibtex main
pdflatex main
pdflatex main      # second pass to resolve cross-refs
```

Produces `main.pdf`. Verify:
- Page count under the venue's limit (NeurIPS 2025 D&B: 9 pp main + 4 pp checklist excluding references)
- All `\ref{...}` and `\citep{...}` calls resolve (no `??` markers)
- Figures render in the correct positions

If you don't have LaTeX locally, [Overleaf](https://www.overleaf.com) accepts a `.zip` of this folder and compiles online.

## NeurIPS D&B submission checklist

Below is the **submission-flow checklist** in approximate order. Some items require user action (marked **YOU**); others can be batch-handled by the LaTeX build (marked **BUILD**).

### Pre-submission (in order)

1. **YOU**: Verify NeurIPS 2026 D&B CFP — actual deadline, page limits, anonymization rules, abstract pre-registration deadline (typically ~1 week before full).
   - https://nips.cc/Conferences/2026/CallForDatasetsBenchmarks
2. **YOU**: Replace `neurips_data_2026.sty` placeholder with the official style file.
3. **BUILD**: Re-run `pdflatex` to verify page count under the new style file.
4. **YOU**: Register OpenReview account (https://openreview.net) if not already.
5. **YOU**: Generate anonymous repo URL via https://anonymous.4open.science (5-min one-click).
6. **YOU**: Paste anonymous URL into `main.tex` — replace `<ANONYMOUS REPO URL>` placeholder.
7. **BUILD**: Re-run `pdflatex` after URL replacement.
8. **YOU**: Pre-register abstract on OpenReview (typically 7-10 days before full deadline).
9. **YOU** + **BUILD**: Final anonymization audit — run the grep above; verify zero hits.
10. **YOU**: Generate `arxiv-supplementary.zip` of the project root (excluding `.git/`, `.env`, `Paper-reviews/`, `_archive/`). Upload to OpenReview supplementary.
11. **YOU**: Upload `main.pdf` + supplementary zip + Croissant `croissant.json` + data card `DATA_CARD.md` via OpenReview.
12. **YOU**: Submit (one-click).

### Post-submission (during review)

- Reviewer questions can be answered via OpenReview comments.
- Major revisions require a new submission cycle (rare for NeurIPS D&B; minor clarifications go through OpenReview comment threads).

### Camera-ready (if accepted)

1. Replace `[final]` flag in `\usepackage{neurips_data_2026}` to reveal authors.
2. Restore acknowledgments.
3. Replace anonymous URL with `github.com/asukul/RAG-Eval-LLM-Judge`.
4. Mint Zenodo DOI (one-click via GitHub-Zenodo integration; pin to a specific GitHub release tag).
5. Apply for ACM Artifact Available + Functional + Reusable badges.

## Why this folder is parallel to `arxiv/` instead of replacing it

The NeurIPS 2026 D&B paper is a **distinct contribution** from the workshop short and the ECIR/journal long versions:
- `arxiv/main.tex` (workshop arXiv version) leads with C1-C4 empirical claims (cross-family κ, within-family bound, open-weight matrix-max, disclosure standard).
- `neurips_db/main.tex` (this bundle) leads with C1-C5 dataset+benchmark contributions (corpus, harness, verifier, Croissant, disclosure standard).

The same data and code support both framings; the venue dictates which framing is appropriate. Per the venue strategy in the project's internal documentation, NeurIPS 2026 D&B is the **primary archival target** for 2026 with a ~mid-June deadline; the workshop arXiv preprint is timed to land **after** D&B submission.

## Citation verification (programmatic)

```bash
py -3 _verify_citations.py
# Expected: UNDEFINED = none
```

If new citations are added to `main.tex`, update `references.bib` and re-run before commit.

## Submission cost / timeline summary

| Step | Cost | Time |
|---|---|---|
| LaTeX build (this bundle) | $0 | ~5 min |
| OpenReview submission | $0 | ~10 min |
| Anonymous URL setup | $0 | ~5 min |
| Style-file swap (placeholder → official) | $0 | ~2 min |
| **Total submission overhead** | **$0** | **~22 min** |

The empirical work (results in the paper) was done in the project's main branch before submission.
