# P4 Slides — Reference & Edit Guide

**Last update:** 2026-04-28
**Folder:** `papers/P4_llm_as_judge/slides/`
**Purpose:** Single source of truth for everything you need when editing the P4 slide decks — file locations, data schemas, headline numbers, edit instructions.

---

## 1. Files in this folder

| File | Purpose |
|---|---|
| `make_p4_slides.py` | Python generator script. Edit this, then re-run, to update the .pptx files. |
| `P4-SHORT.pptx` | 16-slide deck for LLM4Eval @ SIGIR 2027 workshop (15-20 min talk) |
| `P4-LONG.pptx` | 33-slide deck for ECIR 2028 / TOIS full conference (25-30 min talk) |
| `SLIDES_REFERENCE.md` | This file. |

To regenerate both decks after any edit:
```
py -3 -X utf8 papers/P4_llm_as_judge/slides/make_p4_slides.py
```

---

## 2. Slide-by-slide map (which slide → which generator function)

### P4-SHORT (16 slides)

| # | Slide title | Generator function |
|---:|---|---|
| 1 | Title | `slide_title` |
| 2 | The which-judge problem | `slide_motivation` |
| 3 | Four contribution claims | `slide_contribution_overview` |
| 4 | Judge slate (table) | `slide_judge_slate` |
| 5 | Pipeline: 9-judge ThreadPool fan-out | `slide_pipeline_diagram` |
| 6 | Per-judge retrieval metrics (table) | `slide_per_judge_metrics` |
| 7 | 9×9 pairwise κ matrix | `slide_kappa_matrix_9j` |
| 8 | Calibration clusters (topology) | `slide_calibration_topology` |
| 9 | C1 — cross-family κ ≥ 0.75 | `slide_C1` |
| 10 | C3 — cross-org open-weight matrix-highest | `slide_C3` |
| 11 | Mechanism summary | `slide_mechanism_summary` |
| 12 | External validation results (κ table) | `slide_external_validation_results` |
| 13 | TREC-COVID third-corpus replication | `slide_trec_covid_results` |
| 14 | Coverage divergence (3-corpus table) | `slide_coverage_finding` |
| 15 | Takeaway | `slide_takeaway` |
| 16 | Q&A | `slide_qa` |

### P4-LONG (33 slides)

| # | Slide title | Generator function |
|---:|---|---|
| 1 | Title | `slide_title` |
| 2 | Outline | `slide_outline_long` |
| 3 | The which-judge problem | `slide_motivation` |
| 4 | Four contribution claims | `slide_contribution_overview` |
| 5 | Related work — 2025 cluster | `slide_related_work` |
| 6 | Corpus and query set | `slide_corpus_setup` |
| 7 | Judge slate (table) | `slide_judge_slate` |
| 8 | Rubric — bounded 0-3 ordinal | `slide_rubric` |
| 9 | Multi-judge pipeline (norm vs ours) | `slide_pipeline_diagram_long` |
| 10 | Per-judge retrieval metrics | `slide_per_judge_metrics` |
| 11 | Three calibration tiers (table) | `slide_calibration_tiers` |
| 12 | 9×9 pairwise κ matrix | `slide_kappa_matrix_9j` |
| 13 | Calibration topology | `slide_calibration_topology` |
| 14 | C1 — cross-family κ ≥ 0.75 | `slide_C1` |
| 15 | C2 — within-family bounded | `slide_C2` |
| 16 | C3 — cross-org open-weight | `slide_C3` |
| 17 | C4 — toolkit + disclosure | `slide_C4` |
| 18 | §6 mechanism summary | `slide_mechanism_summary` |
| 19 | §6 pair confusion — 4 highest-κ pairs | `slide_pair_confusion_top` |
| 20 | §6 pair confusion — 4 lowest-κ pairs | `slide_pair_confusion_bottom` |
| 21 | §6 regression — panels A+B (R² and ΔR²) | `slide_regression_AB` |
| 22 | §6 regression — panels C+D (coefs and predicted vs actual) | `slide_regression_CD` |
| 23 | §6 tokenizer refutation (table) | `slide_tokenizer_refutation` |
| 24 | §7 external validation setup | `slide_external_validation_setup` |
| 25 | §7 external validation results | `slide_external_validation_results` |
| 26 | §7 TREC-COVID third-corpus replication | `slide_trec_covid_results` |
| 27 | §7 coverage divergence (3 corpora) | `slide_coverage_finding` |
| 28 | §7 Thakur 2025 comparison | `slide_thakur_comparison` |
| 29 | Limitations | `slide_limitations` |
| 30 | Reproducibility (commands) | `slide_reproducibility` |
| 31 | Future work | `slide_future_work` |
| 32 | Takeaway | `slide_takeaway` |
| 33 | Q&A | `slide_qa` |

To find the function in `make_p4_slides.py`, search for `def slide_<name>` (Ctrl+F in any editor).

---

## 3. Headline numbers used in the slides

When you ask "change the κ from 0.49 to X", I edit these constants in `make_p4_slides.py`:

### Within-corpus 9-judge ablation (ISU DSpace)
- **Total pairs:** 570
- **Total score records:** 5,130
- **Wall:** 154 + 175.6 min (canonical 7 + open-weight 2)
- **Cost:** USD 18.30
- **Matrix-highest κ (Qwen ↔ Gemma 4 26B):** 0.80
- **Cross-family reasoning ceiling (Sonnet ↔ GPT-5.5):** 0.79
- **Within-family pairs:** Anthropic 0.71, OpenAI 0.63, Google 0.67, Open-weight cross-org 0.80
- **Lowest off-diagonal κ:** 0.56 (Sonnet ↔ Gemini 3.1 Prev)

### Mechanism (§6)
- **R² joint distribution → κ:** 0.93 (with dispersion + effective rank)
- **R² marginal KL → κ:** 0.33
- **Pearson r (KL, κ):** -0.571 (p = 2.7e-4)
- **Spearman ρ (KL, κ):** -0.625 (p = 4.6e-5)
- **Qwen ↔ Gemma 4 vocabulary Jaccard:** 0.066 (lowest in slate; their κ is highest — refutes shared-tokenizer hypothesis)

### External validation — TREC RAG 2024 (NIST qrels)
- **Sample:** 537 stratified-balanced pairs (135/134/134/134 across labels 0/1/2/3, `random.seed(42)`)
- **Queries:** 86 unique
- **Passages:** 537 unique from MS MARCO v2.1 segmented
- **9-judge ensemble κ:** **0.4941** (moderate, near-substantial)
- **7-judge frontier-only ensemble κ:** **0.5187**
- **Per-judge κ range:** 0.40 (Gemma 4 26B) to 0.55 (Gemini 2.5 Pro on n=92 valid)
- **Wall + cost:** ~7.5 h, USD 35

### External validation — BEIR scifact (precision-only)
- **Sample:** 300 pairs (all-positive qrels, κ undefined methodologically)
- **9-judge ensemble precision (≥ 2):** **63.7%**
- **Per-judge precision range:** 43% (Gemma 4 26B) to 75% (GPT-5.5 reasoning=low)

### External validation — TREC-COVID biomedical scientific (added 2026-04-28)
- **Sample:** 300 pairs (BEIR-distributed; 0/1/2 qrels mapped to 0/2/3 rubric)
- **Frontier-7 ensemble κ:** **0.4462** (Landis-Koch moderate)
- **Per-judge κ range:** 0.22 (Gemini 3.1 Prev) to 0.53 (Opus 4.7)
- **Coverage:** Sonnet/GPT-5.5/GPT-4o 100%; Opus 84%, DSV4 83%; Gemini 3.1 Prev 44%, Gemini 2.5 Pro 6%
- **Open-weight supplement:** Qwen 3.6 Plus + Gemma 4 26B run pending — slide currently shows frontier-7; will be re-rendered as 9-judge once supplement merges
- **Wall + cost:** ~2.5 h, ~$15 (frontier); ~$2 + ~30 min expected for supplement

### Coverage on TREC RAG 2024 (paper-relevant)
| Judge | TREC RAG 2024 | BEIR scifact |
|---|---:|---:|
| GPT-5.5 (reasoning=low) | 100% | 100% |
| GPT-4o | 100% | 100% |
| Claude Opus 4.7 | 100% | 95% |
| Claude Sonnet 4.6 | 100% | 99.7% |
| Qwen 3.6 Plus | 100% | 100% |
| Gemma 4 26B | 100% | 100% |
| DeepSeek V4 Pro | 39% | 84% |
| Gemini 3.1 Pro Preview | 24% | 60% |
| Gemini 2.5 Pro | 17% | 86% |

### Total session cost
- ~$56 across 4 corpus runs
- ~13 h wall total

---

## 4. Source data files (where everything lives)

All paths relative to `D:\RAG-DSpace\isu-research-search\` unless otherwise noted.

### Within-corpus 9-judge JSONs (ISU DSpace)
| File | Description |
|---|---|
| `backend/data/eval/results_dspace_fulltext_vertex_multijudge_9judge_20260425.json` | Canonical 9-judge merged result |
| `backend/data/eval/results_dspace_fulltext_vertex_multijudge_7judge_*.json` | 7-judge ablation (frontier subset) |
| `backend/data/eval/results_dspace_fulltext_vertex_multijudge_5judge_*.json` | 5-judge ablation (preliminary) |

### External-validation JSONs (TREC RAG 2024 + BEIR scifact)
| File | Description |
|---|---|
| `papers/P4_llm_as_judge/_validation_results/trec-rag-2024_judges.json` | **Merged 9-judge file (537 pairs) — the κ headline** |
| `papers/P4_llm_as_judge/_validation_results/trec-rag-2024_judges_p4-frontier.json` | 7-judge frontier subset |
| `papers/P4_llm_as_judge/_validation_results/trec-rag-2024_judges_p4-supplement-openweight.json` | 2-judge open-weight supplement (Qwen + Gemma) |
| `papers/P4_llm_as_judge/_validation_results/trec-rag-2024_kappa_vs_human.json` | κ analysis output |
| `papers/P4_llm_as_judge/_validation_results/beir-scifact_judges.json` | Merged 9-judge BEIR (300 pairs, κ undefined) |
| `papers/P4_llm_as_judge/_validation_results/beir-scifact_judges_p4-frontier.json` | 7-judge frontier |
| `papers/P4_llm_as_judge/_validation_results/beir-scifact_judges_p4-supplement-openweight.json` | 2-judge supplement |
| `papers/P4_llm_as_judge/_validation_results/beir-scifact_judges_smoke.json` | Initial smoke test (Gemma only) |
| `papers/P4_llm_as_judge/_validation_results/beir-scifact_kappa_vs_human.json` | κ output (returns 0 — methodology note) |
| `papers/P4_llm_as_judge/_validation_results/logs/*.log` | All run + analyze logs (8 files) |

### Corpus inputs (TREC RAG 2024)
| File | Description |
|---|---|
| `papers/P4_llm_as_judge/_validation_data/trec-rag-2024/2024-retrieval-qrels.txt` | Official NIST qrels — 20,283 rows × 86 queries × 20,194 passages |
| `papers/P4_llm_as_judge/_validation_data/trec-rag-2024/topics.rag24.test.txt` | TREC RAG 2024 query texts (301 topics, 86 with qrels) |
| `papers/P4_llm_as_judge/_validation_data/trec-rag-2024/sample_537_pairs.tsv` | **Stratified-balanced 537-pair sample** (random.seed(42)) |
| `papers/P4_llm_as_judge/_validation_data/trec-rag-2024/needed_passage_ids.txt` | List of 537 unique MS MARCO v2.1 IDs |
| `papers/P4_llm_as_judge/_validation_data/trec-rag-2024/passages.json` | **537 MS MARCO v2.1 passages** (1.02 MB) |
| `papers/P4_llm_as_judge/_validation_data/trec-rag-2024/fetch_msmarco_progress.json` | Shard-completion checkpoint |
| `papers/P4_llm_as_judge/_validation_data/trec-rag-2024/fetch_msmarco_*.log` | Extraction run log |

### Corpus inputs (BEIR auto-downloaded)
| Folder | Description |
|---|---|
| `papers/P4_llm_as_judge/_validation_data/beir-scifact/scifact/` | BEIR scifact (5,183 docs, 300 queries, 339 qrels — auto-downloaded) |
| `papers/P4_llm_as_judge/_validation_data/beir-trec-covid/trec-covid/` | BEIR TREC-COVID (171,332 docs, 50 queries, 66,336 qrels — auto-downloaded; not used for results) |

### Source PDFs (ISU DSpace)
| Path | Description |
|---|---|
| `tenants/isu/data/pdfs/` (or wherever PDFs live in the live tree) | The 97,441 full-text PDFs that feed the 1.03M Qdrant chunks |
| `backend/data/qdrant/` (Qdrant data dir) | Embedded chunks, M=16 ef_construct=100 cosine, Vertex text-embedding-005 |

### Mechanism analysis outputs
| File | Description |
|---|---|
| `papers/P4_llm_as_judge/judge_kl_kappa_analysis.json` | Per-pair KL, κ, regression coefficients |
| `papers/P4_llm_as_judge/judge_kappa_regression.json` | Full mediation regression diagnostics |
| `papers/P4_llm_as_judge/judge_pair_confusion.json` | All 36 pair confusion matrices |
| `papers/P4_llm_as_judge/judge_tokenizer_overlap.json` | Per-pair Jaccard tokenizer overlap |

### Figures (embedded in slides)
| File | Used in slide |
|---|---|
| `figures/diagram_3_our_multijudge_fanout.png` | Pipeline (SHORT #5) |
| `figures/diagram_4_norm_vs_ours.png` | Pipeline (LONG #9) |
| `figures/diagram_5_calibration_topology.png` | Calibration topology (SHORT #8, LONG #13) |
| `figures/kappa_matrix_9judge.png` | 9×9 κ matrix (SHORT #7, LONG #12) |
| `figures/kappa_matrix_5judge.png` | 5-judge ablation (appendix only) |
| `figures/kappa_matrix_7judge.png` | 7-judge ablation (appendix only) |
| `figures/judge_calibration_mechanism.png` | Mechanism 3-panel (SHORT #11, LONG #18) |
| `figures/judge_pair_confusion_matrices.png` | Original 8-panel confusion (deprecated for slides — overflowed slide boundary) |
| `figures/judge_pair_confusion_top4_highest.png` | Top half of confusion matrices (LONG #19) — 4:1 aspect, fits cleanly |
| `figures/judge_pair_confusion_bottom4_lowest.png` | Bottom half of confusion matrices (LONG #20) — 4:1 aspect |
| `figures/judge_kappa_regression_decomposition.png` | Original regression 4-panel (deprecated for slides — overflowed) |
| `figures/judge_kappa_regression_panels_AB.png` | Top half (panels A+B) of regression (LONG #21) |
| `figures/judge_kappa_regression_panels_CD.png` | Bottom half (panels C+D) of regression (LONG #22) |
| `figures/PROMPTS_INDEX.md` | Index of figure prompts (for AI image regen) |

### Paper drafts
| File | Description |
|---|---|
| `papers/P4_llm_as_judge/short.md` | **v0.4** — workshop short paper, condensed from long.md, ≤2,130 words target |
| `papers/P4_llm_as_judge/long.md` | **v0.1** — master long-form paper, 7,874 words, ECIR 2028 / TOIS target |
| `papers/P4_llm_as_judge/draft.md` | OLDER predecessor (REFINE+Judge — different paper, do not edit for P4) |

### Supporting docs
| File | Description |
|---|---|
| `papers/P4_llm_as_judge/MECHANISTIC_KL_ANALYSIS.md` | Full §6 mechanism narrative |
| `papers/P4_llm_as_judge/PUBLIC_BENCHMARKS_VALIDATION.md` | §7 plan + RESULTS section (per-corpus tables) |
| `papers/P4_llm_as_judge/FINDINGS_5judge.md` | 5-judge ablation findings |
| `papers/P4_llm_as_judge/FINDINGS_7judge.md` | 7-judge ablation findings |
| `papers/P4_llm_as_judge/FINDINGS_9judge.md` | 9-judge canonical ablation findings |
| `papers/P4_llm_as_judge/WRITING_PLAN.md` | 15-section plan for the paper portfolio |
| `papers/P4_llm_as_judge/VIABILITY.md` | Earlier viability analysis |
| `papers/P4_llm_as_judge/DIAGRAMS.md` | Mermaid source for diagrams 1-5 |

### Scripts
| File | Description |
|---|---|
| `backend/scripts/eval_llm_judge.py` | Multi-judge harness (`--judge-preset p4-frontier`, `--judge-preset p4-supplement-openweight`) |
| `papers/P4_llm_as_judge/validate_against_trec.py` | External-validation harness (`--corpus`, `--analyze`) |
| `papers/P4_llm_as_judge/fetch_msmarco_passages.py` | MS MARCO v2.1 passage extractor |
| `papers/P4_llm_as_judge/analyze_kl_vs_kappa.py` | Mechanism analysis (KL on marginals) |
| `papers/P4_llm_as_judge/analyze_pair_confusion.py` | 4×4 confusion matrices |
| `papers/P4_llm_as_judge/analyze_tokenizer_overlap.py` | Tokenizer Jaccard analysis |
| `papers/P4_llm_as_judge/analyze_valid_only_kappa.py` | Re-cluster Gemini judges with None-as-missing |
| `papers/P4_llm_as_judge/build_bibtex.py` | Build references.md from cite blocks |
| `papers/P4_llm_as_judge/slides/make_p4_slides.py` | Slide generator (this folder) |

### Archive (paid-API outputs preserved)
| Path | Description |
|---|---|
| `D:\RAG-DSpace\_archive\20260428_p4_validation_results\` | Snapshot of all paid API outputs as of 2026-04-28 (33 files, 3.0 MB) |
| `D:\RAG-DSpace\_archive\20260428_p4_validation_results.zip` | Zip backup (785 KB) |
| `D:\RAG-DSpace\_archive\20260428_p4_validation_results\README.md` | Archive manifest |
| `D:\RAG-DSpace\_archive\20260428_p4_validation_results\data_dictionary.md` | Schema for every file in the archive |
| `D:\RAG-DSpace\_archive\20260428_p4_validation_results\headline_findings.md` | Plain-text findings summary |

### Memory pointers (auto-loaded into Claude sessions)
| File | Description |
|---|---|
| `~/.claude/projects/D--RAG-DSpace/memory/p4_validation_archive_20260428.md` | Pointer to the validation archive |
| `~/.claude/projects/D--RAG-DSpace/memory/judge_ablation_findings_20260425.md` | 9-judge canonical findings memory |
| `~/.claude/projects/D--RAG-DSpace/memory/p4_short_writing_plan.md` | Writing plan pointer |

---

## 5. Data dictionary (file schemas)

For full schema details see `D:\RAG-DSpace\_archive\20260428_p4_validation_results\data_dictionary.md`. Quick reference:

### Judge-score JSON schema
```json
{
  "pair_index":         [["<query_id>", "<doc_id>"], ...],
  "human_scores":       [<int>, ...],
  "judge_specs":        ["<spec1>", ...],
  "per_judge_scores":   {"<judge_label>": [<int|null>, ...]},
  "per_judge_metadata": {"<judge_label>": {"n_valid": <int>, "n_missing": <int>, "mean_score": <float>}},
  "_merged_from":       {"frontier": "...", "supplement": "...", "merged_at": "..."}    // merged files only
}
```

### Judge spec ↔ label mapping
| Spec ID | Display label | Family | Reasoning |
|---|---|---|---|
| `claude-opus-4.7` | Claude Opus 4.7 (OpenRouter) | Anthropic | yes |
| `claude-sonnet` | Claude Sonnet 4.6 (OpenRouter) | Anthropic | yes |
| `openai-gpt-5.5-low` | GPT-5.5 (reasoning=low) | OpenAI | yes |
| `openai-gpt-4o` | GPT-4o (chat) | OpenAI | no |
| `gemini-3.1-pro` | Gemini 3.1 Pro Preview (OpenRouter) | Google-commercial | yes (thinking) |
| `gemini-2.5-pro` | Gemini 2.5 Pro (OpenRouter) | Google-commercial | yes (thinking) |
| `deepseek-v4-pro` | DeepSeek V4 Pro (OpenRouter) | DeepSeek | yes |
| `qwen-3.6-plus` | Qwen 3.6 Plus (OpenRouter) | Open-weight | no |
| `gemma-4-26b` | Gemma 4 26B (OpenRouter) | Open-weight | no |

### TREC RAG 2024 sample TSV
Tab-separated, header: `query_id <TAB> doc_id <TAB> human_rel`. 537 rows.

### MS MARCO passages.json
Dict keyed by passage ID. Each entry: `{docid, title, headings, text, url}`.

### NIST qrels (TREC format)
Whitespace-separated, no header, 4 columns: `query_id, 0, doc_id, human_rel`. 20,283 rows.

---

## 6. Edit instructions (for future-me / future Claude)

### Common edits

**Change a number:** Find the headline number in `make_p4_slides.py` (search for the value, e.g. "0.4941"). Edit and re-run. The number appears in only 1-2 slide functions; change it in all locations.

**Change a table row:** Each table is a `rows` variable inside its `slide_*` function. Find the function (see slide-by-slide map above), edit the list, re-run.

**Swap a figure:** Replace the .png file at `figures/<name>.png`. Re-run. The slide picks up the new image automatically. To swap which figure is used: edit the `add_image(s, FIGS / "<name>.png", ...)` call in the relevant `slide_*` function.

**Add a new slide:**
1. Write a new `slide_<name>(p, n, total, *, version_label=None)` function in `make_p4_slides.py`. Copy an existing function as a template.
2. Add it to the slide list in `build_short()` and/or `build_long()`.
3. Re-run.

**Remove a slide:** Delete the entry from the slide list in `build_short()` or `build_long()`. The function definition can stay (unused) or be deleted.

**Reorder slides:** Move entries in the slide list. Re-run.

**Change a color:** Edit the `COLOR_*` constants at the top of `make_p4_slides.py`. They use `RGBColor(0xRR, 0xGG, 0xBB)`. The script enforces a consistent visual identity — all titles in `COLOR_TITLE`, all accents in `COLOR_ACCENT`, etc.

**Change the venue / authors / date:** Edit `slide_title()` (top of generator helpers section).

### Re-run after edits
```
py -3 -X utf8 papers/P4_llm_as_judge/slides/make_p4_slides.py
```
Both `.pptx` files are regenerated in ~2 seconds. Idempotent; re-running with no changes produces identical files (except for embedded timestamps which are not visible in PowerPoint).

### When figures change
If you regenerate a figure in `figures/` (e.g. update the κ matrix with new data), re-run the slide generator and both decks pick up the new figure automatically. No code edit needed.

### When data changes (new validation run)
1. Update headline numbers in `make_p4_slides.py` (search-and-replace; tables are explicit lists of strings — easy to update).
2. Re-run the slide generator.
3. Update `SLIDES_REFERENCE.md` (this file) Section 3 with the new numbers.

### Adding the user as a co-author
The author line is in `slide_title()`:
```python
authors_line="Adisak Sukul",
```
Change to e.g. `"Adisak Sukul · [Co-author Name]"`.

---

## 7. Reproducibility commands

For reviewers / future readers:

### Recompute κ headline (within-corpus, ~$18, ~5.5 h):
```
py -3 -X utf8 backend/scripts/eval_llm_judge.py --collection <yours> --judge-preset p4-frontier
py -3 -X utf8 backend/scripts/eval_llm_judge.py --collection <yours> --judge-preset p4-supplement-openweight
```

### Recompute external-validation κ (TREC RAG 2024, ~$35, ~7.5 h):
```
py -3 -X utf8 papers/P4_llm_as_judge/validate_against_trec.py --corpus trec-rag-2024 --judge-preset p4-frontier --max-pairs 537
py -3 -X utf8 papers/P4_llm_as_judge/validate_against_trec.py --corpus trec-rag-2024 --judge-preset p4-supplement-openweight --max-pairs 537
py -3 -X utf8 papers/P4_llm_as_judge/validate_against_trec.py --analyze trec-rag-2024
```

### Recompute mechanism analysis (free, ~30 sec):
```
py -3 -X utf8 papers/P4_llm_as_judge/analyze_kl_vs_kappa.py
py -3 -X utf8 papers/P4_llm_as_judge/analyze_pair_confusion.py
py -3 -X utf8 papers/P4_llm_as_judge/analyze_tokenizer_overlap.py
py -3 -X utf8 papers/P4_llm_as_judge/analyze_valid_only_kappa.py
```

### Re-extract MS MARCO v2.1 passages (free if HF cache hits, ~38 min if cold):
```
py -3 -X utf8 papers/P4_llm_as_judge/fetch_msmarco_passages.py
```

### Regenerate slides (free, ~2 sec):
```
py -3 -X utf8 papers/P4_llm_as_judge/slides/make_p4_slides.py
```

---

## 8. Model versions pinned 2026-04-25

For reproducibility — these are the exact model snapshots used in all runs:

| Family | Spec ID | Model snapshot |
|---|---|---|
| Anthropic | `claude-opus-4.7` | `claude-opus-4-7-20260415` |
| Anthropic | `claude-sonnet` | `claude-sonnet-4-6-20260401` |
| OpenAI | `openai-gpt-5.5-low` | `gpt-5.5-2026-04-12` (reasoning=low) |
| OpenAI | `openai-gpt-4o` | `gpt-4o-2024-08-06` |
| Google | `gemini-3.1-pro` | `gemini-3.1-pro-preview-2026-04` |
| Google | `gemini-2.5-pro` | `gemini-2.5-pro-2025-06-17` |
| DeepSeek | `deepseek-v4-pro` | `deepseek-v4-pro-2026-04-10` |
| Open-weight | `qwen-3.6-plus` | `qwen3.6-plus-2026-03` |
| Open-weight | `gemma-4-26b` | `gemma-4-26b-a4b-it-2026-03` |

API routes:
- **OpenRouter** for: Anthropic, Google, DeepSeek, Qwen, Gemma (uniform billing, gateway approach)
- **OpenAI direct** for: GPT-5.5, GPT-4o (cheaper than OpenRouter passthrough)

---

## 9. Pending / future updates

When new data lands, these slides change:
- **TREC-COVID validation** (planned but not run): would add a column/slide to §7 results
- **LLMJudge benchmark validation** (planned but not run): same
- **TREC RAG 2024 with Thakur 2025's exact 537-pair subset** (pending receipt from authors): would add a comparison row to slide 25
- **P4-LIB derivative paper** (JCDL 2027 target): separate slide deck under a future `slides/p4-lib/` folder
- **P4 v2 with TREC RAG 2025** (when 2025 NIST qrels publish): same passage extractor works (same MS MARCO v2.1 corpus); just needs new sample + run

When the Haiku PRT topic-extraction pipeline finishes (currently 97% complete), no slides change — Haiku is a separate operational pipeline, not part of the P4 paper. But the PRT corpus may become a future second institutional-corpus replication of the 9-judge ablation, complementing ISU DSpace.

---

## 10. Contact & versioning

**Author:** Adisak Sukul (asukul@iastate.edu)
**Slides last regenerated:** 2026-04-28 08:40 CDT
**Generator script version:** v1 (initial release)
**Total project session cost to date:** ~$56 (validation runs only) + minimal harness/figure regeneration cost
