# Changelog

All notable changes to RAG-Eval-LLM-Judge are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-04-29

First public release. Targets the NeurIPS 2026 Datasets & Benchmarks Track (~mid-June 2026 submission), the LLM4Eval @ SIGIR 2027 workshop, and the ECIR 2027 / ACM TOIS journal track.

### Added — datasets and benchmark artifacts

- **Within-corpus 9-judge ablation** on 570 query-document pairs from the Iowa State University DSpace institutional repository (97k full-text PDFs). 5,130 individual LLM-judge score records across 9 judges from 5 model families (Anthropic ×2, OpenAI ×2, Google-commercial ×2, DeepSeek ×1, Open-weight ×2).
- **External validation** of the 9-judge slate against three public corpora with NIST/BEIR human qrels: TREC RAG 2024 (537 stratified-balanced pairs), TREC-COVID biomedical (300 pairs), BEIR scifact (300 pairs). 4,833 valid score records (after pairwise null exclusion).
- **Intra-judge self-consistency subset**: 50 pairs × 9 judges × 3 runs = 1,350 records measuring run-to-run reliability. Mean intra-judge κ = 0.93 across the slate; all 9 judges have intra-κ > κ-vs-human.
- **UMBRELA single-judge baseline** (Upadhyay et al. 2024 verbatim prompt + GPT-4o on the same 537-pair TREC RAG 2024 sample) for direct apples-to-apples comparison; κ vs human = 0.4265, n_valid = 537/537.
- **Bootstrap 95% CIs** on every reported headline κ value (1,000 resamples, seed=42).
- **Quadratic-weighted Gwet AC2** alongside Cohen κ for kappa-paradox triangulation.
- **5×5 family conditional-mean matrix** + per-judge marginal score distributions as bias diagnostics.
- **Croissant 1.0 metadata** (`croissant.json`) — JSON-LD describing the dataset distribution, recordSet schemas; validates against MLCommons mlcroissant validator with no schema warnings.
- **HuggingFace data card** (`DATA_CARD.md`) — YAML frontmatter declaring 8 dataset configurations; sections covering dataset summary, supported tasks, structure, splits, curation rationale, source data, annotations, sensitive info, social impact, biases, limitations, citation, license, maintenance plan.
- **Disclosure-template figure** (`figures/disclosure_template.png`) — proposed community standard for LLM-judged retrieval reporting.

### Added — code

- **`src/eval_llm_judge.py`** (~62 KB) — multi-judge harness with ThreadPool fan-out, retry logic, JUDGE_BUILDERS preset registry. Lazy-imports `qdrant_client` so the `--analyze` path works without it.
- **`src/validate_against_trec.py`** — multi-corpus external-validation harness for TREC RAG 2024, BEIR scifact, TREC-COVID. CORPUS_REGISTRY-driven; supports `--corpus`, `--analyze`, `--download`, `--list`. Path lookup tries both flat and BEIR-style nested data layouts.
- **`src/fetch_msmarco_passages.py`** — HF-streaming MS MARCO v2.1 passage extractor (60 shards, ~400 MB peak disk, ~38 min cold).
- **`src/run_umbrela_baseline.py`** — UMBRELA single-judge baseline runner (verbatim prompt from Upadhyay et al. 2024, §3).
- **`src/intra_judge_consistency.py`** — intra-judge self-consistency runner (imports JUDGE_BUILDERS to guarantee identical API call patterns to the published per-judge JSONs).
- **`src/bootstrap_kappa_cis.py`** — bootstrap 95% CIs on every reported κ.
- **`src/compute_gwet_ac2.py`** — quadratic-weighted Gwet AC2 alongside Cohen κ.
- **`src/bias_diagnostics.py`** — 5×5 family conditional-mean matrix + per-judge marginal distributions + length-stratified κ (skipped: uniform truncation in stored previews).
- **`src/analyze_kl_vs_kappa.py`** — KL-divergence-on-marginals analysis for the §6 mechanism section.
- **`src/analyze_pair_confusion.py`** — 4×4 pairwise confusion matrices for the 36 unique judge pairs.
- **`src/analyze_tokenizer_overlap.py`** — vocabulary Jaccard analysis (refutes the shared-tokenizer hypothesis).
- **`src/analyze_valid_only_kappa.py`** — re-cluster Gemini judges with None-as-missing for §4.2 valid-only mean.
- **`src/make_kappa_heatmap.py`** — Fig. 1 (9×9 κ heatmap) generator.
- **`src/make_disclosure_template.py`** — Fig. 2 (disclosure-template) generator.
- **`src/build_references_md.py`** — converts `arxiv/references.bib` to `papers/references.md` with topical grouping.
- **`src/verify_paper_claims.py`** — programmatic claim verifier; 62 numerical checks against shipped JSONs; < 5 sec to run; exit-code 0 on a clean checkout.

### Added — papers

- **`papers/short.md`** — workshop short-paper draft (4 pp, ≤2,130 words target). Targets LLM4Eval @ SIGIR 2027.
- **`papers/long.md`** — conference / journal full-paper draft (~12 pp LNCS, ~10,000 words). Targets ECIR 2027 / ACM TOIS.
- **`papers/neurips_db.md`** — Datasets & Benchmarks reframe (~9 pp + appendices). Targets NeurIPS 2026 D&B Track.
- **`papers/WALKTHROUGH.md`** — 12-section construction guide explaining how the papers are built (claims map, section walkthrough, figure/table guide, reviewer Q&A, edit instructions).
- **`papers/MECHANISTIC_KL_ANALYSIS.md`** — supporting narrative for §6 (mechanism).
- **`papers/PUBLIC_BENCHMARKS_VALIDATION.md`** — supporting results for §7 (external validation).
- **`papers/references.md`** — auto-generated from `arxiv/references.bib`; 24 entries grouped by topic.

### Added — submission bundles

- **`arxiv/`** — arXiv preprint LaTeX bundle (`main.tex`, `references.bib`, `_verify_citations.py`, `figures/`, `README.md`). Targets the workshop arXiv preprint timed to land after the NeurIPS D&B submission.
- **`neurips_db/`** — NeurIPS 2026 D&B LaTeX bundle, **anonymized** (`main.tex` with author block redacted, repo URL replaced with `<ANONYMOUS REPO URL>`, acknowledgments redacted). Includes a placeholder `neurips_data_2026.sty` to be replaced with the official style file when the CFP drops.

### Added — tests, docs, dashboard

- **`tests/`** — pytest test suite (25 tests across kappa math, ensemble convention, retrieval determinism, shipped-artifact integrity).
- **`pytest.ini`** — pytest configuration.
- **`docs/index.html`** + `docs/figures/` — GitHub Pages static dashboard at https://asukul.github.io/RAG-Eval-LLM-Judge/. KPI tiles, 4 findings, 9-judge κ matrix figure, 1.9× nDCG@10 spread table, mechanism summary, validation summary, reproducibility commands, citation block. Mobile-responsive CSS, no JS frameworks.
- **`slides/`** — `make_p4_slides.py` (python-pptx generator), `P4-SHORT.pptx` (17 slides), `P4-LONG.pptx` (34 slides), `SLIDES_REFERENCE.md` (slide-by-slide reference document).

### Added — repo metadata

- `LICENSE` — MIT for code, CC-BY-4.0 for data/figures/papers.
- `CITATION.cff` — Citation File Format with author, affiliation, and forthcoming arXiv preprint reference.
- `CHANGELOG.md` — this file.
- `requirements.txt` — pinned dependencies for `pip install -r`.
- `.env.template` — placeholder for `OPENAI_API_KEY` and `OPENROUTER_API_KEY` (real `.env` is gitignored).
- `.gitignore` — excludes secrets, build artifacts, IDE caches.

### Verification status (as of v1.0.0)

- **Verifier**: `src/verify_paper_claims.py` reports 62 pass / 0 fail / 0 warn. Latest run captured at `results/verification_log.txt`.
- **Tests**: pytest reports 25 passed in 1.4 sec.
- **Croissant**: `mlcroissant` validates `croissant.json` against the MLCommons 1.0 schema with no warnings (validation log at `results/croissant_validation_log.txt`).
- **Citations**: 23 cited keys in `arxiv/main.tex` and 23 in `neurips_db/main.tex` all resolve to bib entries (no UNDEFINED).

### Reproduction cost from a clean checkout

| Step | Cost | Wall |
|---|---:|---:|
| Within-corpus 7-judge frontier (570 pairs) | $15.50 | 154 min |
| Within-corpus 2-judge open-weight supplement (570 pairs) | $2.80 | 175.6 min |
| TREC RAG 2024 7-judge frontier (537 pairs) | $30 | ~5 h 12 min |
| TREC RAG 2024 2-judge supplement (537 pairs) | $5 | ~2.5 h |
| TREC-COVID 7-judge frontier (300 pairs) | $15 | ~2.5 h |
| TREC-COVID 2-judge supplement (300 pairs) | $2 | ~3 h |
| BEIR scifact 7-judge frontier (300 pairs) | $18 | ~3.5 h |
| BEIR scifact 2-judge supplement (300 pairs) | $3 | ~1.5 h |
| UMBRELA single-judge baseline (537 pairs) | $0.30 | 1.7 min |
| Intra-judge self-consistency (50 × 9 × 3 = 1,350) | $15 | 17.8 min |
| MS MARCO v2.1 passage extraction (537 IDs) | $0 | ~38 min |
| **Total** | **~$95** | **~19 h** |

For comparison: an institution-specific IRB human-rater study on a comparably-sized internal corpus is estimated at $1,500–3,000 over 6–12 weeks, making the validation pipeline ~30× cheaper and ~50× faster.

### Headline numerical claims

- **Within-corpus 9×9 κ matrix**: every off-diagonal cell ≥ 0.56 (substantial or moderate by Landis-Koch). Matrix-max (point estimate): Qwen 3.6 Plus ↔ Gemma 4 26B = 0.80 [95% CI 0.76, 0.83]. Cross-family commercial ceiling: Sonnet ↔ GPT-5.5 = 0.79 [0.75, 0.82]. Within-family pairs: Anthropic 0.71, Google-commercial 0.67, OpenAI 0.63 — all clearly below the cross-organization Open-weight pair (non-overlapping CIs).
- **TREC RAG 2024 9-judge ensemble κ vs NIST human qrels**: 0.4941, 95% CI [0.43, 0.56], moderate near-substantial.
- **TREC RAG 2024 7-judge frontier ensemble κ**: 0.5187 [0.46, 0.58].
- **TREC-COVID 9-judge ensemble κ**: 0.3447 [0.24, 0.45], fair near-moderate.
- **BEIR scifact 9-judge ensemble precision-at-≥2**: 65.7%.
- **UMBRELA single-judge baseline (TREC RAG 2024)**: 0.4265. Both ensembles + best single judge (Sonnet 4.6, κ = 0.5123) outperform UMBRELA by 0.07–0.09 κ.
- **Intra-judge κ across 3 runs (mean across 9 judges)**: 0.93. All judges have intra-κ > κ-vs-human (Δ range +0.51 to +0.70). Open-weight judges (Qwen 0.89, Gemma 0.91) are the *least* self-consistent in the slate.
- **Mechanism**: joint-distribution decomposition of κ into dispersion + effective rank recovers R² = 0.928. The high R² with dispersion is partly mathematical (synthetic-pair simulation gives R² ≈ 0.998 from κ-formula structure alone). The empirically interesting result, descriptively consistent with full mediation, is that structural factors (provider, reasoning-mode, model class) add ≤ 0.005 incremental R² above the decomposition. Shared-tokenizer hypothesis is refuted: Qwen ↔ Gemma 4 vocabulary Jaccard = 0.066 (lowest in slate) yet κ = 0.80 (matrix-highest).
- **Cluster gap**: within-cluster mean κ = 0.74 vs cross-cluster = 0.68 (Welch t = 3.45, p = 0.002 two-sided; Mann-Whitney U one-sided p = 0.004).

### Reviewer-driven revision history (P1–P3)

The v1.0 release reflects a full P1–P3 revision cycle driven by external peer reviewers (ChatGPT 5.5 deep-research and Claude Opus 4.7 deep-review-v2, 2026-04-29):

- **P1** (commit `ddda299`): Critical bug fixes — §5 C1 arithmetic errors corrected, §6.2 reframed to acknowledge the partial-tautology of the dispersion-vs-κ regression, path bugs in `src/` scripts fixed (no longer assumes the legacy monorepo layout), `verify_paper_claims.py` adopted with attribution.
- **P2** (commit `f5e0f24`): Reproducibility gaps closed — within-corpus per-judge JSONs shipped (5,130 records), ensemble convention standardized to upper-median (BEIR ensemble precision: 63.7% → 65.7% for consistency with TREC), §4.3 cluster claim quantified with effect size (0.06 κ, p = 0.002), `papers/references.md` generated, disclosure-template figure drawn, pytest suite added.
- **P3** (commits `5dd0f95` + `ca6a7ec` + `1bbaeba` + `39733d3` + `5c885e4`): Strengthening for venue-quality submission — bootstrap CIs, Gwet AC2, bias diagnostics, UMBRELA baseline, intra-judge self-consistency, NeurIPS D&B reframe with anonymized LaTeX bundle, Croissant + data card.
- **P3-followup** (commits `3b1a9e1` + `a927e5d`): Croissant validation against MLCommons mlcroissant validator (clean), code-archive zip for OpenReview supplementary, this CHANGELOG.

Each P-tier was committed and pushed; the pre-tier verifier state, post-tier verifier state, and complete commit history are available in `git log`.

---

## [Unreleased] — planned for v1.1

- TREC RAG 2025 replication when 2025 NIST qrels publish.
- Self-hosted open-weight inference replication (currently API-routed).
- Provenance-labeled documents for a strict self-preference test in the Panickssery 2024 sense.
- Possible additions: LLMJudge benchmark validation, TREC-COVID full set when expanded qrels publish.

[1.0.0]: https://github.com/asukul/RAG-Eval-LLM-Judge/releases/tag/v1.0.0
