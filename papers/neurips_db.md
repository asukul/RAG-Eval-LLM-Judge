---
title: "RAG-Eval-LLM-Judge: A 5-Family, 9-Judge Cross-Family LLM-Judge Agreement Benchmark for Institutional Retrieval"
authors:
  - name: "Adisak Sukul"
    affil: "Iowa State University"
    email: "asukul@iastate.edu"
venue: "NeurIPS 2026 — Datasets & Benchmarks Track (target submission)"
length_target: "9 pages main + unlimited appendix"
status: "draft v0.1 (P3f reframe of long.md, written 2026-04-29)"
companion: "long.md (ECIR full paper version), short.md (LLM4Eval workshop), papers/WALKTHROUGH.md"
---

# Abstract

We release **RAG-Eval-LLM-Judge**, a fully reproducible benchmark suite for studying inter-LLM-judge agreement on bounded ordinal information-retrieval relevance judgment, plus a reference 9-judge dataset that demonstrates its use. The benchmark addresses a methodological gap that no current public artifact closes simultaneously: a **five-family** pairwise quadratic-weighted Cohen's κ matrix on a **bounded 0–3 ordinal** rubric at **≥ 500 paired observations**, with **within-family controls** and **open-weight peers as first-class participants**. Our reference run scores 9 LLM judges (Anthropic ×2, OpenAI ×2, Google-commercial ×2, DeepSeek ×1, Open-weight ×2) on **570 query-document pairs from the Iowa State University DSpace institutional repository (97k full-text PDFs)** plus **1,137 pairs across three external public corpora with NIST/BEIR human qrels** (TREC RAG 2024, TREC-COVID, BEIR scifact). The release contains: (i) **9,963 individual LLM-judge score records** (5,130 within-corpus + 4,833 external valid + 1,350 intra-judge re-runs subset + UMBRELA single-judge baseline); (ii) **a 1,500-line Python harness** (`eval_llm_judge.py`, `validate_against_trec.py`, `fetch_msmarco_passages.py`) covering scoring, multi-corpus validation, MS MARCO v2.1 streaming extraction, and the cross-run merge; (iii) **a programmatic verifier** (`verify_paper_claims.py`) that asserts every numerical claim in the companion paper against the shipped JSONs (62 claim checks, exit-code 0 on a clean checkout); (iv) **a 25-test pytest suite** for the kappa math, ensemble convention, retrieval determinism, and shipped-artifact integrity; (v) **Croissant 1.0 metadata + HuggingFace data card**; (vi) **bootstrap 95% CIs, Gwet AC2, intra-judge self-consistency, and bias diagnostics** as pre-computed analytical artifacts. Total reproduction cost from a clean checkout is **~$95 in paid API spend** (within-corpus ~$18 + external validation ~$50 + UMBRELA ~$0.30 + intra-judge ~$15) and **~13 hours of wall time**, which replaces an institution-specific IRB human-rater study (estimated $1,500–3,000 / 6–12 weeks). All code is MIT-licensed; all data, figures, and paper text are CC-BY-4.0. The headline empirical finding the dataset establishes — that a cross-organization open-weight pair (Qwen 3.6 Plus + Gemma 4 26B) **matches** the cross-family commercial reasoning ceiling at point-estimate κ = 0.80 (95% CI [0.76, 0.83], overlapping the commercial reasoning pair's [0.75, 0.82]) and **clearly exceeds every commercial within-family pair** (non-overlapping CIs) — supports an ensemble deployment recommendation for sovereign-cloud and privacy-conservative institutions. Repository: https://github.com/asukul/RAG-Eval-LLM-Judge

# 1. Introduction

Modern retrieval-augmented generation (RAG) systems sit on top of vector retrievers tuned over institutional corpora — academic libraries, parliamentary archives, medical records, legal collections. Practitioners want to answer "is this retrieval good?" — but they hit three blockers: there is no gold relevance set (TREC-style ground truth requires hundreds of expert hours per topic), generic IR benchmarks (BEIR, MS-MARCO) do not transfer, and the canonical workaround — **LLM-as-Judge** [zheng2023judge; faggioli2023perspectives; rahmani2024judges; upadhyay2024umbrela] — surfaces a downstream question that has not been answered by a public reproducible artifact: **which LLM judge?**

The recent literature has begun to address this question, but published comparisons cover either: (a) a single provider family [balog2025rankersjudges; farzi2025umbrela_other], (b) two-family pairs without ordinal weighting [thakur2025trecragsupport, who report κ = 0.60 for GPT-4o ↔ Llama-3.1-405B on 537 unweighted pairs], or (c) judge-vs-human only [han2025judgesverdict]. None — to our knowledge — provides a **public, reproducible artifact** containing:

1. A **multi-family pairwise κ matrix** (≥ 3 model families).
2. **Within-family controls** (≥ 1 within-family pair per family represented).
3. **Open-weight peers as first-class participants** (not as a single-bucket supplement).
4. **Bounded ordinal rubric** (the target evaluation regime for IR; not pairwise preference).
5. **≥ 500 paired observations** (enough for stable κ values).
6. **External validation against public NIST / BEIR human qrels** so the harness's outputs can be calibrated against an independent ground-truth.

We close that gap with **RAG-Eval-LLM-Judge**, a complete reproducible release covering all six. The within-corpus reference run is on **570 pairs** from the Iowa State University DSpace repository; external validation runs cover **537 pairs from TREC RAG 2024**, **300 from BEIR scifact**, and **300 from TREC-COVID**, all using identical 9-judge slate, identical rubric, identical pipeline.

This paper presents the **dataset, harness, and validation results as a primary contribution**; the empirical findings about specific judges (cross-family κ ≥ 0.75, cross-organization open-weight matching the commercial reasoning ceiling, kappa-paradox triangulation via Gwet AC2, etc.) are documented as **representative use cases** of what the released harness enables, and pointed at the companion long-form paper (`papers/long.md`) for full mechanistic and limitations analysis.

**Contributions.**
- **C1.** A reproducible 9,963-record cross-family LLM-judge agreement dataset spanning four corpora (one institutional, three public).
- **C2.** An open-source Python harness (`src/eval_llm_judge.py` + `validate_against_trec.py` + `fetch_msmarco_passages.py`) that re-runs the entire experiment from a clean checkout in ~13 hours wall and ~$95 paid-API spend.
- **C3.** A programmatic claim verifier (`src/verify_paper_claims.py`, 62 numerical checks) and a 25-test pytest suite, both runnable in < 5 seconds, that assert every number in the companion paper against the shipped JSONs.
- **C4.** A Croissant 1.0 metadata document and HuggingFace-style data card, supporting machine-readable dataset discovery and ML-platform integration.
- **C5.** A community **disclosure-template** standard for LLM-judged retrieval reporting: *"nDCG@10 = X.XX via [family]/[model]/[reasoning-config], N pairs, DATE"*. The release includes the template as a labeled figure (`figures/disclosure_template.png`).

# 2. Related Work and Position in the Benchmark Landscape

## 2.1 LLM-as-Judge for IR

Faggioli et al. [faggioli2023perspectives], Rahmani et al. [rahmani2024judges], and UMBRELA [upadhyay2024umbrela] established LLMs as practical IR-relevance assessors. UMBRELA in particular has been deployed at TREC scale on the 2024 RAG track and is the closest single-judge baseline to our work; we run it as an explicit comparison point (§5).

## 2.2 The 2025 inter-LLM-judge agreement cluster

A 2025 cluster has begun reporting inter-LLM agreement on relevance — Balog et al. [balog2025rankersjudges] (single-family), Thakur et al. [thakur2025trecragsupport] (two-family pair without ordinal weighting), Han et al. [han2025judgesverdict] (judge-vs-human only). None provides the combination we ship. Parallel evidence from educational measurement [jiao2026essayraters] shows frontier LLMs converging on quadratic-weighted κ with humans on 0-N ordinal scoring, supporting the cross-domain plausibility of our results.

## 2.3 Existing IR benchmarks vs this dataset

BEIR [thakur2021beir] is the dominant public IR benchmark suite but treats each LLM judge as a single-output black box; it does not surface inter-judge agreement structure. The TREC RAG 2024 track [macavaney2025trec_rag] publishes NIST human qrels but has **no published inter-LLM-judge agreement matrix** at the scale we provide. RAGAS [es2024ragas], ARES, RAGChecker, and BERGEN [es2024ragas; ru2024ragchecker; rau2024bergen] are RAG evaluation frameworks that report agreement *against humans*, not pairwise across LLMs. Our release **complements** rather than competes with these: it stress-tests the LLM-as-Judge primitive itself by pitting nine of them against each other and against the human qrels these other benchmarks use.

# 3. Dataset Description

## 3.1 Reference judge slate (9 judges, 5 families)

| # | Judge | Family | Reasoning | API route |
|---|---|---|---|---|
| 1 | Claude Opus 4.7 | Anthropic | yes | OpenRouter |
| 2 | Claude Sonnet 4.6 | Anthropic | yes | OpenRouter |
| 3 | GPT-5.5 (reasoning=low) | OpenAI | yes | OpenAI direct |
| 4 | GPT-4o (chat) | OpenAI | no | OpenAI direct |
| 5 | Gemini 3.1 Pro Preview | Google-commercial | yes (thinking) | OpenRouter |
| 6 | Gemini 2.5 Pro | Google-commercial | yes (thinking) | OpenRouter |
| 7 | DeepSeek V4 Pro | DeepSeek | yes | OpenRouter |
| 8 | Qwen 3.6 Plus | Open-weight | no | OpenRouter |
| 9 | Gemma 4 26B | Open-weight | no | OpenRouter |

Within-family controls: Anthropic (1↔2), OpenAI (3↔4), Google-commercial (5↔6), Open-weight cross-organization (8↔9). DeepSeek V4 Flash dropped due to OpenRouter free-tier 429 throttling. Model versions pinned at **2026-04-25**.

## 3.2 Within-corpus dataset (570 pairs, 5,130 records)

**Corpus**: Iowa State University Digital Repository (DSpace), 97,441 full-text PDFs → 1.03M chunks (≤500 words, 100-word overlap), embedded with Google Vertex `text-embedding-005` in Qdrant 1.13 HNSW. **Queries**: 57 from REFINE [Sukul, forthcoming], a corpus-faithful query-synthesis method, balanced across factoid/methodological/comparative/author-style intents over 20 topic clusters. **Pairs**: top-10 retrieval per query → 570 (query, document) pairs.

**Released artifacts** (in `results/within_corpus/`):
- 9 per-judge JSONs (`judge_<name>.json`, ~340 KB each, total 3 MB) holding the full 57-query × 10-retrieval structure with `judge_score` (int|null) per pair.
- 1 merged multijudge JSON (`multijudge_9judge_merged.json`, ~8 KB) with aggregates + 9×9 pairwise κ matrix.
- Schema documented in `results/within_corpus/README.md`.

**Retrieval determinism**: every per-judge file shares the same `(query_id, rank, point_id)` triple sequence, verified by the merge harness at run time (`retrieval_determinism_mismatches: 0` in the merged JSON). This means score arrays from different judges align positionally without any realignment, which is what makes the pairwise κ across judges directly computable.

## 3.3 External-validation datasets (1,137 pairs, 4,833 valid records)

| Corpus | n_pairs | Sample method | Qrels | Format |
|---|---:|---|---|---|
| TREC RAG 2024 | 537 | Stratified-balanced (135/134/134/134) over 4 NIST labels | NIST 2024 retrieval-qrels (20,283-pair full set) | 0–3 ordinal |
| TREC-COVID | 300 | First 300 from BEIR `GenericDataLoader` | BEIR-distributed TREC-COVID qrels | 0/1/2 mapped to 0/2/3 |
| BEIR scifact | 300 | First 300 from BEIR `GenericDataLoader` | BEIR scifact qrels (all-positive) | binary mapped to 0/3; precision-only |

**Released artifacts** (in `results/`):
- `trec-rag-2024_judges.json`, `trec-covid_judges.json`, `beir-scifact_judges.json` — merged 9-judge files with `pair_index`, `human_scores`, `per_judge_scores`, `per_judge_metadata`.
- `*_judges_p4-frontier.json` — 7-judge frontier-only subset (no open-weight) for ablation comparison.
- `*_judges_p4-supplement-openweight.json` — 2-judge open-weight subset (Qwen + Gemma).
- `*_kappa_vs_human.json` — pre-computed per-judge and ensemble κ vs human.
- `data/sample_537_pairs.tsv` (the stratified TREC RAG 2024 sample), `data/passages.json` (the 537 MS MARCO v2.1 passages extracted via `fetch_msmarco_passages.py`), `data/2024-retrieval-qrels.txt` (NIST official qrels).

## 3.4 Intra-judge self-consistency subset (1,350 records)

**50 pairs × 9 judges × 3 runs** (`results/intra_judge_consistency.json`). Pairs are sampled deterministically (`random.seed(42)`) from the 537-pair TREC RAG 2024 stratified set. Each `runs[i]` array preserves the raw integer/null scores from one independent invocation of the judge. Mean intra-judge κ across 9 judges = **0.93**, range 0.89 (Qwen) to 1.00 (Opus 4.7, Gemini 3.1 Prev, Gemini 2.5 Pro). All 9 judges have intra-κ > κ-vs-human (Δ range +0.51 to +0.70); no judge in the slate is "Rating Roulette unstable" [hong2025rating].

## 3.5 UMBRELA single-judge baseline

**`results/umbrela_baseline_trec_rag_2024.json`**: GPT-4o + UMBRELA verbatim prompt [upadhyay2024umbrela, §3] on the same 537-pair TREC RAG 2024 sample, for direct comparison against our 9-judge ensemble. Result: κ vs human = **0.4265**, n_valid = 537/537 (100%), 1.7 min wall, $0.30. Two independent runs produced 0.4265 and 0.4387 — a 0.012 κ jitter at temperature 0.0 confirming that judges are themselves non-deterministic and motivating the intra-judge subset.

## 3.6 Pre-computed analytical artifacts

- **`bootstrap_kappa_cis.json`** — 1,000-resample bootstrap 95% CIs (seed=42) on every reported headline κ value.
- **`gwet_ac2_alongside_kappa.json`** — quadratic-weighted Gwet AC2 alongside Cohen κ for kappa-paradox triangulation.
- **`bias_diagnostics.json`** — 5×5 family conditional-mean matrix + per-judge marginal score distributions + length-stratification (skipped: uniform truncation in stored previews).
- **`verification_log.txt`** — output of `src/verify_paper_claims.py`, the 62-check programmatic verifier.

# 4. Benchmark Methodology

## 4.1 Rubric and pipeline

**Rubric**: 0 = irrelevant, 1 = topical, 2 = partial, 3 = fully answers. Calibrated at prompt time with five worked examples. Documents truncated to **1,500 chars** [saito2023verbosity-style verbosity control]. Temperature 0.0.

**Pipeline** (`src/eval_llm_judge.py`): fans the 9 judges per pair via `concurrent.futures.ThreadPoolExecutor`, retries on transient errors with exponential backoff, saves per-judge JSONs plus a combined κ matrix. Two-phase deployment supported: 7-judge frontier preset + 2-judge open-weight supplement, merged via `merge_trec_covid_validation.py`-style harness. Total within-corpus run: 5,130 records, 154 + 175.6 minutes wall, USD 18.30.

## 4.2 Metrics

Per-judge **nDCG@10**, **P@5** (s ≥ 2 threshold), and **MRR** with raw judge scores as gain. Inter-judge agreement is pairwise **quadratic-weighted Cohen's κ** [thakur2024judges; cohen1968; fleiss1971], interpreted via Landis-Koch bands [landis1977]. **Gwet AC2** [gwet2008] reported alongside for triangulation under marginal-skew. Where a judge returned a missing/unparseable score (`null`), we exclude that pair from the judge's mean and from any κ pair involving that judge; aggregate metrics treat null as 0 with a corrected valid-only mean reported alongside.

## 4.3 Ensemble convention

All ensemble medians use the **upper-middle** convention: for a sorted vote vector of length *n*, we report `sorted_votes[n // 2]`. Adopted uniformly across TREC RAG 2024, BEIR scifact, and TREC-COVID for cross-corpus consistency (see `src/verify_paper_claims.py` for the canonical implementation).

## 4.4 Reproducibility-determinism

Every released JSON includes a `config` block with the random seed, model versions, prompt template, judge spec list, timestamp, and elapsed time. The verifier asserts that all 62 numerical claims reproduce to within 0.005 absolute error (the kappa-rounding tolerance) when the analyses are re-run on a clean checkout. The merge harness verifies retrieval determinism across runs by checking `(query_id, rank, point_id)` tuple alignment for every pair.

# 5. Validation Results (illustrative use case)

The following are headline numbers the dataset enables; they are documented in detail in the companion long paper (`papers/long.md`) and verified programmatically by `src/verify_paper_claims.py`.

## 5.1 Within-corpus 9-judge κ matrix

The 9-judge run on 570 ISU DSpace pairs produces a 9×9 pairwise κ matrix (Fig. 1) with **every off-diagonal cell ≥ 0.56** (substantial or moderate by Landis-Koch). Selected cells:

| Cell | κ | 95% CI | Interpretation |
|---|---:|---|---|
| Qwen ↔ Gemma 4 (cross-org open-weight) | **0.80** | [0.76, 0.83] | matrix-max (point estimate) |
| Sonnet ↔ GPT-5.5 (cross-family commercial) | 0.79 | [0.75, 0.82] | overlaps open-weight CI |
| Anthropic Opus ↔ Sonnet | 0.71 | [0.67, 0.74] | within-family ceiling |
| Google Gem 3.1 ↔ Gem 2.5 | 0.67 | [0.56, 0.77] | within-family Google |
| OpenAI GPT-5.5 ↔ GPT-4o | 0.63 | [0.59, 0.67] | within-family OpenAI |

Within-cluster mean κ (0.74) exceeds cross-cluster mean κ (0.68) by 0.06 (Welch t = 3.45, p = 0.002 two-sided; Mann-Whitney U one-sided p = 0.004). The cross-organization open-weight pair κ matches the cross-family commercial reasoning ceiling at point-estimate but the CIs overlap; the robust empirical claim is that **the cross-organization open-weight pair clearly exceeds every commercial within-family pair** (non-overlapping CIs).

## 5.2 Per-judge retrieval-metric spread

The same 570 retrieved documents yield nDCG@10 between **0.45 (Gemini 3.1 Prev) and 0.86 (Sonnet 4.6)** — a **1.9× spread** driven entirely by judge selection. This is the empirical hook for the disclosure-template proposal (C5).

## 5.3 External validation against NIST / BEIR human qrels

| Corpus | Ensemble κ vs human | 95% CI | Landis-Koch | UMBRELA single-judge baseline |
|---|---:|---|---|---:|
| TREC RAG 2024 (9-judge) | **0.4941** | [0.43, 0.56] | moderate, near-substantial | 0.4265 |
| TREC RAG 2024 (7-judge frontier) | 0.5187 | [0.46, 0.58] | moderate-substantial | — |
| TREC-COVID (9-judge) | 0.3447 | [0.24, 0.45] | fair, near-moderate | — |
| TREC-COVID (7-judge frontier) | 0.4462 | [0.35, 0.53] | moderate | — |
| BEIR scifact (9-judge ensemble precision-at-≥2) | **65.7%** | — | (κ undefined) | — |

**Both ensembles + best single judge (Sonnet 4.6, κ = 0.5123) outperform the UMBRELA single-judge baseline by 0.07–0.09 κ**, supporting the methodology contribution as a slate + ensemble (not prompt) advance.

## 5.4 Coverage divergence — content-domain reliability axis

| Judge | TREC RAG 2024 (n=537) | BEIR scifact (n=300) | TREC-COVID (n=300) |
|---|---:|---:|---:|
| Anthropic, OpenAI, Qwen, Gemma | 100% | ≥95% | ≥84% |
| DeepSeek V4 Pro | 39% | 84% | 83% |
| Gemini 3.1 Pro Preview | 24% | 60% | 44% |
| Gemini 2.5 Pro | 17% | 86% | 6% |

The **always-works 6-judge subset** (≥95% on all three) is Anthropic + OpenAI + Qwen + Gemma — and the 2 cheapest open-weight judges make this subset.

## 5.5 Intra-judge self-consistency

| Judge | Intra-K | κ vs human | Δ |
|---|---:|---:|---:|
| Mean across 9 | 0.93 | 0.36 | +0.57 |
| Min (Qwen 3.6 Plus) | 0.89 | 0.35 | +0.54 |
| Max (Opus, Gem 3.1, Gem 2.5) | 1.00* | varies | varies |

*Three judges report intra-K = 1.0000; for the Gemini judges this is partly an artifact of small valid-subset overlap. Open-weight judges have the lowest intra-K, supporting an ensemble (not single-call) deployment recommendation.

# 6. Reproducibility

Full reproduction from a clean checkout requires Python 3.11+, ~1 GB disk, and **API keys** for OpenAI + OpenRouter (~$95 total spend; ~13 hours wall).

```bash
git clone https://github.com/asukul/RAG-Eval-LLM-Judge
cd RAG-Eval-LLM-Judge
pip install -r requirements.txt
cp .env.template .env  # fill in OPENAI_API_KEY + OPENROUTER_API_KEY

# Step 0 — verify shipped numerical claims (free, ~3 sec)
py -3 -X utf8 src/verify_paper_claims.py
# → 62 pass / 0 fail / 0 warn

# Step 1 — re-run external validation harness end-to-end (~$50, ~7.5 h)
py -3 -X utf8 src/validate_against_trec.py --corpus trec-rag-2024 --judge-preset p4-frontier --max-pairs 537
py -3 -X utf8 src/validate_against_trec.py --corpus trec-rag-2024 --judge-preset p4-supplement-openweight --max-pairs 537
py -3 -X utf8 src/validate_against_trec.py --analyze trec-rag-2024
# → ensemble_median κ = 0.4941 (matches shipped value to 1e-4)

# Step 2 — UMBRELA single-judge baseline (~$0.30, ~2 min)
py -3 -X utf8 src/run_umbrela_baseline.py
# → κ vs human = 0.4265 (matches shipped value)

# Step 3 — intra-judge self-consistency (~$15, ~18 min)
py -3 -X utf8 src/intra_judge_consistency.py

# Step 4 — bootstrap CIs + Gwet AC2 + bias diagnostics (free, < 1 min)
py -3 -X utf8 src/bootstrap_kappa_cis.py
py -3 -X utf8 src/compute_gwet_ac2.py
py -3 -X utf8 src/bias_diagnostics.py

# Step 5 — pytest test suite (free, ~1.5 sec)
py -3 -m pytest -q
# → 25 passed
```

## 6.1 Re-extracting MS MARCO v2.1 passages from scratch

If `data/passages.json` is missing or you want to verify provenance:

```bash
py -3 -X utf8 src/fetch_msmarco_passages.py
# → streams 60 shards from drexalt/msmarco-2.1-segmented; ~38 min cold, ~$0
```

## 6.2 Within-corpus replication

The within-corpus 9-judge run requires Qdrant + ISU DSpace embeddings and is institution-specific; the released within-corpus per-judge JSONs (`results/within_corpus/`) are sufficient for any analysis that does not require the original 1.03M-chunk vector store.

# 7. Maintenance and Evolution

**Versioning.** Released as v1.0 (2026-04-29). Major version bumps for substantial corpus additions (e.g., LLMJudge 2025 benchmark, TREC RAG 2025 replication when qrels publish). Minor versions for additional analyses on existing corpora. Patch versions for numerical errata or schema fixes.

**Maintenance commitment.** Issues responded within 30 days; pull requests reviewed within 14 days for trivial fixes and 60 days for substantive additions. Hosting via GitHub (primary); a Zenodo mirror with DOI is planned at NeurIPS-camera-ready time if accepted.

**Roadmap (planned additions).**
- TREC RAG 2025 replication (when 2025 NIST qrels are published; the same MS MARCO v2.1 passage extractor works unchanged).
- Self-hosted open-weight replication (current data is API-routed; a small sovereign-cloud follow-up will measure on-prem inference economics directly).
- Provenance-labeled documents (for a strict self-preference test in the Panickssery 2024 sense).
- Additional public corpora (LLMJudge benchmark, TREC-COVID full set when expanded qrels publish).

# 8. Ethics, Licensing, and Responsible Use

**Licensing.** MIT for code; CC-BY-4.0 for data/figures/papers. Third-party redistributed inputs (MS MARCO v2.1 passages, NIST TREC RAG 2024 qrels) retain their original licenses; we redistribute only score arrays and metadata derived under terms of fair-use research.

**Privacy and consent.**
- ISU DSpace content is publicly published academic material (theses, conference proceedings, journal volumes); no PII beyond author names that already appear on public publications. Documents under licenses chosen by their authors and ISU Library.
- TREC RAG 2024 / TREC-COVID / BEIR scifact: pre-published by NIST / BEIR consortium; we redistribute only our derived score records.
- API responses from commercial models obtained under provider terms-of-service for research use; we redistribute only integer/null scores, not raw API response text.

**Intended uses.**
- Reproducibility of the LLM-as-Judge for IR meta-evaluation work.
- Methodology research on inter-rater agreement, bias diagnostics, and ensemble design.
- Practitioner guidance on judge selection for institutional RAG deployments.

**Out-of-scope uses.**
- The dataset is **not** a substitute for human-rater studies on sensitive corpora (medical, legal, security-critical). Use it as a methodology-vetting tool, not a deployment ground-truth.
- The "always-works 6-judge subset" finding applies to the three external corpora measured; institutions deploying on substantially different content (e.g., mathematical reasoning, code, multilingual non-English) should validate before relying on the recommendation.

**Known biases** (see DATA_CARD.md §"Discussion of Biases" for full discussion).

# 9. Limitations

- Single internal corpus for the within-corpus 9×9 matrix (ISU DSpace).
- Stratified-balanced TREC RAG 2024 sample is not directly comparable to natural-distribution-proportional samples (e.g., Thakur 2025).
- DeepSeek V4 Flash dropped due to OpenRouter throttling — incomplete within-DeepSeek-family pair.
- Missing-data accounting: aggregate metrics treat null as 0; valid-only means provided.
- §6.2 mechanism's R² = 0.928 between κ and dispersion+rank is partly mathematical (dispersion appears in the κ-formula numerator); see companion long paper §6 for the empirical-vs-mathematical disclosure.
- Open-weight cost claim ($0.30 vs $18 for commercial) uses API routing, not on-prem inference; the policy claim about sovereign-cloud is provisional pending self-hosted replication.

# 10. Conclusion

We release a complete reproducible benchmark + reference dataset for studying inter-LLM-judge agreement on bounded ordinal IR relevance judgment. The release closes a six-axis gap (multi-family × within-family controls × open-weight first-class × bounded ordinal × ≥ 500 obs × external validation) that no current public artifact addresses simultaneously. The accompanying empirical findings — cross-family κ ≥ 0.75 convergence, cross-organization open-weight matching the commercial reasoning ceiling, kappa-paradox-robust triangulation via Gwet AC2, and an "always-works 6-judge subset" recommendation — are documented as illustrative use cases of what the dataset enables. The full mechanism analysis, full validation tables with bootstrap CIs, and limitations discussion are in the companion long paper (`papers/long.md`); the workshop short version is in `papers/short.md`; the arXiv-ready preprint is in `arxiv/`.

# Acknowledgments

ISU Library and Digital Initiatives team for corpus access. Anthropic, OpenAI, Google DeepMind, DeepSeek, Alibaba, and Google (Gemma) model teams for the model snapshots. OpenRouter for the API gateway. NIST and the TREC RAG 2024 organizers for the public qrels [macavaney2025trec_rag]; Microsoft for the MS MARCO v2.1 corpus, mirrored on Hugging Face by `drexalt/msmarco-2.1-segmented`. The BEIR consortium [thakur2021beir] for the scifact and TREC-COVID datasets. The peer reviewers of round 1 (ChatGPT 5.5 deep-research and Claude Opus 4.7 deep-research-v2, 2026-04-29) for the critical feedback that drove the P1–P3 revision cycle (§5 C1 fixes, §6.2 reframing, path bug fixes, UMBRELA baseline addition, intra-judge self-consistency, bootstrap CIs, Gwet AC2 triangulation, bias diagnostics). API costs (~USD 95) covered by personal research budget. We acknowledge the use of Claude Code as a coding assistant for harness implementation, figure regeneration, and pytest test development; all experimental design, methodology, and analytical interpretation are the author's. Remaining errors are the author's.

# References

See `papers/references.md` for the verified bibliography (24 entries grouped by topic). The companion long paper has identical citations.

# Appendices (online supplementary)

- **A** — Full 9×9 κ matrix with bootstrap 95% CIs and AC2 alongside κ
- **B** — Full per-judge tables for all four corpora
- **C** — Full intra-judge raw score arrays (3 runs × 9 judges × 50 pairs = 1,350 records)
- **D** — Croissant 1.0 metadata document (`croissant.json`)
- **E** — HuggingFace data card (`DATA_CARD.md`)
- **F** — Cost / wall-time accounting and reproducibility checklist
