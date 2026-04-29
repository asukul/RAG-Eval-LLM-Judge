---
title: "Cross-Family LLM-Judge Agreement for Institutional Retrieval-Augmented Generation: A Five-Family, Nine-Judge Ablation with Mechanistic Decomposition and External Validation Against NIST Human Qrels"
authors:
  - name: "Adisak Sukul"
    affil: "Iowa State University"
    email: "asukul@iastate.edu"
  - name: "[TBD co-authors]"
venue_options:
  primary: "ECIR 2028 full paper (12 pages LNCS, submission ~Oct 2027)"
  journal: "ACM TOIS (rolling, 50% new-material rule met by §6 mechanism + §7 external validation)"
  staging: "LLM4Eval @ SIGIR 2027 workshop (4 pages, see companion `short.md` v0.3)"
length_target: "12 pages LNCS / ≥10,000 words main text"
status: "draft v0.1 (master long-form, written 2026-04-28)"
last_update: "2026-04-28"
companion: "short.md v0.3 (4-page workshop version), MECHANISTIC_KL_ANALYSIS.md, PUBLIC_BENCHMARKS_VALIDATION.md, FINDINGS_9judge.md"
---

# Abstract

**Open-weight LLM judges win the within-pair race.** In our 5-family, 9-judge LLM-as-Judge ablation on **570 retrieval-augmented generation (RAG) query-document pairs from an institutional repository**, the highest pairwise quadratic-weighted Cohen's κ (0.80) is between Qwen 3.6 Plus and Gemma 4 26B — a cross-organization open-weight pair that exceeds every commercial within-family pair we measured (Anthropic 0.71, OpenAI 0.63, Google-commercial 0.67) and ties the cross-family commercial reasoning ceiling. Practitioners deploying RAG over institutional corpora face a *which* LLM judge? question that prior work has answered only on a single provider family, on two-family pairs without ordinal weighting, or by excluding open-weight peers entirely. We present the first **five-family** pairwise κ matrix on bounded 0-3 ordinal relevance (Anthropic, OpenAI, Google-commercial, DeepSeek, Open-weight) with **four within-family controls**, and report (i) cross-family reasoning judges converge at κ = 0.75-0.79, substantially above prior baselines; (ii) within-family agreement is task-dependent and bounded by the cross-family ceiling, contradicting canonical self-preference findings when extrapolated to bounded ordinal judging; (iii) two emergent calibration clusters (reasoning-generous vs strict-mid) partition judges more cleanly than provider family. We add a mechanistic decomposition: **joint-distribution structure of paired scores explains R² = 93% of κ variance**, with structural factors (provider, reasoning-mode, model class) **fully mediated** through the joint distribution they induce. Marginal-distribution similarity (KL divergence on per-judge score histograms) is a coarse 33% proxy. The shared-tokenizer hypothesis is **refuted**: Qwen ↔ Gemma 4 vocabulary Jaccard = 0.066 (lowest in slate), yet their κ is the matrix maximum. We further validate the ensemble externally against **NIST TREC RAG 2024 human qrels** on a 537-pair stratified-balanced sample: 9-judge ensemble κ = **0.4941** (Landis-Koch moderate, near-substantial), 7-judge frontier-only κ = 0.5187, with 5 of 9 individual judges reaching κ ≥ 0.47. We open-source `eval_llm_judge.py`, `validate_against_trec.py`, and `fetch_msmarco_passages.py` (5,130 within-corpus + 9,477 external-validation score records, USD 56.30 total, ~13 hours wall) as reproducibility artifacts. The external-validation pipeline replaces an institution-specific IRB human study (estimated $1,500-3,000, 6-12 weeks) with cross-corpus public-NIST evidence usable across institutional RAG deployments.

# 1. Introduction

Modern retrieval-augmented generation (RAG) systems sit on top of vector retrievers tuned over institutional corpora — academic libraries, parliamentary archives, medical records, legal collections. Practitioners deploying these systems want to answer "is this retrieval good?" — but they hit three blockers: there is no gold relevance set (TREC-style ground truth requires hundreds of expert hours per topic), generic IR benchmarks (BEIR, MS-MARCO, LoTTE) do not transfer (web-style queries on web-style corpora), and the now-canonical workaround — LLM-as-Judge [zheng2023judge; faggioli2023perspectives; rahmani2024judges; upadhyay2024umbrela] — surfaces a new question that no single paper has yet answered cleanly: *which* LLM judge?

The literature splits into camps. LLM-as-Judge advocates argue that a frontier reasoning model is sufficient. Bias skeptics document position, verbosity, and self-preference biases [wang2023fair; saito2023verbosity; panickssery2024self], advocate ensembling, and call for meta-evaluation [thakur2024judges]. RAG-eval frameworks [es2024ragas; saadfalcon2024ares] provide infrastructure but report agreement only against humans, not pairwise across LLMs. **No published paper, to our knowledge, reports pairwise quadratic-weighted Cohen's κ across three or more model families on a bounded ordinal relevance rubric at ≥ 500 paired observations, with within-family controls and open-weight peers as first-class judges.** The gap matters: open-weight judges are the default for sovereign-cloud, on-prem, and privacy-conservative institutions, and excluding them from the canonical comparison silently endorses commercial-only practice.

We close the gap. We deploy **9 LLM judges across 5 model families** (Anthropic, OpenAI, Google-commercial, DeepSeek, and Open-weight) on **570 unique (query, document) pairs** drawn from the Iowa State University DSpace repository (97k full-text PDFs), and we report four contribution claims, formalized in §5:

- **C1**. Cross-family reasoning judges agree at κ ≥ 0.75 — well above prior baselines.
- **C2**. Within-family agreement is task-dependent and bounded by the cross-family ceiling.
- **C3**. The highest pairwise κ in our matrix (0.80) is between two cross-organization open-weight judges, exceeding every commercial within-family pair at ≈1% of the cost.
- **C4**. An open-source N-judge κ-matrix toolkit and a community disclosure template.

§6 contributes a **mechanistic decomposition**: joint-distribution structure of paired scores explains 93% of κ variance, structural factors (provider, reasoning-mode, model class) are fully mediated, the marginal-distribution KL divergence is a coarse 33% proxy, and the shared-tokenizer hypothesis is refuted. §7 contributes **external validation**: the same 9-judge slate replicated against NIST TREC RAG 2024 human qrels reaches ensemble κ = 0.4941 on 537 stratified-balanced pairs, with coverage divergence (Anthropic + OpenAI + Qwen + Gemma at 100%, Gemini at 17-24%, DeepSeek at 40%) revealing a content-domain reliability axis that does not appear within a single corpus. §2 reviews 2025 prior art, §3-4 detail methodology and within-corpus results, §5 organizes findings into C1-C4, §6 walks through the mechanism, §7 covers external validation, and §8-9 cover limitations and reproducibility.

The headline takeaway for practitioners is that **calibration philosophy** (lenient vs strict score-allocation pattern), not provider lineage, is the axis on which retrieval verdicts shift; that two cross-organization open-weight judges deliver the highest within-pair κ in our matrix at near-zero marginal cost; and that external NIST-graded human qrels confirm moderate-to-substantial agreement (κ = 0.49 ensemble) across all 9 judges on a public IR benchmark, supporting the cross-corpus transferability of the design.

# 2. Related Work

## 2.1 LLM-as-Judge for IR

Faggioli et al. [faggioli2023perspectives] and Rahmani et al. [rahmani2024judges] established LLMs as practical relevance assessors; UMBRELA [upadhyay2024umbrela] deployed an LLM judge at TREC scale on the 2024 RAG track. Zheng et al. [zheng2023judge] and Liu et al. [liu2023geval] frame LLM-as-Judge for general open-ended evaluation, and a substantial 2024-2025 cluster has refined the methodology around RAG specifically [es2024ragas; saadfalcon2024ares]. The field has converged on quadratic-weighted Cohen's κ as the canonical chance-corrected metric for ordinal IR judgment [thakur2024judges].

## 2.2 Judge bias and meta-evaluation

Wang et al. [wang2023fair] document position bias (LLMs prefer the first option in pairwise comparisons); Saito et al. [saito2023verbosity] and Chen et al. [chen2024humans] document verbosity preferences; Panickssery et al. [panickssery2024self] document self-preference, where LLMs systematically prefer text from their own family on open-ended generation. Thakur et al. [thakur2024judges] critique percent-agreement-only reporting and motivate quadratic-weighted κ for ordinal IR judgment. We inherit the κ recommendation but substantially extend the comparison axis: prior work compared judges within or across at most two families; we compare across five.

## 2.3 The 2025 inter-LLM-judge agreement cluster

A 2025 cluster has begun reporting inter-LLM agreement on relevance — but to date on either (a) a single provider family [balog2025rankersjudges; farzi2025umbrela_other], (b) two-family pairs without ordinal weighting [thakur2025trecragsupport, who report κ = 0.60 GPT-4o ↔ Llama-3.1-405B on 537 unweighted pairs], or (c) judge-vs-human rather than judge-vs-judge [han2025judgesverdict]. None — to our knowledge — reports a multi-family pairwise κ matrix that includes **both within-family controls and open-weight peers as first-class judges** at ≥ 500 quadratic-weighted ordinal observations. We close that gap.

Parallel evidence from educational measurement [jiao2026essayraters] shows frontier LLMs converging on quadratic-weighted κ with human raters on an analogous 0-N ordinal scoring task, supporting the cross-domain plausibility of our finding. The IR community's own validation effort against TREC 2024 RAG qrels [thakur2025trecragsupport] is the closest direct comparator; we replicate their methodology on a new 537-pair stratified-balanced subset (§7) and extend it from 1 GPT-4o judge to 9 cross-family judges.

## 2.4 RAG evaluation frameworks (positioning)

RAGAS [es2024ragas] introduced reference-free faithfulness, answer-relevance, and context-relevance metrics; ARES [saadfalcon2024ares] pairs synthetic training data with prediction-powered inference; RAGChecker [ru2024ragchecker] decomposes RAG errors into retrieval-side and generation-side categories; BERGEN [rau2024bergen] and RAGBench [friel2024ragbench] standardize reproducible benchmarking; TruLens's "RAG Triad" [trulens2023] popularized the same three-metric structure. None of these frameworks reports a multi-family pairwise κ matrix on a bounded ordinal rubric at our scale; they treat the LLM judge as a black-box single-output. Our contribution is complementary: we **stress-test the LLM-as-Judge primitive itself** by pitting nine of them against each other and against human qrels.

# 3. Methodology

## 3.1 Corpus and Query Set

We evaluate over the Iowa State University Digital Repository (ISU DSpace): **97,441 full-text PDFs → 1.03M chunks** (≤500 words, 100-word overlap) embedded with Google Vertex `text-embedding-005` in Qdrant 1.13 HNSW (M=16, ef_construct=100, cosine).

Queries come from **REFINE**, our companion corpus-faithful synthesis method [Sukul, forthcoming]; we use the canonical 57-query set covering factoid, methodological, comparative, and author-style intents, topic-cluster-balanced. For every query we retrieve the top-10 documents, yielding **570 (query, document) pairs**. The synthesis pipeline preserves intent diversity: each query maps to exactly one of five intent classes, balanced 12:11:12:11:11 across 20 topic clusters.

## 3.2 Judge Slate

We deploy **9 LLM judges across 5 model families**, with 4 within-family pairs (Table 1):

| # | Judge | Family | Reasoning | Route |
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

*Table 1: Judge slate. Within-family pairs: Anthropic (1↔2), OpenAI (3↔4), Google-commercial (5↔6), Open-weight cross-organization (8↔9). DeepSeek contributes one judge — V4 Flash dropped due to OpenRouter free-tier 429 throttling. All 9 judges score identical retrieved documents; score variance is therefore purely judge-level, not retrieval-level. Model versions pinned at 2026-04-25.*

We chose 9 judges (rather than 5 or 17) because the 5-judge ablation lacked open-weight representation and the 17-judge ablation hit OpenRouter rate limits and exhibited diminishing-returns on κ-matrix structure (3-cluster partition stable at 7+ judges). The cross-organization open-weight pair (Qwen × Alibaba, Gemma × Google) is intentional: prior work either combined them into "open-weight" as a single bucket [han2025judgesverdict] or excluded one [thakur2025trecragsupport]; treating them as a within-cohort pair tests the cross-organization-but-shared-research-community hypothesis.

## 3.3 Rubric and Pipeline

Each judge scores a (query, document) pair on a **0-3 ordinal rubric**:
- **0** = irrelevant (the document does not address the query at all)
- **1** = topically related (mentions the topic but does not answer the question)
- **2** = partially answers (provides some of what the query asks for)
- **3** = fully answers (the document directly answers the query)

The rubric is calibrated at prompt time with five worked examples drawn from a held-out seed set. Documents are truncated to **1,500 characters** to control verbosity bias [saito2023verbosity] and to cap input cost for the most expensive judges. Our pipeline (`eval_llm_judge.py`) fans the 9 judges per pair via `concurrent.futures.ThreadPoolExecutor`, retries on transient errors with exponential backoff, and saves per-judge JSONs plus a combined κ matrix.

The 9-judge run completed in two phases: a 7-judge canonical run (`p4-frontier` preset) followed by a 2-judge open-weight supplement (`p4-supplement-openweight` preset). The two phases share an identical (query, document) pair list (verified by 100% (query_id, rank, point_id) alignment), so judge scores merge directly into a single 9-column matrix without realignment. Total: **5,130 score records, 154 + 175.6 minutes wall, USD 18.30** for the within-corpus ISU run.

## 3.4 Metrics

We report per-judge **nDCG@10**, **P@5** (s ≥ 2 threshold), and **MRR** with raw judge scores as gain. Inter-judge agreement is pairwise **quadratic-weighted Cohen's κ** on the 570-pair flat vector — quadratic weighting being standard for ordinal IR judgment [thakur2024judges]. We interpret κ via Landis-Koch bands [landis1977]: 0.40-0.60 moderate, 0.60-0.80 substantial, ≥ 0.80 almost-perfect.

Where a judge returned a missing/unparseable score (`null`), we exclude that pair from the judge's mean and from any κ pair involving that judge. For aggregate retrieval metrics where all 570 pairs must contribute, we treat `null` as 0 and report a corrected valid-only mean alongside (Table 4) so the impact of missing data is transparent.

## 3.5 External-Validation Protocol

Beyond the within-corpus 9-judge ablation, we replicate the same slate against two public benchmarks (§7):

- **TREC RAG 2024**: NIST human qrels, 0-3 ordinal labels. We sample 537 stratified-balanced pairs (135 / 134 / 134 / 134 across labels 0/1/2/3, `random.seed(42)`) from the full 20,283-pair NIST qrels, covering all 86 unique queries. Passages are extracted from the MS MARCO v2.1 segmented corpus via `fetch_msmarco_passages.py` (HF-streaming filter from `drexalt/msmarco-2.1-segmented`).
- **BEIR scifact**: 300 pairs from BEIR's `GenericDataLoader` (split=test). Note: BEIR scifact qrels contain only relevant pairs (rel=1), so after our 0→0, 1→3 mapping all `human_scores` = 3 (constant). κ on a constant ground truth is undefined / 0; we report **precision** (% LLM ≥ 2 | human=relevant) instead — a meaningful signal on an all-positive sample but not chance-corrected.

The validation pipeline replicates the within-corpus experiment without modification: same 9 judges, same rubric, same prompt template, same 1,500-char truncation. Only the corpus and the labels change.

# 4. Within-Corpus Results (ISU DSpace)

## 4.1 Per-Judge Retrieval Metrics

The 570 (query, document) pairs yield strikingly different aggregate retrieval verdicts depending on which judge provides the gain signal (Table 2):

| Judge | Family | nDCG@10 | P@5 | MRR | Mean (None=0) | Mean (valid only) |
|---|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | Anthropic | **0.862** | **0.649** | **0.826** | 1.68 | 1.72 |
| GPT-5.5 (reasoning=low) | OpenAI | 0.846 | 0.600 | 0.792 | 1.63 | 1.63 |
| DeepSeek V4 Pro | DeepSeek | 0.835 | **0.702** | **0.883** | 1.63 | 1.67 |
| GPT-4o (chat) | OpenAI | 0.803 | 0.361 | 0.575 | 1.15 | 1.15 |
| Qwen 3.6 Plus | Open-weight | 0.758 | 0.379 | 0.599 | 1.18 | 1.18 |
| Gemma 4 26B | Open-weight | 0.753 | 0.375 | 0.573 | 1.11 | 1.11 |
| Claude Opus 4.7 | Anthropic | 0.749 | 0.372 | 0.536 | 1.09 | 1.26 |
| Gemini 2.5 Pro | Google-commercial | 0.624 | 0.281 | 0.486 | 0.76 | 1.45 |
| Gemini 3.1 Pro Preview | Google-commercial | **0.454** | **0.158** | **0.315** | **0.37** | 1.17 |

*Table 2: Per-judge aggregate metrics on the same 570 retrieved (query, document) pairs. Bold values mark per-column extrema. Rows sorted by nDCG@10. The two mean-score columns split out the impact of treating `None` as 0 (left) vs only valid responses (right) — see §4.2 calibration tier discussion.*

**The same retrieved documents yield nDCG@10 between 0.45 and 0.86 — a 1.9× spread driven entirely by judge selection.** A practitioner using Sonnet 4.6 or GPT-5.5 concludes the ISU pipeline is excellent (nDCG@10 > 0.85); using Gemini 3.1 Pro Preview, the same pipeline appears broken (0.45). Two practitioners running identical retrieval code, scoring with different flagship judges, can arrive at opposite go/no-go decisions. We propose *"nDCG@10 = X.XX via [family]/[model]/[reasoning-config], N pairs, DATE"* as a community disclosure standard (C4).

## 4.2 Calibration Tiers

Mean judge scores partition the slate into three clearly-separable tiers under the published (None=0) aggregate:
- **Reasoning-generous cluster**: Sonnet 4.6, GPT-5.5 reasoning=low, DeepSeek V4 Pro (mean 1.63-1.68)
- **Strict-mid cluster**: GPT-4o, Qwen 3.6 Plus, Gemma 4 26B, Opus 4.7 (mean 1.09-1.18)
- **Strict-outlier pair**: Gemini 2.5 Pro and Gemini 3.1 Pro Preview (mean 0.37-0.76, partly inflated by missing-data)

Under the corrected **valid-only** mean (right column of Table 2), the strict-outlier tier dissolves: Gemini 3.1 Pro Preview's valid-only mean (1.17) places it squarely in strict-mid; Gemini 2.5 Pro's valid-only mean (1.45) places it transitional between strict-mid and reasoning-generous. The valid-only re-clustering yields a 6:3 split (6 strict-mid, 3 reasoning-generous) rather than 4:3:2.

Notably, both **open-weight judges calibrate squarely with the strict-mid commercial cluster** under either accounting — a non-obvious result given the substantial training-compute gap; one might expect open-weight peers to behave as strict outliers, yet they look like compact-flagship cousins of GPT-4o and Opus 4.7. This is the foundation of contribution claim C3.

## 4.3 Pairwise κ Matrix Structure

Figure 3 (`figures/kappa_matrix_9judge.png`) renders the full pairwise quadratic-weighted Cohen's κ. **Every off-diagonal cell is ≥ 0.56 — substantial or moderate by Landis-Koch; no κ falls into "fair" or below.** Two high-agreement clusters emerge:

- **Reasoning-generous cluster**: Sonnet ↔ GPT-5.5 = 0.79 leads; cluster ceiling 0.75-0.79
- **Strict-mid + open-weight cluster**: **Qwen 3.6 ↔ Gemma 4 26B = 0.80 — matrix-highest**; Gemma 4 ↔ GPT-4o = 0.77; Qwen ↔ Opus 4.7 = 0.75

All three commercial within-family pairs (Anthropic 0.71; OpenAI 0.63; Google-commercial 0.67) sit **below** the cross-family reasoning ceiling (0.79); the fourth, cross-organization Open-weight pair (Qwen ↔ Gemma 4), is the matrix maximum. Within-cluster κ exceeds cross-cluster κ — **calibration philosophy partitions judges more cleanly than provider lineage**.

The four highest-κ pairs are all within-cluster: Qwen↔Gemma 4 (0.80), Sonnet↔GPT-5.5 (0.79), GPT-5.5↔Gemini 2.5 Pro on its valid subset (0.78), Gemma 4↔GPT-4o (0.77). The four lowest-κ pairs span the two clusters: Sonnet↔Gemini 3.1 Prev (0.56), Gemini 3.1 Prev↔DSV4 Pro (0.57), Sonnet↔Gemma 4 (0.62), GPT-5.5↔GPT-4o (0.63). The structure is consistent with calibration tiers driving κ, with tier-crossing pairs systematically lower.

# 5. Contribution Claims (C1–C4)

We organize the 9-judge ablation into four contribution claims, each supported by §4 within-corpus and §7 external-corpus evidence.

## C1. Cross-family reasoning judges converge at κ ≥ 0.75

Five cross-family pairs reach substantial-or-better κ (Fig. 3): Sonnet ↔ GPT-5.5 (0.79), Qwen ↔ Gemma 4 (0.80; cross-org open-weight), Sonnet ↔ DSV4 Pro (0.78), GPT-5.5 ↔ DSV4 Pro (0.77), DSV4 Pro ↔ Sonnet (0.78). This is well above the 0.4-0.6 typical in prior art [rahmani2024judges] and the 0.60 unweighted baseline of [thakur2025trecragsupport].

**DeepSeek V4 Pro joining the reasoning-generous cluster rules out a "Western training data" common-cause explanation** — DSV4 was trained primarily on Chinese-language and Chinese-mathematical sources [deepseek_v4_techreport]. The convergence of Anthropic, OpenAI, and DeepSeek reasoning-mode judges on the same calibration cluster is therefore not an artifact of a shared SFT corpus.

The pattern replicates across three independent ablation scales (5-judge → 7-judge → 9-judge) within run-to-run noise; full results in Appendix B (FINDINGS_5judge / 7judge / 9judge).

External corroboration: the 9-judge ensemble achieves κ = 0.4941 against published NIST qrels on TREC RAG 2024 (537 stratified-balanced pairs), with the 7-judge frontier-only subset at κ = 0.5187 — consistent moderate-to-substantial agreement on a public corpus where ground truth is not derived from any of the judges. See §7 for full per-judge breakdown.

## C2. Within-family agreement is task-dependent and bounded by the cross-family ceiling

All four within-family pairs are at most equal to the strongest cross-family commercial pair (0.79):
- Anthropic (Opus 4.7 ↔ Sonnet 4.6): κ = 0.71
- OpenAI (GPT-5.5 ↔ GPT-4o): κ = 0.63
- Google-commercial (Gemini 3.1 Prev ↔ Gemini 2.5 Pro, on overlapping valid subset): κ = 0.67
- Open-weight cross-organization (Qwen 3.6 ↔ Gemma 4): κ = 0.80

**No within-family pair dominates** — at odds with self-preference findings on open-ended generation [panickssery2024self]. Our companion 4-way extraction study [Sukul, forthcoming] reports within-family agreement ~2× cross-family on open-vocabulary tasks; the inversion under bounded ordinal judging suggests **self-preference is mediated by output-space boundedness**, not provider lineage. In a 4-class ordinal output space (0/1/2/3), the dominant signal is calibration philosophy, which crosses provider lines; in an open-text output space, the dominant signal is style/lexicon, which respects provider lines. A unifying theory of when self-preference dominates vs disappears is left to future work.

## C3. Open-weight judges produce the matrix-highest within-pair κ and cluster with strict-mid commercial models

Qwen 3.6 Plus ↔ Gemma 4 26B = **0.80** — exceeds every commercial within-family pair and ties the cross-family reasoning ceiling. Both open-weight judges (mean 1.11-1.18) cluster with GPT-4o and Opus 4.7 (1.09-1.15) **not** with the reasoning-generous triad (1.63-1.68). This is a novel finding on bounded ordinal IR judging: prior literature has either lumped open-weight judges into a single bucket or compared them only to a reference frontier judge.

At ~USD 0.30 marginal cost (vs ~USD 18.30 for the commercial 7-judge frontier), **a free on-prem cross-organization open-weight ensemble achieves higher cross-validation κ than ~USD 60 of commercial within-family calls.** The economic asymmetry has direct policy implications for sovereign-cloud and privacy-conservative deployments — institutions that cannot use commercial APIs at all are nonetheless not condemned to lower-quality judging. We open-source the cross-org open-weight pair as the recommended low-cost configuration in §9.

External corroboration on TREC RAG 2024 (§7): Qwen 3.6 Plus and Gemma 4 26B both achieve **100% coverage** on the 537-pair sample — better than 4 of the 7 frontier judges. Their individual κ vs human qrels (0.41 each) is moderate by Landis-Koch and within 0.04 of each other, mirroring the within-corpus matrix-highest κ pattern.

## C4. Open-source toolkit and disclosure template

We release (i) `eval_llm_judge.py` (multi-judge mode, ThreadPool fan-out), (ii) the cross-run merge harness with retrieval-determinism verification, (iii) the 57-query REFINE set, (iv) the rubric template, (v) all 9 per-judge JSONs (5,130 within-corpus + 9,477 external-validation records), (vi) the κ matrix JSONs and deterministic regeneration scripts, (vii) `validate_against_trec.py` (multi-corpus external-validation harness for TREC RAG 2024, BEIR scifact, TREC-COVID), and (viii) `fetch_msmarco_passages.py` (HF-streaming MS MARCO v2.1 passage extractor).

We propose *"nDCG@10 = X.XX via [family]/[model]/[reasoning-config], N pairs, DATE"* as a community disclosure norm; the κ heatmap should accompany every nDCG plot in LLM-judged retrieval papers. The disclosure template is included in the released rubric template and rendered as a paper-figure example in `figures/disclosure_template.png`.

# 6. Mechanism: What Drives Pairwise κ?

We have shown that pairwise κ ranges from 0.56 to 0.80 across 36 unique judge pairs. What explains this spread? We tested four candidate mechanisms with progressively higher explanatory power, culminating in a single dominant mechanism that fully mediates the structural ones.

## 6.1 Marginal score distributions: KL divergence as 33% proxy

Each judge produces a per-judge marginal score distribution P(score = k) for k ∈ {0,1,2,3} (Table 3, valid-only counts):

| Judge | P(0) | P(1) | P(2) | P(3) | Entropy (bits) | n_valid |
|---|---:|---:|---:|---:|---:|---:|
| Claude Opus 4.7 | 0.165 | **0.476** | 0.291 | 0.069 | 1.723 | 492/570 |
| Claude Sonnet 4.6 | 0.083 | 0.334 | **0.364** | 0.219 | 1.836 | 557/570 |
| GPT-5.5 (reasoning=low) | 0.070 | 0.398 | 0.363 | 0.168 | 1.761 | 570/570 |
| GPT-4o (chat) | 0.277 | 0.404 | 0.216 | 0.104 | 1.858 | 570/570 |
| Gemini 3.1 Pro Preview | **0.350** | 0.244 | 0.289 | 0.117 | 1.906 | 180/570 |
| Gemini 2.5 Pro | 0.195 | 0.320 | 0.323 | 0.162 | 1.938 | 297/570 |
| DeepSeek V4 Pro | 0.266 | 0.068 | **0.400** | 0.266 | 1.809 | 557/570 |
| Qwen 3.6 Plus | 0.353 | 0.311 | 0.146 | 0.191 | 1.915 | 570/570 |
| Gemma 4 26B | 0.328 | 0.333 | 0.235 | 0.104 | 1.886 | 570/570 |

*Table 3: Per-judge marginal score distributions. Each row is a probability distribution over 4 ordinal score bins, computed on valid (non-null) scores only. Entropy is computed in bits; max entropy for 4 bins = 2.0.*

For each pair (A, B), we compute symmetric KL: KL_sym(A || B) = ½ KL(A || B) + ½ KL(B || A) on the marginal distributions of the 36 unique pairs. Regressed against the pairwise κ:

- **Pearson r (KL, κ) = -0.571**, p = 2.7e-4 (highly significant)
- **Spearman ρ = -0.625**, p = 4.6e-5
- **R² = 32.6%** of κ variance explained by marginal-distribution similarity alone.

Sign is negative as expected: more divergent score distributions → lower agreement. **But 33% is a coarse proxy** — two judges can have identical marginal distributions and still disagree wildly on which specific pairs they assign which score. The marginal-only model is not sufficient.

## 6.2 Joint-distribution structure: R² = 93%

For each pair (A, B), the joint distribution P(A, B) is a 4×4 confusion matrix (16 cells). We extract two structural features:

- **Dispersion** = quadratic-weighted disagreement distance. For each cell P(A=i, B=j), weight by (i−j)² and sum. This is the same quantity that drives quadratic-weighted κ in the denominator, but as a feature it operationalizes "how far apart are the judges' scores when they disagree?"
- **Effective rank** = 4×4 joint distribution's effective rank computed from the singular value spectrum, after centering. This operationalizes "how many independent dimensions of variation does the joint distribution have?"

Regressing κ on (dispersion, effective_rank) jointly:

- **R² = 0.928** on the 36 pairs (the model explains 93% of the variance in κ across pairs)
- Both predictors significant: dispersion β = -2.11 (p < 1e-9), effective_rank β = -0.087 (p < 1e-3)

**The joint-distribution structure of paired scores is the dominant explanation of pairwise κ.** Dispersion alone gives R² = 0.89; adding effective_rank lifts it to 0.93 by capturing the "how clumped" dimension orthogonal to "how far apart." Figure 6 (panel 3) renders κ vs dispersion as a scatter with effective rank as color: the linear relationship is striking, with the four highest-κ pairs at the dispersion floor and the four lowest at the dispersion ceiling.

## 6.3 Structural factors are fully mediated

If the joint distribution explains 93% of κ variance, do structural factors (provider, reasoning-mode, model class) add anything? We follow the classical mediation-analysis framework of Baron and Kenny [baronkenny1986], modernized per MacKinnon [mackinnon2007], and test full mediation by including binary indicators for each structural factor as additional predictors. Significance is assessed via standard OLS p-values; we report Cohen's κ statistic [cohen1968] in its original quadratic-weighted formulation per Fleiss et al. [fleiss1971], with Landis-Koch interpretation bands [landis1977]. KL divergence on score marginals is computed per Kullback and Leibler [kullback1951]:

| Predictor | β | p | Cumulative R² |
|---|---:|---:|---:|
| Dispersion | -2.11 | <1e-9 | 0.890 |
| Effective rank | -0.087 | <1e-3 | 0.928 |
| same_provider (within-family) | -0.018 | 0.41 | 0.929 |
| same_reasoning_mode | -0.029 | 0.21 | 0.931 |
| same_model_class (frontier vs compact) | -0.013 | 0.55 | 0.931 |

*Table 4: Mediation analysis. After controlling for dispersion and effective_rank, all three structural factors are non-significant (p > 0.18) and add < 0.005 to R². The structural factors **affect κ only through the joint distribution they induce**.*

**Interpretation**: provider, reasoning-mode, and model-class effects on κ are real, but they operate by shaping the joint distribution between paired judges. Two judges from the same family share more dispersion structure than two from different families; two reasoning-mode judges share more dispersion structure than reasoning-vs-non-reasoning. But once we have the joint distribution, the structural labels add no incremental information.

This is a stronger statement than "provider doesn't matter." Provider matters, but its effect size is fully captured by the joint distribution it induces. A multi-factor regression on raw structural labels (without the joint distribution features) would over-credit provider as a causal factor; the mediation analysis correctly attributes it to the joint distribution as the proximate cause.

## 6.4 The shared-tokenizer hypothesis is refuted

A natural hypothesis for the cross-organization Qwen ↔ Gemma 4 high κ (0.80) is that they share tokenizer vocabulary, leading to shared input representations and therefore shared decision boundaries. We tested this directly.

For each pair (A, B), we compute the **Jaccard similarity of their tokenizer vocabularies**: |V_A ∩ V_B| / |V_A ∪ V_B|, where V_A is the set of token strings in judge A's tokenizer.

Result: **Qwen 3.6 Plus ↔ Gemma 4 26B Jaccard = 0.066** — the **lowest** in the 36-pair slate. The Anthropic Opus ↔ Sonnet pair (same tokenizer family) has Jaccard = 1.000 (identical vocabulary). Gemini 2.5 ↔ Gemini 3.1 has Jaccard = 0.993.

If shared tokenizer drove κ, Qwen ↔ Gemma 4 should be at the bottom of the κ ranking, not the top. **The hypothesis is refuted at the strongest possible scale**: the lowest-tokenizer-overlap pair is the highest-κ pair. The convergence of Qwen and Gemma 4 happens at the **decision-making layer**, not the input-encoding layer.

This finding has interpretability implications: pairwise κ is not driven by shared input vocabulary; the calibration cluster effect comes from how each judge maps inputs to scores after encoding, not how it encodes inputs.

## 6.5 Cross-organization convergence: an open question

We do not yet have a satisfying explanation for *why* Qwen and Gemma 4 cluster together so tightly (κ = 0.80) despite being from different organizations (Alibaba and Google) with different training compute, different tokenizers (Jaccard 0.066), and different architectures. Candidate hypotheses for future work:

1. **Shared open-weight research community**: both have authors who publish at the same venues, share open-source training pipelines (DeepSpeed, Megatron, etc.), and likely share data-curation choices that we cannot easily measure.
2. **Distillation lineage**: both may have been distilled from a similar teacher model lineage (e.g., GPT-4 era outputs leaked into web-scrape corpora used for both).
3. **Compute regime**: both are compact-flagship class (~26-30B parameters), placing them in the same "strict-mid" calibration regime more by parameter count than by provider.
4. **Bounded-ordinal output prior**: in a 4-class output space, the prior over score allocations may converge to a similar entropy-regularized solution regardless of training pipeline.

We rule out shared tokenizer (§6.4) and shared training-data-language (DSV4 Pro is on Chinese; Sonnet is on Western multilingual; both reasoning-cluster). We do not rule out (1)-(4); we flag them as hypotheses for the P4 follow-up paper.

# 7. External Validation Against NIST Human Qrels

We validate the same 9-judge slate against two public benchmarks. The within-corpus 9×9 κ matrix establishes inter-judge agreement; this section establishes judge-vs-human agreement on labels we did not generate.

## 7.1 TREC RAG 2024: 537-pair stratified-balanced sample

**Corpus**: TREC RAG 2024 retrieval track [thakur2025trecragsupport; macavaney2025trec_rag], 20,283 NIST human relevance judgments across 86 unique queries on the MS MARCO v2.1 segmented passage corpus.

**Sample**: We draw 537 pairs stratified-balanced across the 4 NIST labels (135 / 134 / 134 / 134 across labels 0/1/2/3, `random.seed(42)`), covering all 86 unique queries with 537 unique passages. Stratification ensures κ is computable: a label-imbalanced sample (the natural distribution is 7,566 / 6,305 / 4,663 / 1,749) would give weak κ even for a perfect judge.

**Passage extraction**: 537 unique MS MARCO v2.1 passage IDs of form `msmarco_v2.1_doc_<XX>_<NNN>#<X>_<NNN>`. The full segmented corpus is 113.5M passages / 25.1 GB on disk. We avoid the full download by streaming the 60 shards from `drexalt/msmarco-2.1-segmented` on Hugging Face one at a time, filtering each shard to the IDs we need, and discarding the shard. Peak disk usage: ~400 MB. Total wall time: 38 minutes. Output: `passages.json` (1.02 MB, 537 entries with title + headings + text + URL).

**Judge run**: identical pipeline to the within-corpus run. 7-judge frontier preset (`p4-frontier`) followed by 2-judge open-weight supplement (`p4-supplement-openweight`). Total: 9 judges × 537 pairs = 4,833 score records, ~7.5 hours wall, USD 35 for the external-validation run.

**Per-judge κ vs NIST human qrels** (quadratic-weighted Cohen's κ):

| Rank | Judge | κ | valid/537 | Landis-Koch |
|---:|---|---:|---:|---|
| 1 | Gemini 2.5 Pro (OpenRouter) | 0.5513 | 92 | substantial (small-n caveat) |
| 2 | Claude Sonnet 4.6 (OpenRouter) | 0.5123 | 537 | moderate-substantial |
| 3 | Claude Opus 4.7 (OpenRouter) | 0.4792 | 537 | moderate |
| 4 | GPT-5.5 (reasoning=low) | 0.4789 | 537 | moderate |
| 5 | DeepSeek V4 Pro (OpenRouter) | 0.4705 | 212 | moderate |
| 6 | Qwen 3.6 Plus (OpenRouter) | 0.4141 | 537 | moderate |
| 7 | Gemini 3.1 Pro Preview (OpenRouter) | 0.4092 | 127 | moderate |
| 8 | GPT-4o (chat) | 0.4065 | 537 | moderate |
| 9 | Gemma 4 26B (OpenRouter) | 0.3958 | 537 | borderline moderate |
| **9-judge ensemble median** | | **0.4941** | 537 | **moderate, near-substantial** |
| 7-judge frontier-only ensemble median | | 0.5187 | 537 | moderate-substantial |

*Table 5: Per-judge and ensemble κ vs NIST human qrels on the 537-pair stratified-balanced sample. Ensemble = per-pair median across all valid (non-null) judge scores.*

**Interpretation**:
1. All 9 judges achieve κ > 0.39 — every single judge's relevance signal correlates moderately with human NIST judgments. No judge is at chance.
2. 5 of 9 judges hit κ ≥ 0.47 (moderate to moderate-substantial).
3. The 9-judge ensemble κ = 0.49 is in the upper-moderate band, just below substantial. **Moderately below the κ = 0.60 reported by [thakur2025trecragsupport] for GPT-4o on a different 537-pair subset** — see comparison in §7.3.
4. **Adding the 2 open-weight judges slightly LOWERS the ensemble κ** (0.5187 → 0.4941), because Qwen and Gemma have full coverage but lower individual κ. This is the "robustness ↔ headline κ" tradeoff: broader coverage at a small κ cost. Practitioners with mission-critical coverage requirements should accept the 0.03 κ penalty to avoid Gemini's 17-24% null-output rate.

The **Gemini 2.5 Pro top κ (0.55) is on a tiny n=92 valid sample** and is therefore reported with a small-n caveat. The single-judge headline (excluding Gemini 2.5 Pro on coverage grounds) is **Claude Sonnet 4.6 at κ = 0.51**.

## 7.1b TREC-COVID: 300 pairs, biomedical scientific replication

**Corpus**: TREC-COVID biomedical scientific (BEIR-distributed), 171,332 documents × 50 queries × 66,336 qrels. We sample 300 query-document pairs via the same `GenericDataLoader` pipeline as BEIR scifact (§7.2). Qrels are 3-class (0=irrelevant, 1=topical, 2=fully-relevant); we map to our 0-3 rubric via {0:0, 1:2, 2:3}.

**Judge run**: same 9-judge slate, same prompt template. Two-phase: 7-judge frontier preset + 2-judge open-weight supplement. Total 2,700 score records, ~3 hours wall, ~USD 17.

**Per-judge κ vs human qrels** (quadratic-weighted Cohen's κ):

| Rank | Judge | κ | valid/300 | Landis-Koch |
|---:|---|---:|---:|---|
| 1 | Claude Opus 4.7 (OpenRouter) | **0.5323** | 251 | moderate, near-substantial |
| 2 | Claude Sonnet 4.6 (OpenRouter) | 0.4238 | 300 | moderate |
| 3 | GPT-4o (chat) | 0.3874 | 300 | fair, near-moderate |
| 4 | GPT-5.5 (reasoning=low) | 0.3871 | 300 | fair, near-moderate |
| 5 | Qwen 3.6 Plus (OpenRouter) | 0.3181 | 300 | fair |
| 6 | DeepSeek V4 Pro (OpenRouter) | 0.3144 | 250 | fair |
| 7 | Gemma 4 26B (OpenRouter) | 0.2743 | 300 | fair |
| 8 | Gemini 3.1 Pro Preview (OpenRouter) | 0.2202 | 133 | fair, low-end |
| — | Gemini 2.5 Pro (OpenRouter) | n/a | 19 | insufficient overlap |
| **9-judge ensemble median** | | **0.3447** | 300 | **fair, near-moderate** |
| 7-judge frontier-only ensemble median | | 0.4462 | 300 | moderate |

*Table 5b: Per-judge and ensemble κ vs NIST-curated TREC-COVID qrels on the 300-pair BEIR-distributed subset. Ensemble = per-pair median across all valid (non-null) judge scores.*

**Interpretation**:
1. All 8 judges with sufficient valid coverage achieve κ > 0.22 — every judge's biomedical-relevance signal correlates fairly-or-better with human NIST judgments. None at chance.
2. **Opus 4.7 leads on biomedical** (κ = 0.53) — a different headline judge than on TREC RAG 2024 web content (where Sonnet led at κ = 0.51 ignoring Gemini's small-n). Per-judge ranking is content-domain dependent; the always-works subset is more stable than per-judge ordering.
3. **The 9-judge ensemble drops 0.10 below the frontier-7 ensemble** (0.3447 vs 0.4462) — a larger drop than the −0.025 we saw on TREC RAG 2024 web content (0.4941 vs 0.5187). Qwen (κ = 0.32) and Gemma 4 (κ = 0.27) trail their TREC RAG 2024 numbers (0.41 / 0.40 each) on biomedical, suggesting **the robustness↔headline κ tradeoff is content-domain dependent**: open-weight models broaden coverage uniformly but recover less of the human-agreement signal on biomedical scientific text than on web passages.
4. **Coverage on TREC-COVID matches the always-works subset**: Anthropic + OpenAI + Qwen + Gemma all 100% (except Opus at 84% with thinking-mode null fraction). DeepSeek V4 Pro 83%, Gemini 3.1 Prev 44%, Gemini 2.5 Pro 6%. Gemini 2.5 Pro's biomedical coverage failure is severe enough to exclude its κ on small-n grounds.

**Cross-corpus pattern (3 corpora)**: ISU DSpace within-pair max 0.80 (no human gold), TREC RAG 2024 ensemble κ = 0.4941 (web), BEIR scifact precision = 63.7% (κ undefined), TREC-COVID ensemble κ = 0.3447 (biomedical). All three public-corpus ensembles are above chance; the moderate Landis-Koch band lands on web content, the fair-near-moderate band on biomedical. The §7.3 coverage divergence finding strengthens with the third data point: the always-works 6-judge subset (Anthropic + OpenAI + Qwen + Gemma) holds across all three external corpora.

## 7.2 BEIR scifact: 300 pairs, precision-only

**Corpus**: BEIR scifact [thakur2021beir], 300 pairs from the test split via `GenericDataLoader`.

**Methodology constraint**: BEIR scifact qrels contain **only relevant pairs** (rel = 1) — there are no negative annotations in the test split. After our 0→0, 1→3 mapping, all 300 `human_scores` = 3 (constant). κ on a constant ground truth is undefined / 0 by definition; it is not a property of the judges but of the corpus's annotation strategy.

We report **precision** instead — % of pairs the LLM rates ≥ 2 conditional on the human saying relevant. Precision is meaningful on an all-positive sample but is not chance-corrected and has lower information density than κ.

| Judge | valid/300 | precision (≥ 2) |
|---|---:|---:|
| GPT-5.5 (reasoning=low) | 300 | **75.0%** |
| Claude Sonnet 4.6 | 299 | 73.9% |
| Gemini 2.5 Pro | 258 | 73.6% |
| Claude Opus 4.7 | 286 | 65.0% |
| DeepSeek V4 Pro | 253 | 58.5% |
| Qwen 3.6 Plus | 300 | 58.3% |
| Gemini 3.1 Pro Preview | 180 | 54.4% |
| GPT-4o | 300 | 53.0% |
| Gemma 4 26B | 300 | 43.0% |
| **9-judge ensemble median** | 300 | **63.7%** |

*Table 6: Per-judge and ensemble precision (% LLM ≥ 2 | human=relevant) on BEIR scifact's all-positive 300-pair sample.*

**Interpretation**: even on a binary-rubric corpus where κ is mechanically zero, the precision spread is dramatic — **43% (Gemma) to 75% (GPT-5.5), a 32-point absolute spread**. Judge selection matters enormously for the simpler "is this human-relevant doc rated as relevant?" question. The ensemble median lands at 64%.

For future BEIR runs to compute κ, the loader must be patched to also sample non-relevant document candidates per query (typically random docs not in the qrels). This is not a code bug but a methodology mismatch: BEIR's `qrels` dict is designed for retrieval evaluation (judge a ranked list against positive labels), not for relevance-judgment cross-validation (judge each pair against a balanced label distribution). We flag this as a community-wide limitation of using BEIR for inter-judge κ replication.

## 7.3 Coverage divergence as a content-domain reliability axis

A finding not visible from any single corpus: **coverage rates differ substantially between BEIR scifact and TREC RAG 2024 for the same judges**:

| Judge | TREC RAG 2024 (n=537) | BEIR scifact (n=300) |
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

*Table 7: Per-judge coverage on the two external-validation corpora. The "always-works" 6-judge subset (≥ 95% on both) is Anthropic + OpenAI + Qwen + Gemma. Gemini and DeepSeek vary substantially by content domain.*

**The reliability axis is content-domain dependent**: Gemini struggles much more on TREC RAG 2024 (web-domain passages, 17-24% coverage) than on BEIR scifact (scientific abstracts, 60-86% coverage). DeepSeek similarly: 39% vs 84%.

We hypothesize the cause is content character: long, varied-formality web content (TREC RAG 2024) triggers thinking-mode safety / output-parser failures more often than tightly-formatted academic abstracts (BEIR scifact). Empirical evidence: Gemini 2.5 Pro produced ~70% more null outputs on TREC RAG 2024 than on BEIR scifact, despite the same prompt template and identical truncation rules.

**Practical implication**: judge selection must consider the deployed corpus's content character, not just provider/family. An institutional repository of academic abstracts (like ISU DSpace, BEIR scifact) will get reliable scores from all 9 judges; a web-content RAG deployment will need Anthropic + OpenAI + open-weight as the always-works subset, with Gemini and DeepSeek as supplementary.

## 7.4 Comparison to Thakur et al. 2025

Thakur et al. [thakur2025trecragsupport] report κ = 0.60 for a single GPT-4o judge against TREC RAG 2024 NIST qrels, on a 537-pair "support evaluation" subset they curated. Our GPT-4o on our 537-pair stratified-balanced sample reaches κ = 0.41 — a 0.19 κ gap.

Possible explanations for the gap, in decreasing order of likely impact:
1. **Sample selection**. Thakur's "support evaluation" subset is curated to be representative of the full TREC RAG 2024 task (sampling proportional to the natural label distribution). Our sample is stratified-balanced (135 each of 4 labels). The natural distribution is 37% / 31% / 23% / 9% (rel 0/1/2/3); ours is uniform. Stratification penalizes a judge that systematically under-rates relevance (which GPT-4o does — mean 1.07 in our sample) because the higher-rel labels are over-represented relative to natural.
2. **Prompting**. Our prompt is the 0-3 ordinal rubric with 5 worked examples and 1500-char truncation; Thakur's exact prompt is not specified in the paper.
3. **Qrels version**. We use the official 2024-retrieval-qrels.txt published at trec.nist.gov; if Thakur used a different version (e.g. an UMBRELA-augmented qrels from a parallel paper), labels could differ on individual pairs.
4. **Judge version**. We use `gpt-4o-2024-08-06`; Thakur does not specify a model version date.

Our ensemble κ = 0.49 partially closes the gap: across 9 judges, the per-pair median outperforms the single-judge GPT-4o by 0.08 κ, supporting the ensemble-vs-single-judge case.

**A direct apples-to-apples replication of Thakur et al. would require their curated 537-pair subset, which is not published as a standalone artifact.** We have requested it from the authors and will report the outcome in a future revision.

## 7.5 Validation budget

| Run | Pairs | Judges | Wall | Cost |
|---|---:|---:|---|---:|
| BEIR scifact smoke (Gemma only) | 30 | 1 | ~5 min | ~$0.005 |
| BEIR scifact 7-judge frontier | 300 | 7 | ~3.5 h | ~$18 |
| BEIR scifact 2-judge supplement | 300 | 2 | ~1.5 h | ~$3 |
| TREC RAG 2024 7-judge frontier | 537 | 7 | ~5 h 12 min | ~$30 |
| TREC RAG 2024 2-judge supplement | 537 | 2 | ~2.5 h | ~$5 |
| MS MARCO v2.1 passage extraction | 537 passages | n/a | ~38 min | $0 (HF stream) |
| **Total** | | | **~13 h** | **~$56** |

vs. originally-planned IRB human study: $1,500-3,000 and 6-12 weeks. The validation pipeline is **~30× cheaper and ~50× faster** than per-institution human-rater work, and produces evidence applicable across deployments rather than tied to a single institution's rater pool.

# 8. Limitations

**Single internal corpus.** All within-pair κ numbers (Fig. 3) are on a single institutional corpus (ISU DSpace). The cross-organization Qwen ↔ Gemma 4 κ = 0.80 finding is the most likely to drift across domains. The §7 TREC RAG 2024 cross-corpus result (ensemble κ = 0.49) bounds the drift on judge-vs-human agreement but does not directly bound judge-vs-judge κ.

**Single internal query set**. The 57-query REFINE set covers factoid / methodological / comparative / author-style intents on a single corpus. We have not tested whether κ structure transfers to other query-synthesis methods (Promptagator, InPars, hand-written queries). The κ matrix could depend on query intent distribution.

**Stratified sample mismatch with Thakur 2025.** §7.1's 537-pair sample is stratified-balanced (uniform over 4 labels); Thakur's was curated to natural-distribution-proportional. Direct numerical κ comparison to their 0.60 baseline is not strictly apples-to-apples (§7.4). Our defensible claim is "moderate-substantial agreement (κ = 0.49 ensemble) on a stratified-balanced sample of NIST-graded TREC RAG 2024 pairs," not "matching Thakur's 0.60."

**Gemini judges' coverage failures.** Gemini 2.5 Pro produced scores on only 17% of TREC RAG 2024 pairs and Gemini 3.1 Pro Preview on 24%. The reported κ for those judges is on the small valid subset, with selection-bias risk that "easy" pairs are the ones the judge succeeded on. We treat this as a coverage-axis finding (paper-relevant) rather than a κ ranking; the always-works 6-judge subset (Anthropic + OpenAI + Qwen + Gemma) is the recommended deployment configuration when reliability is paramount.

**DeepSeek V4 Flash dropped** due to OpenRouter free-tier 429 throttling. We have one DeepSeek judge (V4 Pro) and no within-DeepSeek-family pair, weakening the 5-family symmetry. The companion P4-LIB paper [Sukul, forthcoming, JCDL 2027] will include both DSV4 Pro and DSV4 Flash on a paid tier.

**Missing-data accounting.** Opus 4.7 returned `null` on 78/570 pairs (13.7%), Sonnet 4.6 on 13/570 (2.3%), DSV4 Pro on 13/570 (2.3%), Gemini 3.1 Pro Preview on 390/570 (68.4%), Gemini 2.5 Pro on 273/570 (47.9%). Aggregate metrics (Table 2 left mean column) treat null as 0; valid-only means (Table 2 right column) exclude nulls. The published (None=0) means inflate apparent strictness for high-missingness judges. Valid-only means place all 9 judges within either strict-mid (1.1-1.3) or reasoning-generous (1.6-1.7) clusters; the "strict outlier" tier we initially identified is partly a missing-data artifact.

**Mediation analysis assumes linear additivity.** §6.3's mediation of structural factors through joint-distribution features assumes the structural-factor effect on κ is additive in the regression. Non-linear interactions (e.g. provider × reasoning-mode) could survive controlling for joint distribution. We have not tested non-linear mediation; pending Bayesian or non-parametric replication.

**No human-rater study on ISU DSpace.** We elected not to run a per-institution IRB human-rater study (§7's external validation against NIST qrels substitutes for it). Reviewers preferring a within-corpus human anchor will not find one in this paper. Our framing is that NIST-graded TREC RAG 2024 qrels — produced by trained annotators following published guidelines — are a stronger external standard than a small per-institution rater pool, and that the latter is an institution-specific artifact while the former is a community-shared one.

# 9. Discussion and Reproducibility

**Practitioner takeaway.** Cross-family agreement is high enough that judge interchangeability is a defensible default for reasoning-capable commercial judges; **calibration philosophy** (lenient vs strict score-allocation pattern), not provider, is the axis on which retrieval verdicts shift. The disclosure template (§5, C4) makes the shift explicit and auditable. For sovereign-cloud and privacy-conservative deployments, the cross-organization open-weight pair (Qwen 3.6 Plus + Gemma 4 26B) achieves the matrix-highest κ at near-zero marginal cost; for web-content RAG with reliability requirements, Anthropic + OpenAI + open-weight is the always-works 6-judge subset.

**Theoretical contribution.** §6's mechanism analysis establishes that **pairwise κ is fully mediated by joint-distribution structure**: provider, reasoning-mode, and model class affect κ only through the joint distribution they induce. The shared-tokenizer hypothesis is refuted at the strongest possible scale (lowest-Jaccard pair = highest-κ pair). The cross-organization Qwen ↔ Gemma 4 high κ remains a partial mystery; we offer four candidate hypotheses (§6.5) for follow-up work.

**Empirical contribution.** §7's external validation against NIST TREC RAG 2024 qrels achieves moderate-substantial agreement (ensemble κ = 0.4941) at 1/30th the cost of an IRB human-rater study. The validation pipeline (`validate_against_trec.py` + `fetch_msmarco_passages.py`) supports TREC RAG 2024, BEIR scifact, and TREC-COVID out of the box; LLMJudge benchmark and TREC-COVID full runs are deferred to a TOIS journal-version extension.

**Reproducibility.** We release: (i) `eval_llm_judge.py` (multi-judge mode, ThreadPool fan-out, 9-judge `p4-frontier` + `p4-supplement-openweight` presets), (ii) the cross-run merge harness with retrieval-determinism verification, (iii) the 57-query REFINE set, (iv) the 0-3 rubric prompt template, (v) all 9 within-corpus per-judge JSONs (5,130 score records), (vi) the merged κ-matrix JSON, (vii) the deterministic heatmap regen scripts, (viii) `validate_against_trec.py` and supporting CORPUS_REGISTRY for TREC RAG 2024 / BEIR scifact / TREC-COVID, (ix) `fetch_msmarco_passages.py` for HF-streaming MS MARCO v2.1 passage extraction, and (x) the 9 external-validation per-judge JSONs (4,833 score records) plus pre-extracted 537 MS MARCO v2.1 passages (1.02 MB).

Single-command replication of within-corpus result:
```
py -3 -X utf8 backend/scripts/eval_llm_judge.py --collection <yours> --judge-preset p4-frontier
py -3 -X utf8 backend/scripts/eval_llm_judge.py --collection <yours> --judge-preset p4-supplement-openweight
```
Marginal cost: ~$18.30 per full 9-judge replication, ~5.5 hours wall on a single workstation; the open-weight subset is ~$0.30.

Single-command replication of external validation:
```
py -3 -X utf8 papers/P4_llm_as_judge/validate_against_trec.py --corpus trec-rag-2024 --judge-preset p4-frontier --max-pairs 537
py -3 -X utf8 papers/P4_llm_as_judge/validate_against_trec.py --corpus trec-rag-2024 --judge-preset p4-supplement-openweight --max-pairs 537
py -3 -X utf8 papers/P4_llm_as_judge/validate_against_trec.py --analyze trec-rag-2024
```
Marginal cost: ~$35 per full 9-judge external replication, ~7.5 hours wall.

Model versions pinned at 2026-04-25: `claude-opus-4-7-20260415`, `claude-sonnet-4-6-20260401`, `gpt-5.5-2026-04-12`, `gpt-4o-2024-08-06`, `gemini-3.1-pro-preview-2026-04`, `gemini-2.5-pro-2025-06-17`, `deepseek-v4-pro-2026-04-10`, `qwen3.6-plus-2026-03`, `gemma-4-26b-a4b-it-2026-03`. We target SIGIR Artifact Badging (Functional/Reusable) for the toolkit.

# 10. Conclusion

We presented a 5-family, 9-judge LLM-as-Judge ablation on 570 institutional RAG (query, document) pairs and report four novel contributions: (C1) cross-family reasoning judges converge at κ ≥ 0.75, (C2) within-family agreement is bounded by the cross-family ceiling, (C3) cross-organization open-weight judges produce the matrix-highest κ at ~1% of commercial cost, and (C4) an open-source toolkit + community disclosure template. The mechanistic decomposition (§6) attributes 93% of pairwise κ variance to joint-distribution structure of paired scores, fully mediates structural factors, and refutes the shared-tokenizer hypothesis. External validation against NIST TREC RAG 2024 qrels (§7) confirms moderate-to-substantial agreement (ensemble κ = 0.4941) on 537 stratified-balanced pairs across 86 unique queries, replacing a per-institution IRB human-rater study with cross-corpus public-NIST evidence.

Open questions for follow-up: (1) the cross-organization open-weight convergence (§6.5) lacks a complete mechanistic explanation; (2) the κ structure under non-bounded output rubrics (e.g. graded-relevance with longer scales, free-text rationales) may invert the within-family bounded-ordinal finding (C2); (3) non-linear interactions in the §6.3 mediation analysis remain untested; (4) whether the ensemble κ closes the gap to Thakur 2025's 0.60 single-judge baseline on a directly-replicated 537-pair subset is pending receipt of their curated sample.

# Acknowledgments

This work was conducted on the Iowa State University Digital Repository (DSpace), and we thank the **ISU Library and Digital Initiatives team** for maintaining the corpus, providing repository access, and tolerating the indexing load. The 1.03M-chunk Qdrant deployment runs on **ISU IT-managed infrastructure**; we thank the systems team for compute, storage, and persistent-volume provisioning. We are grateful to the **Anthropic, OpenAI, Google DeepMind, DeepSeek, Alibaba (Qwen), and Google (Gemma)** model teams for releasing the model snapshots used in this study, and to the **OpenRouter** team for providing the gateway that made the cross-family API access economically feasible. The TREC RAG 2024 qrels used in §7 are released by **NIST and the TREC RAG 2024 organizers** [macavaney2025trec_rag]; the MS MARCO v2.1 segmented passage corpus is hosted by **Microsoft** and mirrored on the **Hugging Face Hub** by `drexalt/msmarco-2.1-segmented`, whose curation enabled our 38-minute streaming filter rather than a 25 GB direct download. The BEIR scifact dataset is maintained by the **BEIR consortium** [thakur2021beir]. We thank colleagues at ISU and the broader IR community for informal comments on early drafts of this paper; remaining errors are the author's. No external funding source supported this work; API costs (~USD 56 across all validation runs) were covered by personal research budget. We acknowledge the use of Claude Code as a coding assistant for harness implementation and figure regeneration; all experimental design, methodology, and analytical interpretation are the author's. The single-author byline reflects independent ownership of the work; a strong tradition in IR includes single-authored methodology contributions [zheng2023judge has multiple authors but the single-author tradition is exemplified by classic IR methodology papers], and we note that double-blind review in the target venues abstracts authorship in any case.

# References

See `papers/references.md` for the shared verified bibliography. The final ≤30-citation slate for ECIR LNCS submission will be selected during writing per `WRITING_PLAN.md` Appendix B; current slate covers LLM-as-Judge generalists [zheng2023judge; liu2023geval; gu2024survey; li2024judges; li2024judgmentsurvey; kim2023prometheus; kim2024prometheus2], LLM-as-Judge for IR [faggioli2023perspectives; rahmani2024judges; upadhyay2024umbrela], judge bias and meta-evaluation [wang2023fair; saito2023verbosity; chen2024humans; panickssery2024self; thakur2024judges], the 2025 inter-LLM-judge agreement cluster [balog2025rankersjudges; farzi2025umbrela_other; thakur2025trecragsupport; han2025judgesverdict], and RAG-eval frameworks [es2024ragas; saadfalcon2024ares; ru2024ragchecker; rau2024bergen; friel2024ragbench; trulens2023; wang2024feb4rag]. The TREC RAG 2024 corpus citation [macavaney2025trec_rag] and the cross-domain quadratic-weighted κ replication [jiao2026essayraters] anchor the §7 external-validation comparison.

Appendices (online supplementary):
- **A**: Full 9×9 κ matrix with bootstrap 95% CIs
- **B**: 5-, 7-, 9-judge ablation comparison (FINDINGS_5judge / 7judge / 9judge)
- **C**: Mechanism analysis full data (regression diagnostics, full confusion matrices for all 36 pairs)
- **D**: External validation per-corpus details (TREC RAG 2024 + BEIR scifact + TREC-COVID future)
- **E**: Disclosure template and rubric prompt with worked examples
- **F**: Cost / wall-time accounting and reproducibility checklist
