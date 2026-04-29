---
title: "Cross-Family LLM-Judge Agreement for Institutional RAG: A 5-Family, 9-Judge Ablation"
authors:
  - name: "Adisak Sukul"
    affil: "Iowa State University"
    email: "asukul@iastate.edu"
  - name: "[TBD co-authors]"
venue: "LLM4Eval @ SIGIR 2027 (workshop short paper, 4 pages)"
length_target: "4 pages (≤2,130 words main text)"
status: "draft v0.4 — condensed from long.md (2026-04-28)"
last_update: "2026-04-28"
companion: "long.md (full conference version, ECIR 2028 / TOIS target)"
abstract_opener_hook: "Open-weight LLM judges win the within-pair race."
---

# Abstract

**Open-weight LLM judges win the within-pair race.** In our 5-family, 9-judge LLM-as-Judge ablation on **570 RAG query-document pairs from an institutional repository**, the highest pairwise quadratic-weighted Cohen's κ (0.80) is between Qwen 3.6 Plus and Gemma 4 26B — a cross-organization open-weight pair that exceeds every commercial within-family pair (Anthropic 0.71, OpenAI 0.63, Google-commercial 0.67) and ties the cross-family commercial reasoning ceiling. We present the first **five-family** pairwise κ matrix on bounded 0-3 ordinal relevance with **four within-family controls** and report (i) cross-family reasoning judges converge at κ = 0.75-0.79, (ii) within-family agreement is task-dependent and bounded by the cross-family ceiling, contradicting canonical self-preference findings under bounded ordinal judging, (iii) two emergent calibration clusters (reasoning-generous vs strict-mid) partition judges more cleanly than provider family. Mechanistic decomposition (companion long version) attributes **93% of κ variance to joint-distribution structure of paired scores**; structural factors (provider, reasoning-mode) are fully mediated; the shared-tokenizer hypothesis is refuted (Qwen↔Gemma vocabulary Jaccard = 0.066, lowest in slate, yet matrix-highest κ). External validation against **NIST TREC RAG 2024 human qrels** on a 537-pair stratified-balanced sample reaches 9-judge ensemble **κ = 0.4941** (Landis-Koch moderate, near-substantial); 5 of 9 individual judges hit κ ≥ 0.47. We open-source `eval_llm_judge.py` and `validate_against_trec.py` (5,130 within-corpus + 4,833 external score records, USD 56.30, ~13 hours wall) as reproducibility artifacts.

# 1. Introduction

Modern RAG systems sit on top of vector retrievers tuned over institutional corpora — academic libraries, parliamentary archives, medical records, legal collections. Practitioners want to answer "is this retrieval good?" — but they hit three blockers: there is no gold relevance set (TREC-style ground truth is prohibitively expensive), generic IR benchmarks (BEIR, MS-MARCO) do not transfer, and the canonical workaround — LLM-as-Judge [zheng2023judge; faggioli2023perspectives; rahmani2024judges; upadhyay2024umbrela] — surfaces a new question: *which* LLM judge?

The literature splits into camps. LLM-as-Judge advocates argue a frontier reasoning model is sufficient. Bias skeptics document position, verbosity, and self-preference biases [wang2023fair; saito2023verbosity; panickssery2024self], advocate ensembling, and call for meta-evaluation [thakur2024judges]. RAG-eval frameworks [es2024ragas] provide infrastructure but report agreement only against humans, not pairwise across LLMs. **No published paper, to our knowledge, reports pairwise quadratic-weighted Cohen's κ across three or more model families on a bounded ordinal relevance rubric at ≥ 500 paired observations, with within-family controls and open-weight peers as first-class judges.** Open-weight judges are the default for sovereign-cloud and privacy-conservative institutions; excluding them silently endorses commercial-only practice.

We close the gap. We deploy **9 LLM judges across 5 model families** (Anthropic, OpenAI, Google-commercial, DeepSeek, Open-weight) on **570 (query, document) pairs** from the Iowa State University DSpace repository (97k full-text PDFs), and report four claims: **(C1)** cross-family reasoning judges agree at κ ≥ 0.75; **(C2)** within-family agreement is bounded by the cross-family ceiling; **(C3)** the highest pairwise κ (0.80) is between two cross-organization open-weight judges, exceeding every commercial within-family pair at ≈1% of the cost; **(C4)** an open-source N-judge κ-matrix toolkit and disclosure template. §6 contributes a mechanistic decomposition; §7 reports external validation against NIST TREC RAG 2024 qrels.

# 2. Related Work

**LLM-as-Judge for IR.** Faggioli et al. [faggioli2023perspectives], Rahmani et al. [rahmani2024judges], and UMBRELA [upadhyay2024umbrela] established LLMs as practical relevance assessors. Bias work documents position [wang2023fair], verbosity [saito2023verbosity; chen2024humans], and self-preference biases [panickssery2024self]; Thakur et al. [thakur2024judges] motivate quadratic-weighted κ for ordinal IR judgment.

**2025 inter-LLM-judge agreement.** Recent work reports inter-LLM agreement on relevance but covers either a single provider family [balog2025rankersjudges; farzi2025umbrela_other], two-family pairs without ordinal weighting [thakur2025trecragsupport, κ = 0.60 GPT-4o ↔ Llama-3.1-405B on 537 unweighted pairs], or judge-vs-human only [han2025judgesverdict]. None — to our knowledge — reports a multi-family pairwise κ matrix with **both within-family controls and open-weight peers as first-class judges** at ≥ 500 quadratic-weighted observations. Parallel evidence from educational measurement [jiao2026essayraters] shows frontier LLMs converging on quadratic-weighted κ with humans on 0-N ordinal scoring.

# 3. Methodology

**Corpus.** Iowa State University Digital Repository: **97,441 full-text PDFs → 1.03M chunks** (≤500 words, 100-word overlap), Google Vertex `text-embedding-005` in Qdrant 1.13 HNSW. Queries from **REFINE** [Sukul, forthcoming]; canonical 57-query set covering factoid / methodological / comparative / author-style intents, balanced across 20 topic clusters. Top-10 retrieval per query → **570 (query, document) pairs**.

**Judges.** 9 across 5 families with 4 within-family pairs (Table 1):

| # | Judge | Family | Reasoning |
|---|---|---|---|
| 1-2 | Claude Opus 4.7 / Sonnet 4.6 | Anthropic | yes |
| 3-4 | GPT-5.5 (low) / GPT-4o | OpenAI | yes / no |
| 5-6 | Gemini 3.1 Pro Prev / Gemini 2.5 Pro | Google | yes (thinking) |
| 7 | DeepSeek V4 Pro | DeepSeek | yes |
| 8-9 | Qwen 3.6 Plus / Gemma 4 26B | Open-weight | no |

*Table 1: Judge slate. Within-family pairs: Anthropic, OpenAI, Google-commercial, Open-weight cross-organization. DSV4 Flash dropped due to OpenRouter 429s. Versions pinned 2026-04-25.*

**Rubric and metrics.** 0-3 ordinal (0=irrelevant, 1=topical, 2=partial, 3=fully answers), calibrated with five worked examples; documents truncated to 1,500 chars [saito2023verbosity]. `eval_llm_judge.py` fans 9 judges/pair via ThreadPool. Two-phase run (canonical 7 + open-weight 2), 5,130 records, ~5.5 h wall, USD 18.30; merge harness verified 100% (query_id, rank, point_id) alignment. Per-judge nDCG@10, P@5 (s≥2), MRR; pairwise **quadratic-weighted Cohen's κ** [thakur2024judges]; Landis-Koch bands. Fig. 3: 9×9 κ heatmap.

# 4. Results

## 4.1 Per-Judge Retrieval Metrics

The 570 pairs yield strikingly different aggregate verdicts (Table 2):

| Judge | Family | nDCG@10 | P@5 | MRR | Mean |
|---|---|---:|---:|---:|---:|
| Sonnet 4.6 | Anthropic | **0.862** | **0.649** | **0.826** | 1.68 |
| GPT-5.5 (low) | OpenAI | 0.846 | 0.600 | 0.792 | 1.63 |
| DSV4 Pro | DeepSeek | 0.835 | **0.702** | **0.883** | 1.63 |
| GPT-4o | OpenAI | 0.803 | 0.361 | 0.575 | 1.15 |
| Qwen 3.6 | Open-weight | 0.758 | 0.379 | 0.599 | 1.18 |
| Gemma 4 26B | Open-weight | 0.753 | 0.375 | 0.573 | 1.11 |
| Opus 4.7 | Anthropic | 0.749 | 0.372 | 0.536 | 1.09 |
| Gemini 2.5 Pro | Google | 0.624 | 0.281 | 0.486 | 0.76 |
| Gemini 3.1 Prev | Google | **0.454** | **0.158** | **0.315** | **0.37** |

*Table 2: Same 570 retrieved documents; bold = per-column extrema; sorted by nDCG@10. **The same documents yield nDCG@10 between 0.45 and 0.86 — a 1.9× spread driven entirely by judge selection.** A practitioner using Sonnet 4.6 or GPT-5.5 concludes the pipeline is excellent; using Gemini 3.1 Prev, the same pipeline appears broken.* We propose *"nDCG@10 = X.XX via [family]/[model]/[reasoning-config], N pairs, DATE"* as a community disclosure standard (C4).

## 4.2 Calibration Tiers and κ Matrix Structure

Mean scores partition into three tiers: a **reasoning-generous cluster** (Sonnet, GPT-5.5, DSV4 Pro; mean 1.63-1.68), a **strict-mid cluster** (GPT-4o, Qwen, Gemma 4, Opus 4.7; mean 1.09-1.18), and a strict-outlier pair (Gemini, partly inflated by missing-data; valid-only means place both near strict-mid — see §6). Both **open-weight judges calibrate squarely with strict-mid commercial models**, not as outliers — non-obvious given the training-compute gap.

Fig. 3 renders the full pairwise κ heatmap. **Every off-diagonal cell is ≥ 0.56 — substantial or moderate by Landis-Koch.** Two clusters: reasoning-generous (Sonnet ↔ GPT-5.5 = 0.79 leads; ceiling 0.75-0.79) and strict-mid + open-weight (**Qwen ↔ Gemma 4 = 0.80 — matrix-highest**; Gemma 4 ↔ GPT-4o = 0.77; Qwen ↔ Opus 4.7 = 0.75). All three commercial within-family pairs (Anthropic 0.71, OpenAI 0.63, Google 0.67) sit **below** the cross-family reasoning ceiling (0.79); the fourth, cross-organization Open-weight pair, is the matrix maximum. **Calibration philosophy partitions judges more cleanly than provider lineage.**

# 5. Findings (C1–C4)

**C1. Cross-family reasoning judges converge at κ ≥ 0.75.** Five cross-family pairs reach substantial-or-better κ (Fig. 3) — well above 0.4-0.6 typical [rahmani2024judges] and the 0.60 unweighted baseline of [thakur2025trecragsupport]. **DeepSeek V4 Pro joining the cluster rules out a "Western training data" common-cause explanation,** and the pattern replicates across three independent ablation scales (5-, 7-, 9-judge) within run-to-run noise. The 9-judge ensemble achieves κ = 0.49 against published NIST qrels on TREC RAG 2024 (537 stratified-balanced pairs; §7), so the convergence is not specific to our institutional corpus.

**C2. Within-family agreement is task-dependent and bounded by the cross-family ceiling.** All four within-family pairs are at most equal to the strongest cross-family commercial pair (0.79): Anthropic 0.71, OpenAI 0.63, Google 0.67, cross-org Open-weight 0.80. **No within-family pair dominates** — at odds with self-preference findings on open-ended generation [panickssery2024self]. Our companion 4-way extraction study reports within-family ~2× cross-family on open-vocabulary tasks; the inversion under bounded ordinal judging suggests **self-preference is mediated by output-space boundedness**, not provider lineage.

**C3. Open-weight judges produce the matrix-highest within-pair κ and cluster with strict-mid commercial models.** Qwen ↔ Gemma 4 = **0.80** — exceeds every commercial within-family pair and ties the cross-family reasoning ceiling. Both open-weight judges (mean 1.11-1.18) cluster with GPT-4o and Opus 4.7 (1.09-1.15) **not** with the reasoning-generous triad (1.63-1.68). At ~USD 0.30 marginal cost, **a free on-prem cross-organization open-weight ensemble achieves higher cross-validation κ than ~USD 18 of commercial within-family calls**. External corroboration: Qwen and Gemma 4 both achieve **100% coverage** on TREC RAG 2024 (better than 4 of 7 frontier judges), with individual κ vs human qrels of 0.41 each.

**C4. Open-source toolkit and disclosure template.** We release `eval_llm_judge.py` (multi-judge mode, ThreadPool fan-out), `validate_against_trec.py` (multi-corpus external-validation harness for TREC RAG 2024 / BEIR scifact / TREC-COVID), `fetch_msmarco_passages.py` (HF-streaming MS MARCO v2.1 passage extractor), the cross-run merge harness, the 57-query REFINE set, the rubric template, all 9 within-corpus per-judge JSONs (5,130 records), and the 9 external-validation JSONs (4,833 records).

# 6. Mechanism (Summary)

**Joint-distribution structure of paired scores explains R² = 93% of κ variance** (dispersion and effective rank dominate, both p < 0.001). Marginal-distribution KL divergence [kullback1951] is a coarse 33% proxy. **Structural factors (reasoning-mode, provider, model class) are fully mediated** [following Baron-Kenny [baronkenny1986] and MacKinnon [mackinnon2007]]: they affect κ only through the joint distribution they induce (all p > 0.18 after controlling). The **shared-tokenizer hypothesis is refuted**: Qwen ↔ Gemma 4 vocabulary Jaccard = 0.066 (lowest in slate) yet κ = 0.80 (matrix-highest); convergence is at the decision-making layer. We use Cohen's κ [cohen1968] in its original quadratic-weighted formulation per Fleiss [fleiss1971] with Landis-Koch [landis1977] bands. Full analysis with regression diagnostics, 36 confusion matrices, and per-pair tokenizer-overlap table: companion `long.md` §6.

# 7. External Validation Against NIST Human Qrels

To address single-corpus exposure, we replicated the 9-judge slate on **TREC RAG 2024** (537 stratified-balanced pairs from the official 20,283-pair NIST qrels, balanced 135/134/134/134 across labels 0/1/2/3, 86 queries; MS MARCO v2.1 passages extracted via HF-streaming filter from `drexalt/msmarco-2.1-segmented`).

**Results.** Per-judge κ vs human qrels ranges 0.40-0.55; **9-judge ensemble median κ = 0.4941**, **7-judge frontier-only κ = 0.5187** (open-weight broadens coverage, slightly lowers κ — a robustness/headline tradeoff). 5 of 9 judges hit κ ≥ 0.47. Compared to Thakur et al.'s GPT-4o κ = 0.60 [thakur2025trecragsupport], our GPT-4o = 0.41 — gap likely from sample (ours stratified-balanced; theirs natural-proportional), prompting, qrels version.

**Coverage divergence is a content-domain reliability axis.** On TREC RAG 2024, Anthropic + OpenAI + Qwen + Gemma all return scores on **100%** of pairs; **Gemini 3.1 Pro Preview 24%, Gemini 2.5 Pro 17%** (thinking-mode parse aborts on long web-domain passages), DeepSeek V4 Pro 40%. On BEIR scifact (300 pairs, all-positive qrels — κ undefined; 9-judge ensemble precision at ≥2 = **63.7%**, range 43-75%), the same Gemini judges hit 60-86% — reliability depends on content character. The **always-works 6-judge subset** (≥95% both corpora) is Anthropic + OpenAI + Qwen + Gemma — **the 2 cheapest judges make this subset**. Validation pipeline replaces a per-institution IRB study (~$1,500-3,000, 6-12 weeks) with public-NIST evidence (~$56, ~13 h). Per-corpus tables: `PUBLIC_BENCHMARKS_VALIDATION.md`.

# 8. Limitations and Reproducibility

**Single internal corpus.** Within-pair κ numbers (Fig. 3) are on ISU DSpace; the open-weight κ ceiling is the most likely finding to drift across domains, though §7's TREC RAG 2024 result bounds the drift on judge-vs-human agreement. **Stratified vs natural-distribution sampling.** Our 537-pair sample is balanced; Thakur 2025's was natural-distribution-proportional, so direct numerical comparison is not strictly apples-to-apples. **DSV4 Flash dropped** (OpenRouter 429). **Missing-data caveats.** Opus 4.7 returned None on 78/570 (13.7%); Gemini 3.1 Prev on 390/570 (68.4%); Gemini 2.5 Pro on 273/570 (47.9%). Aggregate metrics treat None as 0; valid-only means place both Gemini judges within the strict-mid cluster (1.17 / 1.45) rather than as extreme outliers. **Practitioner takeaway.** Judge interchangeability is a defensible default for reasoning-capable commercial judges; **calibration philosophy** (lenient vs strict), not provider, is the axis on which retrieval verdicts shift; for sovereign-cloud deployments, the cross-organization open-weight pair achieves matrix-highest κ at near-zero marginal cost.

**Reproducibility.** Single-command within-corpus replication: `py eval_llm_judge.py --collection <yours> --judge-preset p4-frontier`; ~USD 18.30, ~5.5 h wall. Single-command external replication: `py validate_against_trec.py --corpus trec-rag-2024 --judge-preset p4-frontier --max-pairs 537`; ~USD 35, ~7.5 h wall. Model versions pinned 2026-04-25. Target SIGIR Artifact Badging (Functional/Reusable). Companion long version (`long.md`) contains full mechanism analysis, expanded validation tables, and 6 supplementary appendices.

# Acknowledgments

This work was conducted on the Iowa State University Digital Repository (DSpace); we thank the **ISU Library and Digital Initiatives team** for corpus access and the **ISU IT systems team** for the Qdrant deployment infrastructure. We thank the **Anthropic, OpenAI, Google DeepMind, DeepSeek, Alibaba, and Google (Gemma)** model teams for the model snapshots used here, and the **OpenRouter** team for the gateway that made cross-family API access economically feasible. §7 external validation uses the **NIST TREC RAG 2024** qrels [macavaney2025trec_rag] and the MS MARCO v2.1 segmented corpus mirrored on Hugging Face by `drexalt/msmarco-2.1-segmented`. The BEIR scifact dataset is maintained by the **BEIR consortium** [thakur2021beir]. API costs (~USD 56) were covered by personal research budget; we acknowledge the use of Claude Code as a coding assistant for harness implementation. Remaining errors are the author's.

# References

See `papers/references.md` for the shared verified bibliography. The 18-citation slate now includes [baronkenny1986; mackinnon2007] for §6 mediation analysis, [cohen1968; fleiss1971; kullback1951; landis1977] for the κ statistic and KL divergence, [macavaney2025trec_rag; thakur2021beir] for §7 corpora, and the LLM-as-Judge / IR / RAG-eval cluster cited in §2.
