# P4 Paper Walkthrough — How Each Paper Works

**Last update:** 2026-04-28
**Audience:** Reviewers, co-authors, future readers, anyone who needs to understand *how each paper is constructed* before editing it
**Files explained:** `papers/short.md` (workshop, 4 pp), `papers/long.md` (conference / journal, 12 pp LNCS)

This document is **not the paper**. It explains *how each paper works*: what claims it makes, what evidence supports each claim, how the sections connect, and which design choices are load-bearing vs decorative. Read this before editing the papers; the paper text is dense and removing the wrong sentence breaks the argument structure.

---

## 1. The two papers — at a glance

| | **short.md** | **long.md** |
|---|---|---|
| **Target** | LLM4Eval @ SIGIR 2027 workshop | ECIR 2028 full paper / TOIS journal |
| **Length** | 4 pages, ≤2,130 words main | 12 pages LNCS, ~10,000 words main |
| **Status** | v0.4 (2026-04-28) | v0.1 (2026-04-28) |
| **Core claims** | C1, C2, C3, C4 | C1, C2, C3, C4 + mechanism + external validation |
| **Word budget** | tight — every sentence load-bearing | comfortable — adds full mechanism §6 + full validation §7 + 6 appendices |
| **§6 mechanism** | summary paragraph only (R²=93%, KL=33% proxy, tokenizer refuted) | full section: §6.1 KL → §6.2 joint-distribution → §6.3 mediation → §6.4 tokenizer refutation → §6.5 open question |
| **§7 validation** | one paragraph per corpus | full per-judge tables, per-corpus discussion, Thakur comparison, validation budget |
| **Figures referenced** | Fig. 3 (κ matrix) only | Figs. 3, 6 (regression scatter), plus appendix figures |
| **References** | ~18 cites | ~30 cites (LNCS limit) + appendix expansions |

**The two papers share the same scaffolding.** Sections 1–5 (Intro, Related Work, Methodology, Within-Corpus Results, Findings) are structurally identical; long.md just has more detail per section. Sections 6 and 7 are where they diverge sharply: short.md compresses each to one paragraph; long.md gives each a full section.

---

## 2. The argument graph — what supports what

The papers are not 10 independent observations. They are one argument with four conclusions (C1–C4), each grounded in a specific evidence slice. Reading a section out of context can mislead — the right way to read this paper is **claims-first** (§5), then trace each claim back to the evidence that supports it.

```
                        ┌─ §3 Methodology (corpus, judges, rubric, pipeline)
                        │
                        ├─ §4.1 Per-judge metrics (1.9× nDCG@10 spread)
        §5 Claims ──────┤
        (C1, C2, C3, C4)│
                        ├─ §4.2-4.3 Calibration tiers + κ matrix (Fig. 3)
                        │       │
                        │       └─ §6 Mechanism (long only): WHY κ varies
                        │           - 6.1 KL: 33% R² (coarse)
                        │           - 6.2 Joint: 93% R² (dominant)
                        │           - 6.3 Mediation: structural factors absorbed
                        │           - 6.4 Tokenizer hypothesis: REFUTED
                        │
                        └─ §7 External validation (NIST TREC RAG 2024 qrels)
                            - 7.1 Per-judge κ vs human (Table 5)
                            - 7.2 BEIR scifact precision-only
                            - 7.3 Coverage divergence
                            - 7.4 Thakur 2025 comparison
```

**Key dependency:** §6 (mechanism) and §7 (validation) are NOT a single block of "supplementary results." They are two independent robustness checks for different attacks:
- **§6 protects against:** "Your κ matrix is artifactual — the structure could be explained by a hidden factor like provider lineage or tokenizer overlap."
- **§7 protects against:** "Your κ matrix is institution-specific — your numbers won't transfer to a corpus with public ground truth."

A reviewer hitting one objection should be redirected to the relevant section, not the other.

---

## 3. The four claims (C1–C4) — what each one says and why

Each claim has a specific evidence slice. If a claim breaks, only that section needs revision.

### C1 — Cross-family reasoning judges converge at κ ≥ 0.75

**What it says:** Five cross-family pairs (Sonnet↔GPT-5.5, Qwen↔Gemma 4, Sonnet↔DSV4, GPT-5.5↔DSV4, etc.) reach substantial-or-better quadratic-weighted κ on bounded ordinal IR judgment. This is well above the 0.4–0.6 typical in prior literature.

**Evidence:** Fig. 3 (κ heatmap), Table 5 (TREC RAG 2024 corroboration), three-scale replication (5/7/9-judge ablations).

**The DeepSeek detail is load-bearing.** The C1 paragraph specifically calls out that DSV4 Pro joining the reasoning-generous cluster *rules out a "Western training data" common-cause explanation* — DeepSeek was trained primarily on Chinese-language and Chinese-mathematical sources. Without DSV4, a critic could argue the convergence is just "models trained on similar Western web-scale text agree." With DSV4, that hypothesis is harder to sustain.

**What weakens C1:** A single corpus (ISU DSpace). §7's external validation (κ=0.4941 on TREC RAG 2024) extends C1 to a public corpus — moderate, not substantial, but cross-corpus confirmation that judges are not at chance.

### C2 — Within-family agreement is bounded by the cross-family ceiling

**What it says:** Three commercial within-family pairs (Anthropic 0.71, OpenAI 0.63, Google-commercial 0.67) all sit *below* the strongest cross-family pair (Sonnet↔GPT-5.5 at 0.79). The cross-organization Open-weight pair (Qwen↔Gemma 4) is the matrix maximum at 0.80.

**Evidence:** Fig. 3 (κ matrix), comparison to [panickssery2024self] (open-ended generation finds within-family self-preference).

**Why it's surprising:** It contradicts canonical self-preference findings. The literature on open-ended generation [panickssery2024self] finds that LLMs prefer their own family's outputs. Our finding is the opposite: under bounded ordinal judging, within-family agreement does *not* dominate cross-family agreement. C2 is the paper's most theoretically novel claim.

**The proposed mechanism (in long.md §5):** *self-preference is mediated by output-space boundedness.* In a 4-class ordinal output space, the dominant signal is calibration philosophy (lenient vs strict), which crosses provider lines. In an open-text output space, the dominant signal is style/lexicon, which respects provider lines. C2 does not claim to *prove* this mechanism — it flags it as a hypothesis for follow-up work.

### C3 — Open-weight judges produce the matrix-highest κ

**What it says:** Qwen 3.6 Plus ↔ Gemma 4 26B = **0.80** — exceeds every commercial within-family pair, ties the cross-family reasoning ceiling. Both calibrate with strict-mid commercial models (mean 1.11–1.18) not with reasoning-generous (1.63–1.68).

**Evidence:** Fig. 3 cell (Qwen↔Gemma 4 = 0.80), Table 2 (means), §6.4 (tokenizer refutation rules out the obvious "shared encoding" explanation), §7 corroboration (both Qwen and Gemma 4 hit 100% coverage on TREC RAG 2024 — better than 4 of 7 frontier judges).

**Why it matters practically:** $0.30 marginal cost to run Qwen + Gemma 4 vs $18.30 for the commercial 7-judge frontier. C3 is the paper's most actionable claim for sovereign-cloud / privacy-conservative institutions — they're not condemned to lower-quality judging by being unable to use commercial APIs.

**Why it's the most fragile claim across domains:** §8 explicitly flags this. The 0.80 number is on a single corpus; it could drift across domains. §7 partially bounds the drift via TREC RAG 2024 (Qwen κ=0.41, Gemma κ=0.40 — both individually moderate, neither matrix-max), but the within-pair κ on TREC-COVID is a planned follow-up.

### C4 — Open-source toolkit and disclosure template

**What it says:** Two artifacts:
1. **Toolkit:** `eval_llm_judge.py` (multi-judge mode), `validate_against_trec.py` (external-validation harness), `fetch_msmarco_passages.py` (HF-streaming MS MARCO v2.1 extractor), and the merge harness — open-sourced under MIT.
2. **Disclosure template:** `nDCG@10 = X.XX via [family]/[model]/[reasoning-config], N pairs, DATE` as a community norm, with the κ heatmap as a recommended companion to every nDCG plot.

**Evidence:** The repo itself (`github.com/asukul/RAG-Eval-LLM-Judge`), Table 2's 1.9× nDCG@10 spread (the motivation for disclosure).

**The 1.9× spread is C4's empirical hook.** Without it, "we propose a disclosure standard" sounds editorial. With it, the standard is a response to a measurable problem: two practitioners running identical retrieval code can arrive at opposite go/no-go decisions depending on which judge they pick. Without disclosure, "our retriever scores 0.86 nDCG@10" is not a comparable measurement.

---

## 4. Section-by-section walkthrough

### §1 Introduction (both papers)

**Three blockers framing.** Practitioners want to evaluate institutional RAG but face: (a) no gold relevance set, (b) generic IR benchmarks don't transfer, (c) LLM-as-Judge surfaces "which judge?" Each blocker maps to a paper section: (a) → §7, (b) → §3 (institutional corpus methodology), (c) → §4–§5 (the κ matrix).

**The literature-gap sentence is load-bearing:** *"No published paper, to our knowledge, reports pairwise quadratic-weighted Cohen's κ across three or more model families on a bounded ordinal relevance rubric at ≥ 500 paired observations, with within-family controls and open-weight peers as first-class judges."* Three quantifiers — *three or more families*, *bounded ordinal*, *≥500 obs* — are each individually defensible against the closest prior work (Thakur 2025 has 2 families, Han 2025 is judge-vs-human, Balog 2025 is single-family). Removing any one quantifier weakens the gap claim.

### §2 Related Work

**short.md** compresses §2 to two paragraphs (LLM-as-Judge for IR + 2025 cluster). **long.md** splits into four subsections:
- §2.1 LLM-as-Judge for IR
- §2.2 Judge bias and meta-evaluation
- §2.3 The 2025 inter-LLM-judge agreement cluster
- §2.4 RAG evaluation frameworks (positioning)

The 2025 cluster section (§2.3) is the most important — it's where the paper draws contrast with closest competitors (Thakur 2025, Han 2025, Balog 2025, Farzi 2025). Each is dispatched with one sentence explaining why the gap remains: *single family, two-family without ordinal weighting, judge-vs-human only*. Long.md adds [jiao2026essayraters] from educational measurement as cross-domain support.

### §3 Methodology

**Five subsections (long.md):** corpus, judge slate (Table 1), rubric, pipeline, external-validation protocol. **Short.md** collapses these to ~6 paragraphs.

**Critical methodology choices:**
- **9 judges, not 5 or 17.** Why 9? §3.2 explicitly defends: 5 lacked open-weight, 17 hit OpenRouter rate limits and showed diminishing returns. The 5/7/9-judge ablation pattern stays stable past 7 judges.
- **Cross-organization open-weight pair is intentional.** Qwen × Alibaba, Gemma × Google. Prior work either bucketed open-weight or excluded one. Treating them as a *within-cohort* pair tests the cross-organization-but-shared-research-community hypothesis.
- **1,500-char document truncation.** Controls verbosity bias [saito2023verbosity] and caps cost. Without it, judges with longer context windows would systematically agree more (more shared signal).
- **Two-phase run (frontier 7 + supplement 2).** Allows commercial-only ablation as a fallback. Reproducibility note: scores merge by (query_id, rank, point_id) tuple — exact-match required, no interpolation.

### §4 Within-Corpus Results

**§4.1 — Table 2 (per-judge metrics):** This is the table that motivates the entire paper. Ordered by nDCG@10, *bold = per-column extrema*. The 1.9× spread (0.45 to 0.86) is the headline: **same retrieved documents, judge selection alone produces a 1.9× verdict spread.**

**§4.2 — Calibration tiers:** Mean scores partition the slate into three tiers (reasoning-generous 1.63–1.68, strict-mid 1.09–1.18, strict-outlier Gemini). The valid-only mean column is critical: it shows the strict-outlier tier is partly a missing-data artifact. Without the valid-only column, Gemini judges look like outliers; with it, they're transitional strict-mid. **The calibration-tier story is the bridge to C2:** if calibration tiers cross provider lines (and they do — Sonnet, GPT-5.5, DSV4 all reasoning-generous; GPT-4o, Qwen, Gemma 4, Opus all strict-mid), then within-family pairs span tier boundaries while some cross-family pairs share tiers — explaining why within-family agreement is bounded by cross-family agreement.

**§4.3 — κ matrix structure:** Fig. 3 is the paper's primary figure. Two clusters; within-cluster κ exceeds cross-cluster κ. **Calibration philosophy partitions judges more cleanly than provider lineage** — this sentence is the paper's mechanistic punchline before §6 elaborates.

### §5 Findings (C1–C4)

Each claim is restated formally with the supporting evidence pointer. **Read §5 first if you only have 5 minutes** — it's the executive summary.

### §6 Mechanism (long.md only; short.md has 1 paragraph)

The mechanism story progresses from coarsest to finest:
- **§6.1 KL divergence on marginals:** R² = 33%. *Two judges can have identical marginal distributions and still disagree wildly on which specific pairs they assign which score.* Marginal-only is insufficient.
- **§6.2 Joint-distribution structure:** R² = 93%. Two features — *dispersion* (quadratic-weighted disagreement distance) and *effective rank* (4×4 joint matrix's effective rank from SVD). Dispersion alone gives R² = 0.89; +rank lifts to 0.93.
- **§6.3 Mediation analysis:** After controlling for dispersion + rank, all structural factors (provider, reasoning-mode, model class) are non-significant (p > 0.18). They affect κ *only through the joint distribution they induce*. This is a stronger statement than "provider doesn't matter" — provider matters, but its effect size is fully captured by the joint distribution.
- **§6.4 Tokenizer refutation:** Qwen↔Gemma 4 vocabulary Jaccard = 0.066 (lowest in slate) yet κ = 0.80 (matrix-highest). Convergence happens at the *decision-making layer*, not input encoding.
- **§6.5 Open question:** Why do Qwen and Gemma 4 cluster so tightly despite different organizations / tokenizers / training compute? Four candidate hypotheses (shared open-weight community, distillation lineage, compute regime, bounded-ordinal output prior), none confirmed.

**Methodological note:** Mediation analysis follows Baron-Kenny [baronkenny1986] / MacKinnon [mackinnon2007]. The classical framework requires three regression steps; we simplify by using a multi-predictor OLS with structural factors as additional indicators after dispersion + rank. This is full-mediation testing, not partial.

### §7 External Validation (long.md full; short.md compressed)

**§7.1 — TREC RAG 2024:** The headline number is here. 537 stratified-balanced pairs (135/134/134/134 across labels 0/1/2/3, `random.seed(42)`), 86 unique queries, MS MARCO v2.1 passages extracted via HF-streaming. **Per-judge κ table is Table 5 in long.md.** All 9 judges have κ > 0.39; ensemble κ = 0.4941 (moderate, near-substantial). 7-judge frontier ensemble κ = 0.5187.

**§7.2 — BEIR scifact:** 300 pairs, all-positive qrels → κ undefined. Report precision-at-≥2 instead. 9-judge ensemble = 63.7%. Notable spread: Gemma 43%, GPT-5.5 75% — judge selection matters even for the simpler "is this rated relevant?" question.

**§7.3 — Coverage divergence (paper-relevant finding):** Coverage rates differ between corpora. Gemini struggles on TREC RAG 2024 web content (17–24%) but works on BEIR scifact (60–86%). DeepSeek similarly. **The "always-works 6-judge subset" (≥95% on both) is Anthropic + OpenAI + Qwen + Gemma — the 2 cheapest judges make this subset.** This is a content-domain reliability axis that's only visible from cross-corpus comparison.

**§7.4 — Thakur 2025 comparison:** Their GPT-4o κ = 0.60; ours = 0.41 on a stratified-balanced subset. Four candidate explanations for the 0.19 gap; the paper does not claim to match Thakur's number, only to report ours under a different sampling regime.

**§7.5 — Validation budget:** ~$56 total, ~13 hours wall. Compared against an estimated $1,500–3,000 / 6–12 weeks for an IRB human-rater study. ~30× cheaper, ~50× faster.

### §8 Limitations (long.md only)

Eight limitations called out. The most important to defend against:
- **Single internal corpus** (mitigated by §7's external validation, but not fully)
- **Stratified vs natural-distribution sample** (limits direct comparison to Thakur 2025)
- **Gemini coverage failures** (selection-bias risk on the small valid subset)
- **Mediation analysis assumes linear additivity** (non-linear interactions untested)
- **No human-rater study on ISU DSpace** (NIST qrels in §7 substitute, but reviewers may still want a within-corpus human anchor)

**Reviewer-2 anticipation:** §8 frames the no-human-anchor decision proactively. *NIST-graded TREC RAG 2024 qrels — produced by trained annotators following published guidelines — are a stronger external standard than a small per-institution rater pool, and the latter is an institution-specific artifact while the former is a community-shared one.*

### §9 Discussion and Reproducibility

Practitioner takeaway, theoretical contribution, empirical contribution, single-command replication. The reproducibility commands are exact: `py -3 -X utf8 ...` with the precise judge presets and arguments. Marginal cost / wall time per replication is documented.

### §10 Conclusion (long.md only)

Restates C1–C4, mechanism summary, validation summary, four open questions for follow-up.

---

## 5. How to read each figure

Figures are referenced from the paper text but their interpretation isn't always obvious. Here's how to read each one.

### Fig. 3 — `kappa_matrix_9judge.png` (the headline figure)

9×9 heatmap of pairwise quadratic-weighted Cohen's κ on 570 ISU DSpace pairs.

- **Color scale:** RdYlGn (red = low agreement, green = high). Diagonal cells are 1.00 by construction.
- **Cell text:** numerical κ + Landis-Koch band (substantial / moderate / etc.) in italic underneath.
- **Two emergent clusters:** look for the two diagonal-bordered green blocks. Reasoning-generous (Sonnet, GPT-5.5, DSV4) and strict-mid + open-weight (GPT-4o, Opus 4.7, Qwen, Gemma 4).
- **What to point at when explaining:** Qwen↔Gemma 4 = 0.80 (matrix-max), Anthropic Opus↔Sonnet = 0.71 (within-family), Sonnet↔GPT-5.5 = 0.79 (cross-family ceiling). Compare 0.80 vs 0.79 vs 0.71 to make C2 concrete.
- **Bottom-row labels are 45° tilted** to avoid overlap (regenerated 2026-04-28 via `src/make_kappa_heatmap.py`).

### Fig. 6 (long.md only) — `judge_kappa_regression_decomposition.png` / `_panels_AB.png` / `_panels_CD.png`

Four panels supporting §6.2:
- **Panel A:** R² stacked bar chart. Marginal KL alone = 0.33; +dispersion = 0.89; +effective_rank = 0.93. Visual proof of joint > marginal.
- **Panel B:** ΔR² of structural factors after joint controls. All ≤ 0.005. Visual proof of full mediation.
- **Panel C:** Coefficient plot with 95% CIs. Dispersion β = -2.11 (large negative), effective_rank β = -0.087 (small negative). Provider / reasoning / class CIs straddle zero.
- **Panel D:** Predicted-vs-actual κ scatter. R² = 0.928 line of perfect fit visible.

The 4-panel image was split into top (A+B) and bottom (C+D) halves for slide use because the original overflowed slide boundaries.

### `judge_calibration_mechanism.png` (referenced in §6, both papers)

3-panel figure: per-judge marginal score histogram (left), pairwise dispersion vs κ scatter (middle), full-slate calibration topology (right). Used in slide deck for both SHORT and LONG.

### `judge_pair_confusion_matrices.png` / `_top4_highest.png` / `_bottom4_lowest.png`

8 individual 4×4 confusion matrices for the highest-κ and lowest-κ pairs. Highest-κ pairs (Qwen↔Gemma, Sonnet↔GPT-5.5, Sonnet↔DSV4, Gemma↔GPT-4o) show diagonal concentration (judges agree on score). Lowest-κ pairs show off-diagonal mass (systematic disagreement). Visual evidence for the dispersion claim.

### Diagram figures (`diagram_1_*` through `diagram_5_*`)

Conceptual flow diagrams, not data figures:
- **Diagram 1:** Standard RAG flow (retriever → generator)
- **Diagram 2:** Standard eval (single LLM judge)
- **Diagram 3:** Our 9-judge ThreadPool fan-out (used in slide pipeline section)
- **Diagram 4:** Norm vs ours (single-judge eval vs 9-judge κ matrix)
- **Diagram 5:** Calibration topology (the 2-cluster structure)

Each has a `.txt` companion file with the prompt used to generate the diagram, for AI-assisted regeneration.

---

## 6. How to read each table

### Table 1 — Judge slate (§3.2)

5 families × 9 judges. Within-family pairs: rows 1-2 (Anthropic), 3-4 (OpenAI), 5-6 (Google), 8-9 (Open-weight). Reasoning column matters for C1 (cross-family reasoning convergence).

### Table 2 — Per-judge retrieval metrics (§4.1)

The table that motivates C4 (disclosure). Bold = per-column extrema. **The 1.9× nDCG@10 spread (0.45 to 0.86) is the headline number.** Two mean columns (None=0 vs valid-only) split out the missing-data effect on Gemini judges.

### Table 3 — Per-judge marginal score distributions (long.md §6.1)

Each row is a 4-bin probability distribution + entropy. The bolded cell in each row is the modal score for that judge. *Visual signal:* reasoning-generous judges (Sonnet, GPT-5.5, DSV4) have modal scores of 1 or 2 (high mean). Strict-mid (Opus, GPT-4o) have modal score 1 (low mean).

### Table 4 — Mediation analysis (long.md §6.3)

5-row regression: dispersion, effective_rank, same_provider, same_reasoning_mode, same_model_class. **Cumulative R² column is the story:** 0.890 → 0.928 with dispersion+rank, then ≤0.003 incremental from each structural factor. p-values for structural factors all > 0.18. Visual proof of full mediation.

### Table 5 — TREC RAG 2024 per-judge κ vs human qrels (long.md §7.1)

9 rows + ensemble + frontier-only. **Sort by κ** to see the per-judge ranking. Notice valid/537 column: Gemini judges have small valid subsets (small-n caveat applies to their κ). The bottom two rows are the headline: 9-judge ensemble = 0.4941, 7-judge frontier = 0.5187.

### Table 6 — BEIR scifact precision-only (long.md §7.2)

Same structure as Table 5, but precision instead of κ (κ undefined on all-positive qrels). 9-judge ensemble = 63.7%.

### Table 7 — Coverage divergence (long.md §7.3)

The "always-works 6-judge subset" finding. Two columns (TREC RAG 2024 coverage, BEIR scifact coverage). The 6 judges that hit ≥95% on both define the recommended deployment configuration when reliability is paramount.

---

## 7. Editing the paper — what's load-bearing vs decorative

Things that **are** load-bearing (do not edit without thinking through downstream impact):

1. **The literature-gap sentence in §1** (three quantifiers: families × ordinal × ≥500 obs)
2. **The 1.9× nDCG@10 spread** — motivates C4 (without it, disclosure sounds editorial)
3. **The DeepSeek detail in C1** (rules out "Western training data" common cause)
4. **The 537 stratified-balanced sample size and `random.seed(42)`** (reproducibility hook)
5. **The cross-organization framing of Qwen↔Gemma 4** (intentional vs incidental)
6. **The "always-works 6-judge subset" sentence** — gives the paper a deployable recommendation
7. **The "fully mediated" language in §6.3** (precise statistical claim, not "provider doesn't matter")

Things that **are** decorative (can be trimmed if word budget tightens):

1. The §2.4 "RAG evaluation frameworks (positioning)" subsection (long.md only) — valuable context but cuttable if pages overflow
2. §6.5 open-question hypotheses 1–4 — useful for future-work framing, but can be condensed to a single sentence
3. The §7.4 Thakur comparison's 4-explanation list — can be compressed to "sample selection, prompting, qrels version, judge version"
4. Acknowledgments paragraph (long.md) — currently long; can trim to one paragraph for camera-ready

Things that **must NOT** be edited without re-running analysis:

1. **Numerical κ values, R² values, p-values** — these come from the JSON files in `results/` and the analysis scripts in `src/`. To change one, re-run the relevant analysis.
2. **Table 2 / Table 5 / Table 6 row counts and orderings** — sorted by specific metrics (nDCG@10, κ, precision). Changing the sort changes the implicit narrative.
3. **The Landis-Koch interpretation bands** — these are from [landis1977], not paper-specific. Don't relabel "moderate" as "fair" without checking the Landis-Koch source.

---

## 8. The version-control story

| Version | Date | Change |
|---|---|---|
| `long.md` v0.1 | 2026-04-28 | Initial master draft, 8,212 words, 10 sections + 6 appendices |
| `short.md` v0.4 | 2026-04-28 | Condensed from long.md, ≤2,130 words target, ~110 lines |
| `arxiv/main.tex` | 2026-04-28 | LaTeX version of `short.md` ready for arXiv submission |
| `papers/MECHANISTIC_KL_ANALYSIS.md` | — | Standalone narrative for §6 (used as input to long.md §6) |
| `papers/PUBLIC_BENCHMARKS_VALIDATION.md` | — | Standalone results document for §7 (per-corpus tables) |

When a number changes (e.g. you re-run the analysis and ensemble κ becomes 0.4955 instead of 0.4941), the change must be propagated to:
1. `papers/short.md` (abstract + §7 paragraph)
2. `papers/long.md` (abstract + §7.1 Table 5 + §10 conclusion)
3. `arxiv/main.tex` (abstract + §7 paragraph)
4. `arxiv/README.md` (abstract block)
5. `docs/index.html` (KPI tile + §7 list)
6. `slides/make_p4_slides.py` (regenerate both decks)
7. `slides/SLIDES_REFERENCE.md` (Section 3 headline numbers)

A grep for the old number across the repo will surface all occurrences — `grep -r "0.4941" .` is the standard sanity check before committing.

---

## 9. Companion artifacts and where each one fits

| Artifact | Purpose | Used by |
|---|---|---|
| `papers/short.md` | Workshop paper text | Workshop submission |
| `papers/long.md` | Conference / journal paper text | ECIR / TOIS submission |
| `papers/MECHANISTIC_KL_ANALYSIS.md` | §6 source narrative | Long-paper §6, slide deck |
| `papers/PUBLIC_BENCHMARKS_VALIDATION.md` | §7 source results tables | Long-paper §7, slide deck |
| `arxiv/main.tex` + `references.bib` + `figures/` | arXiv-ready LaTeX bundle | arXiv preprint upload |
| `arxiv/README.md` | arXiv submission checklist + abstract for the form | Author reference |
| `slides/P4-SHORT.pptx` (15 slides) | Workshop talk (15-20 min) | Conference presentation |
| `slides/P4-LONG.pptx` (32 slides) | Conference talk (25-30 min) | Conference presentation |
| `slides/make_p4_slides.py` | Slide generator (python-pptx) | Regenerate slides after data changes |
| `slides/SLIDES_REFERENCE.md` | Slide-by-slide map + data-source pointers | Slide editing guide |
| `docs/index.html` | Static results dashboard | GitHub Pages: `asukul.github.io/RAG-Eval-LLM-Judge/` |
| `results/*.json` | Raw judge scores + κ matrices + analysis outputs | Analysis scripts read these |
| `src/eval_llm_judge.py` | Multi-judge harness | Reproducibility |
| `src/validate_against_trec.py` | External-validation harness | Reproducibility |
| `src/make_kappa_heatmap.py` | κ matrix figure generator | Regenerate figures |
| `src/analyze_*.py` | Per-mechanism analysis scripts | Section 6 inputs |

---

## 10. Quick-start: how to edit each paper

**To change a sentence in short.md:**
1. Edit `papers/short.md` directly
2. If the same sentence appears in `arxiv/main.tex`, edit there too
3. Commit + push

**To change a numerical value (e.g., re-run gives κ = 0.4955 not 0.4941):**
1. `grep -r "0.4941" .` to find all occurrences
2. Update each location (papers, arxiv, dashboard, slides reference)
3. Update the `expected = {...}` dict in `src/verify_paper_claims.py` to match the new value
4. Re-run `slides/make_p4_slides.py` to update PowerPoint
5. Run `py -3 src/verify_paper_claims.py` — must report 0 fail before commit
6. Commit all changes in one batch with a message explaining the source of the new number

The verifier `src/verify_paper_claims.py` is the single source of truth for "every numerical claim in every paper draft + dashboard is consistent with the shipped data." Run it before every commit that touches numbers.

**To add a new finding (e.g., TREC-COVID validation):**
1. Add to `papers/long.md` §7 (full paragraph + per-judge table per user's preference)
2. Add to `papers/short.md` §7 (one sentence — workshop word budget)
3. Add to `arxiv/main.tex` §7 (one sentence)
4. Add to `docs/index.html` validation section
5. Update `slides/make_p4_slides.py` (add a slide or extend an existing one)
6. Update `slides/SLIDES_REFERENCE.md` Section 3 (headline numbers)
7. Run `_verify_citations.py` if a new citation was added

**To add a new figure:**
1. Add the PNG to `figures/`
2. Reference it in the relevant paper section via `figures/<name>.png`
3. Add a slide function in `slides/make_p4_slides.py` and add to `build_short()` or `build_long()`
4. Document in `slides/SLIDES_REFERENCE.md` Section 4 ("Figures embedded in slides")
5. If using in arXiv, copy to `arxiv/figures/` and update `arxiv/main.tex`

**To add a new citation:**
1. Add the BibTeX entry to `arxiv/references.bib`
2. Add a `\citep{newkey}` (or `\citet{newkey}`) reference in `arxiv/main.tex`
3. Run `arxiv/_verify_citations.py` to confirm all keys resolve
4. Mirror in `papers/short.md` and `papers/long.md` if used there

---

## 11. Reviewer Q&A — what to expect and where to point

**"Why only one corpus?"** → §7 (external validation against NIST TREC RAG 2024 + BEIR scifact + TREC-COVID)

**"How do I know the κ matrix isn't artifactual?"** → §6 (mechanism: 93% R² from joint distribution; structural factors fully mediated; tokenizer hypothesis refuted)

**"Why these 9 judges and not others?"** → §3.2 (5/7/9-judge ablation pattern stable past 7; cross-organization open-weight intentional; DSV4 Flash dropped due to throttling — disclosed)

**"Your κ = 0.49 is below Thakur 2025's 0.60 — why?"** → §7.4 (sample selection: stratified-balanced vs natural-distribution-proportional; prompting may differ; qrels version may differ; judge version may differ)

**"Your finding contradicts self-preference [panickssery2024self]"** → C2 + §5 (proposed mechanism: self-preference is mediated by output-space boundedness; companion 4-way extraction study supports the inversion under bounded ordinal)

**"How does this transfer to a different deployment?"** → §9 + §7.3 (always-works 6-judge subset = Anthropic + OpenAI + Qwen + Gemma; calibration philosophy is the axis, not provider; recommended sovereign-cloud config = Qwen + Gemma)

**"Is the open-weight κ = 0.80 robust?"** → §6.4 (tokenizer refutation rules out the obvious "shared encoding" explanation), §7 (both judges hit 100% coverage on TREC RAG 2024 and individual κ = 0.41 each), and §8 limitation (the most fragile claim across domains; flagged for follow-up replication)

**"What's the budget?"** → §9 + §7.5 (~$56 total, ~13 hours wall; ~30× cheaper / ~50× faster than IRB human-rater study)

---

## 12. Summary

The papers make four claims (C1–C4) about LLM-judge agreement on bounded ordinal IR judgment, each anchored in a specific evidence slice. **§6 and §7 are independent robustness checks against different attacks** — §6 against artifactual-structure objections, §7 against single-corpus objections.

The argument graph flows: §3 methodology → §4 within-corpus results (Tables 2, Fig. 3) → §5 claims (C1–C4) → §6 mechanism (long only) → §7 external validation. Section 8 limitations is reviewer-facing pre-emption. Section 9 reproducibility is the artifact contribution (C4).

**The paper is one connected argument, not a list of findings.** Every sentence in the four-claim section connects to evidence elsewhere; every methodology choice is defended; every limitation is matched by a counter-argument. Edit with the argument graph in mind.
