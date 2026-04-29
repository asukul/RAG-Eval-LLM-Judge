---
license: cc-by-4.0
language:
  - en
size_categories:
  - 1K<n<10K
task_categories:
  - text-classification
task_ids:
  - text-scoring
  - relevance-judgment
pretty_name: "RAG-Eval-LLM-Judge: Cross-Family LLM-Judge Agreement Dataset"
tags:
  - llm-as-judge
  - retrieval-augmented-generation
  - information-retrieval
  - inter-rater-agreement
  - meta-evaluation
  - cohens-kappa
  - trec-rag
  - beir
configs:
  - config_name: within_corpus_isu_dspace
    data_files: "results/within_corpus/judge_*.json"
  - config_name: trec_rag_2024
    data_files: "results/trec-rag-2024_judges.json"
  - config_name: trec_covid
    data_files: "results/trec-covid_judges.json"
  - config_name: beir_scifact
    data_files: "results/beir-scifact_judges.json"
  - config_name: bootstrap_cis
    data_files: "results/bootstrap_kappa_cis.json"
  - config_name: gwet_ac2
    data_files: "results/gwet_ac2_alongside_kappa.json"
  - config_name: bias_diagnostics
    data_files: "results/bias_diagnostics.json"
  - config_name: intra_judge_consistency
    data_files: "results/intra_judge_consistency.json"
---

# Dataset Card: RAG-Eval-LLM-Judge — Cross-Family LLM-Judge Agreement Dataset

**Release:** v1.0 (2026-04-29) · **License:** MIT (code) + CC-BY-4.0 (data, figures, papers) · **Repository:** https://github.com/asukul/RAG-Eval-LLM-Judge · **Author:** Adisak Sukul (Iowa State University, asukul@iastate.edu)

---

## Dataset Summary

This dataset is a **comprehensive empirical artifact for studying inter-LLM-judge agreement on bounded ordinal information-retrieval relevance judgment**. It contains **9,963 individual LLM-judge score records** across:

- **Within-corpus 9-judge ablation** on 570 query-document pairs from the Iowa State University DSpace institutional repository (97k full-text academic PDFs): **9 judges × 570 pairs = 5,130 records**.
- **External validation** of the same 9-judge slate against three public corpora with NIST/BEIR human qrels: TREC RAG 2024 (537 stratified-balanced pairs), TREC-COVID biomedical (300 pairs), BEIR scifact (300 pairs). **9 × (537 + 300 + 300) = 12,033 records, with 4,833 valid (the rest are pairwise null returns)**.
- **Intra-judge self-consistency** subset: 50 pairs × 9 judges × 3 runs = **1,350 records** measuring run-to-run reliability.
- **UMBRELA single-judge baseline** (Upadhyay et al. 2024) on the same 537-pair TREC RAG 2024 sample for direct comparison.

The **9 judges** span 5 model families with 4 within-family pairs:

| # | Judge | Family | Reasoning mode |
|---|---|---|---|
| 1 | Claude Opus 4.7 | Anthropic | yes |
| 2 | Claude Sonnet 4.6 | Anthropic | yes |
| 3 | GPT-5.5 (reasoning=low) | OpenAI | yes |
| 4 | GPT-4o (chat) | OpenAI | no |
| 5 | Gemini 3.1 Pro Preview | Google-commercial | yes (thinking) |
| 6 | Gemini 2.5 Pro | Google-commercial | yes (thinking) |
| 7 | DeepSeek V4 Pro | DeepSeek | yes |
| 8 | Qwen 3.6 Plus | Open-weight | no |
| 9 | Gemma 4 26B | Open-weight | no |

The dataset is **the first published five-family pairwise quadratic-weighted Cohen's κ matrix** on a bounded 0–3 ordinal relevance rubric at ≥ 500 paired observations with within-family controls and open-weight peers as first-class participants.

## Supported Tasks

- **Inter-rater agreement studies** (`text-scoring`): the score arrays + pair indices support reproducing the 9×9 κ matrix and any subset thereof.
- **LLM judge meta-evaluation**: per-judge κ vs human qrels (4 corpora) supports comparison of new judges against the published 9-judge baseline.
- **Bias and self-consistency research**: the intra-judge-consistency subset (50 × 9 × 3) and 5×5 family conditional-mean matrix support new analyses of judge stability and family-level calibration drift.
- **Mechanism studies**: the joint-distribution analysis (dispersion, effective rank, KL divergence on marginals) is reproducible from the score records.
- **Cross-corpus validation**: judge slates can be applied to additional public corpora and compared against our published per-judge κ.

## Dataset Structure

The release is organized as **8 logical configurations** (corresponding to the `configs:` section in the YAML header), each backed by JSON files in `results/`.

### Configuration: `within_corpus_isu_dspace`
Files: `results/within_corpus/judge_*.json` (9 files, ~340 KB each, total ~3 MB) + `results/within_corpus/multijudge_9judge_merged.json` (~8 KB).

Each per-judge file is shaped:

```json
{
  "config": {
    "judge_label": "<display name>",
    "judge_spec":  "<spec ID>",
    "collection":  "dspace_fulltext_vertex",
    "top_k":       10,
    "max_chunk_chars": 1500,
    "n_queries":    57,
    "timestamp":    "20260425_HHMMSS"
  },
  "aggregate": {
    "ndcg@10":          <float>,
    "precision@5":      <float>,
    "mrr":              <float>,
    "mean_judge_score": <float>,
    "n_judged":         <int>
  },
  "queries": [
    {
      "query_id": "<id>",
      "query":    "<text>",
      "tags":     ["intent_class", "topic_cluster", ...],
      "metrics":  { "ndcg@10": <float>, "precision@5": <float>, "mrr": <float> },
      "retrieved": [
        {
          "rank":         <int 1..10>,
          "qdrant_score": <float>,
          "point_id":     "<UUID>",
          "title":        "<document title>",
          "text_preview": "<240-char snippet>",
          "judge_score":  <int 0..3 | null>
        }, ...
      ]
    }, ...
  ]
}
```

The merged file has `config + aggregates_per_judge + kappa_matrix` (no per-pair raw scores).

### Configurations: `trec_rag_2024`, `trec_covid`, `beir_scifact`
External-validation files use a flatter shape:

```json
{
  "pair_index":   [["<query_id>", "<doc_id>"], ...],
  "human_scores": [<int>, ...],
  "judge_specs":  ["<spec1>", ...],
  "per_judge_scores":   {"<judge_label>": [<int|null>, ...]},
  "per_judge_metadata": {"<judge_label>": {"n_valid": <int>, "n_missing": <int>, "mean_score": <float>}}
}
```

`pair_index` and `human_scores` are aligned positionally; each `per_judge_scores[label]` array has the same length as `pair_index`.

| Corpus | n_pairs | n_judges | Total records | Human qrels source |
|---|---:|---:|---:|---|
| TREC RAG 2024 (stratified-balanced) | 537 | 9 | 4,833 | NIST 2024 retrieval-qrels |
| TREC-COVID (BEIR-distributed) | 300 | 9 | 2,700 | BEIR scifact `qrels/test.tsv` |
| BEIR scifact | 300 | 9 | 2,700 | BEIR scifact `qrels/test.tsv` |

### Configuration: `intra_judge_consistency`
File: `results/intra_judge_consistency.json` (~30 KB).

Shape: `{config, pair_index, human_scores, per_judge: {<label>: {runs: [[scores], [scores], [scores]], mean_intra_judge_kappa, mean_kappa_vs_human, ...}}}`. The `runs` field has 3 score arrays (one per re-run), each length 50.

### Configurations: `bootstrap_cis`, `gwet_ac2`, `bias_diagnostics`
Pre-computed analytical outputs that derive from the raw score files; provided for fast verification of paper claims without running the analysis.

## Data Splits

There is **no train/test split** — the entire dataset is observational (LLM-judge scores). Researchers re-using the dataset for new analyses should pick splits relevant to their question (e.g., by query intent class, by document length quartile, by judge family).

## Curation Rationale

LLM-as-Judge has become a de-facto evaluation standard for retrieval-augmented generation and IR systems, but **prior published work compared judges either within a single provider family, on two-family pairs without ordinal weighting, or excluded open-weight peers entirely**. Our motivation was to build the smallest dataset that closes those three gaps simultaneously: five model families × within-family controls × open-weight peers as first-class participants × a bounded 0–3 ordinal relevance rubric × ≥ 500 paired observations.

## Source Data

- **Iowa State University DSpace** (within-corpus): 97,441 full-text academic PDFs harvested from a US-public-university institutional repository, embedded with Google Vertex `text-embedding-005` in Qdrant 1.13. Documents are public scholarly content (theses, conference proceedings, journal volumes, etc.); see `data/dspace_index_stats.json` for the collection breakdown.
- **MS MARCO v2.1 segmented** (TREC RAG 2024): 537 passages extracted via HF-streaming filter from `drexalt/msmarco-2.1-segmented`. The 537 IDs are stratified-balanced over the 4 NIST relevance labels (135/134/134/134 with `random.seed(42)`).
- **TREC-COVID** (biomedical): 300 pairs from `BeirGenericDataLoader` on the TREC-COVID test split; 0/1/2 NIST-qrels mapped to our 0/2/3 rubric via `{0:0, 1:2, 2:3}`.
- **BEIR scifact** (scientific fact-verification): 300 pairs from `BeirGenericDataLoader` on the scifact test split; all-positive qrels (κ undefined → precision-at-≥2 reported instead).

## Annotations

**Annotators**: 9 LLM judges (Anthropic, OpenAI, Google, DeepSeek, Open-weight) — see judge slate above. **No human annotations were collected for this work**; we use existing NIST and BEIR human qrels as the ground-truth anchor for §7 external validation.

**Rubric**: 0 = irrelevant, 1 = topical, 2 = partial, 3 = fully answers. Calibrated at prompt time with five worked examples from a held-out seed set. Documents truncated to 1,500 chars (judge-input cap) before scoring.

**Inter-rater statistic**: pairwise quadratic-weighted Cohen's κ on the 36 unique judge pairs (within-corpus) and judge-vs-human κ for external validation.

## Personal and Sensitive Information

- **ISU DSpace**: public scholarly content; no personally identifiable information beyond author names that appear on public publications. Documents are released under licenses chosen by their authors and the ISU Library.
- **TREC RAG 2024**: MS MARCO v2.1 web passages; pre-published by NIST/Microsoft.
- **TREC-COVID, BEIR scifact**: pre-published biomedical/scientific corpora distributed by the BEIR consortium.
- **No human-rater data was collected** during this work; all human relevance labels are derived from previously-published NIST or BEIR qrels.
- **API responses** from commercial models (Anthropic, OpenAI, Google, DeepSeek, OpenRouter) were obtained under the respective providers' terms of service for research use; we redistribute only the integer/null score outputs, not raw API response text.

## Considerations for Using the Data

### Social Impact

- **Positive**: enables IR practitioners and researchers to triangulate LLM-judge choice without per-institution IRB studies (which are estimated at $1,500–3,000 and 6–12 weeks). The "always-works 6-judge subset" finding (§7.3) gives a concrete deployment recommendation for sovereign-cloud / privacy-conservative institutions.
- **Negative**: relying on LLM judges (any of them) introduces evaluation biases that may differ systematically from human annotators. We document the κ vs human qrels gap (per-judge κ ranges 0.40–0.55 on TREC RAG 2024) so users can calibrate expectations.

### Discussion of Biases

- **Verbosity bias** (Saito 2023): mitigated via 1,500-char truncation cap; not directly measured because stored `text_preview` is uniformly truncated at 240 chars (storage-side artifact).
- **Position bias** (Wang 2023): not directly applicable — our rubric scores one document at a time, not pairwise alternatives.
- **Self-preference** (Panickssery 2024): the 5×5 family conditional-mean matrix in `results/bias_diagnostics.json` shows calibration drift, *not* self-preference, on this dataset (no family's diagonal dominates its row).
- **Coverage divergence** (paper-relevant): on TREC RAG 2024 web passages, Gemini judges return scores on only 17–24% of pairs (thinking-mode parse aborts); on biomedical content (TREC-COVID), only 6–44%. Anthropic + OpenAI + Qwen + Gemma form an "always-works 6-judge subset" with ≥ 95% coverage on all three external corpora.
- **Marginal skew** (kappa paradox): Cohen's κ values can be deflated under skewed marginal distributions. We report Gwet's AC2 alongside κ in `results/gwet_ac2_alongside_kappa.json` for triangulation; AC2 runs ≈ 0.25 higher than κ across the slate.

### Other Known Limitations

- **Single internal corpus** for the within-corpus 9×9 matrix (ISU DSpace). Cross-corpus replication is in §7.
- **Stratified vs natural-distribution sampling**: our 537-pair TREC RAG 2024 sample is balanced over the 4 ordinal labels (135/134/134/134); Thakur 2025's sample was natural-proportional. Direct numerical comparison is not strictly apples-to-apples (see paper §7.4).
- **DeepSeek V4 Flash dropped** due to OpenRouter free-tier 429 throttling. We have one DeepSeek judge (V4 Pro), no within-DeepSeek-family pair.
- **Missing-data accounting**: aggregate metrics treat null as 0; valid-only means are reported alongside (long.md Table 4) so the impact is transparent.
- **Intra-judge κ on n = 50 pairs has wide bootstrap CIs**; absolute values are point estimates suitable for ranking, not for asserting "judge X is materially more self-consistent than judge Y."

## Additional Information

### Citation

```bibtex
@misc{sukul2026rageval,
  title         = {{RAG-Eval-LLM-Judge: Cross-Family LLM-Judge Agreement Dataset for Institutional Retrieval-Augmented Generation}},
  author        = {Sukul, Adisak},
  year          = {2026},
  note          = {Companion dataset to the LLM4Eval @ SIGIR 2027 / ECIR 2027 paper. arXiv preprint forthcoming.},
  url           = {https://github.com/asukul/RAG-Eval-LLM-Judge},
  doi           = {forthcoming}
}
```

See `CITATION.cff` for machine-readable metadata.

### License

- **Code** (`src/`, `slides/make_p4_slides.py`, `arxiv/_verify_citations.py`, all build scripts): **MIT License**.
- **Data, figures, and paper text** (`results/`, `figures/`, `papers/`, `arxiv/`, `docs/`, `data/`): **Creative Commons Attribution 4.0 International (CC-BY-4.0)**.
- **Third-party redistributed inputs**:
  - **MS MARCO v2.1 segmented passages** (`data/passages.json`, 537 entries): redistributed under MS MARCO's license terms; original corpus by Microsoft / NIST.
  - **NIST TREC RAG 2024 qrels** (`data/2024-retrieval-qrels.txt`): public-domain government work, redistributed unchanged.
  - **BEIR scifact and BEIR-distributed TREC-COVID** are auto-downloaded by `src/validate_against_trec.py` from the BEIR consortium URLs; we ship only our derived score records, not the raw corpora.

### Maintenance Plan

- **Versioning**: this dataset card is for **v1.0** (2026-04-29). Future versions will be tagged via GitHub releases. Significant additions (e.g., LLMJudge 2025 benchmark, TREC RAG 2025 replication) will bump the major version.
- **Updates**: corrections (numerical errata, misattribution fixes) will be patched and the change documented in the commit log + a `CHANGELOG.md`.
- **Contact**: open a GitHub issue at https://github.com/asukul/RAG-Eval-LLM-Judge/issues or email asukul@iastate.edu. Response target: 30 days for substantive issues.
- **Hosting**: GitHub (primary). A mirror on Zenodo (with DOI) is planned at NeurIPS-camera-ready time if accepted.
- **Reproducibility tooling**: every numerical claim in the paper is verified by `src/verify_paper_claims.py` against the shipped JSONs. Latest verifier run: `results/verification_log.txt`. Pytest tests: `tests/`. Both run in < 5 seconds.

### Contributions

- **Adisak Sukul** (Iowa State University): all experimental design, methodology, harness implementation, paper drafts, and dataset curation.
- **AI assistance**: we acknowledge the use of Claude Code as a coding assistant for harness implementation, figure regeneration, and pytest test development. All experimental design, methodology, and analytical interpretation are the author's.
- **Reviewers (round 1, 2026-04-29)**: ChatGPT 5.5 deep-research and Claude Opus 4.7 deep-research-v2 provided critical feedback that drove the P1–P3 revision cycle (§5 C1 arithmetic fixes, §6.2 reframing, path bug fixes, UMBRELA baseline, intra-judge self-consistency, bootstrap CIs). Their reviews + my responses are not part of this public release but are preserved in the project's internal Paper-reviews/ folder.
