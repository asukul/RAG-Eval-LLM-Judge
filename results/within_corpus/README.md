# Within-corpus 9-judge ablation — raw per-judge JSONs

This directory holds the raw per-judge score files for the **within-corpus 9-judge ablation on Iowa State University DSpace** (97k full-text PDFs → 1.03M chunks → top-10 retrieval per query, 57 queries → 570 (query, document) pairs). Each file contains every judge's score on every retrieved (query, document) pair, so the within-corpus 9×9 κ matrix in `figures/kappa_matrix_9judge.png` and the §4 / §6 numerical claims in `papers/long.md` are now independently reproducible from this folder alone.

## Files

| File | Judge | Phase | Records |
|---|---|---|---|
| `judge_claude_opus_4.7.json` | Claude Opus 4.7 (OpenRouter) | frontier | 570 |
| `judge_claude_sonnet_4.6.json` | Claude Sonnet 4.6 (OpenRouter) | frontier | 570 |
| `judge_gpt-5.5_reasoning_low.json` | GPT-5.5 (reasoning=low) | frontier | 570 |
| `judge_gpt-4o.json` | GPT-4o (chat) | frontier | 570 |
| `judge_gemini_3.1_pro_preview.json` | Gemini 3.1 Pro Preview (OpenRouter) | frontier | 570 |
| `judge_gemini_2.5_pro.json` | Gemini 2.5 Pro (OpenRouter) | frontier | 570 |
| `judge_deepseek_v4_pro.json` | DeepSeek V4 Pro (OpenRouter) | frontier | 570 |
| `judge_qwen_3.6_plus.json` | Qwen 3.6 Plus (OpenRouter) | open-weight supplement | 570 |
| `judge_gemma_4_26b.json` | Gemma 4 26B (OpenRouter) | open-weight supplement | 570 |
| `multijudge_9judge_merged.json` | (combined) | merged 9-judge canonical | aggregates + κ matrix only |

Total: 9 per-judge files × 570 records = **5,130 within-corpus score records.**
File timestamps: frontier-7 = `20260425_034626`; open-weight supplement = `20260425_102101` (renamed for readability).

## Schema (per-judge file)

```json
{
  "config": {
    "judge_label": "<display name, e.g. Claude Sonnet 4.6 (OpenRouter)>",
    "judge_spec":  "<spec ID, e.g. claude-sonnet>",
    "collection":  "dspace_fulltext_vertex",
    "qdrant_url":  "...",
    "top_k":       10,
    "max_chunk_chars": 1500,
    "queries_file": "...",
    "timestamp":    "YYYYMMDD_HHMMSS",
    "elapsed_seconds": <int>,
    "n_queries":    57
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
      "query":    "<query text>",
      "tags":     ["intent_class", "topic_cluster", ...],
      "metrics":  { "ndcg@10": <float>, "precision@5": <float>, "mrr": <float> },
      "retrieved": [
        {
          "rank":         <int 1..10>,
          "qdrant_score": <float>,
          "point_id":     "<UUID-style document ID>",
          "title":        "<document title>",
          "text_preview": "<truncated content snippet sent to the judge>",
          "judge_score":  <int 0..3 | null>
        },
        ...
      ]
    },
    ...
  ]
}
```

## Schema (`multijudge_9judge_merged.json`)

The merged file is **aggregates + κ matrix only** (no per-pair raw scores; those live in the per-judge files above):

```json
{
  "config": {
    "judge_labels": ["Claude Opus 4.7 ...", ..., "Gemma 4 26B ..."],
    "n_queries":          57,
    "n_retrieved_pairs":  570,
    "merged_from":        ["20260425_034626", "20260425_102101"],
    "retrieval_determinism_mismatches": 0,
    "mode":               "merged-multi-judge"
  },
  "aggregates_per_judge": {
    "<judge_label>": { "ndcg@10": ..., "precision@5": ..., "mrr": ..., "mean_judge_score": ..., "n_judged": ... },
    ...
  },
  "kappa_matrix": {
    "<row_judge_label>": { "<col_judge_label>": <κ float | null>, ... },
    ...
  }
}
```

## How to recompute the 9×9 κ matrix from these files

```python
import json
from pathlib import Path
HERE = Path("results/within_corpus")

# Load per-judge files into {label: [score_per_pair_in_consistent_order]}
def flat_scores(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["config"]["judge_label"], [
        r["judge_score"]
        for q in d["queries"]
        for r in sorted(q["retrieved"], key=lambda x: x["rank"])
    ]

per_judge = dict(flat_scores(p) for p in HERE.glob("judge_*.json"))

# Pairwise κ; reuse cohen_kappa_quadratic from src/eval_llm_judge.py
import sys; sys.path.insert(0, "src")
from eval_llm_judge import cohen_kappa_quadratic
labels = list(per_judge)
matrix = {a: {b: cohen_kappa_quadratic(per_judge[a], per_judge[b]) for b in labels} for a in labels}
```

The `src/verify_paper_claims.py` script does this automatically and asserts:
- Recomputed within-pair κ values match those reported in the paper text
- The four cited highest-κ pairs and four lowest-κ pairs come out exactly as cited
- Off-diagonal min ≥ 0.56 holds

## Retrieval determinism

All 9 per-judge files share the same `(query_id, rank, point_id)` tuple sequence — verified by the merge harness at run time (`retrieval_determinism_mismatches: 0` in the merged JSON). This means score arrays from different judges align positionally without any realignment step, which is why pairwise κ across judges is computable directly.

## Provenance

Generated 2026-04-25 by `src/eval_llm_judge.py` using the `p4-frontier` and `p4-supplement-openweight` judge presets. The full run log lives at `results/logs/` (frontier and supplement logs from the 2026-04-24 / 2026-04-25 session). Total cost: ~USD 18.30. Total wall: 154 + 175.6 minutes.

## License

Code in `src/` is MIT. Data in this folder is **CC-BY-4.0** (with the underlying ISU DSpace document content credited to the original authors and the Iowa State University Library and Digital Initiatives team). When citing this dataset, use the entry in `CITATION.cff` plus a clear statement that the score values are author-generated LLM judgments and the underlying document content is third-party.
