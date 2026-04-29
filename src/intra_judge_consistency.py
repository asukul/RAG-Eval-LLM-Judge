"""Intra-judge self-consistency: re-run each judge 3 times on a 50-pair subset.

Addresses the "Rating Roulette" finding (EMNLP Findings 2025) that LLM
judges can be much less self-consistent than commonly assumed.

Method:
  - Take 50 pairs from the 537-pair TREC RAG 2024 stratified-balanced
    sample (random.seed(42) for reproducibility).
  - For each of the 9 judges, run 3 independent calls on those 50 pairs
    via the same JUDGE_BUILDERS used in the within-corpus run.
  - Compute pairwise intra-judge kappa across the 3 runs per judge.
  - Compare to mean kappa vs human qrels on the same 50-pair subset.

Run from repo root:
    py -3 src/intra_judge_consistency.py [--n-pairs 50] [--n-runs 3]

Output: results/intra_judge_consistency.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Import production judge infrastructure -- guaranteed to use the same
# API call patterns as the published per-judge JSONs.
from eval_llm_judge import (  # noqa: E402
    JUDGE_BUILDERS, _load_dotenv_manual, cohen_kappa_quadratic,
)

DOTENV = REPO / ".env"
if DOTENV.exists():
    _load_dotenv_manual(DOTENV)


def load_50_pair_subset(n_pairs, seed):
    """Load n pairs from the 537-pair stratified sample, deterministically."""
    sample_path = REPO / "data" / "sample_537_pairs.tsv"
    passages_path = REPO / "data" / "passages.json"
    topics_path = REPO / "data" / "topics.rag24.test.txt"

    queries = {}
    for line in topics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            queries[parts[0].strip()] = parts[1].strip()

    passages = json.loads(passages_path.read_text(encoding="utf-8"))

    all_pairs = []
    with sample_path.open(encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, docid, rel = parts[0], parts[1], int(parts[2])
            if qid not in queries or docid not in passages:
                continue
            psg = passages[docid]
            text = (psg.get("title", "") + "\n\n"
                    + (psg.get("headings", "") or "") + "\n\n"
                    + (psg.get("text", "") or ""))[:1500]
            all_pairs.append({
                "query_id": qid, "doc_id": docid,
                "query": queries[qid], "passage": text,
                "human_rel": rel,
            })

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(all_pairs)), n_pairs))
    return [all_pairs[i] for i in indices]


def run_judge_round(judge_spec, pairs, workers=8):
    """Build one judge from JUDGE_BUILDERS and score all pairs in parallel."""
    label, judge = JUDGE_BUILDERS[judge_spec]()
    out = [None] * len(pairs)

    def _do(i, pair):
        try:
            return i, judge.score(pair["query"], pair["passage"])
        except Exception:  # noqa: BLE001
            return i, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_do, i, p) for i, p in enumerate(pairs)]
        for fut in as_completed(futures):
            i, score = fut.result()
            out[i] = score
    return label, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pairs = load_50_pair_subset(args.n_pairs, args.seed)
    print(f"Loaded {len(pairs)} pairs from TREC RAG 2024 (seed={args.seed})", flush=True)

    judge_specs = [
        "claude-opus-4.7", "claude-sonnet",
        "openai-gpt-5.5-low", "openai-gpt-4o",
        "gemini-3.1-pro", "gemini-2.5-pro",
        "deepseek-v4-pro",
        "qwen-3.6-plus", "gemma-4-26b",
    ]
    print(f"Running {len(judge_specs)} judges x {args.n_runs} rounds = "
          f"{len(judge_specs) * args.n_runs * len(pairs)} total calls", flush=True)
    print("=" * 72, flush=True)

    out = {
        "config": {
            "n_pairs": len(pairs),
            "n_runs": args.n_runs,
            "seed": args.seed,
            "subset_source": "data/sample_537_pairs.tsv",
            "method": "intra-judge kappa across N runs of same judge on same pairs",
            "judge_specs": judge_specs,
        },
        "pair_index": [(p["query_id"], p["doc_id"]) for p in pairs],
        "human_scores": [p["human_rel"] for p in pairs],
        "per_judge": {},
    }

    t0 = time.time()
    for ji, spec in enumerate(judge_specs, start=1):
        print(f"\n[{ji}/{len(judge_specs)}] {spec} ...", flush=True)
        runs = []
        label = spec
        for run_i in range(args.n_runs):
            tj = time.time()
            label, scores = run_judge_round(spec, pairs, workers=args.workers)
            valid = sum(1 for s in scores if s is not None)
            print(f"  run {run_i+1}/{args.n_runs}: valid={valid}/{len(pairs)}  "
                  f"({time.time()-tj:.1f}s)", flush=True)
            runs.append(scores)

        intra_kappas = []
        for i in range(args.n_runs):
            for j in range(i + 1, args.n_runs):
                k = cohen_kappa_quadratic(runs[i], runs[j])
                if k is not None:
                    intra_kappas.append(k)
        mean_intra = (sum(intra_kappas) / len(intra_kappas)
                      if intra_kappas else None)
        kappa_vs_humans = []
        for r in runs:
            k = cohen_kappa_quadratic(r, out["human_scores"])
            if k is not None:
                kappa_vs_humans.append(k)
        mean_vs_human = (sum(kappa_vs_humans) / len(kappa_vs_humans)
                         if kappa_vs_humans else None)

        out["per_judge"][label] = {
            "spec": spec,
            "runs": runs,
            "mean_intra_judge_kappa": (round(mean_intra, 4)
                                        if mean_intra is not None else None),
            "mean_kappa_vs_human": (round(mean_vs_human, 4)
                                     if mean_vs_human is not None else None),
            "intra_kappa_pairs": [round(k, 4) for k in intra_kappas],
            "n_valid_per_run": [sum(1 for s in r if s is not None) for r in runs],
        }
        if mean_intra is not None:
            print(f"  -> mean intra-judge kappa = {mean_intra:.4f}", flush=True)
        if mean_vs_human is not None:
            print(f"  -> mean kappa vs human   = {mean_vs_human:.4f}", flush=True)

    elapsed = time.time() - t0
    out["config"]["wall_seconds"] = round(elapsed, 1)

    print(f"\n=== Summary (wall={elapsed/60:.1f} min) ===", flush=True)
    print(f"{'judge':<35s}  {'intra-K':>7s}  {'vs-human':>9s}  {'delta':>7s}", flush=True)
    for j, m in out["per_judge"].items():
        i_k = m["mean_intra_judge_kappa"]
        v_k = m["mean_kappa_vs_human"]
        delta = (i_k - v_k) if (i_k is not None and v_k is not None) else None
        i_str = f"{i_k:.4f}" if i_k is not None else "n/a"
        v_str = f"{v_k:.4f}" if v_k is not None else "n/a"
        d_str = f"{delta:+.4f}" if delta is not None else "n/a"
        print(f"  {j[:33]:<33s}  {i_str:>7s}  {v_str:>9s}  {d_str:>7s}", flush=True)

    out_path = REPO / "results" / "intra_judge_consistency.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
