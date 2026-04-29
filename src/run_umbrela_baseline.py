"""UMBRELA baseline on the 537-pair TREC RAG 2024 sample.

UMBRELA (Upadhyay et al. 2024) is the open-source single-judge baseline
deployed at the TREC 2024 RAG track. We re-run their published prompt
using GPT-4o (UMBRELA's default backbone) on the same 537 stratified-
balanced pairs that our 9-judge ensemble used, so the comparison is
prompt-and-rubric-only (same model, same pairs, same human qrels).

What this gives us for the paper:

  - A direct apples-to-apples baseline column for the per-judge kappa
    table (Table 5). The reviewers (ChatGPT 5.5 + Claude Opus 4.7,
    2026-04-29) flagged "no baselines vs UMBRELA" as the single most
    likely rejection reason at ECIR/NeurIPS.
  - Evidence that our ensemble outperforms a single-judge UMBRELA-style
    setup, supporting C4 (the disclosure-template + ensemble proposal).

The UMBRELA prompt is reproduced verbatim from Upadhyay et al. 2024
(arXiv:2406.06519), Section 3.

Run from repo root:
    py -3 src/run_umbrela_baseline.py [--max-pairs 537]

Output: results/umbrela_baseline_trec_rag_2024.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Reuse the same .env loader so API keys come from the standard location.
from eval_llm_judge import _load_dotenv_manual, cohen_kappa_quadratic  # noqa: E402

DOTENV = REPO / ".env"
if DOTENV.exists():
    _load_dotenv_manual(DOTENV)


# UMBRELA prompt — verbatim from Upadhyay et al. 2024, Section 3
UMBRELA_PROMPT_TEMPLATE = """\
Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query
1 = represents that the passage seems related to the query but does not answer it
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies score it as 0.

Query: {query}
Passage: {passage}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning.
"""


def parse_umbrela_score(text: str):
    """Extract integer 0-3 from UMBRELA-format response. Returns None on failure."""
    if not text:
        return None
    m = re.search(r"##\s*final\s*score:\s*([0-3])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: any 0-3 digit at end of text
    m = re.search(r"\b([0-3])\b\s*\Z", text.strip())
    if m:
        return int(m.group(1))
    # Last resort: scan whole text for first 0-3 digit
    m = re.search(r"\b([0-3])\b", text)
    if m:
        return int(m.group(1))
    return None


def call_gpt4o_with_prompt(prompt: str, max_retries: int = 3,
                           backoff_seconds: float = 2.0):
    """Call GPT-4o via OpenAI API with the UMBRELA prompt. Returns text or None."""
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    client = OpenAI(api_key=api_key)

    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (2 ** attempt))
    print(f"  WARN: gave up after {max_retries} retries: {last_exc}")
    return None


def load_537_sample():
    """Load the same 537-pair stratified-balanced TREC RAG 2024 sample."""
    sample_path = REPO / "data" / "sample_537_pairs.tsv"
    passages_path = REPO / "data" / "passages.json"
    topics_path = REPO / "data" / "topics.rag24.test.txt"
    if not (sample_path.exists() and passages_path.exists()
            and topics_path.exists()):
        raise FileNotFoundError(
            "TREC RAG 2024 inputs missing from data/. Need "
            "sample_537_pairs.tsv, passages.json, topics.rag24.test.txt.")

    queries = {}
    for line in topics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            queries[parts[0].strip()] = parts[1].strip()

    passages = json.loads(passages_path.read_text(encoding="utf-8"))

    pairs = []
    with sample_path.open(encoding="utf-8") as f:
        f.readline()  # header
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
            pairs.append({
                "query_id": qid, "doc_id": docid,
                "query": queries[qid], "passage": text,
                "human_rel": rel,
            })
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=537)
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent API calls (OpenAI tier 2+ supports >=8)")
    args = ap.parse_args()

    pairs = load_537_sample()[:args.max_pairs]
    print(f"Loaded {len(pairs)} pairs from TREC RAG 2024 stratified sample")
    print("Running UMBRELA prompt (Upadhyay et al. 2024) with gpt-4o-2024-08-06")
    print("=" * 72)

    t0 = time.time()
    scores = [None] * len(pairs)
    raw_responses = [None] * len(pairs)

    def run_one(i, pair):
        prompt = UMBRELA_PROMPT_TEMPLATE.format(query=pair["query"],
                                                  passage=pair["passage"])
        text = call_gpt4o_with_prompt(prompt)
        return i, text, parse_umbrela_score(text)

    n_done = 0
    n_valid = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, i, p) for i, p in enumerate(pairs)]
        for fut in as_completed(futures):
            i, text, score = fut.result()
            raw_responses[i] = text
            scores[i] = score
            n_done += 1
            if score is not None:
                n_valid += 1
            if n_done % 20 == 0 or n_done == len(pairs):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 0.001)
                eta = (len(pairs) - n_done) / max(rate, 0.001)
                print(f"  [{n_done:4d}/{len(pairs)}] valid={n_valid}  "
                      f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    human = [p["human_rel"] for p in pairs]
    kappa = cohen_kappa_quadratic(scores, human)
    n_valid_total = sum(1 for s in scores if s is not None)

    print(f"\n=== UMBRELA baseline result ===")
    print(f"  pairs:       {len(pairs)}")
    print(f"  valid:       {n_valid_total}/{len(pairs)} "
          f"({100*n_valid_total/len(pairs):.1f}%)")
    print(f"  kappa vs human: {kappa:.4f}" if kappa is not None
          else "  kappa: undefined")
    print(f"  wall:        {elapsed/60:.1f} min")

    out = {
        "method": "UMBRELA (Upadhyay et al. 2024) prompt + gpt-4o-2024-08-06",
        "model": "gpt-4o-2024-08-06",
        "prompt_source": "arXiv:2406.06519, Section 3 (verbatim)",
        "n_pairs": len(pairs),
        "n_valid": n_valid_total,
        "kappa_vs_human": kappa,
        "wall_seconds": round(elapsed, 1),
        "pair_index": [(p["query_id"], p["doc_id"]) for p in pairs],
        "umbrela_scores": scores,
        "human_scores": human,
        "config": {
            "max_tokens": 64,
            "temperature": 0.0,
            "max_chunk_chars": 1500,
            "workers": args.workers,
            "seed_in_sample": 42,
        },
    }
    out_path = REPO / "results" / "umbrela_baseline_trec_rag_2024.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
