"""Bootstrap 95% confidence intervals for every reported kappa value.

Resamples pairs with replacement (n=1000 by default) and reports the
2.5th / 97.5th percentile of the resulting kappa distribution. Adds
statistical machinery the reviewers (ChatGPT 5.5 + Claude Opus 4.7,
2026-04-29) flagged as critical for ECIR/NeurIPS-quality submission.

Coverage:
  - TREC RAG 2024: per-judge kappa vs human + 9-judge ensemble kappa
  - TREC-COVID:   per-judge kappa vs human + 9-judge ensemble kappa
  - Within-corpus: pairwise kappa for the four headline cells
                   (Qwen-Gemma, Sonnet-GPT-5.5, Opus-Sonnet, GPT-5.5-GPT-4o)

Run from repo root:
    py -3 src/bootstrap_kappa_cis.py [--n-resamples 1000] [--seed 42]

Output: results/bootstrap_kappa_cis.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from eval_llm_judge import cohen_kappa_quadratic  # noqa: E402
from verify_paper_claims import ensemble_upper_median  # noqa: E402


def bootstrap_kappa(y1: List[Optional[int]],
                    y2: List[Optional[int]],
                    n_resamples: int = 1000,
                    seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap CI for cohen_kappa_quadratic(y1, y2).

    Returns (point_estimate, ci_low_2.5, ci_high_97.5).
    """
    point = cohen_kappa_quadratic(y1, y2)
    rng = random.Random(seed)
    n = len(y1)
    samples = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a = [y1[i] for i in idx]
        b = [y2[i] for i in idx]
        k = cohen_kappa_quadratic(a, b)
        if k is not None:
            samples.append(k)
    if not samples:
        return point, float("nan"), float("nan")
    samples.sort()
    ci_low = samples[int(0.025 * len(samples))]
    ci_high = samples[int(0.975 * len(samples))]
    return point, ci_low, ci_high


def bootstrap_external_corpus(corpus_id: str, n_resamples: int, seed: int):
    """Per-judge + ensemble bootstrap for an external-validation corpus."""
    fp = REPO / "results" / f"{corpus_id}_judges.json"
    if not fp.exists():
        return None
    d = json.loads(fp.read_text(encoding="utf-8"))
    human = d["human_scores"]
    out = {"per_judge": {}, "ensemble_9j": None}

    for label, scores in d["per_judge_scores"].items():
        point, lo, hi = bootstrap_kappa(scores, human, n_resamples, seed)
        out["per_judge"][label] = {
            "kappa": point,
            "ci_95_low": round(lo, 4),
            "ci_95_high": round(hi, 4),
            "n_valid": sum(1 for s in scores if s is not None),
        }

    e9 = ensemble_upper_median(d["per_judge_scores"])
    point, lo, hi = bootstrap_kappa(e9, human, n_resamples, seed)
    out["ensemble_9j"] = {
        "kappa": point,
        "ci_95_low": round(lo, 4),
        "ci_95_high": round(hi, 4),
    }

    frontier = {k: v for k, v in d["per_judge_scores"].items()
                if "Qwen" not in k and "Gemma" not in k}
    e7 = ensemble_upper_median(frontier)
    point, lo, hi = bootstrap_kappa(e7, human, n_resamples, seed)
    out["ensemble_7j_frontier"] = {
        "kappa": point,
        "ci_95_low": round(lo, 4),
        "ci_95_high": round(hi, 4),
    }
    return out


def bootstrap_within_corpus(n_resamples: int, seed: int):
    """Bootstrap pairwise kappa for headline within-corpus cells."""
    within = REPO / "results" / "within_corpus"
    if not within.exists() or not any(within.glob("judge_*.json")):
        return None

    per_judge = {}
    for p in sorted(within.glob("judge_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        per_judge[d["config"]["judge_label"]] = [
            r["judge_score"]
            for q in d["queries"]
            for r in sorted(q["retrieved"], key=lambda x: x["rank"])
        ]

    def find(s):
        return next(l for l in per_judge if s in l)

    cells = {
        "Qwen <-> Gemma 4 (open-weight cross-org, matrix max)":
            (find("Qwen"), find("Gemma")),
        "Sonnet <-> GPT-5.5 (cross-family commercial ceiling)":
            (find("Sonnet"), find("GPT-5.5")),
        "Opus <-> Sonnet (Anthropic within-family)":
            (find("Opus"), find("Sonnet")),
        "GPT-5.5 <-> GPT-4o (OpenAI within-family)":
            (find("GPT-5.5"), find("GPT-4o")),
        "Gem 3.1 <-> Gem 2.5 (Google-commercial within-family)":
            (find("3.1 Pro Preview"), find("2.5 Pro")),
        "Sonnet <-> DSV4 Pro (cross-family, post-fix from C1)":
            (find("Sonnet"), find("DeepSeek")),
    }
    out = {}
    for name, (a, b) in cells.items():
        point, lo, hi = bootstrap_kappa(per_judge[a], per_judge[b],
                                        n_resamples, seed)
        out[name] = {
            "kappa": point,
            "ci_95_low": round(lo, 4),
            "ci_95_high": round(hi, 4),
            "n_pairs": len(per_judge[a]),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Bootstrap CIs (n={args.n_resamples}, seed={args.seed})")
    print("=" * 72)

    out = {
        "config": {
            "n_resamples": args.n_resamples,
            "seed": args.seed,
            "method": "case-resampling bootstrap (resample pairs with replacement)",
            "ci_method": "percentile 2.5 / 97.5",
        }
    }

    print("\nTREC RAG 2024 (n=537):")
    out["trec_rag_2024"] = bootstrap_external_corpus("trec-rag-2024",
                                                     args.n_resamples, args.seed)
    if out["trec_rag_2024"]:
        e = out["trec_rag_2024"]["ensemble_9j"]
        print(f"  9-judge ensemble: kappa={e['kappa']:.4f}  "
              f"95% CI [{e['ci_95_low']:.4f}, {e['ci_95_high']:.4f}]")
        e7 = out["trec_rag_2024"]["ensemble_7j_frontier"]
        print(f"  7-judge frontier: kappa={e7['kappa']:.4f}  "
              f"95% CI [{e7['ci_95_low']:.4f}, {e7['ci_95_high']:.4f}]")

    print("\nTREC-COVID (n=300):")
    out["trec_covid"] = bootstrap_external_corpus("trec-covid",
                                                  args.n_resamples, args.seed)
    if out["trec_covid"]:
        e = out["trec_covid"]["ensemble_9j"]
        print(f"  9-judge ensemble: kappa={e['kappa']:.4f}  "
              f"95% CI [{e['ci_95_low']:.4f}, {e['ci_95_high']:.4f}]")
        e7 = out["trec_covid"]["ensemble_7j_frontier"]
        print(f"  7-judge frontier: kappa={e7['kappa']:.4f}  "
              f"95% CI [{e7['ci_95_low']:.4f}, {e7['ci_95_high']:.4f}]")

    print("\nWithin-corpus pairwise (ISU DSpace, n=570):")
    out["within_corpus"] = bootstrap_within_corpus(args.n_resamples, args.seed)
    if out["within_corpus"]:
        for name, ci in out["within_corpus"].items():
            print(f"  {name[:55]:<55s} kappa={ci['kappa']:.4f}  "
                  f"95% CI [{ci['ci_95_low']:.4f}, {ci['ci_95_high']:.4f}]")

    out_path = REPO / "results" / "bootstrap_kappa_cis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
