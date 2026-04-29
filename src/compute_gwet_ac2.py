"""Quadratic-weighted Gwet AC2 alongside Cohen kappa.

Gwet's AC2 (Gwet, 2008) is a chance-corrected agreement coefficient
designed to be robust to the "kappa paradox" (high observed agreement
but low kappa under skewed marginal distributions). On bounded ordinal
ratings it can be computed with quadratic weights analogous to Cohen
kappa. The Bloomberg Law EARL@RecSys 2025 paper recommends reporting
both Cohen kappa and Gwet AC2 to triangulate inter-rater agreement
under marginal-skew conditions.

Quadratic-weighted Gwet AC2 formula (Gwet 2008, Wongpakaran et al. 2013):

    AC2_q = (P_a - P_e) / (1 - P_e)
    where:
        P_a = sum over (i,j) of w_ij * P(rater1=i, rater2=j)        (weighted agreement)
        P_e = (T(T-1))^-1 * sum over k of pi_k * (1 - pi_k) * w_kk' (chance term)
              -- but the standard simplification for ordinal-N categories is:
        P_e (Gwet) = sum over (i,j) of w_ij * pi_i * (1 - pi_j) * 2 / (N*(N-1))
              -- where pi_k is the marginal proportion of rating k (averaged across raters)
              -- and w_ij is the quadratic weight = 1 - ((i-j)/(N-1))^2
              -- N is the number of categories

Implementation note: AC2 differs from quadratic-weighted Cohen kappa
primarily in the chance-correction term P_e. Cohen kappa uses
P_e = sum(p_row * p_col), which can collapse to near 1 (forcing kappa
to ~0) under skewed marginals. AC2 uses a category-distribution-aware
chance term that avoids this paradox.

Run from repo root:
    py -3 src/compute_gwet_ac2.py

Output: results/gwet_ac2_alongside_kappa.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from eval_llm_judge import cohen_kappa_quadratic  # noqa: E402
from verify_paper_claims import ensemble_upper_median  # noqa: E402


def gwet_ac2_quadratic(y1: List[Optional[int]],
                       y2: List[Optional[int]],
                       n_categories: int = 4) -> Optional[float]:
    """Quadratic-weighted Gwet AC2 for ordinal categories 0..n_categories-1.

    Reference: Gwet, K. (2008). Computing inter-rater reliability and its
    variance in the presence of high agreement. British Journal of
    Mathematical and Statistical Psychology, 61(1), 29-48.

    Implementation follows the bounded-ordinal formulation in
    Wongpakaran et al. (2013) "A comparison of Cohen's Kappa and Gwet's
    AC1 when calculating inter-rater reliability coefficients..."
    extended to quadratic weighting.
    """
    paired = [(a, b) for a, b in zip(y1, y2) if a is not None and b is not None]
    if not paired:
        return None
    N = len(paired)
    if N < 2:
        return None

    # Quadratic weights w_ij = 1 - ((i-j)/(n-1))^2
    n_minus_1 = max(1, n_categories - 1)
    w = [[1.0 - ((i - j) / n_minus_1) ** 2 for j in range(n_categories)]
         for i in range(n_categories)]

    # Observed weighted agreement
    p_a = 0.0
    for a, b in paired:
        if 0 <= a < n_categories and 0 <= b < n_categories:
            p_a += w[a][b]
    p_a /= N

    # Marginal probability of each category, averaged across raters
    pi = [0.0] * n_categories
    for a, b in paired:
        if 0 <= a < n_categories:
            pi[a] += 1
        if 0 <= b < n_categories:
            pi[b] += 1
    total = sum(pi)
    if total == 0:
        return None
    pi = [x / total for x in pi]

    # Gwet's chance-correction for AC2 (quadratic-weighted ordinal):
    # P_e = sum_{i!=j} w_ij * pi_i * pi_j  +  (1/(N-1)) * adjustment
    # We use the simpler bounded formulation:
    #   P_e = sum_{i,j} w_ij * pi_i * (1 - pi_j) / (n_categories - 1)
    p_e = 0.0
    for i in range(n_categories):
        for j in range(n_categories):
            if i != j:
                p_e += w[i][j] * pi[i] * (1 - pi[j])
    p_e /= max(1, n_categories - 1)

    if p_e >= 1:
        return None
    ac2 = (p_a - p_e) / (1 - p_e)
    return round(ac2, 4)


def compute_for_corpus(corpus_id: str):
    fp = REPO / "results" / f"{corpus_id}_judges.json"
    if not fp.exists():
        return None
    d = json.loads(fp.read_text(encoding="utf-8"))
    human = d["human_scores"]
    out = {"per_judge": {}, "ensemble_9j": None, "ensemble_7j_frontier": None}

    for label, scores in d["per_judge_scores"].items():
        kappa = cohen_kappa_quadratic(scores, human)
        ac2 = gwet_ac2_quadratic(scores, human)
        out["per_judge"][label] = {
            "kappa": kappa, "ac2": ac2,
            "delta_ac2_minus_kappa": (round(ac2 - kappa, 4)
                                       if ac2 is not None and kappa is not None
                                       else None),
        }

    e9 = ensemble_upper_median(d["per_judge_scores"])
    out["ensemble_9j"] = {
        "kappa": cohen_kappa_quadratic(e9, human),
        "ac2": gwet_ac2_quadratic(e9, human),
    }

    frontier = {k: v for k, v in d["per_judge_scores"].items()
                if "Qwen" not in k and "Gemma" not in k}
    e7 = ensemble_upper_median(frontier)
    out["ensemble_7j_frontier"] = {
        "kappa": cohen_kappa_quadratic(e7, human),
        "ac2": gwet_ac2_quadratic(e7, human),
    }
    return out


def main():
    print("Quadratic-weighted Gwet AC2 alongside Cohen kappa")
    print("=" * 72)

    out = {
        "method": ("Gwet (2008) AC2 with quadratic weighting; Cohen kappa "
                   "from src/eval_llm_judge.py for comparison"),
    }

    for corpus_id in ("trec-rag-2024", "trec-covid"):
        print(f"\n{corpus_id}:")
        out[corpus_id] = compute_for_corpus(corpus_id)
        if out[corpus_id]:
            for label, m in out[corpus_id]["per_judge"].items():
                k = m["kappa"]
                a = m["ac2"]
                k_str = f"{k:.4f}" if k is not None else "n/a"
                a_str = f"{a:.4f}" if a is not None else "n/a"
                print(f"  {label[:35]:<35s}  kappa={k_str}  AC2={a_str}")
            e = out[corpus_id]["ensemble_9j"]
            print(f"  9-judge ensemble:                    "
                  f"kappa={e['kappa']:.4f}  AC2={e['ac2']:.4f}")
            e7 = out[corpus_id]["ensemble_7j_frontier"]
            print(f"  7-judge frontier ensemble:           "
                  f"kappa={e7['kappa']:.4f}  AC2={e7['ac2']:.4f}")

    out_path = REPO / "results" / "gwet_ac2_alongside_kappa.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
