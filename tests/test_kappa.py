"""Unit tests for cohen_kappa_quadratic.

These tests cover the canonical mathematical properties of the
quadratic-weighted kappa statistic that the paper relies on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from eval_llm_judge import cohen_kappa_quadratic  # noqa: E402


class TestKappaIdentities:
    def test_perfect_agreement_returns_one(self):
        kappa = cohen_kappa_quadratic([0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3])
        assert kappa is not None
        assert abs(kappa - 1.0) < 1e-10

    def test_constant_arrays_returns_none_or_undefined(self):
        # Both judges always say 2: no variance in either rater -> kappa undefined.
        kappa = cohen_kappa_quadratic([2] * 10, [2] * 10)
        # Implementation choice: returns None when denominator is 0.
        assert kappa is None

    def test_max_disagreement_negative(self):
        # Quadratic-weighted: maximally distant ratings (0 vs 3) on every pair
        # gives the largest possible disagreement.
        kappa = cohen_kappa_quadratic([0, 0, 3, 3], [3, 3, 0, 0])
        assert kappa is not None
        assert kappa < 0  # negative = worse than chance

    def test_symmetry(self):
        a = [0, 1, 2, 3, 1, 2, 0, 3, 2, 1]
        b = [0, 2, 2, 3, 0, 1, 1, 3, 2, 2]
        kab = cohen_kappa_quadratic(a, b)
        kba = cohen_kappa_quadratic(b, a)
        assert kab is not None and kba is not None
        assert abs(kab - kba) < 1e-10

    def test_handles_nones_by_pairwise_exclusion(self):
        # None entries should be excluded pair-by-pair, not arrayed against 0.
        a = [0, 1, None, 3, 2]
        b = [0, 1, 2, 3, 2]
        # Only 4 valid pairs: (0,0), (1,1), (3,3), (2,2). Perfect agreement.
        kappa = cohen_kappa_quadratic(a, b)
        assert kappa is not None
        assert abs(kappa - 1.0) < 1e-10

    def test_all_none_returns_none(self):
        kappa = cohen_kappa_quadratic([None, None, None], [1, 2, 3])
        assert kappa is None

    def test_matches_published_trec_rag_2024_sonnet(self):
        """End-to-end: recompute Sonnet vs human on TREC RAG 2024 from
        shipped JSON; must match the paper's claim of 0.5123."""
        import json
        fp = REPO / "results" / "trec-rag-2024_judges.json"
        if not fp.exists():
            pytest.skip(f"shipped JSON not available at {fp}")
        d = json.loads(fp.read_text(encoding="utf-8"))
        sonnet_label = next(l for l in d["per_judge_scores"]
                            if "Sonnet" in l)
        kappa = cohen_kappa_quadratic(d["per_judge_scores"][sonnet_label],
                                       d["human_scores"])
        assert kappa is not None
        assert abs(kappa - 0.5123) < 1e-3
