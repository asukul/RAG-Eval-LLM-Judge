"""Tests for the ensemble-median convention used across all corpora.

The repository standardized on UPPER-median in P2c (2026-04-29). These
tests pin that convention so it cannot drift undetected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from verify_paper_claims import ensemble_upper_median  # noqa: E402


class TestEnsembleConvention:
    def test_upper_median_odd_count(self):
        # 3 judges all return scores, sorted votes [0,1,2]; index 1 = 1.
        scores = {"j1": [0], "j2": [1], "j3": [2]}
        result = ensemble_upper_median(scores)
        assert result == [1]

    def test_upper_median_even_count(self):
        # 4 judges all return scores, sorted votes [0,1,2,3]; index 4//2 = 2.
        scores = {"j1": [0], "j2": [1], "j3": [2], "j4": [3]}
        result = ensemble_upper_median(scores)
        assert result == [2]

    def test_upper_median_excludes_nones(self):
        # 3 judges with one None: sorted votes [0,2]; index 2//2 = 1.
        scores = {"j1": [0], "j2": [None], "j3": [2]}
        result = ensemble_upper_median(scores)
        assert result == [2]

    def test_upper_median_all_nones_returns_none(self):
        scores = {"j1": [None], "j2": [None], "j3": [None]}
        result = ensemble_upper_median(scores)
        assert result == [None]

    def test_trec_rag_9judge_ensemble_kappa_0_4941(self):
        """Reproduce the headline 9-judge ensemble kappa = 0.4941."""
        import json
        fp = REPO / "results" / "trec-rag-2024_judges.json"
        if not fp.exists():
            pytest.skip(f"shipped JSON not available at {fp}")
        from eval_llm_judge import cohen_kappa_quadratic
        d = json.loads(fp.read_text(encoding="utf-8"))
        ensemble = ensemble_upper_median(d["per_judge_scores"])
        kappa = cohen_kappa_quadratic(ensemble, d["human_scores"])
        assert kappa is not None
        assert abs(kappa - 0.4941) < 1e-3
