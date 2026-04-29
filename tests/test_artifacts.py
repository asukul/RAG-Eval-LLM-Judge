"""Integrity tests for shipped data artifacts.

Verifies that every per-judge JSON in the results/ directory has a
self-consistent schema and the expected number of pairs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


class TestExternalValidationFiles:
    @pytest.mark.parametrize("corpus,expected_n", [
        ("trec-rag-2024", 537),
        ("trec-covid", 300),
        ("beir-scifact", 300),
    ])
    def test_corpus_judges_json_has_expected_n_pairs(self, corpus, expected_n):
        fp = REPO / "results" / f"{corpus}_judges.json"
        if not fp.exists():
            pytest.skip(f"{fp} not shipped")
        d = json.loads(fp.read_text(encoding="utf-8"))
        assert len(d["pair_index"]) == expected_n
        for label, scores in d["per_judge_scores"].items():
            assert len(scores) == expected_n, \
                f"judge {label} has {len(scores)} scores, expected {expected_n}"
        assert len(d["human_scores"]) == expected_n

    def test_trec_rag_2024_stratified_balance(self):
        """The 537-pair sample must hit 135/134/134/134 across labels."""
        fp = REPO / "results" / "trec-rag-2024_judges.json"
        if not fp.exists():
            pytest.skip(f"{fp} not shipped")
        d = json.loads(fp.read_text(encoding="utf-8"))
        from collections import Counter
        dist = Counter(d["human_scores"])
        assert dist[0] == 135
        assert dist[1] == 134
        assert dist[2] == 134
        assert dist[3] == 134

    def test_trec_rag_2024_has_9_judges(self):
        fp = REPO / "results" / "trec-rag-2024_judges.json"
        if not fp.exists():
            pytest.skip(f"{fp} not shipped")
        d = json.loads(fp.read_text(encoding="utf-8"))
        assert len(d["per_judge_scores"]) == 9


class TestWithinCorpusFiles:
    def test_within_corpus_directory_exists(self):
        d = REPO / "results" / "within_corpus"
        if not d.exists():
            pytest.skip("results/within_corpus/ not shipped")
        assert d.is_dir()

    def test_nine_per_judge_files(self):
        d = REPO / "results" / "within_corpus"
        if not d.exists():
            pytest.skip("results/within_corpus/ not shipped")
        files = list(d.glob("judge_*.json"))
        assert len(files) == 9

    def test_each_per_judge_file_has_570_pairs(self):
        d = REPO / "results" / "within_corpus"
        if not d.exists():
            pytest.skip("results/within_corpus/ not shipped")
        for fp in d.glob("judge_*.json"):
            j = json.loads(fp.read_text(encoding="utf-8"))
            n = sum(len(q.get("retrieved", [])) for q in j["queries"])
            assert n == 570, \
                f"{fp.name}: {n} retrieved entries, expected 570"

    def test_retrieval_determinism_across_per_judge_files(self):
        """All 9 per-judge files should agree on (query_id, rank, point_id)
        triples — this is the determinism guarantee that lets per-pair
        scores from different judges be aligned positionally."""
        d = REPO / "results" / "within_corpus"
        if not d.exists():
            pytest.skip("results/within_corpus/ not shipped")
        files = sorted(d.glob("judge_*.json"))
        if len(files) < 2:
            pytest.skip("need at least 2 per-judge files")
        ref = json.loads(files[0].read_text(encoding="utf-8"))
        ref_keys = [(q["query_id"], r["rank"], r["point_id"])
                    for q in ref["queries"]
                    for r in sorted(q["retrieved"], key=lambda x: x["rank"])]
        for fp in files[1:]:
            cur = json.loads(fp.read_text(encoding="utf-8"))
            cur_keys = [(q["query_id"], r["rank"], r["point_id"])
                        for q in cur["queries"]
                        for r in sorted(q["retrieved"], key=lambda x: x["rank"])]
            assert ref_keys == cur_keys, \
                f"{fp.name} has different (qid, rank, point_id) keys than {files[0].name}"


class TestDataInputs:
    def test_trec_rag_2024_qrels_present(self):
        fp = REPO / "data" / "2024-retrieval-qrels.txt"
        assert fp.exists(), "TREC RAG 2024 qrels missing from data/"

    def test_trec_rag_2024_passages_present_and_537(self):
        fp = REPO / "data" / "passages.json"
        if not fp.exists():
            pytest.skip("passages.json not shipped (re-extract with src/fetch_msmarco_passages.py)")
        d = json.loads(fp.read_text(encoding="utf-8"))
        assert len(d) == 537, f"expected 537 passages, got {len(d)}"

    def test_sample_537_pairs_present_and_correct_size(self):
        fp = REPO / "data" / "sample_537_pairs.tsv"
        assert fp.exists()
        lines = fp.read_text(encoding="utf-8").strip().split("\n")
        # 1 header + 537 data rows
        assert len(lines) == 538


class TestVerifierScript:
    def test_verifier_runs_clean(self):
        """The headline-claims verifier must exit with 0 (all pass)."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-X", "utf8",
             str(REPO / "src" / "verify_paper_claims.py")],
            capture_output=True, text=True, env={**__import__("os").environ,
                                                  "PYTHONIOENCODING": "utf-8"},
        )
        assert result.returncode == 0, \
            f"verifier failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
