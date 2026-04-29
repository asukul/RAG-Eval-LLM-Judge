"""Multi-corpus public-benchmark validation of the 9-judge ensemble.

Validates our 9-judge LLM ensemble against publicly-released human qrels from
TREC RAG 2024, TREC-COVID, BEIR scifact, and (optionally) LLMJudge benchmark.
Replaces the IRB-pending human study with cross-corpus cross-validation.

See `PUBLIC_BENCHMARKS_VALIDATION.md` in this directory for the full experiment
plan, rationale, and acceptance criteria.

Usage:
    # Download data (run once):
    py -3 papers/P4_llm_as_judge/validate_against_trec.py --download trec-rag-2024
    py -3 papers/P4_llm_as_judge/validate_against_trec.py --download trec-covid
    py -3 papers/P4_llm_as_judge/validate_against_trec.py --download beir-scifact

    # Run validation against a corpus:
    py -3 papers/P4_llm_as_judge/validate_against_trec.py --corpus trec-rag-2024 --judge-preset p4-frontier --max-pairs 537
    py -3 papers/P4_llm_as_judge/validate_against_trec.py --corpus beir-scifact --judge-preset p4-supplement-openweight --max-pairs 300

    # Compare ensemble vs human qrels:
    py -3 papers/P4_llm_as_judge/validate_against_trec.py --analyze trec-rag-2024
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "_validation_data"
RESULTS_DIR = ROOT / "_validation_results"

# Reuse judge infrastructure from the main eval harness
sys.path.insert(0, str(REPO / "backend" / "scripts"))
try:
    from eval_llm_judge import (  # type: ignore
        JUDGE_BUILDERS,
        JUDGE_PRESETS,
        cohen_kappa_quadratic,
        compute_kappa_matrix,
        build_judges,
        _load_dotenv_manual,
    )
    _load_dotenv_manual(REPO / ".env")
except ImportError as e:
    print(f"ERROR: cannot import from eval_llm_judge.py: {e}")
    print("This script depends on backend/scripts/eval_llm_judge.py being importable.")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Corpus registry — what to download, how to parse, how to map score scales
# -----------------------------------------------------------------------------
CORPUS_REGISTRY = {
    "trec-rag-2024": {
        "description": "TREC RAG 2024 retrieval track (NIST human qrels) — 20,283 pairs across 86 queries",
        "qrels_url": "https://trec.nist.gov/data/rag/2024-retrieval-qrels.txt",
        "topics_url": "https://trec-rag.github.io/assets/txt/topics.rag24.test.txt",
        "qrels_format": "trec",  # trec qrels: query_id  0  doc_id  rel
        "score_scale": "0-3",  # actual scale (verified 2026-04-26): 0=irrelevant, 1=marginal, 2=relevant, 3=highly-relevant
        "score_mapping": {0: 0, 1: 1, 2: 2, 3: 3},  # identity: matches our rubric exactly
        "expected_pairs": 20283,
        "n_queries_judged": 86,
        "doc_corpus": "msmarco_v2.1",  # passages from MS MARCO v2.1 — needs separate ~50GB download for full text
    },
    "trec-rag-2024-conditions": {
        "description": "TREC RAG 2024 with assessment-condition variants",
        "qrels_url": "https://trec.nist.gov/data/rag/2024-retrieval-conditions-qrels.txt",
        "topics_url": "https://trec-rag.github.io/assets/txt/topics.rag24.test.txt",
        "qrels_format": "trec",
        "score_scale": "0-3",
        "score_mapping": {0: 0, 1: 1, 2: 2, 3: 3},  # identity
        "expected_pairs": 20283,
    },
    "trec-covid": {
        "description": "TREC-COVID biomedical scientific (BEIR-distributed)",
        "qrels_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
        "qrels_format": "beir",
        "score_scale": "0-2",
        "score_mapping": {0: 0, 1: 2, 2: 3},
        "expected_pairs": 200,  # subset
    },
    "beir-scifact": {
        "description": "BEIR scifact scholarly fact-verification (binary qrels)",
        "qrels_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "qrels_format": "beir",
        "score_scale": "binary",
        "score_mapping": {0: 0, 1: 3},  # binary → ordinal: collapse to extremes
        "expected_pairs": 300,
    },
    "llmjudge-2025": {
        "description": "LLMJudge benchmark (Farzi & Dietz 2025)",
        "qrels_url": "TBD - contact authors or check publication supplementary",
        "qrels_format": "trec",
        "score_scale": "0-3",
        "score_mapping": {0: 0, 1: 1, 2: 2, 3: 3},
        "expected_pairs": 500,
        "status": "manual-download-required",
    },
}


# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
def cmd_download(corpus_id: str) -> int:
    """Download corpus qrels + topics to _validation_data/<corpus_id>/."""
    if corpus_id not in CORPUS_REGISTRY:
        print(f"ERROR: unknown corpus '{corpus_id}'. Available: {list(CORPUS_REGISTRY)}")
        return 1
    cfg = CORPUS_REGISTRY[corpus_id]
    if cfg.get("status") == "manual-download-required":
        print(f"NOTE: {corpus_id} requires manual download. URL: {cfg['qrels_url']}")
        print(f"Save files to: {DATA_DIR / corpus_id}/")
        return 0

    out_dir = DATA_DIR / corpus_id
    out_dir.mkdir(parents=True, exist_ok=True)

    qrels_url = cfg["qrels_url"]
    qrels_path = out_dir / Path(qrels_url).name
    print(f"Downloading qrels: {qrels_url}")
    print(f"  → {qrels_path}")
    try:
        # NIST blocks default urllib UA; use a browser-like UA
        req = urllib.request.Request(
            qrels_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        qrels_path.write_bytes(data)
        size_mb = qrels_path.stat().st_size / 1024 / 1024
        print(f"  done ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Manual fallback: download {qrels_url} via browser; save to {qrels_path}")
        return 1

    if cfg.get("topics_url"):
        topics_url = cfg["topics_url"]
        topics_path = out_dir / Path(topics_url).name
        print(f"Downloading topics: {topics_url}")
        try:
            urllib.request.urlretrieve(topics_url, topics_path)
            print(f"  done ({topics_path.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            print(f"  topics download FAILED: {e}")
            print(f"  (qrels still usable; topics may be obtainable via ir_datasets)")

    print(f"\nNext step: download document corpus separately (huge file).")
    print(f"  TREC RAG 2024: see https://trec-rag.github.io/")
    print(f"  BEIR datasets: pip install beir; auto-downloads via BEIR loader")
    return 0


# -----------------------------------------------------------------------------
# Load qrels
# -----------------------------------------------------------------------------
def parse_trec_qrels(path: Path) -> list[dict]:
    """Parse TREC qrels format: query_id 0 doc_id relevance"""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
            rows.append({"query_id": qid, "doc_id": doc_id, "human_rel": rel})
    return rows


def parse_beir_qrels(beir_root: Path, split: str = "test") -> list[dict]:
    """Parse BEIR qrels: TSV with header query-id, corpus-id, score."""
    qrels_path = beir_root / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        # BEIR's GenericDataLoader extracts to a nested <dataset>/qrels/<split>.tsv;
        # fall back to a glob one level deep.
        candidates = list(beir_root.glob(f"*/qrels/{split}.tsv"))
        if candidates:
            qrels_path = candidates[0]
    if not qrels_path.exists():
        raise FileNotFoundError(f"BEIR qrels not found at {qrels_path}")
    rows = []
    with qrels_path.open(encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, rel = parts[0], parts[1], int(parts[2])
            rows.append({"query_id": qid, "doc_id": doc_id, "human_rel": rel})
    return rows


# -----------------------------------------------------------------------------
# Score-scale mapping
# -----------------------------------------------------------------------------
def map_score_to_rubric(human_rel: int, mapping: dict[int, int]) -> int:
    """Map source-corpus relevance score to our 0-3 rubric."""
    return mapping.get(human_rel, human_rel)


# -----------------------------------------------------------------------------
# κ computation
# -----------------------------------------------------------------------------
def compute_kappa_vs_human(
    judge_scores: dict[str, list[Optional[int]]],
    human_scores: list[int],
) -> dict[str, float]:
    """Compute Cohen's κ for each judge vs human qrels.

    judge_scores: {judge_label: [score_or_None for each pair]}
    human_scores: [human_score for each pair]
    """
    out = {}
    n_pairs = len(human_scores)
    for judge_label, scores in judge_scores.items():
        valid_pairs = [
            (s, h) for s, h in zip(scores, human_scores)
            if s is not None and h is not None
        ]
        if len(valid_pairs) < 30:
            out[judge_label] = None
            continue
        a, b = zip(*valid_pairs)
        kappa = cohen_kappa_quadratic(list(a), list(b))
        out[judge_label] = float(kappa) if kappa is not None else None
    return out


def compute_ensemble_median(judge_scores: dict[str, list[Optional[int]]]) -> list[Optional[int]]:
    """Median vote across judges per pair."""
    n_pairs = len(next(iter(judge_scores.values())))
    out = []
    for i in range(n_pairs):
        votes = [scores[i] for scores in judge_scores.values() if scores[i] is not None]
        if not votes:
            out.append(None)
        else:
            sorted_votes = sorted(votes)
            out.append(sorted_votes[len(sorted_votes) // 2])
    return out


# -----------------------------------------------------------------------------
# Analyze (read existing per-judge JSONs and compute κ vs human)
# -----------------------------------------------------------------------------
def cmd_analyze(corpus_id: str) -> int:
    """Compute κ between each judge and human qrels for an already-completed run."""
    cfg = CORPUS_REGISTRY.get(corpus_id)
    if cfg is None:
        print(f"ERROR: unknown corpus '{corpus_id}'")
        return 1

    results_path = RESULTS_DIR / f"{corpus_id}_judges.json"
    qrels_path = DATA_DIR / corpus_id / Path(cfg["qrels_url"]).name
    if not results_path.exists():
        print(f"ERROR: judge results not found at {results_path}")
        print(f"  Run validation first: --corpus {corpus_id} --judge-preset p4-frontier")
        return 1
    if not qrels_path.exists():
        print(f"ERROR: qrels not downloaded. Run: --download {corpus_id}")
        return 1

    print(f"Loading judge results from {results_path}...")
    judge_data = json.loads(results_path.read_text(encoding="utf-8"))
    judge_scores = judge_data["per_judge_scores"]  # {judge_label: [score per pair]}
    pair_index = judge_data["pair_index"]  # [(query_id, doc_id), ...]

    print(f"Parsing human qrels from {qrels_path}...")
    if cfg["qrels_format"] == "trec":
        qrels = parse_trec_qrels(qrels_path)
    elif cfg["qrels_format"] == "beir":
        qrels = parse_beir_qrels(qrels_path.parent)
    else:
        print(f"ERROR: unknown qrels format: {cfg['qrels_format']}")
        return 1

    qrels_lookup = {(r["query_id"], r["doc_id"]): r["human_rel"] for r in qrels}
    human_scores = []
    aligned_pairs = []
    for qid, doc_id in pair_index:
        if (qid, doc_id) in qrels_lookup:
            mapped = map_score_to_rubric(qrels_lookup[(qid, doc_id)], cfg["score_mapping"])
            human_scores.append(mapped)
            aligned_pairs.append((qid, doc_id))
    print(f"Aligned {len(aligned_pairs)} pairs (judge-rated AND human-rated)")
    print()

    print("=== Per-judge κ vs human qrels ===")
    kappa_per_judge = compute_kappa_vs_human(judge_scores, human_scores)
    for judge_label, kappa in sorted(kappa_per_judge.items(), key=lambda x: -(x[1] or 0)):
        print(f"  {judge_label[:35]:<35s}  κ = {kappa:.4f}" if kappa is not None else f"  {judge_label}  (insufficient overlap)")
    print()

    ensemble_median = compute_ensemble_median(judge_scores)
    ensemble_kappa = compute_kappa_vs_human({"ensemble_median": ensemble_median}, human_scores)
    print(f"=== Ensemble (9-judge median) vs human qrels ===")
    print(f"  ensemble_median  κ = {ensemble_kappa['ensemble_median']:.4f}")
    print()

    out = {
        "corpus": corpus_id,
        "n_aligned_pairs": len(aligned_pairs),
        "kappa_per_judge_vs_human": kappa_per_judge,
        "kappa_ensemble_median_vs_human": ensemble_kappa["ensemble_median"],
        "interpretation": _interpret_kappa(ensemble_kappa["ensemble_median"]),
    }
    out_path = RESULTS_DIR / f"{corpus_id}_kappa_vs_human.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


def _interpret_kappa(kappa: Optional[float]) -> str:
    if kappa is None:
        return "insufficient data"
    if kappa < 0.20:
        return "slight (Landis-Koch)"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost-perfect"


# -----------------------------------------------------------------------------
# Corpus loaders: load (query_id, doc_id, query_text, doc_text, human_rel) tuples
# -----------------------------------------------------------------------------
def load_trec_rag_corpus(max_pairs: int) -> list[dict]:
    """Load TREC RAG 2024 from the pre-extracted 537-pair stratified subset.

    Reads sample_537_pairs.tsv (qid, docid, rel) plus passages.json (per-docid
    text from the MS MARCO v2.1 segmented corpus, fetched via fetch_msmarco_passages.py)
    and topics.rag24.test.txt (qid → query text). Returns aligned pair dicts in
    the same shape as load_beir_corpus.
    """
    base = DATA_DIR / "trec-rag-2024"
    sample_path = base / "sample_537_pairs.tsv"
    passages_path = base / "passages.json"
    topics_path = base / "topics.rag24.test.txt"

    if not sample_path.exists() or not passages_path.exists():
        raise FileNotFoundError(
            f"Missing TREC RAG 2024 inputs. Expected at {base}:\n"
            f"  - sample_537_pairs.tsv (run the sampler first)\n"
            f"  - passages.json (run fetch_msmarco_passages.py first)\n"
        )

    queries: dict[str, str] = {}
    for line in topics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            queries[parts[0].strip()] = parts[1].strip()
    print(f"  loaded {len(queries)} TREC RAG 2024 topics")

    passages = json.loads(passages_path.read_text(encoding="utf-8"))
    print(f"  loaded {len(passages)} MS MARCO v2.1 passages from passages.json")

    pairs = []
    skipped_no_query = 0
    skipped_no_passage = 0
    with sample_path.open(encoding="utf-8") as f:
        header = f.readline()  # skip "query_id\tdoc_id\thuman_rel"
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, docid, rel_s = parts[0], parts[1], parts[2]
            if qid not in queries:
                skipped_no_query += 1
                continue
            if docid not in passages:
                skipped_no_passage += 1
                continue
            psg = passages[docid]
            doc_text = " ".join(filter(None, [
                psg.get("title", ""),
                psg.get("headings", ""),
                psg.get("text", ""),
            ])).strip()
            pairs.append({
                "query_id": qid,
                "doc_id": docid,
                "query": queries[qid],
                "doc_text": doc_text,
                "human_rel": int(rel_s),
            })
            if len(pairs) >= max_pairs:
                break

    print(
        f"  built {len(pairs)} TREC RAG 2024 pairs "
        f"(skipped {skipped_no_query} no-query, {skipped_no_passage} no-passage)"
    )
    return pairs


def load_beir_corpus(dataset_id: str, max_pairs: int) -> list[dict]:
    """Load BEIR dataset (auto-downloads if missing) and return aligned pairs.

    dataset_id: BEIR dataset name without "beir-" prefix (e.g., "scifact", "trec-covid")
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    out_dir = DATA_DIR / f"beir-{dataset_id}"
    data_path_dir = out_dir / dataset_id
    if not data_path_dir.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_id}.zip"
        print(f"Downloading BEIR {dataset_id} from {url}...")
        out_dir.mkdir(parents=True, exist_ok=True)
        util.download_and_unzip(url, str(out_dir))

    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path_dir)).load(split="test")
    print(f"  loaded BEIR {dataset_id}: {len(corpus)} docs, {len(queries)} queries, "
          f"{sum(len(v) for v in qrels.values())} qrels")

    # Flatten qrels into pair list
    pairs = []
    for qid, doc_to_rel in qrels.items():
        if qid not in queries:
            continue
        query_text = queries[qid]
        for doc_id, rel in doc_to_rel.items():
            if doc_id not in corpus:
                continue
            doc = corpus[doc_id]
            doc_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            pairs.append({
                "query_id": qid,
                "doc_id": doc_id,
                "query": query_text,
                "doc_text": doc_text,
                "human_rel": int(rel),
            })
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    return pairs


# -----------------------------------------------------------------------------
# Run judges on pairs
# -----------------------------------------------------------------------------
def run_judges_on_pairs(pairs: list[dict], judge_specs: list[str]) -> dict:
    """Run each judge in `judge_specs` against each pair; return per-judge scores.

    Returns:
        {
            "pair_index": [(query_id, doc_id), ...],
            "human_scores": [int, ...],
            "per_judge_scores": {judge_label: [int_or_None, ...]},
            "per_judge_metadata": {judge_label: {"n_valid": ..., "n_missing": ...}},
        }
    """
    judges = build_judges(judge_specs)
    print(f"Built {len(judges)} judges:")
    for label, _ in judges:
        print(f"  - {label}")
    print()

    pair_index = [(p["query_id"], p["doc_id"]) for p in pairs]
    human_scores = [p["human_rel"] for p in pairs]
    per_judge_scores: dict[str, list] = {label: [] for label, _ in judges}

    n_pairs = len(pairs)
    for i, p in enumerate(pairs):
        for label, judge in judges:
            try:
                score = judge.score(p["query"], p["doc_text"])
            except Exception as e:
                print(f"  pair {i+1}/{n_pairs}, judge={label[:30]}: error {e}")
                score = None
            per_judge_scores[label].append(score)
        if (i + 1) % 5 == 0 or (i + 1) == n_pairs:
            print(f"  [{i+1:>4d}/{n_pairs}] processed")

    per_judge_metadata = {
        label: {
            "n_valid": sum(1 for s in scores if s is not None),
            "n_missing": sum(1 for s in scores if s is None),
            "mean_score": (sum(s for s in scores if s is not None) / max(1, sum(1 for s in scores if s is not None))),
        }
        for label, scores in per_judge_scores.items()
    }
    return {
        "pair_index": pair_index,
        "human_scores": human_scores,
        "per_judge_scores": per_judge_scores,
        "per_judge_metadata": per_judge_metadata,
        "judge_specs": judge_specs,
    }


def cmd_validate(corpus_id: str, judge_preset: str, max_pairs: int, judges_arg: Optional[list[str]] = None) -> int:
    """Run judges on a corpus and save per-judge scores."""
    cfg = CORPUS_REGISTRY[corpus_id]

    # Load pairs
    if corpus_id.startswith("beir-") or corpus_id == "trec-covid":
        # TREC-COVID is also distributed via BEIR
        beir_id = "trec-covid" if corpus_id == "trec-covid" else corpus_id.replace("beir-", "", 1)
        pairs = load_beir_corpus(beir_id, max_pairs)
    elif corpus_id == "trec-rag-2024":
        pairs = load_trec_rag_corpus(max_pairs)
    elif corpus_id.startswith("trec-rag-2024"):
        print(f"ERROR: variant {corpus_id} not yet supported.")
        return 1
    else:
        print(f"ERROR: no loader for {corpus_id}")
        return 1

    print(f"Loaded {len(pairs)} aligned pairs from {corpus_id}")
    print(f"Judge preset: {judge_preset} or judges: {judges_arg}")
    print()

    # Resolve judge specs
    if judges_arg:
        specs = judges_arg
    elif judge_preset in JUDGE_PRESETS:
        specs = JUDGE_PRESETS[judge_preset]
    else:
        print(f"ERROR: judge preset '{judge_preset}' not in JUDGE_PRESETS")
        return 1

    # Apply score-mapping to human scores (so they're in our 0-3 rubric)
    for p in pairs:
        p["human_rel"] = map_score_to_rubric(p["human_rel"], cfg["score_mapping"])

    # Run judges
    print(f"Running judges on {len(pairs)} pairs...")
    results = run_judges_on_pairs(pairs, specs)

    # Save
    out_path = RESULTS_DIR / f"{corpus_id}_judges.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Show summary
    print("\n=== Per-judge summary ===")
    for label, meta in results["per_judge_metadata"].items():
        print(f"  {label[:40]:<40s}  valid={meta['n_valid']:3d}/{len(pairs)}  mean={meta['mean_score']:.2f}")

    print(f"\nNext step: --analyze {corpus_id} to compute κ vs human qrels")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Add beir-scifact and trec-covid to corpus choices since they aren't in CORPUS_REGISTRY by default
    corpus_choices = list(CORPUS_REGISTRY.keys()) + ["beir-scifact", "trec-covid"]
    p.add_argument("--download", choices=list(CORPUS_REGISTRY.keys()), help="Download a corpus's qrels and topics")
    p.add_argument("--corpus", choices=corpus_choices, help="Run validation against a corpus")
    p.add_argument("--analyze", choices=corpus_choices, help="Compute κ vs human qrels for an already-run corpus")
    p.add_argument("--judge-preset", default="p4-frontier", help="Judge preset (see eval_llm_judge.py JUDGE_PRESETS)")
    p.add_argument("--judges", type=lambda s: s.split(","), help="Explicit judge list (comma-separated specs from JUDGE_BUILDERS), overrides --judge-preset")
    p.add_argument("--max-pairs", type=int, default=300, help="Max pairs to validate")
    p.add_argument("--list", action="store_true", help="List available corpora")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("Available corpora:")
        for cid, cfg in CORPUS_REGISTRY.items():
            status = cfg.get("status", "auto")
            print(f"  {cid:<28s}  pairs≈{cfg['expected_pairs']:>4d}  scale={cfg['score_scale']:<8s}  status={status}")
            print(f"  {' '*30}{cfg['description']}")
        return 0

    if args.download:
        return cmd_download(args.download)
    if args.corpus:
        return cmd_validate(args.corpus, args.judge_preset, args.max_pairs, args.judges)
    if args.analyze:
        return cmd_analyze(args.analyze)

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
