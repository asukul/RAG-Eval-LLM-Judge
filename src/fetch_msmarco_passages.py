"""Fetch MS MARCO v2.1 segmented passages for TREC RAG 2024 qrels subset.

Streams shards from drexalt/msmarco-2.1-segmented on Hugging Face, filters to
the 537 unique passage IDs in needed_passage_ids.txt, saves to a compact
local JSON keyed by docid. Shards are downloaded one at a time and deleted
after processing to keep peak disk usage bounded (~400 MB).

Usage:
  py -3 -X utf8 papers/P4_llm_as_judge/fetch_msmarco_passages.py

Output:
  _validation_data/trec-rag-2024/passages.json (one entry per needed docid)
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "_validation_data" / "trec-rag-2024"
NEEDED_IDS_FILE = DATA / "needed_passage_ids.txt"
OUT_JSON = DATA / "passages.json"
PROGRESS_FILE = DATA / "fetch_msmarco_progress.json"

REPO_ID = "drexalt/msmarco-2.1-segmented"
SHARD_TEMPLATE = "msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_{:02d}.json.gz"


def shard_for_id(passage_id: str) -> int:
    return int(passage_id.split("_")[3])


def main() -> int:
    if not NEEDED_IDS_FILE.exists():
        print(f"ERROR: needed IDs file not found at {NEEDED_IDS_FILE}", file=sys.stderr)
        return 1

    needed = set(NEEDED_IDS_FILE.read_text(encoding="utf-8").strip().splitlines())
    print(f"Targeting {len(needed)} unique passage IDs")

    by_shard: dict[int, set[str]] = {}
    for pid in needed:
        s = shard_for_id(pid)
        by_shard.setdefault(s, set()).add(pid)
    print(f"Shards to scan: {len(by_shard)} (shards {min(by_shard)}-{max(by_shard)})")

    if OUT_JSON.exists():
        passages = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        print(f"Resuming from existing passages.json ({len(passages)} already fetched)")
    else:
        passages = {}

    if PROGRESS_FILE.exists():
        progress = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        completed_shards = set(progress.get("completed_shards", []))
    else:
        completed_shards = set()
    print(f"Already-completed shards: {sorted(completed_shards)}")

    t_start = time.time()
    for shard_idx in sorted(by_shard.keys()):
        if shard_idx in completed_shards:
            print(f"[shard {shard_idx:02d}] skipped (already done)")
            continue

        shard_targets = by_shard[shard_idx]
        already_have = sum(1 for pid in shard_targets if pid in passages)
        if already_have == len(shard_targets):
            print(f"[shard {shard_idx:02d}] all {len(shard_targets)} passages already in cache, marking done")
            completed_shards.add(shard_idx)
            PROGRESS_FILE.write_text(
                json.dumps({"completed_shards": sorted(completed_shards)}, indent=2),
                encoding="utf-8",
            )
            continue

        rel_path = SHARD_TEMPLATE.format(shard_idx)
        print(
            f"[shard {shard_idx:02d}] need {len(shard_targets)} passages "
            f"({already_have} already cached); downloading {rel_path}..."
        )
        t_dl = time.time()
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=rel_path,
            repo_type="dataset",
        )
        size_mb = Path(local_path).stat().st_size / 1024 / 1024
        print(f"  downloaded {size_mb:.1f} MB in {time.time()-t_dl:.1f}s; scanning...")

        n_scanned = 0
        n_matched_this_shard = 0
        t_scan = time.time()
        with gzip.open(local_path, "rt", encoding="utf-8") as f:
            for line in f:
                n_scanned += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                docid = obj.get("docid") or obj.get("id")
                if not docid or docid not in shard_targets:
                    continue
                passages[docid] = {
                    "docid": docid,
                    "title": obj.get("title", ""),
                    "headings": obj.get("headings", ""),
                    "text": obj.get("segment") or obj.get("text", ""),
                    "url": obj.get("url", ""),
                }
                n_matched_this_shard += 1
                if n_matched_this_shard >= len(shard_targets):
                    break  # we have all the passages from this shard
        scan_dur = time.time() - t_scan
        print(
            f"  scanned {n_scanned:,} lines in {scan_dur:.1f}s; "
            f"matched {n_matched_this_shard}/{len(shard_targets)} target passages from this shard"
        )

        # Persist progress + delete shard cache file (keep peak disk small)
        OUT_JSON.write_text(json.dumps(passages, indent=2), encoding="utf-8")
        try:
            Path(local_path).unlink()
        except OSError:
            pass

        completed_shards.add(shard_idx)
        PROGRESS_FILE.write_text(
            json.dumps({"completed_shards": sorted(completed_shards)}, indent=2),
            encoding="utf-8",
        )
        elapsed = time.time() - t_start
        remaining = len(by_shard) - len(completed_shards)
        eta_min = (elapsed / max(1, len(completed_shards))) * remaining / 60
        print(
            f"  ✓ shard {shard_idx:02d} done. Total cached: {len(passages)}/{len(needed)}. "
            f"Remaining shards: {remaining}. ETA: ~{eta_min:.0f} min"
        )

    missing = sorted(needed - set(passages.keys()))
    print()
    print("=" * 60)
    print(f"Total cached: {len(passages)}/{len(needed)}")
    print(f"Missing: {len(missing)}")
    if missing:
        print("First 10 missing IDs:")
        for m in missing[:10]:
            print(f"  {m}")
    print(f"Saved to: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
