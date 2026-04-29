"""Tokenizer overlap test: Qwen 3.6 + Gemma 4 + reference baselines.

Tests the "shared open ecosystem" hypothesis behind the matrix-highest κ
between Qwen 3.6 Plus and Gemma 4 26B. If their tokenizers have unusually
high vocabulary overlap, this corroborates training-data-overlap as a
mechanism.

For comparison, also computes Qwen vs Llama-3 (a different open ecosystem)
and Gemma 4 vs Llama-3.

Note: we use the canonical Hugging Face tokenizers for Qwen 3 (predecessor of
3.6 — same SentencePiece family) and Gemma 4 (the actual deployed open-weight).
For Qwen 3.6 Plus the tokenizer should be identical to Qwen 2.5 / Qwen 3 series.

Usage:
    py -3 papers/P4_llm_as_judge/analyze_tokenizer_overlap.py
"""
from __future__ import annotations
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "judge_tokenizer_overlap.json"

# Use cache dir within repo to avoid leaking to user profile
os.environ.setdefault("HF_HOME", str(ROOT / "_hf_cache"))

# Reference tokenizers we'll compare. Use non-gated mirrors where possible.
TOKENIZERS = {
    # Open-weight in our slate
    "Qwen 3 (proxy for 3.6 Plus)": "Qwen/Qwen3-32B",  # newest available; Qwen 3.6 not on HF yet
    "Gemma 2 9B (proxy for 4 26B)": "unsloth/gemma-2-9b-it",  # third-party non-gated mirror; same tokenizer as Gemma 2 27B
    # Cross-ecosystem references
    "Llama 3 (NousResearch mirror)": "NousResearch/Meta-Llama-3-8B",  # non-gated mirror
    "Mistral Small": "mistralai/Mistral-Small-Instruct-2409",
    "Phi 3.5 mini": "microsoft/Phi-3.5-mini-instruct",  # MS open
}


def main() -> int:
    print("Loading tokenizers from Hugging Face (cached at first use)...\n")
    from transformers import AutoTokenizer
    tokenizers = {}
    vocabs = {}
    for name, hf_id in TOKENIZERS.items():
        print(f"  Loading {name} ({hf_id})...")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, use_fast=True)
            vocab = tok.get_vocab()
            tokenizers[name] = tok
            vocabs[name] = set(vocab.keys())
            print(f"    vocab size: {len(vocab)}")
        except Exception as e:
            print(f"    [FAIL] {e}")
            tokenizers[name] = None
            vocabs[name] = None
    print()

    # Pairwise Jaccard similarity on vocabulary tokens
    print("Pairwise vocabulary Jaccard similarity:\n")
    print(f"{'Pair':<55s}  {'|A|':>8s}  {'|B|':>8s}  {'|A∩B|':>8s}  {'Jaccard':>9s}")

    names = list(TOKENIZERS.keys())
    overlap_data = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                va, vb = vocabs[a], vocabs[b]
                if va is None or vb is None:
                    continue
                inter = len(va & vb)
                union = len(va | vb)
                jaccard = inter / union if union > 0 else 0.0
                pair_label = f"{a} / {b}"
                print(f"{pair_label:<55s}  {len(va):>8d}  {len(vb):>8d}  {inter:>8d}  {jaccard:>9.4f}")
                overlap_data[pair_label] = {
                    "vocab_a_size": len(va),
                    "vocab_b_size": len(vb),
                    "intersection": inter,
                    "union": union,
                    "jaccard": jaccard,
                }
    print()

    # Sanity: top 30 tokens that are unique to each pair vs shared
    print("Sample shared subwords between Qwen and Gemma (random 25):\n")
    if vocabs.get("Qwen 3 (proxy for 3.6 Plus)") and vocabs.get("Gemma 2 27B (proxy for 4 26B)"):
        shared = vocabs["Qwen 3 (proxy for 3.6 Plus)"] & vocabs["Gemma 2 27B (proxy for 4 26B)"]
        sample = sorted(shared)[:25]
        print("  ", ", ".join(sample))
        print()

    out = {
        "config": {
            "tokenizers": TOKENIZERS,
            "method": "Jaccard similarity on vocabulary token strings",
        },
        "vocab_sizes": {name: len(v) if v else None for name, v in vocabs.items()},
        "pairwise_overlap": overlap_data,
        "interpretation": (
            "Jaccard similarity on raw token strings. "
            "Note: small differences in special-token formatting can dominate Jaccard "
            "even when underlying subword units overlap heavily. Treat as a coarse signal."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
