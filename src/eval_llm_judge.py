"""
LLM-as-judge evaluation framework for RAG retrieval quality.

For each test query in a JSON file, this script:
  1. Embeds the query with the backend that matches the target Qdrant collection
     (Ollama for `dspace_fulltext_nomic`, Vertex for `dspace_fulltext_vertex` or
      `dspace_chunks`).
  2. Searches Qdrant for the top-K most similar chunks.
  3. Asks Claude to score each chunk's relevance on a 0-3 scale:
       0 = Not relevant
       1 = Marginally relevant
       2 = Relevant
       3 = Highly relevant
  4. Computes standard IR metrics — nDCG@5, nDCG@10, Precision@1/5/10, MRR —
     per-query and averaged across the test set.
  5. Saves a timestamped JSON report under results/.

The LLM judge supports two transports:
  A) Direct `anthropic` SDK using ANTHROPIC_API_KEY (fast, default when available)
  B) Fallback: shell out to the `claude` CLI (`claude -p "..."`) if no key is set

Supported collections (--collection):
  - dspace_chunks          (Vertex text-embedding-005, task=RETRIEVAL_QUERY)
  - dspace_fulltext_vertex (Vertex text-embedding-005, task=RETRIEVAL_QUERY)
  - dspace_fulltext_nomic  (Ollama nomic-embed-text, task=query)

Usage:
    # Minimal — will auto-create a sample test_queries.json on first run
    python scripts/eval_llm_judge.py \
        --collection dspace_fulltext_nomic \
        --queries data/eval/test_queries.json \
        --top-k 10

    # Compare models
    python scripts/eval_llm_judge.py --collection dspace_fulltext_nomic  --queries data/eval/test_queries.json
    python scripts/eval_llm_judge.py --collection dspace_fulltext_vertex --queries data/eval/test_queries.json
    python scripts/eval_llm_judge.py --collection dspace_chunks          --queries data/eval/test_queries.json

    # Use the Claude CLI instead of the SDK (no API key needed)
    python scripts/eval_llm_judge.py --judge-backend cli --collection dspace_fulltext_nomic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# qdrant-client is needed for the within-corpus retrieval path (--collection),
# but NOT for the --analyze path that just reads shipped per-judge JSONs.
# Make the import lazy so external-validation analysis (validate_against_trec.py
# --analyze) works on machines that haven't installed qdrant_client.
try:
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    _QDRANT_AVAILABLE = True
except ImportError:  # pragma: no cover
    QdrantClient = None  # type: ignore[assignment,misc]
    _QDRANT_AVAILABLE = False

# ----------------------------------------------------------------------
# Paths and defaults
# ----------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
# Standalone-repo paths. The harness was originally written for an outer
# isu-research-search/backend/scripts/ layout; in the public RAG-Eval-LLM-Judge
# repo, src/ sits directly under the repo root and results/ is the equivalent
# of the old backend/data/eval/ output directory.
REPO = HERE.parent
EVAL_DIR = REPO / "results"
DEFAULT_QUERIES = EVAL_DIR / "test_queries.json"

# Backwards-compatible alias for any external code that imported BACKEND.
BACKEND = REPO

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6334")
DEFAULT_TOP_K = 10
DEFAULT_CLAUDE_MODEL = os.getenv("CLAUDE_JUDGE_MODEL", "claude-opus-4-6")

# Project .env (API keys for OpenAI/Gemini/OpenRouter/Anthropic SDK usage).
# In the standalone repo, .env sits at the repo root.
PROJECT_ROOT = REPO
DOTENV_PATH = PROJECT_ROOT / ".env"


def _load_dotenv_manual(path: Path) -> int:
    """Lightweight .env loader: no python-dotenv dependency.
    Skips blanks/comments. Does NOT overwrite already-set env vars."""
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n

# Per-collection embedding configuration. These mirror the scripts that built
# the collections (embed_pdfs_ollama.py, embed_vertex_batch_real.py,
# embed_abstracts_to_qdrant.py).
COLLECTION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "dspace_chunks": {
        "embedder": "vertex",
        "model": "text-embedding-005",
        "dim": 768,
        "task_type": "RETRIEVAL_QUERY",
        "vector_name": None,  # uses the unnamed default vector
        "payload_text_fields": ("text", "text_preview", "snippet"),
        "payload_title_fields": ("doc_title", "title"),
    },
    "dspace_fulltext_vertex": {
        "embedder": "vertex",
        "model": "text-embedding-005",
        "dim": 768,
        "task_type": "RETRIEVAL_QUERY",
        "vector_name": "dense",  # named vector per current Qdrant collection schema
        "payload_text_fields": ("text", "text_preview", "snippet"),
        "payload_title_fields": ("title", "doc_title"),
    },
    "dspace_fulltext_nomic": {
        "embedder": "ollama",
        "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        "dim": 768,
        "task_type": "query",  # nomic uses "search_query: " prefix
        "vector_name": None,
        "payload_text_fields": ("text", "text_preview", "snippet"),
        "payload_title_fields": ("title", "doc_title"),
    },
}

# Shared 5-query sample that gets written to disk if the queries file is absent.
# Chosen to cover the ISU DSpace corpus: agriculture, engineering, policy, CS, vet med.
SAMPLE_QUERIES: List[Dict[str, Any]] = [
    {
        "id": "q1_soybean_drought",
        "query": "How does drought stress affect soybean yield in Iowa farms?",
        "tags": ["agriculture", "plant_science"],
    },
    {
        "id": "q2_wind_turbine_fatigue",
        "query": "Fatigue analysis of wind turbine blades composite materials",
        "tags": ["engineering", "wind_energy"],
    },
    {
        "id": "q3_food_security_policy",
        "query": "Rural food insecurity policy interventions effectiveness",
        "tags": ["policy", "social_science"],
    },
    {
        "id": "q4_machine_learning_crops",
        "query": "Machine learning methods for crop disease detection",
        "tags": ["computer_science", "agriculture"],
    },
    {
        "id": "q5_veterinary_swine",
        "query": "Porcine reproductive and respiratory syndrome virus diagnostics",
        "tags": ["veterinary", "swine"],
    },
]


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_sample_queries(path: Path) -> None:
    """Write a 5-query placeholder file if `path` doesn't exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Placeholder sample test queries for the LLM-judge RAG eval. "
            "Regenerate this file with the synthetic-query generator to replace with real data."
        ),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries": SAMPLE_QUERIES,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote sample queries to {path} ({len(SAMPLE_QUERIES)} queries)")


def load_queries(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON queries file. Accepts either a top-level list or {queries: [...]}"""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        queries = data
    elif isinstance(data, dict):
        queries = data.get("queries") or data.get("items") or []
    else:
        queries = []
    # Normalise each entry to {id, query, tags}
    out = []
    for i, q in enumerate(queries):
        if isinstance(q, str):
            out.append({"id": f"q{i+1}", "query": q, "tags": []})
        elif isinstance(q, dict):
            out.append({
                "id": q.get("id") or f"q{i+1}",
                "query": q.get("query") or q.get("question") or q.get("text") or "",
                "tags": q.get("tags") or [],
            })
    # Drop blanks
    return [q for q in out if q["query"].strip()]


# ----------------------------------------------------------------------
# Embedders
# ----------------------------------------------------------------------

class OllamaEmbedder:
    """
    Minimal Ollama embedding client. Mirrors the logic in
    services/embedding_service.py — specifically the `search_query: ` prefix
    that nomic-embed-text requires for query-side embeddings.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def embed_query(self, text: str) -> Optional[List[float]]:
        if "nomic-embed" in self.model.lower():
            text = "search_query: " + text
        body = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                emb = data.get("embedding")
                if not emb:
                    # Newer Ollama versions return under "embeddings"
                    embs = data.get("embeddings") or []
                    if embs:
                        return embs[0]
                return emb
        except Exception as e:
            log(f"  Ollama embed failed: {e}")
            return None


class VertexEmbedder:
    """
    Wraps google-genai Vertex client for query embeddings.
    Uses RETRIEVAL_QUERY task type and text-embedding-005 by default.
    """

    def __init__(self, model: str, dim: int = 768,
                 project: str = "rag-dspace", location: str = "us-central1"):
        self.model = model
        self.dim = dim
        self.project = project
        self.location = location
        self._client = None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            from google import genai
            self._client = genai.Client(
                vertexai=True, project=self.project, location=self.location,
            )
            return True
        except Exception as e:
            log(f"  Vertex client init failed: {e}")
            return False

    def embed_query(self, text: str) -> Optional[List[float]]:
        if not self._ensure_client():
            return None
        try:
            from google.genai import types as genai_types
            result = self._client.models.embed_content(
                model=self.model,
                contents=[text],
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self.dim,
                ),
            )
            if not result or not result.embeddings:
                return None
            return list(result.embeddings[0].values)
        except Exception as e:
            log(f"  Vertex embed failed: {e}")
            return None


def build_embedder(config: Dict[str, Any]):
    """Return an embedder instance keyed off the collection config."""
    if config["embedder"] == "ollama":
        return OllamaEmbedder(model=config["model"])
    if config["embedder"] == "vertex":
        return VertexEmbedder(model=config["model"], dim=config.get("dim", 768))
    raise ValueError(f"Unknown embedder type: {config['embedder']}")


# ----------------------------------------------------------------------
# Qdrant retrieval
# ----------------------------------------------------------------------

def retrieve(
    client: QdrantClient,
    collection: str,
    query_vector: List[float],
    top_k: int,
    vector_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search a Qdrant collection and return normalized hits.

    Uses `query_points` which works with both named and default vectors.
    Falls back silently if the collection uses an unusual vector config.
    """
    try:
        if vector_name:
            res = client.query_points(
                collection_name=collection,
                query=query_vector,
                using=vector_name,
                limit=top_k,
                with_payload=True,
            )
        else:
            res = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
    except Exception as e:
        log(f"  Qdrant query failed on {collection}: {e}")
        return []

    out = []
    for hit in res.points:
        payload = hit.payload or {}
        out.append({
            "id": str(getattr(hit, "id", "")),
            "score": float(getattr(hit, "score", 0.0) or 0.0),
            "payload": payload,
        })
    return out


def extract_text_and_title(
    payload: Dict[str, Any],
    text_fields: Tuple[str, ...],
    title_fields: Tuple[str, ...],
) -> Tuple[str, str]:
    """Pull text + title from a payload using a fallback chain."""
    text = ""
    for f in text_fields:
        v = payload.get(f)
        if v:
            text = str(v)
            break
    title = ""
    for f in title_fields:
        v = payload.get(f)
        if v:
            title = str(v)
            break
    return text, title


# ----------------------------------------------------------------------
# LLM judge
# ----------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = (
    "Query: {query}\n"
    "Document chunk: {text}\n"
    "Score (0-3) how relevant this chunk is to answering the query.\n"
    "0 = Not relevant, 1 = Marginally relevant, 2 = Relevant, 3 = Highly relevant.\n"
    "Output ONLY a single digit 0, 1, 2, or 3."
)

_DIGIT_RE = re.compile(r"[0-3]")


def parse_score(text: str) -> Optional[int]:
    """Extract the first 0-3 digit from a Claude response. None on failure."""
    if not text:
        return None
    m = _DIGIT_RE.search(text.strip())
    if not m:
        return None
    try:
        v = int(m.group(0))
        return v if 0 <= v <= 3 else None
    except Exception:
        return None


class LLMJudge:
    """
    Score relevance via Claude. Two backends:
      - 'sdk' : anthropic Python SDK (requires ANTHROPIC_API_KEY)
      - 'cli' : shell out to `claude -p "..."` (requires the Claude CLI on PATH)
    `auto` picks SDK when the key is set and falls back to CLI otherwise.
    """

    def __init__(
        self,
        backend: str = "auto",
        model: str = DEFAULT_CLAUDE_MODEL,
        max_chars: int = 500,
    ):
        self.model = model
        self.max_chars = max_chars
        self.backend = self._select_backend(backend)
        self._sdk_client = None
        if self.backend == "sdk":
            self._init_sdk()

    # -- backend selection -------------------------------------------------

    def _select_backend(self, requested: str) -> str:
        if requested == "sdk":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "judge-backend=sdk requires ANTHROPIC_API_KEY to be set"
                )
            return "sdk"
        if requested == "cli":
            if not shutil.which("claude"):
                raise RuntimeError(
                    "judge-backend=cli requires the `claude` CLI to be on PATH"
                )
            return "cli"
        # auto
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # noqa: F401
                return "sdk"
            except ImportError:
                pass
        if shutil.which("claude"):
            return "cli"
        raise RuntimeError(
            "No LLM judge backend available: set ANTHROPIC_API_KEY (with `anthropic` "
            "installed) or install the `claude` CLI and re-run."
        )

    def _init_sdk(self) -> None:
        import anthropic
        self._sdk_client = anthropic.Anthropic()
        log(f"LLMJudge: using Anthropic SDK (model={self.model})")

    # -- scoring -----------------------------------------------------------

    def _build_prompt(self, query: str, chunk_text: str) -> str:
        trimmed = (chunk_text or "").strip()[: self.max_chars]
        return JUDGE_PROMPT_TEMPLATE.format(query=query, text=trimmed)

    def _score_sdk(self, prompt: str) -> Optional[int]:
        # Retry once on any exception; most SDK failures are transient.
        for attempt in range(3):
            try:
                resp = self._sdk_client.messages.create(
                    model=self.model,
                    max_tokens=4,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = ""
                for block in resp.content or []:
                    t = getattr(block, "text", None)
                    if t:
                        text += t
                score = parse_score(text)
                if score is not None or attempt == 2:
                    return score
                # Empty/unparseable — brief sleep then retry
                time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                if attempt == 2:
                    log(f"  judge sdk error (final attempt {attempt+1}): {e}")
                    return None
                log(f"  judge sdk error (attempt {attempt+1}): {e} — retrying")
                time.sleep(1.0 * (attempt + 1))
        return None

    def _score_cli(self, prompt: str) -> Optional[int]:
        """`claude -p "prompt"` prints the assistant reply to stdout and exits.
        Retry up to 3 times on non-zero exit or empty output — most CLI failures
        are transient (subprocess buffering, auth refresh, brief rate-limit)."""
        for attempt in range(3):
            try:
                result = subprocess.run(
                    ["claude", "-p", prompt],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    encoding="utf-8",
                    errors="replace",
                    stdin=subprocess.DEVNULL,
                )
                if result.returncode != 0:
                    stderr = (result.stderr or "").strip()
                    stdout = (result.stdout or "").strip()
                    if attempt == 2:
                        log(f"  judge cli error (final attempt): rc={result.returncode} "
                            f"stderr={stderr[:200]!r} stdout[:80]={stdout[:80]!r}")
                        return None
                    log(f"  judge cli error (attempt {attempt+1}/3): rc={result.returncode} "
                        f"stderr={stderr[:120]!r} — retrying")
                    time.sleep(1.0 * (attempt + 1))
                    continue
                score = parse_score(result.stdout)
                if score is None and attempt < 2:
                    # Retry once on parse failure (CLI sometimes outputs extra noise)
                    log(f"  judge cli parse failed (attempt {attempt+1}/3) — retrying")
                    time.sleep(0.5)
                    continue
                return score
            except subprocess.TimeoutExpired:
                if attempt == 2:
                    log("  judge cli timeout (final attempt)")
                    return None
                log(f"  judge cli timeout (attempt {attempt+1}/3) — retrying")
                continue
            except Exception as e:
                if attempt == 2:
                    log(f"  judge cli error (final): {e}")
                    return None
                log(f"  judge cli error (attempt {attempt+1}/3): {e} — retrying")
                time.sleep(1.0)
        return None

    def score(self, query: str, chunk_text: str) -> Optional[int]:
        prompt = self._build_prompt(query, chunk_text)
        if self.backend == "sdk":
            return self._score_sdk(prompt)
        return self._score_cli(prompt)


# ----------------------------------------------------------------------
# Additional judge backends (Gemini via Vertex; OpenAI SDK; OpenRouter)
# ----------------------------------------------------------------------

class GeminiVertexJudge:
    """Gemini judge via google-genai SDK + Vertex AI. Uses gcloud default
    credentials (project=rag-dspace, location=us-central1 by default).
    Retries on HTTP 429 RESOURCE_EXHAUSTED with exponential backoff."""
    backend = "gemini-vertex"

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        project: str = "rag-dspace",
        location: str = "us-central1",
        thinking_budget: Optional[int] = None,
        max_out_tok: int = 16,
        max_chars: int = 500,
    ):
        from google import genai  # lazy import
        self.model = model
        self.max_chars = max_chars
        self.thinking_budget = thinking_budget
        self.max_out_tok = max_out_tok
        self._client = genai.Client(vertexai=True, project=project, location=location)
        log(f"GeminiVertexJudge: model={model} thinking_budget={thinking_budget} max_out_tok={max_out_tok}")

    def score(self, query: str, chunk_text: str) -> Optional[int]:
        from google.genai import types
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query, text=(chunk_text or "").strip()[: self.max_chars]
        )
        cfg_kwargs = {"max_output_tokens": self.max_out_tok, "temperature": 0.0}
        if self.thinking_budget is not None:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        for attempt in range(3):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                text = (resp.text or "").strip() if resp else ""
                score = parse_score(text)
                if score is not None or attempt == 2:
                    return score
                # Empty/unparseable — brief retry
                time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                msg = str(e)[:250]
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < 2:
                    delay = [5, 15][attempt]
                    log(f"  GeminiVertex 429 (attempt {attempt+1}/3); sleeping {delay}s")
                    time.sleep(delay)
                    continue
                # Other transient errors — short backoff + retry
                if attempt < 2:
                    log(f"  GeminiVertex error (attempt {attempt+1}/3) ({self.model}): {msg} — retrying")
                    time.sleep(1.0 * (attempt + 1))
                    continue
                log(f"  GeminiVertex error final ({self.model}): {msg}")
                return None
        return None


class OpenAISdkJudge:
    """OpenAI SDK judge. Supports both Responses API (reasoning models) and
    Chat Completions (classic chat models).

    Also used as the base class for OpenRouterJudge since OpenRouter exposes
    an OpenAI-compatible API."""
    backend = "openai-sdk"

    def __init__(
        self,
        model: str = "gpt-5.5",
        reasoning_effort: Optional[str] = None,
        max_out_tok: int = 256,
        max_chars: int = 500,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        import openai  # lazy import
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_out_tok = max_out_tok
        self.max_chars = max_chars
        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        loc = "openrouter" if base_url and "openrouter" in base_url else "openai"
        log(f"OpenAISdkJudge[{loc}]: model={model} reasoning={reasoning_effort} max_out_tok={max_out_tok}")

    def score(self, query: str, chunk_text: str) -> Optional[int]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query, text=(chunk_text or "").strip()[: self.max_chars]
        )
        for attempt in range(3):
            try:
                if self.reasoning_effort:
                    resp = self._client.responses.create(
                        model=self.model,
                        input=prompt,
                        max_output_tokens=self.max_out_tok,
                        reasoning={"effort": self.reasoning_effort},
                    )
                    text = (getattr(resp, "output_text", "") or "").strip()
                    if not text:
                        for item in getattr(resp, "output", []) or []:
                            for b in getattr(item, "content", []) or []:
                                t = getattr(b, "text", None)
                                if t:
                                    text = t.strip()
                                    break
                            if text:
                                break
                else:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=min(self.max_out_tok, 16),
                        temperature=0.0,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                score = parse_score(text)
                if score is not None or attempt == 2:
                    return score
                # Empty text, likely transient — brief retry
                time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                msg = str(e)[:250]
                if attempt < 2:
                    # Rate-limit-style errors get longer backoff
                    is_rate = "429" in msg or "rate" in msg.lower()
                    delay = (5 if is_rate else 1) * (attempt + 1)
                    log(f"  OpenAI judge error (attempt {attempt+1}/3) ({self.model}): {msg} — retrying in {delay}s")
                    time.sleep(delay)
                    continue
                log(f"  OpenAI judge error final ({self.model}): {msg}")
                return None
        return None


class OpenRouterJudge(OpenAISdkJudge):
    """OpenRouter wrapper. Uses OPENROUTER_API_KEY and model IDs like
    `google/gemini-3.1-pro-preview` or `anthropic/claude-sonnet-4.6`."""
    backend = "openrouter-sdk"

    def __init__(self, model: str, **kwargs):
        super().__init__(
            model=model,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )


# Judge-spec registry. Each entry returns (label, judge_instance).
# Used by build_judges() for --judges / --judge-preset dispatch.
JUDGE_BUILDERS: Dict[str, Any] = {
    # ---- Anthropic Claude (legacy CLI/SDK path) ----
    "claude-sonnet-cli":
        lambda: ("Claude Sonnet 4.6 (CLI)",
                 LLMJudge(backend="auto", model="claude-sonnet-4-6")),
    "claude-haiku-cli":
        lambda: ("Claude Haiku 4.5 (CLI)",
                 LLMJudge(backend="auto", model="claude-haiku-4-5")),
    "claude-opus-cli":
        lambda: ("Claude Opus 4.6 (CLI)",
                 LLMJudge(backend="auto", model="claude-opus-4-6")),

    # ---- Anthropic Claude via OpenRouter (preferred — no Claude Code policy layer) ----
    "claude-sonnet":
        lambda: ("Claude Sonnet 4.6 (OpenRouter)",
                 OpenRouterJudge(model="anthropic/claude-sonnet-4.6", max_out_tok=16)),
    "claude-haiku":
        lambda: ("Claude Haiku 4.5 (OpenRouter)",
                 OpenRouterJudge(model="anthropic/claude-haiku-4.5", max_out_tok=16)),
    "claude-opus":
        lambda: ("Claude Opus 4.6 (OpenRouter)",
                 OpenRouterJudge(model="anthropic/claude-opus-4.6", max_out_tok=16)),
    "claude-opus-4.7":
        lambda: ("Claude Opus 4.7 (OpenRouter)",
                 OpenRouterJudge(model="anthropic/claude-opus-4.7", max_out_tok=16)),
    "deepseek-v4-pro":
        lambda: ("DeepSeek V4 Pro (OpenRouter)",
                 OpenRouterJudge(model="deepseek/deepseek-v4-pro", max_out_tok=16)),
    "deepseek-v4-flash":
        lambda: ("DeepSeek V4 Flash (OpenRouter)",
                 OpenRouterJudge(model="deepseek/deepseek-v4-flash", max_out_tok=16)),
    # Open-weight flagships (added 2026-04-25 for P4-SHORT supplementary run)
    "qwen-3.6-plus":
        lambda: ("Qwen 3.6 Plus (OpenRouter)",
                 OpenRouterJudge(model="qwen/qwen3.6-plus", max_out_tok=16)),
    "gemma-4-26b":
        lambda: ("Gemma 4 26B (OpenRouter)",
                 OpenRouterJudge(model="google/gemma-4-26b-a4b-it", max_out_tok=16)),

    # ---- Google Gemini via Vertex (gcloud creds, thinking support) ----
    "gemini-vertex-pro":
        lambda: ("Gemini 2.5 Pro (Vertex, thinking=1024)",
                 GeminiVertexJudge(model="gemini-2.5-pro", thinking_budget=1024, max_out_tok=2048)),
    "gemini-vertex-flash":
        lambda: ("Gemini 2.5 Flash (Vertex, thinking=0)",
                 GeminiVertexJudge(model="gemini-2.5-flash", thinking_budget=0, max_out_tok=16)),
    "gemini-vertex-flash-lite":
        lambda: ("Gemini 2.5 Flash-Lite (Vertex)",
                 GeminiVertexJudge(model="gemini-2.5-flash-lite", max_out_tok=16)),

    # ---- Google Gemini via OpenRouter (preferred uniform-auth path) ----
    "gemini-2.5-pro":
        lambda: ("Gemini 2.5 Pro (OpenRouter)",
                 OpenRouterJudge(model="google/gemini-2.5-pro", max_out_tok=2048)),
    "gemini-2.5-flash":
        lambda: ("Gemini 2.5 Flash (OpenRouter)",
                 OpenRouterJudge(model="google/gemini-2.5-flash", max_out_tok=16)),
    "gemini-2.5-flash-lite":
        lambda: ("Gemini 2.5 Flash-Lite (OpenRouter)",
                 OpenRouterJudge(model="google/gemini-2.5-flash-lite", max_out_tok=16)),
    "gemini-3.1-pro":
        lambda: ("Gemini 3.1 Pro Preview (OpenRouter)",
                 OpenRouterJudge(model="google/gemini-3.1-pro-preview", max_out_tok=2048)),
    "gemini-3.1-flash-lite":
        lambda: ("Gemini 3.1 Flash-Lite Preview (OpenRouter)",
                 OpenRouterJudge(model="google/gemini-3.1-flash-lite-preview", max_out_tok=16)),
    # Deprecated aliases (kept for back-compat with earlier experiments)
    "gemini-openrouter-3.1-pro":
        lambda: ("Gemini 3.1 Pro Preview (OpenRouter)",
                 OpenRouterJudge(model="google/gemini-3.1-pro-preview", max_out_tok=2048)),
    "gemini-openrouter-3.1-flash-lite":
        lambda: ("Gemini 3.1 Flash-Lite Preview (OpenRouter)",
                 OpenRouterJudge(model="google/gemini-3.1-flash-lite-preview", max_out_tok=16)),

    # ---- OpenAI (direct SDK — cheaper than OpenRouter markup for OpenAI models) ----
    "openai-gpt-5.5-low":
        lambda: ("GPT-5.5 (reasoning=low)",
                 OpenAISdkJudge(model="gpt-5.5", reasoning_effort="low", max_out_tok=256)),
    "openai-gpt-5-low":
        lambda: ("GPT-5 (reasoning=low)",
                 OpenAISdkJudge(model="gpt-5", reasoning_effort="low", max_out_tok=256)),
    "openai-gpt-4o":
        lambda: ("GPT-4o (chat)",
                 OpenAISdkJudge(model="gpt-4o", max_out_tok=16)),
}


JUDGE_PRESETS: Dict[str, List[str]] = {
    # ===== P4 §6.4.2 cross-family judge ablation =====
    # Original 5-judge slate (kept for reproducibility of the 2026-04-24 run).
    # Claude+Gemini via OpenRouter, OpenAI direct (OR markup for OpenAI is high).
    "p4-ablation": [
        "claude-sonnet",             # Anthropic via OpenRouter
        "gemini-2.5-pro",            # Google 2.5 flagship via OpenRouter
        "gemini-3.1-pro",            # Google 3.1 preview via OpenRouter
        "openai-gpt-5.5-low",        # OpenAI flagship direct
        "openai-gpt-4o",             # OpenAI chat baseline direct
    ],
    # ----- 7-judge frontier slate (canonical for the camera-ready) -----
    # Big-4 commercial frontier (Opus 4.7 + GPT-5.5 + Gemini 3.1 Pro + DeepSeek V4 Pro)
    # plus 3 within-family controls (Opus↔Sonnet, GPT-5.5↔GPT-4o, Gemini 3.1↔2.5).
    # DeepSeek V4 Flash dropped 2026-04-25 due to OpenRouter free-tier upstream
    # 429 throttling that hung an in-flight 8-judge run; can be added later via
    # supplemental run if a paid DeepSeek key is provided.
    "p4-frontier": [
        "claude-opus-4.7",           # Anthropic flagship via OpenRouter
        "claude-sonnet",             # Anthropic mid (within-family ctrl)
        "openai-gpt-5.5-low",        # OpenAI flagship direct
        "openai-gpt-4o",             # OpenAI chat (within-family ctrl)
        "gemini-3.1-pro",            # Google flagship preview via OpenRouter
        "gemini-2.5-pro",            # Google flagship-prev (within-family ctrl)
        "deepseek-v4-pro",           # DeepSeek flagship via OpenRouter (single, no within-pair)
    ],
    # Minimal fast preset for smoke tests — 3 judges, low cost.
    "fast-smoke": [
        "claude-sonnet",             # OpenRouter (no CLI policy issues)
        "gemini-2.5-flash-lite",     # OpenRouter
        "openai-gpt-4o",             # direct
    ],
    # ----- 2-judge open-weight supplementary run (2026-04-25) -----
    # Scores the SAME 570 pairs as p4-frontier, then merges offline to produce
    # a 9-judge canonical κ matrix. Adds two open-weight families (Qwen, Gemma)
    # to strengthen Finding 4 (DeepSeek-joins-reasoning-cluster) by ruling out
    # "commercial-only training data" as the cause of cross-family convergence.
    "p4-supplement-openweight": [
        "qwen-3.6-plus",
        "gemma-4-26b",
    ],
    # Same as p4-ablation but everything through OpenRouter (uniform billing).
    "p4-ablation-or-only": [
        "claude-sonnet",
        "gemini-2.5-pro",
        "gemini-3.1-pro",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
    ],
    # Degenerate single-judge (backward compat with legacy mode).
    "single": ["claude-sonnet"],
}


def build_judges(specs: List[str]) -> List[Tuple[str, Any]]:
    """Resolve a list of judge-spec IDs to (label, judge) pairs.

    A single-element list containing a preset name expands to the preset's specs."""
    if len(specs) == 1 and specs[0] in JUDGE_PRESETS:
        specs = JUDGE_PRESETS[specs[0]]
    out: List[Tuple[str, Any]] = []
    for spec in specs:
        if spec not in JUDGE_BUILDERS:
            raise ValueError(
                f"Unknown judge spec: {spec!r}. "
                f"Valid: {sorted(JUDGE_BUILDERS.keys())} "
                f"or preset: {sorted(JUDGE_PRESETS.keys())}"
            )
        label, judge = JUDGE_BUILDERS[spec]()
        out.append((label, judge))
    return out


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def dcg(scores: List[int]) -> float:
    """Discounted cumulative gain (using 2^rel - 1 numerator)."""
    total = 0.0
    for i, s in enumerate(scores):
        if s <= 0:
            continue
        total += (2 ** s - 1) / math.log2(i + 2)
    return total


def ndcg_at_k(scores: List[int], k: int) -> float:
    if not scores:
        return 0.0
    actual = dcg(scores[:k])
    ideal = dcg(sorted(scores, reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return actual / ideal


def precision_at_k(scores: List[int], k: int, threshold: int = 2) -> float:
    """
    Precision@k using `threshold` as the "relevant" cutoff. Defaults to 2 —
    treat scores 2-3 as relevant, 0-1 as non-relevant.
    """
    if not scores or k <= 0:
        return 0.0
    top = scores[:k]
    if not top:
        return 0.0
    hits = sum(1 for s in top if s >= threshold)
    return hits / min(k, len(top))


def reciprocal_rank(scores: List[int], threshold: int = 2) -> float:
    """First rank whose score >= threshold, as 1/rank. 0 if none qualify."""
    for i, s in enumerate(scores, start=1):
        if s >= threshold:
            return 1.0 / i
    return 0.0


def average(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# ----------------------------------------------------------------------
# Evaluation loop
# ----------------------------------------------------------------------

@dataclass
class QueryResult:
    query_id: str
    query: str
    tags: List[str]
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    scores: List[Optional[int]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def evaluate_query(
    query: Dict[str, Any],
    qclient: QdrantClient,
    collection: str,
    config: Dict[str, Any],
    embedder,
    judge: LLMJudge,
    top_k: int,
) -> QueryResult:
    qid = query["id"]
    text = query["query"]

    qr = QueryResult(query_id=qid, query=text, tags=query.get("tags", []))

    log(f"  [{qid}] {text[:80]}")
    qv = embedder.embed_query(text)
    if not qv:
        log(f"    embedding failed — skipping")
        qr.metrics = {"error": 1.0}
        return qr

    hits = retrieve(qclient, collection, qv, top_k, config.get("vector_name"))
    if not hits:
        log(f"    no hits returned")
        qr.metrics = {"error": 1.0}
        return qr

    text_fields = config["payload_text_fields"]
    title_fields = config["payload_title_fields"]

    # Judge every hit sequentially — these calls are cheap in aggregate
    # (50 queries * 10 hits * 4 tokens each = cheap on opus).
    scores: List[Optional[int]] = []
    retrieved_log: List[Dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        chunk_text, title = extract_text_and_title(hit["payload"], text_fields, title_fields)
        score = judge.score(text, chunk_text)
        scores.append(score)
        retrieved_log.append({
            "rank": rank,
            "qdrant_score": round(hit["score"], 4),
            "point_id": hit["id"],
            "title": title[:160],
            "text_preview": (chunk_text or "")[:240],
            "judge_score": score,
        })

    # Treat unscored (None) as 0 when computing metrics so nothing is NaN —
    # we still persist the raw None in retrieved_log so issues are visible.
    clean_scores = [s if s is not None else 0 for s in scores]
    qr.retrieved = retrieved_log
    qr.scores = scores
    qr.metrics = {
        "ndcg@5": round(ndcg_at_k(clean_scores, 5), 4),
        "ndcg@10": round(ndcg_at_k(clean_scores, 10), 4),
        "precision@1": round(precision_at_k(clean_scores, 1), 4),
        "precision@5": round(precision_at_k(clean_scores, 5), 4),
        "precision@10": round(precision_at_k(clean_scores, 10), 4),
        "mrr": round(reciprocal_rank(clean_scores), 4),
        "mean_judge_score": round(average([float(s) for s in clean_scores]), 4),
        "unscored": sum(1 for s in scores if s is None),
    }
    log(
        f"    nDCG@10={qr.metrics['ndcg@10']:.3f}  "
        f"P@5={qr.metrics['precision@5']:.3f}  "
        f"MRR={qr.metrics['mrr']:.3f}  "
        f"mean={qr.metrics['mean_judge_score']:.2f}"
    )
    return qr


def aggregate_metrics(results: List[QueryResult]) -> Dict[str, float]:
    keys = ["ndcg@5", "ndcg@10", "precision@1", "precision@5", "precision@10", "mrr",
            "mean_judge_score"]
    agg: Dict[str, float] = {}
    for k in keys:
        vals = [r.metrics[k] for r in results if k in r.metrics]
        agg[k] = round(average(vals), 4) if vals else 0.0
    agg["queries_evaluated"] = float(len(results))
    agg["queries_with_errors"] = float(sum(1 for r in results if r.metrics.get("error")))
    return agg


# ----------------------------------------------------------------------
# Multi-judge support: Cohen's κ + per-doc-pair scoring loop
# ----------------------------------------------------------------------

def cohen_kappa_quadratic(
    y1: List[Optional[int]],
    y2: List[Optional[int]],
    n_categories: int = 4,
) -> Optional[float]:
    """Quadratic-weighted Cohen's κ for ordinal data in {0..n_categories-1}.

    Pairs with None on either side are dropped (paired missingness).
    Returns None if all pairs drop out or if the expected agreement matches N.
    """
    paired = [(a, b) for a, b in zip(y1, y2) if a is not None and b is not None]
    if not paired:
        return None
    N = len(paired)
    O = [[0] * n_categories for _ in range(n_categories)]
    for a, b in paired:
        if 0 <= a < n_categories and 0 <= b < n_categories:
            O[a][b] += 1
    r = [sum(O[i]) for i in range(n_categories)]              # row totals
    c = [sum(O[i][j] for i in range(n_categories))            # col totals
         for j in range(n_categories)]
    n_minus_1 = max(1, n_categories - 1)
    w = [[1.0 - ((i - j) / n_minus_1) ** 2 for j in range(n_categories)]
         for i in range(n_categories)]
    O_sum = sum(O[i][j] * w[i][j]
                for i in range(n_categories) for j in range(n_categories))
    E_sum = sum((r[i] * c[j] / N) * w[i][j]
                for i in range(n_categories) for j in range(n_categories))
    denom = N - E_sum
    if denom <= 0:
        return None
    return round((O_sum - E_sum) / denom, 4)


def compute_kappa_matrix(
    labels: List[str],
    per_judge_scores: Dict[str, List[Optional[int]]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Pairwise κ matrix across judges (symmetric; diagonal = 1.0)."""
    matrix: Dict[str, Dict[str, Optional[float]]] = {l: {} for l in labels}
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i == j:
                matrix[l1][l2] = 1.0
            elif i < j:
                k = cohen_kappa_quadratic(per_judge_scores[l1], per_judge_scores[l2])
                matrix[l1][l2] = k
                matrix[l2][l1] = k  # symmetric
    return matrix


def evaluate_query_multi(
    query: Dict[str, Any],
    qclient: QdrantClient,
    collection: str,
    config: Dict[str, Any],
    embedder,
    judges: List[Tuple[str, Any]],
    top_k: int,
    parallel_judges: bool = True,
) -> Dict[str, QueryResult]:
    """Retrieve once, score with every judge. Returns label -> QueryResult.

    When `parallel_judges=True` (default), all N judges score each (query, doc)
    pair concurrently via ThreadPoolExecutor. Per-pair wall time becomes
    max(judge_latencies) instead of sum(judge_latencies). Each judge's SDK
    client is thread-safe so this is robust.

    The retrieved docs are shared across judges so any per-judge variance is
    purely attributable to judge scoring behavior (not retrieval noise)."""
    from concurrent.futures import ThreadPoolExecutor

    qid = query["id"]
    text = query["query"]

    log(f"  [{qid}] {text[:80]}")
    qv = embedder.embed_query(text)
    if not qv:
        log(f"    embedding failed — skipping")
        return {label: QueryResult(query_id=qid, query=text,
                                   tags=query.get("tags", []),
                                   metrics={"error": 1.0})
                for label, _ in judges}

    hits = retrieve(qclient, collection, qv, top_k, config.get("vector_name"))
    if not hits:
        log(f"    no hits returned")
        return {label: QueryResult(query_id=qid, query=text,
                                   tags=query.get("tags", []),
                                   metrics={"error": 1.0})
                for label, _ in judges}

    text_fields = config["payload_text_fields"]
    title_fields = config["payload_title_fields"]

    shared_hits = [
        (rank, hit, extract_text_and_title(hit["payload"], text_fields, title_fields))
        for rank, hit in enumerate(hits, start=1)
    ]

    # Per-judge accumulators (indexed by label)
    per_judge_scores: Dict[str, List[Optional[int]]] = {label: [] for label, _ in judges}
    per_judge_retrieved: Dict[str, List[Dict[str, Any]]] = {label: [] for label, _ in judges}

    if parallel_judges and len(judges) > 1:
        # Fire all judges in parallel for each doc. One pool per query (cheap),
        # max_workers == number of judges so every judge gets its own thread.
        with ThreadPoolExecutor(max_workers=len(judges)) as ex:
            for rank, hit, (chunk_text, title) in shared_hits:
                futures = {
                    label: ex.submit(judge.score, text, chunk_text)
                    for label, judge in judges
                }
                for label, fut in futures.items():
                    try:
                        score = fut.result()
                    except Exception as e:
                        log(f"    [{label}] worker exception: {type(e).__name__}: {str(e)[:120]}")
                        score = None
                    per_judge_scores[label].append(score)
                    per_judge_retrieved[label].append({
                        "rank": rank,
                        "qdrant_score": round(hit["score"], 4),
                        "point_id": hit["id"],
                        "title": title[:160],
                        "text_preview": (chunk_text or "")[:240],
                        "judge_score": score,
                    })
    else:
        # Serial fallback (also used when only 1 judge).
        for label, judge in judges:
            for rank, hit, (chunk_text, title) in shared_hits:
                score = judge.score(text, chunk_text)
                per_judge_scores[label].append(score)
                per_judge_retrieved[label].append({
                    "rank": rank,
                    "qdrant_score": round(hit["score"], 4),
                    "point_id": hit["id"],
                    "title": title[:160],
                    "text_preview": (chunk_text or "")[:240],
                    "judge_score": score,
                })

    # Build per-judge QueryResult
    per_judge_qr: Dict[str, QueryResult] = {}
    for label, _ in judges:
        scores = per_judge_scores[label]
        clean = [s if s is not None else 0 for s in scores]
        qr = QueryResult(query_id=qid, query=text, tags=query.get("tags", []))
        qr.retrieved = per_judge_retrieved[label]
        qr.scores = scores
        qr.metrics = {
            "ndcg@5": round(ndcg_at_k(clean, 5), 4),
            "ndcg@10": round(ndcg_at_k(clean, 10), 4),
            "precision@1": round(precision_at_k(clean, 1), 4),
            "precision@5": round(precision_at_k(clean, 5), 4),
            "precision@10": round(precision_at_k(clean, 10), 4),
            "mrr": round(reciprocal_rank(clean), 4),
            "mean_judge_score": round(average([float(s) for s in clean]), 4),
            "unscored": sum(1 for s in scores if s is None),
        }
        log(f"    [{label[:38]:<38}] nDCG@10={qr.metrics['ndcg@10']:.3f}  "
            f"P@5={qr.metrics['precision@5']:.3f}  mean={qr.metrics['mean_judge_score']:.2f}")
        per_judge_qr[label] = qr
    return per_judge_qr


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        required=True,
        choices=sorted(COLLECTION_CONFIGS.keys()),
        help="Qdrant collection to evaluate",
    )
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES),
        help="Path to a JSON file with test queries",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to retrieve per query",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Qdrant URL (default: http://localhost:6334)",
    )
    parser.add_argument(
        "--judge-backend",
        default="auto",
        choices=["auto", "sdk", "cli"],
        help="Single-judge Claude transport (legacy/backward-compat)",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_CLAUDE_MODEL,
        help="Claude model name for single-judge mode (used by the SDK backend)",
    )
    parser.add_argument(
        "--judges",
        default=None,
        help="Comma-separated judge-spec IDs for MULTI-JUDGE mode. "
             "Example: --judges claude-sonnet,gemini-vertex-pro,openai-gpt-4o. "
             f"Available specs: {','.join(sorted(JUDGE_BUILDERS.keys()))}. "
             "If set (or --judge-preset), overrides single-judge mode.",
    )
    parser.add_argument(
        "--judge-preset",
        default=None,
        choices=sorted(JUDGE_PRESETS.keys()),
        help="Shortcut for common multi-judge combinations",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=500,
        help="Truncate each chunk to this many characters before sending to the judge",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override the output path (default auto-generated under results/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate the first N queries (0 = all)",
    )
    args = parser.parse_args()

    # Load .env for API keys — safe to call whether or not file exists.
    loaded = _load_dotenv_manual(DOTENV_PATH)
    if loaded:
        log(f"Loaded {loaded} env vars from {DOTENV_PATH}")

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    queries_path = Path(args.queries)
    if not queries_path.is_absolute():
        queries_path = BACKEND / queries_path
    ensure_sample_queries(queries_path)
    queries = load_queries(queries_path)
    if args.limit and args.limit > 0:
        queries = queries[: args.limit]
    if not queries:
        log(f"No queries loaded from {queries_path}")
        return 1

    config = COLLECTION_CONFIGS[args.collection]
    log("=" * 70)
    log(f"LLM-as-judge RAG evaluation")
    log(f"  Collection  : {args.collection}")
    log(f"  Embedder    : {config['embedder']} ({config['model']})")
    log(f"  Qdrant URL  : {args.qdrant_url}")
    log(f"  Queries     : {queries_path} ({len(queries)} queries)")
    log(f"  Top-K       : {args.top_k}")
    log(f"  Judge model : {args.judge_model}")
    log("=" * 70)

    # Probe Qdrant and make sure the collection exists
    try:
        qclient = QdrantClient(url=args.qdrant_url, timeout=60)
        info = qclient.get_collection(args.collection)
        log(f"Qdrant OK: {args.collection} has {getattr(info, 'points_count', '?')} points")
    except Exception as e:
        log(f"ERROR: cannot reach Qdrant or collection {args.collection}: {e}")
        return 2

    embedder = build_embedder(config)
    if config["embedder"] == "ollama" and not embedder.available():
        log("ERROR: Ollama server is not reachable — start `ollama serve` and retry")
        return 3

    # ------------------------------------------------------------------
    # MULTI-JUDGE MODE (preferred) — triggered by --judges or --judge-preset
    # ------------------------------------------------------------------
    multi_specs: Optional[List[str]] = None
    if args.judges:
        multi_specs = [s.strip() for s in args.judges.split(",") if s.strip()]
    elif args.judge_preset:
        multi_specs = [args.judge_preset]  # single-element list → preset expansion

    if multi_specs:
        try:
            judges = build_judges(multi_specs)
        except (RuntimeError, ValueError) as e:
            log(f"ERROR: {e}")
            return 4
        log(f"MULTI-JUDGE mode: {len(judges)} judges")
        for label, j in judges:
            log(f"  - {label}  [{getattr(j, 'backend', '?')}]")

        t0 = time.time()
        # label -> list of QueryResult (same order as queries)
        per_judge_results: Dict[str, List[QueryResult]] = {label: [] for label, _ in judges}
        # per (query_idx, rank) flattened list of scores per judge, for κ computation
        flat_scores: Dict[str, List[Optional[int]]] = {label: [] for label, _ in judges}

        for q in queries:
            per_judge_qr = evaluate_query_multi(
                query=q, qclient=qclient, collection=args.collection,
                config=config, embedder=embedder, judges=judges, top_k=args.top_k,
            )
            for label, qr in per_judge_qr.items():
                per_judge_results[label].append(qr)
                flat_scores[label].extend(qr.scores or [])

        elapsed = time.time() - t0
        labels = [l for l, _ in judges]
        aggregates = {label: aggregate_metrics(per_judge_results[label]) for label in labels}
        kappa_matrix = compute_kappa_matrix(labels, flat_scores)

        # ---- Console summary ----
        log("")
        log("=" * 80)
        log(f"Per-judge aggregate metrics over {len(queries)} queries (total elapsed {elapsed/60:.1f} min):")
        header = f"  {'metric':<18} " + "  ".join(f"{l[:20]:>20}" for l in labels)
        log(header)
        for mkey in ("ndcg@10", "precision@5", "mrr", "mean_judge_score", "queries_with_errors"):
            row = f"  {mkey:<18} " + "  ".join(
                f"{aggregates[l].get(mkey, 0.0):>20.4f}" for l in labels
            )
            log(row)

        log("")
        log("Pairwise quadratic-weighted Cohen's κ (all retrieved docs):")
        header = f"  {'vs':<26} " + "  ".join(f"{l[:20]:>20}" for l in labels)
        log(header)
        for l1 in labels:
            cells = []
            for l2 in labels:
                v = kappa_matrix[l1][l2]
                cells.append(f"{'n/a':>20}" if v is None else f"{v:>20.4f}")
            log(f"  {l1[:24]:<26} " + "  ".join(cells))
        log("=" * 80)

        # ---- Per-judge JSON + combined multi-judge JSON ----
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:40]
        for label in labels:
            out_path = EVAL_DIR / f"results_{args.collection}_{safe(label)}_{ts}.json"
            report = {
                "config": {
                    "collection": args.collection,
                    "collection_config": {
                        "embedder": config["embedder"],
                        "model": config["model"],
                        "dim": config.get("dim"),
                    },
                    "qdrant_url": args.qdrant_url,
                    "top_k": args.top_k,
                    "judge_label": label,
                    "max_chunk_chars": args.max_chunk_chars,
                    "queries_file": str(queries_path),
                    "timestamp": ts,
                    "elapsed_seconds": int(elapsed),
                    "mode": "multi-judge",
                },
                "aggregate": aggregates[label],
                "queries": [
                    {
                        "query_id": r.query_id,
                        "query": r.query,
                        "tags": r.tags,
                        "metrics": r.metrics,
                        "retrieved": r.retrieved,
                    }
                    for r in per_judge_results[label]
                ],
            }
            out_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            log(f"Saved: {out_path.name}")

        combined_path = EVAL_DIR / f"results_{args.collection}_multijudge_{ts}.json"
        combined = {
            "config": {
                "collection": args.collection,
                "qdrant_url": args.qdrant_url,
                "top_k": args.top_k,
                "judge_specs": multi_specs,
                "judge_labels": labels,
                "max_chunk_chars": args.max_chunk_chars,
                "queries_file": str(queries_path),
                "timestamp": ts,
                "elapsed_seconds": int(elapsed),
                "n_queries": len(queries),
                "n_retrieved_pairs": len(flat_scores[labels[0]]) if labels else 0,
            },
            "aggregates_per_judge": aggregates,
            "kappa_matrix": kappa_matrix,
        }
        combined_path.write_text(
            json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(f"Saved combined: {combined_path.name}")
        return 0

    # ------------------------------------------------------------------
    # SINGLE-JUDGE MODE (legacy) — original behavior preserved
    # ------------------------------------------------------------------
    try:
        judge = LLMJudge(
            backend=args.judge_backend,
            model=args.judge_model,
            max_chars=args.max_chunk_chars,
        )
    except RuntimeError as e:
        log(f"ERROR: {e}")
        return 4
    log(f"LLMJudge backend: {judge.backend}")

    results: List[QueryResult] = []
    t0 = time.time()
    for q in queries:
        qr = evaluate_query(
            query=q,
            qclient=qclient,
            collection=args.collection,
            config=config,
            embedder=embedder,
            judge=judge,
            top_k=args.top_k,
        )
        results.append(qr)

    elapsed = time.time() - t0
    aggregate = aggregate_metrics(results)

    log("")
    log("=" * 70)
    log(f"Aggregate metrics over {len(results)} queries (elapsed {elapsed/60:.1f} min):")
    for k in ["ndcg@5", "ndcg@10", "precision@1", "precision@5", "precision@10",
              "mrr", "mean_judge_score"]:
        log(f"  {k:<16s} {aggregate[k]:.4f}")
    if aggregate["queries_with_errors"]:
        log(f"  (errors on {int(aggregate['queries_with_errors'])} queries)")
    log("=" * 70)

    # ----------------------- save report -----------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = EVAL_DIR / f"results_{args.collection}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {
            "collection": args.collection,
            "collection_config": {
                "embedder": config["embedder"],
                "model": config["model"],
                "dim": config.get("dim"),
            },
            "qdrant_url": args.qdrant_url,
            "top_k": args.top_k,
            "judge_backend": judge.backend,
            "judge_model": args.judge_model,
            "max_chunk_chars": args.max_chunk_chars,
            "queries_file": str(queries_path),
            "timestamp": ts,
            "elapsed_seconds": int(elapsed),
        },
        "aggregate": aggregate,
        "queries": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "tags": r.tags,
                "metrics": r.metrics,
                "retrieved": r.retrieved,
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
