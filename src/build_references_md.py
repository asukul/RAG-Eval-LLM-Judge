"""Convert arxiv/references.bib to papers/references.md.

A small standalone bib-to-markdown converter so the two paper drafts
can point at a single human-readable bibliography (`papers/references.md`)
while the LaTeX build uses the BibTeX file (`arxiv/references.bib`)
directly. Re-run after editing the .bib file to keep the .md in sync.

Usage (from repo root):
    py -3 src/build_references_md.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIB = REPO / "arxiv" / "references.bib"
OUT = REPO / "papers" / "references.md"


def parse_bib(text: str):
    """Tiny BibTeX parser for our specific @entry{key, field = {value}, ...} format."""
    entries = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "@":
            i += 1
            continue
        m = re.match(r"@(\w+)\s*\{\s*([^,]+),", text[i:])
        if not m:
            i += 1
            continue
        etype, ekey = m.group(1).strip().lower(), m.group(2).strip()
        i += m.end()

        fields = {"type": etype, "key": ekey}
        depth = 1
        buf = []
        while i < n and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            i += 1
        body = "".join(buf)

        depth = 0
        cur = []
        parts = []
        for ch in body:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
        if cur:
            parts.append("".join(cur))

        for p in parts:
            if "=" not in p:
                continue
            k, _, v = p.partition("=")
            k = k.strip().lower()
            v = v.strip()
            while v.startswith("{") and v.endswith("}"):
                v = v[1:-1].strip()
            while v.startswith('"') and v.endswith('"'):
                v = v[1:-1].strip()
            v = re.sub(r"\{([^{}]*)\}", r"\1", v)
            fields[k] = v

        entries.append(fields)
        i += 1
    return entries


GROUPS = {
    "LLM-as-Judge for IR (relevance assessment)": [
        "faggioli2023perspectives", "rahmani2024judges", "upadhyay2024umbrela",
        "thakur2024judges", "thakur2025trecragsupport",
        "balog2025rankersjudges", "farzi2025umbrela_other", "han2025judgesverdict",
    ],
    "LLM-as-Judge generalist methodology": ["zheng2023judge", "liu2023geval"],
    "Judge bias (position, verbosity, self-preference)": [
        "wang2023fair", "saito2023verbosity", "chen2024humans", "panickssery2024self",
    ],
    "RAG evaluation frameworks": ["es2024ragas"],
    "Statistical foundations (kappa, KL, mediation)": [
        "cohen1968", "fleiss1971", "landis1977", "kullback1951",
        "baronkenny1986", "mackinnon2007",
    ],
    "External-validation corpora": ["macavaney2025trec_rag", "thakur2021beir"],
    "Cross-domain agreement evidence": ["jiao2026essayraters"],
}


def format_entry(e: dict) -> str:
    key = e["key"]
    authors = e.get("author", "n/a")
    author_list = [a.strip() for a in authors.split(" and ")]
    if len(author_list) >= 3:
        authors = author_list[0] + " et al."
    elif len(author_list) == 2:
        authors = " & ".join(author_list)

    year = e.get("year", "n.d.")
    title = e.get("title", "Untitled")
    venue = (e.get("booktitle") or e.get("journal")
             or e.get("howpublished") or e.get("publisher") or "")
    doi = e.get("doi", "")
    url = e.get("url", "")
    eprint = e.get("eprint", "")

    parts = [f"- **`{key}`** &mdash; {authors} ({year}). _{title}_."]
    if venue:
        parts.append(f" {venue}.")
    if eprint and "arxiv" in (e.get("archiveprefix", "") + url).lower():
        parts.append(f" arXiv:{eprint}.")
    if doi:
        parts.append(f" [doi:{doi}](https://doi.org/{doi})")
    elif url:
        parts.append(f" [link]({url})")
    return "".join(parts)


def main() -> int:
    if not BIB.exists():
        print(f"ERROR: {BIB} not found", file=sys.stderr)
        return 1

    text = BIB.read_text(encoding="utf-8")
    entries = parse_bib(text)
    by_key = {e["key"]: e for e in entries}

    out = [
        "# References - RAG-Eval-LLM-Judge",
        "",
        ("Auto-generated from `arxiv/references.bib` by "
         "`src/build_references_md.py`. Re-run after editing the .bib file "
         "to keep this file in sync."),
        "",
        ("**Use:** `papers/short.md` and `papers/long.md` cite by BibTeX key "
         "(e.g. `[faggioli2023perspectives]`); the full entry for any key "
         "can be found below in topical order. The LaTeX preprint "
         "(`arxiv/main.tex`) compiles directly against the .bib file via "
         "`natbib + plainnat`."),
        "",
        f"**Entry count:** {len(entries)} "
        "(verify with `py -3 arxiv/_verify_citations.py`).",
        "",
    ]

    used = set()
    for group_name, keys in GROUPS.items():
        out.append(f"## {group_name}")
        out.append("")
        for k in keys:
            if k in by_key:
                out.append(format_entry(by_key[k]))
                used.add(k)
            else:
                out.append(f"- **`{k}`** &mdash; (NOT FOUND in references.bib)")
        out.append("")

    leftover = [e for e in entries if e["key"] not in used]
    if leftover:
        out.append("## Other")
        out.append("")
        for e in leftover:
            out.append(format_entry(e))
        out.append("")

    out.append("---")
    out.append("")
    out.append(("**Audit trail.** All BibTeX keys cited in `arxiv/main.tex` "
                "are verified resolvable by `arxiv/_verify_citations.py`. "
                "The `papers/short.md` and `papers/long.md` drafts cite the "
                "same keys via in-text `[key]` markers. The slate is curated "
                "for the LLM-as-Judge for IR + bias + RAG-eval literature "
                "plus statistical-foundations citations for the mechanism "
                "section and external-validation citations for the public "
                "benchmarks section."))
    out.append("")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(REPO)} ({len(entries)} entries, "
          f"{sum(len(v) for v in GROUPS.values())} grouped, "
          f"{len(leftover)} ungrouped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
