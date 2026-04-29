# arXiv preprint bundle — P4-SHORT

This folder contains the LaTeX source for submitting **"Cross-Family LLM-Judge Agreement for Institutional RAG: A 5-Family, 9-Judge Ablation"** to arXiv as a preprint, concurrent with the LLM4Eval @ SIGIR 2027 workshop submission.

## Bundle contents

```
arxiv/
├── main.tex          ← article-class LaTeX source
├── references.bib    ← 24 BibTeX entries (natbib / plainnat)
├── figures/
│   └── kappa_matrix_9judge.png   ← Figure 1 (the only figure included)
└── README.md         ← this file
```

## Local build (optional — arXiv compiles server-side)

If you have `pdflatex` + `bibtex` installed (TeX Live, MiKTeX, etc.):

```bash
cd arxiv/
pdflatex main
bibtex main
pdflatex main
pdflatex main      # second pass to resolve cross-refs
```

This produces `main.pdf`. If you don't have LaTeX locally, **skip this step** — arXiv will build it for you when you upload the source bundle.

## arXiv submission checklist

1. **Create / log into arXiv account** — https://arxiv.org/user/
2. **Start new submission** — https://arxiv.org/submit
3. **Category**:
   - Primary: `cs.IR` (Information Retrieval)
   - Secondary: `cs.CL` (Computation and Language)
4. **License**: CC BY 4.0 (matches repo's data/papers license)
5. **Upload**: zip the contents of this folder (NOT the folder itself):
   ```bash
   cd arxiv/
   zip -r ../arxiv-submission.zip main.tex references.bib figures/
   ```
   Upload `arxiv-submission.zip` via the arXiv "Files" tab.
6. **Title**: `Cross-Family LLM-Judge Agreement for Institutional RAG: A 5-Family, 9-Judge Ablation`
7. **Authors**: `Adisak Sukul (Iowa State University)`
8. **Abstract**: Copy from the `\begin{abstract}...\end{abstract}` block in `main.tex` (also reproduced below for convenience).
9. **Comments field**: `Companion repository: https://github.com/asukul/RAG-Eval-LLM-Judge ; companion long version under preparation for ECIR 2028 / TOIS.`
10. **Workshop status disclosure**: Add a footnote on the title page or in the comments field: "Concurrently submitted to LLM4Eval @ SIGIR 2027 workshop."
11. **Preview** the arXiv-built PDF carefully before clicking Submit. arXiv assigns the identifier on submission (`arXiv:2604.XXXXX` for April 2026).
12. **After accepted**: update `CITATION.cff` and the GitHub README with the assigned arXiv ID.

## Abstract (copy/paste for the arXiv form)

> **Open-weight LLM judges win the within-pair race.** In our 5-family, 9-judge LLM-as-Judge ablation on 570 RAG query-document pairs from an institutional repository, the highest pairwise quadratic-weighted Cohen's kappa (0.80) is between Qwen 3.6 Plus and Gemma 4 26B — a cross-organization open-weight pair that exceeds every commercial within-family pair (Anthropic 0.71, OpenAI 0.63, Google-commercial 0.67) and ties the cross-family commercial reasoning ceiling. We present the first five-family pairwise kappa matrix on bounded 0-3 ordinal relevance with four within-family controls and report (i) cross-family reasoning judges converge at kappa = 0.75-0.79, (ii) within-family agreement is task-dependent and bounded by the cross-family ceiling, contradicting canonical self-preference findings under bounded ordinal judging, (iii) two emergent calibration clusters partition judges more cleanly than provider family. Mechanistic decomposition attributes 93% of kappa variance to joint-distribution structure of paired scores; structural factors are fully mediated; the shared-tokenizer hypothesis is refuted (Qwen-Gemma vocabulary Jaccard = 0.066, lowest in slate, yet matrix-highest kappa). External validation against NIST TREC RAG 2024 human qrels on a 537-pair stratified-balanced sample reaches 9-judge ensemble kappa = 0.4941 (Landis-Koch moderate, near-substantial); 5 of 9 individual judges hit kappa >= 0.47. We open-source eval_llm_judge.py and validate_against_trec.py (5,130 within-corpus + 4,833 external score records, USD 56.30, ~13 hours wall) as reproducibility artifacts.

## Pros / cons of preprinting now

**Pros**
- Establishes priority on the C3 finding (open-weight matrix-max kappa) before competing groups publish similar results
- Citation count starts accruing immediately; arXiv versions are commonly cited months before workshop proceedings
- Forces a final pass on writing quality (good)
- Reviewer-2 risk mitigation: visible artifact + community discussion before formal review

**Cons**
- Some workshops disallow concurrent preprints; LLM4Eval @ SIGIR 2027 has not stated either way as of 2026-04-28 — verify on workshop CFP before submitting
- Preprint version is locked once posted; later corrections need a v2 (which is fine but does add a bit of churn)
- If anything in the §7 numbers shifts after additional validation runs, the preprint becomes out-of-date

**Recommendation**: Submit only after (a) the user verifies the workshop CFP allows concurrent arXiv, and (b) the user is satisfied that the §7 TREC RAG 2024 numbers are final (no more validation reruns planned). Otherwise wait until workshop notification.

## Categories — why these

- **`cs.IR` (Information Retrieval) — primary**. The paper is fundamentally about how to evaluate IR pipelines (RAG retrieval). The kappa-matrix methodology, the disclosure standard, and the external validation against TREC RAG 2024 all sit squarely in IR.
- **`cs.CL` (Computation and Language) — secondary**. LLM-as-Judge methodology, model comparison across families, and tokenizer-overlap analysis touch NLP. Cross-listing brings the paper to a community that cares about the open-weight vs. commercial findings.
- Considered but not chosen: `stat.AP` (the mediation analysis is statistical, but R^2-decomposition is supplementary; primary contribution is IR) and `cs.LG` (model behavior is studied, but training is not).

## After arXiv submission

1. arXiv assigns identifier (e.g., `arXiv:2604.12345`) within ~24 h.
2. Update repo:
   ```bash
   # In repo root, update CITATION.cff:
   #   identifiers:
   #     - type: doi
   #       value: 10.48550/arXiv.2604.12345
   ```
3. Add badge to `README.md`:
   ```markdown
   [![arXiv](https://img.shields.io/badge/arXiv-2604.12345-b31b1b.svg)](https://arxiv.org/abs/2604.12345)
   ```
4. Update GitHub Pages dashboard `docs/index.html` to link the arXiv version under "Cite".
