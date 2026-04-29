"""Bias diagnostics that can be computed without new API calls.

Produces three diagnostics from existing within-corpus per-judge data:

1. **5x5 family-pair score-mean delta matrix** — for each ordered (family_A,
   family_B) pair, the mean rating that judges in family A give to documents
   that judges in family B rate highly. Symmetric difference reveals
   per-family score-allocation asymmetry. NOT a self-preference test in the
   Panickssery 2024 sense (we do not have a "from family X" provenance label
   on every retrieved document) but it's the closest free analogue, and it
   directly tests whether each family's judges have a systematic offset
   relative to other families.

2. **Length-stratified kappa** — within-corpus 36 pairwise kappa values
   recomputed on quartiles of document length (text_preview char count).
   Reveals whether agreement is sensitive to document length (verbosity
   bias proxy from Saito 2023).

3. **Calibration-tier confusion matrix** — a 9x9 matrix showing how often
   each judge assigns each ordinal score, normalized per judge. Visualizes
   the "reasoning-generous vs strict-mid" cluster structure directly.

Run from repo root:
    py -3 src/bias_diagnostics.py

Output: results/bias_diagnostics.json + figures/bias_diagnostics_panel.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from eval_llm_judge import cohen_kappa_quadratic  # noqa: E402

# Family assignments
JUDGE_FAMILY = {
    "Claude Opus 4.7":          "Anthropic",
    "Claude Sonnet 4.6":        "Anthropic",
    "GPT-5.5":                  "OpenAI",
    "GPT-4o":                   "OpenAI",
    "Gemini 3.1":               "Google",
    "Gemini 2.5":               "Google",
    "DeepSeek":                 "DeepSeek",
    "Qwen":                     "Open-weight",
    "Gemma":                    "Open-weight",
}

FAMILIES = ["Anthropic", "OpenAI", "Google", "DeepSeek", "Open-weight"]


def load_within_corpus():
    """Load all 9 per-judge files into {label: flat scores aligned by pair}."""
    within = REPO / "results" / "within_corpus"
    if not within.exists() or not any(within.glob("judge_*.json")):
        return None, None
    per_judge = {}
    pair_lengths = None
    for fp in sorted(within.glob("judge_*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        label = d["config"]["judge_label"]
        per_judge[label] = []
        local_lengths = []
        for q in d["queries"]:
            for r in sorted(q["retrieved"], key=lambda x: x["rank"]):
                per_judge[label].append(r["judge_score"])
                local_lengths.append(len(r.get("text_preview") or ""))
        if pair_lengths is None:
            pair_lengths = local_lengths
    return per_judge, pair_lengths


def assign_family(label: str) -> str:
    for prefix, fam in JUDGE_FAMILY.items():
        if prefix in label:
            return fam
    return "Other"


def family_score_mean_matrix(per_judge):
    """5x5 matrix: M[A][B] = mean score given by family-A judges on the
    pairs that family-B judges rate >= 2.

    Diagonal = self-conditional mean. Off-diagonal asymmetry reveals
    per-family score-allocation differences."""
    family_judges = {f: [l for l in per_judge if assign_family(l) == f]
                     for f in FAMILIES}
    M = np.full((len(FAMILIES), len(FAMILIES)), np.nan)
    for i, fa in enumerate(FAMILIES):
        ja = family_judges[fa]
        if not ja:
            continue
        for j, fb in enumerate(FAMILIES):
            jb = family_judges[fb]
            if not jb:
                continue
            # Pairs where any family-B judge rated >= 2
            pos_pairs = set()
            for label in jb:
                for k, s in enumerate(per_judge[label]):
                    if s is not None and s >= 2:
                        pos_pairs.add(k)
            if not pos_pairs:
                continue
            # Mean score that family-A judges give on those pairs
            scores = []
            for label in ja:
                for k in pos_pairs:
                    s = per_judge[label][k]
                    if s is not None:
                        scores.append(s)
            if scores:
                M[i, j] = sum(scores) / len(scores)
    return M


def length_stratified_kappa(per_judge, pair_lengths):
    """Pairwise kappa within length quartiles.

    Returns None if pair lengths are uniform (no spread to stratify on).
    The shipped within-corpus per-judge JSONs store text_preview uniformly
    truncated at 240 characters, so there is no length-spread to analyze
    here -- the actual 1500-char judge input is not preserved per-pair.
    """
    if pair_lengths is None:
        return None
    lengths = np.array(pair_lengths)
    if lengths.max() == lengths.min():
        return {
            "skipped": True,
            "reason": (f"all pair text_previews are exactly {int(lengths[0])} chars; "
                       "no length spread to stratify on. The 1500-char judge-input "
                       "cap was applied at API-call time but not preserved in the "
                       "stored per-judge JSONs. Length-stratified kappa would "
                       "require re-extracting raw chunk text from Qdrant or the "
                       "embeddings parquet."),
            "uniform_length_chars": int(lengths[0]),
        }
    qs = np.quantile(lengths, [0.25, 0.5, 0.75])
    quartiles = np.digitize(lengths, qs)

    labels = list(per_judge)
    pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]]
    out = {}
    for q in range(4):
        idx = [k for k, qq in enumerate(quartiles) if qq == q]
        kappas = []
        for a, b in pairs:
            sa = [per_judge[a][k] for k in idx]
            sb = [per_judge[b][k] for k in idx]
            k = cohen_kappa_quadratic(sa, sb)
            if k is not None:
                kappas.append(k)
        out[f"Q{q+1}"] = {
            "n_pairs": len(idx),
            "length_range": [int(lengths[idx].min()), int(lengths[idx].max())] if idx else None,
            "mean_pairwise_kappa": round(sum(kappas) / len(kappas), 4) if kappas else None,
            "n_pair_kappas_computed": len(kappas),
        }
    return out


def calibration_per_judge_distribution(per_judge):
    """Per-judge marginal score distribution (the table that visualizes
    reasoning-generous vs strict-mid cluster structure)."""
    out = {}
    for label, scores in per_judge.items():
        valid = [s for s in scores if s is not None]
        if not valid:
            continue
        bins = [0, 0, 0, 0]
        for s in valid:
            if 0 <= s <= 3:
                bins[s] += 1
        n = sum(bins)
        out[label] = {
            "P(0)": round(bins[0] / n, 4),
            "P(1)": round(bins[1] / n, 4),
            "P(2)": round(bins[2] / n, 4),
            "P(3)": round(bins[3] / n, 4),
            "mean_score": round(sum(valid) / len(valid), 4),
            "n_valid": n,
        }
    return out


def render_panel_figure(M, length_strat, calib, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), dpi=150)

    # Panel 1: family-score-mean matrix
    ax = axes[0]
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=2.5, aspect="equal")
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_yticks(range(len(FAMILIES)))
    ax.set_xticklabels(FAMILIES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(FAMILIES, fontsize=9)
    for i in range(len(FAMILIES)):
        for j in range(len(FAMILIES)):
            v = M[i, j]
            if not np.isnan(v):
                color = "white" if v < 1.0 or v > 2.0 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=color)
    ax.set_title("Mean score that family A gives\n"
                 "on pairs where family B rated >= 2",
                 fontsize=10)
    ax.set_xlabel("Conditioning family (B)", fontsize=9)
    ax.set_ylabel("Scoring family (A)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: length-stratified kappa (or "skipped" placeholder)
    ax = axes[1]
    if length_strat and length_strat.get("skipped"):
        ax.text(0.5, 0.55,
                "Length-stratified kappa\nNOT AVAILABLE",
                ha="center", va="center", fontsize=14,
                color="#57606a", transform=ax.transAxes,
                fontweight="bold")
        ax.text(0.5, 0.30,
                f"text_preview uniformly truncated\nat "
                f"{length_strat.get('uniform_length_chars', '?')} chars in "
                f"shipped JSONs;\nno length spread to stratify on.\n"
                f"Verbosity-bias diagnostic deferred\nto re-extraction "
                f"of raw chunk text.",
                ha="center", va="center", fontsize=8.5,
                color="#57606a", transform=ax.transAxes, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#d0d7de")
        ax.set_title("Length-stratified mean pairwise kappa\n"
                     "(verbosity-bias diagnostic)", fontsize=10)
    elif length_strat:
        qs = list(length_strat.keys())
        means = [length_strat[q]["mean_pairwise_kappa"] for q in qs]
        ranges = [length_strat[q]["length_range"] for q in qs]
        # Filter to only quartiles with actual data
        valid = [(q, m, r) for q, m, r in zip(qs, means, ranges)
                 if m is not None]
        if valid:
            qs_v, means_v, ranges_v = zip(*valid)
            ax.bar(qs_v, means_v, color="#0e4a86", alpha=0.85)
            for i, m in enumerate(means_v):
                ax.text(i, m + 0.005, f"{m:.3f}", ha="center", fontsize=9)
            ax.set_ylim(0.55, 0.80)
            ax.set_ylabel("Mean pairwise kappa\n(across 36 judge pairs)",
                          fontsize=9)
            labels = [f"{q}\n[{r[0]}-{r[1]}\nchars]" for q, _, r in valid]
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title("Length-stratified mean pairwise kappa\n"
                         "(verbosity-bias diagnostic)", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

    # Panel 3: per-judge marginal score distribution stack
    ax = axes[2]
    judges = list(calib.keys())
    n_j = len(judges)
    P0 = [calib[j]["P(0)"] for j in judges]
    P1 = [calib[j]["P(1)"] for j in judges]
    P2 = [calib[j]["P(2)"] for j in judges]
    P3 = [calib[j]["P(3)"] for j in judges]
    x = np.arange(n_j)
    ax.bar(x, P0, label="0 (irrelevant)", color="#c8102e")
    ax.bar(x, P1, bottom=P0, label="1 (topical)", color="#f4a261")
    ax.bar(x, P2, bottom=[a + b for a, b in zip(P0, P1)],
           label="2 (partial)", color="#a8dadc")
    ax.bar(x, P3, bottom=[a + b + c for a, b, c in zip(P0, P1, P2)],
           label="3 (fully)", color="#0e4a86")
    short = [j.split(" (")[0][:18] for j in judges]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score-allocation distribution", fontsize=9)
    ax.set_title("Per-judge marginal score distribution\n"
                 "(reasoning-generous vs strict-mid)", fontsize=10)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Bias diagnostics (free recompute from within-corpus per-judge files)")
    print("=" * 72)

    per_judge, pair_lengths = load_within_corpus()
    if per_judge is None:
        print("ERROR: no within-corpus per-judge files found")
        return 1

    print(f"\nLoaded {len(per_judge)} judges, "
          f"{len(next(iter(per_judge.values())))} pairs each")

    print("\n[1/3] 5x5 family score-mean matrix...")
    M = family_score_mean_matrix(per_judge)
    matrix_dict = {fa: {fb: float(M[i, j]) if not np.isnan(M[i, j]) else None
                        for j, fb in enumerate(FAMILIES)}
                   for i, fa in enumerate(FAMILIES)}

    print("\n[2/3] length-stratified kappa...")
    length_strat = length_stratified_kappa(per_judge, pair_lengths)
    if length_strat:
        if length_strat.get("skipped"):
            print(f"  SKIPPED: {length_strat['reason']}")
        else:
            for q, m in length_strat.items():
                lr = m["length_range"]
                lr_str = f"[{lr[0]}, {lr[1]}]" if lr else "[empty]"
                print(f"  {q}: chars in {lr_str}, "
                      f"mean pairwise kappa = {m['mean_pairwise_kappa']}, "
                      f"n_pairs={m['n_pairs']}")

    print("\n[3/3] per-judge marginal score distribution...")
    calib = calibration_per_judge_distribution(per_judge)
    for label, m in calib.items():
        print(f"  {label[:30]:<30s}  P=[{m['P(0)']:.2f}, {m['P(1)']:.2f}, "
              f"{m['P(2)']:.2f}, {m['P(3)']:.2f}]  mean={m['mean_score']:.2f}")

    out_path = REPO / "results" / "bias_diagnostics.json"
    out = {
        "family_score_mean_matrix": {
            "description": ("M[A][B] = mean score family-A judges give on pairs where "
                            "family-B judges rated >= 2. Diagonal = self-conditional "
                            "mean. Off-diagonal differences reveal per-family score "
                            "allocation asymmetry."),
            "families": FAMILIES,
            "matrix": matrix_dict,
        },
        "length_stratified_kappa": length_strat,
        "per_judge_marginal_distribution": calib,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    fig_path = REPO / "figures" / "bias_diagnostics_panel.png"
    render_panel_figure(M, length_strat, calib, fig_path)
    print(f"Saved: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
