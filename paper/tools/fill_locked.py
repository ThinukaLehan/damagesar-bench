"""Turn the locked run's output files (k01) into checked numbers and LaTeX for the paper.

    python paper/tools/fill_locked.py <k01 run folder> [--out paper/locked_results]

The run folder is the one k01 writes to Drive (04_outputs/runs/<date>_k01-locked-confirmatory-test/), copied
locally. The script reads only files the locked analysis wrote; it re-derives every verdict from the frozen decision
table and stops if anything disagrees with the frozen plan (level, events, which comparison is confirmatory).

Writes:
* ``locked_results.md``: every number with its source file, for checking and for the experiment log;
* ``table_locked.tex``: the rows of Table XIV (per-event AUROC);
* ``values.tex``: one \\newcommand per number used in the text (estimate, interval, verdict, counts, p-values).

Interpretation is not automated: the sentences that interpret the result are written by the authors from these
numbers, against the development estimate and the minimum detectable gain stated in the plan.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from damagesar_bench.stats.event_level import decide, t_interval

CONFIRMATORY = ["haiti", "turkey", "morocco", "noto", "bata"]
NAMES = {"haiti": "Haiti", "turkey": "T\\\"urkiye", "morocco": "Morocco", "noto": "Noto", "bata": "Bata"}
RQ1 = "RQ1 DINOv3 vs ImageNet (outline)"
ROWS = [("dinov3_vitl16_sat@native", "outline", "DINOv3 ViT-L (satellite)"),
        ("imagenet_vitb16@native", "outline", "ImageNet ViT-B"),
        ("dinov3_vitb16@native", "outline", "DINOv3 ViT-B"),
        ("dinov3_vitl16_sat@native", "ring10m", "DINOv3 ViT-L (satellite), 10 m ring"),
        ("imagefree", "geometry", "Image-free control$^{\\dagger}$")]
LEVEL = 0.975


class FillError(RuntimeError):
    pass


def _need(p: Path) -> Path:
    if not p.exists():
        raise FillError(f"missing {p}")
    return p


def _f(x: float, signed: bool = False) -> str:
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("-", "$-$") if signed else s


def _md(df: pd.DataFrame) -> str:
    """Plain markdown table (no optional dependencies)."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(out)


def load(run: Path) -> dict:
    bench = run / "bench_run"
    return {
        "comps": pd.read_csv(_need(bench / "comparisons.csv")),
        "metrics": pd.read_csv(_need(bench / "event_metrics.csv")),
        "manifest": json.loads(_need(bench / "manifest.json").read_text()) if (bench / "manifest.json").exists() else {},
        "per_event": pd.read_csv(_need(run / "rq1_per_event.csv")),
        "supporting": json.loads(_need(run / "rq1_supporting.json").read_text()),
        "c3": pd.read_csv(_need(run / "c3_confirmatory.csv")),
    }


def check_and_derive(d: dict) -> dict:
    c = d["comps"].set_index("label")
    if RQ1 not in c.index:
        raise FillError(f"no row {RQ1!r} in comparisons.csv")
    if list(d["comps"].loc[d["comps"]["strength"] == "confirmatory", "label"]) != [RQ1]:
        raise FillError("exactly one confirmatory comparison (RQ1) expected; Gate 1 did not pass, so no RQ2")
    if not (d["comps"]["level"].round(6) == LEVEL).all():
        raise FillError(f"every comparison must be at the frozen level {LEVEL}")
    r = c.loc[RQ1]
    events = str(r["events"]).split("+")
    if sorted(events) != sorted(CONFIRMATORY):
        raise FillError(f"RQ1 must use the five confirmatory events, got {events}")
    gains = json.loads(r["gains"])
    iv = t_interval(list(gains.values()), level=LEVEL)
    for name, mine, theirs in [("estimate", iv.estimate, r["estimate"]), ("lower", iv.lower, r["lower"]),
                               ("upper", iv.upper, r["upper"])]:
        if not math.isclose(mine, theirs, abs_tol=1e-5):   # gains are stored rounded to 6 decimals
            raise FillError(f"RQ1 {name}: recomputed {mine} != recorded {theirs}")
    v = decide(float(r["lower"]), float(r["estimate"]), float(r["upper"]))
    if v.meaning != r["verdict"]:
        raise FillError(f"RQ1 verdict: decision table gives {v.meaning!r}, recorded {r['verdict']!r}")
    sd = pd.Series(list(gains.values())).std(ddof=1)
    sup = d["supporting"]
    if sup["n_events"] != 5:
        raise FillError("supporting evidence must cover 5 events")
    iv = type(iv)(float(r["estimate"]), float(r["lower"]), float(r["upper"]), iv.level, iv.n, iv.df, iv.se, iv.method)
    return {"rq1": r, "gains": gains, "interval": iv, "verdict": v, "sd": float(sd), "supporting": sup,
            "exploratory": d["comps"][d["comps"]["strength"] != "confirmatory"]}


def write(d: dict, x: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    m = d["metrics"]
    auc = {(e, p, ev): a for e, p, ev, a in zip(m["encoder"], m["pooling"], m["event"], m["auroc"])}

    rows = []
    for enc, pool, label in ROWS:
        vals = [auc.get((enc, pool, ev)) for ev in CONFIRMATORY]
        if any(v is None for v in vals):
            continue
        mean = sum(vals) / len(vals)
        rows.append(f"{label} & " + " & ".join(_f(v) for v in vals) + f" & {_f(mean)} \\\\")
    (out / "table_locked.tex").write_text(
        "% generated by paper/tools/fill_locked.py\n"
        "Model & " + " & ".join(NAMES[e] for e in CONFIRMATORY) + " & Mean \\\\\n\\midrule\n" + "\n".join(rows) + "\n",
        encoding="utf-8")

    iv, sup, r = x["interval"], x["supporting"], x["rq1"]
    vals = {
        "LockedDate": d["manifest"].get("date", "[date]"),
        "LockedCommit": str(d["manifest"].get("git_commit", "[commit]"))[:7],
        "RQoneEst": _f(iv.estimate, True), "RQoneLow": _f(iv.lower, True), "RQoneHigh": _f(iv.upper, True),
        "RQoneSD": f"{x['sd']:.3f}", "RQoneVerdict": x["verdict"].meaning,
        "RQoneSignFlipTwo": f"{r['sign_flip_p']:.3f}",
        "RQoneSatBetter": str(sup["n_a_better"]), "RQoneImnBetter": str(sup["n_b_better"]),
        "RQoneSignFlipOne": f"{sup['sign_flip_one_sided_p']:.4f}",
    }
    (out / "values.tex").write_text(
        "% generated by paper/tools/fill_locked.py\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals.items()), encoding="utf-8")

    lines = ["# Locked confirmatory test: numbers for the paper", "",
             f"Run date {vals['LockedDate']}, commit {vals['LockedCommit']}; level {LEVEL}; events "
             + ", ".join(CONFIRMATORY) + ".", "",
             "## RQ1 (confirmatory): satellite DINOv3 ViT-L minus ImageNet ViT-B, outline, native", "",
             f"- Per-event gains: " + ", ".join(f"{k} {v:+.4f}" for k, v in x["gains"].items()),
             f"- Estimate {iv.estimate:+.4f}, 97.5% interval [{iv.lower:+.4f}, {iv.upper:+.4f}], SD {x['sd']:.4f}",
             f"- Verdict (decision table, re-derived): **{x['verdict'].meaning}** (row {x['verdict'].row})",
             f"- Two-sided sign-flip p {r['sign_flip_p']:.4f}",
             f"- Supporting: satellite ahead (bootstrap interval > 0) in {sup['n_a_better']} of 5; ImageNet in "
             f"{sup['n_b_better']} of 5; one-sided sign-flip p {sup['sign_flip_one_sided_p']:.4f}", "",
             "## Per-event paired differences (supporting)", "", _md(d["per_event"].round(4)), "",
             "## Exploratory comparisons", "",
             _md(x["exploratory"][["label", "estimate", "lower", "upper", "verdict", "sign_flip_p"]].round(4)), "",
             "## RQ3 (exploratory): label-free prevalence, RQ1 model", "",
             _md(d["c3"].round(3)), "",
             "Image-free control: not blind on these events (31 Oct audit computed a building-count AUROC).", ""]
    (out / "locked_results.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=Path("paper/locked_results"))
    a = ap.parse_args(argv)
    try:
        d = load(a.run)
        x = check_and_derive(d)
    except FillError as ex:
        print("STOP:", ex, file=sys.stderr)
        return 1
    write(d, x, a.out)
    print(f"RQ1 {x['interval'].estimate:+.4f} [{x['interval'].lower:+.4f}, {x['interval'].upper:+.4f}] -> "
          f"{x['verdict'].meaning}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
