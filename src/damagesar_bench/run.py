"""``bench run``: building features -> probes -> event metrics -> comparisons (FR7, FR10, NFR1).

One run folder holds everything needed to check or rebuild the report:

    fold_scores.parquet   probe score per held-out building
    probe_fits.csv        one row per fold (training events, sizes, iterations)
    event_metrics.csv     AUROC, AP, DeLong SE, tile-cluster bootstrap per (setting, event)
    comparisons.csv       per comparison: events, per-event gains, event-level interval, verdict, sign-flip
    run_manifest.json     date, phase, tool version, git commit, seed, settings, SHA-256 of every input
    report.md             human-readable summary (``bench report`` rebuilds it from the files above)

Phase: before the 10 Jan 2027 freeze only development events are held out, so every comparison is
**exploratory**. After the freeze, comparisons default to the fixed confirmatory event set, and every
confirmatory event must be present, so none can drop out silently.
"""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from . import __version__
from .reproduce import environment
from .config import REPO_ROOT, EventsConfig
from .data.events import labels_unlocked
from .probe.logistic import run_probes
from .stats.compare import compare, event_metrics


class RunError(ValueError):
    pass


@dataclass(frozen=True)
class ComparisonSpec:
    a: tuple[str, str]   # (encoder, pooling)
    b: tuple[str, str]
    label: str = ""
    primary: bool = True   # False: never reported as confirmatory, even in the locked run (e.g. RQ3/RQ4 checks)

    @classmethod
    def parse(cls, text: str) -> "ComparisonSpec":
        """``label=encA/poolA:encB/poolB`` or ``encA/poolA:encB/poolB``."""
        label, _, body = text.rpartition("=")
        try:
            left, right = body.split(":")
            a = tuple(left.split("/"))
            b = tuple(right.split("/"))
        except ValueError:
            raise RunError(f"bad comparison {text!r}; use encoder/pooling:encoder/pooling") from None
        if len(a) != 2 or len(b) != 2 or not all(a + b):
            raise RunError(f"bad comparison {text!r}; use encoder/pooling:encoder/pooling")
        return cls(a, b, label or f"{a[0]}/{a[1]} vs {b[0]}/{b[1]}")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - git missing
        return "unknown"


def comparison_events(events: EventsConfig, today: date | None, event_set: str) -> list[str] | None:
    """Which held-out events a comparison uses. ``None`` = every event both settings have."""
    unlocked = labels_unlocked(events, today)
    if event_set == "auto":
        event_set = "confirmatory" if unlocked else "development"
    if event_set == "development":
        return events.keys_with_role("development")
    if event_set == "confirmatory":
        if not unlocked:
            raise RunError("confirmatory comparisons are refused before the freeze")
        return events.keys_with_role("confirmatory")
    if event_set == "all":
        return None
    raise RunError(f"unknown event set {event_set!r} (auto, development, confirmatory, all)")


def feature_settings(feature_files: list[str | Path]) -> list[tuple[str, str]]:
    """Every (encoder, pooling) in the files, read without loading the feature columns."""
    found: set[tuple[str, str]] = set()
    for p in feature_files:
        keys = pd.read_parquet(p, columns=["encoder", "pooling"])
        found |= set(zip(keys["encoder"], keys["pooling"]))
    return sorted(found)


def read_setting(feature_files: list[str | Path], encoder: str, pooling: str) -> pd.DataFrame:
    """The rows of one (encoder, pooling) setting from every file (parquet row filters, so only
    that setting is loaded)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    filters = [("encoder", "==", encoder), ("pooling", "==", pooling)]
    tables = [pq.read_table(p, filters=filters) for p in feature_files]
    tables = [t for t in tables if t.num_rows]
    if not tables:
        raise RunError(f"no rows for setting {encoder}/{pooling}")
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    del tables
    block = table.to_pandas(split_blocks=True, self_destruct=True)   # avoids a second full copy
    del table
    block["damaged"] = block["damaged"].astype("boolean")
    return block


def run_benchmark(
    feature_files: list[str | Path], comparisons: list[ComparisonSpec], events: EventsConfig, out_dir: str | Path,
    *, today: date | None = None, level: float = 0.95, n_boot: int = 2000, seed: int = 0,
    event_set: str = "auto", only_compared: bool = False,
) -> Path:
    if not feature_files:
        raise RunError("give at least one features file")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = feature_settings(feature_files)
    for c in comparisons:
        for side in (c.a, c.b):
            if side not in settings:
                raise RunError(f"comparison {c.label!r}: setting {side[0]}/{side[1]} not in the features "
                               f"(have {[f'{e}/{p}' for e, p in settings]})")
    if only_compared:   # probe only the settings some comparison uses (others in the files are skipped)
        used = {side for c in comparisons for side in (c.a, c.b)}
        settings = [s for s in settings if s in used]
    # One (encoder, pooling) setting in memory at a time: a whole BRIGHT features file with several
    # settings does not fit in a free Colab session alongside the copies the probes make.
    score_parts, fits = [], []
    for enc, pool in settings:
        block = read_setting(feature_files, enc, pool)
        s, f = run_probes(block, events, today, seed=seed)
        score_parts.append(s)
        fits.extend(f)
        del block
        gc.collect()
    scores = pd.concat(score_parts, ignore_index=True)
    metrics = event_metrics(scores, events, n_boot=n_boot, seed=seed)
    ev_list = comparison_events(events, today, event_set)
    phase = "locked" if labels_unlocked(events, today) else "development"
    rows = []
    for c in comparisons:
        r = compare(metrics, c.a, c.b, events=ev_list, level=level)
        rows.append({
            "label": c.label, "a": "/".join(c.a), "b": "/".join(c.b),
            "strength": "confirmatory" if (c.primary and phase == "locked" and ev_list is not None
                                           and set(ev_list) == set(events.keys_with_role("confirmatory")))
            else "exploratory",
            "n_events": r.interval.n, "events": "+".join(r.gains.index),
            "gains": json.dumps({k: round(float(v), 6) for k, v in r.gains.items()}),
            "estimate": r.interval.estimate, "lower": r.interval.lower, "upper": r.interval.upper,
            "level": level, "df": r.interval.df, "verdict_row": r.verdict.row, "verdict": r.verdict.meaning,
            "sign_flip_p": r.sign_flip.p_value, "sign_flip_min_p": r.sign_flip.min_p,
        })
    scores.to_parquet(out / "fold_scores.parquet", index=False)
    pd.DataFrame([asdict(f) for f in fits]).to_csv(out / "probe_fits.csv", index=False)
    metrics.to_csv(out / "event_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "comparisons.csv", index=False)
    manifest = {
        "date": (today or date.today()).isoformat(), "phase": phase, "tool_version": __version__,
        "git_commit": git_commit(), "seed": seed, "n_boot": n_boot, "level": level, "event_set": event_set,
        "comparison_events": ev_list, "settings": [f"{e}/{p}" for e, p in settings],
        "comparisons": [c.label for c in comparisons],
        "inputs": {str(p): sha256(p) for p in feature_files},
        "environment": environment(),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(out)
    return out


# ---------------------------------------------------------------- report v0

def _fmt(x, nd=3):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def write_report(run_dir: str | Path) -> Path:
    """Markdown report from a run folder (``bench report``)."""
    d = Path(run_dir)
    man = json.loads((d / "run_manifest.json").read_text())
    m = pd.read_csv(d / "event_metrics.csv")
    comps = pd.read_csv(d / "comparisons.csv")
    lines = [
        "# DamageSAR-Bench run report", "",
        f"- Date: {man['date']} · phase: **{man['phase']}** · tool {man['tool_version']} · commit `{man['git_commit']}`",
        f"- Seed {man['seed']}, bootstrap {man['n_boot']} replicates, interval level {man['level']:.3g}",
        f"- Settings: {', '.join(man['settings'])}",
    ]
    if man["phase"] == "development":
        lines += ["", "> Development phase (before the 10 Jan 2027 freeze): only development events are held out. "
                  "Every comparison below is **exploratory**; no confirmatory event is used."]
    lines += ["", "## Per-event results (held-out events)", "",
              "| Setting | Event | Role | Buildings | Damaged | AUROC | 95% tile-bootstrap | AP | Note |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in m.sort_values(["encoder", "pooling", "event"]).iterrows():
        note = "" if r["boot_reliable"] else "bootstrap unreliable (many one-class replicates)"
        lines.append(f"| {r['encoder']}/{r['pooling']} | {r['event']} | {r['role']} | {int(r['n'])} | "
                     f"{int(r['n_damaged'])} | {_fmt(r['auroc'])} | {_fmt(r['boot_lower'])}–{_fmt(r['boot_upper'])} | "
                     f"{_fmt(r['ap'])} | {note} |")
    lines += ["", "## Comparisons (event-level)", ""]
    if comps.empty:
        lines.append("No comparisons requested.")
    else:
        lines += ["| Comparison | Strength | Events | Mean gain | Interval | Verdict | Sign-flip p (min possible) |",
                  "|---|---|---|---|---|---|---|"]
        for _, c in comps.iterrows():
            verdict = f"{int(c['verdict_row'])}. {c['verdict']}"
            if c["strength"] != "confirmatory":
                verdict += " (exploratory reading, not a claim)"
            lines.append(f"| {c['label']} | {c['strength']} | {int(c['n_events'])} | {_fmt(c['estimate'])} | "
                         f"{_fmt(c['lower'])} to {_fmt(c['upper'])} ({c['level']:.3g}, t, {int(c['df'])} df) | "
                         f"{verdict} | {_fmt(c['sign_flip_p'])} ({_fmt(c['sign_flip_min_p'])}) |")
        lines += ["", "Verdict rows (checked in order): 1 lower > 0 and gain ≥ +0.02 · 2 lower > 0, gain < +0.02 · "
                  "3 lower ≤ 0 and upper < +0.02 · 4 otherwise (inconclusive). The sign-flip test is supporting "
                  "evidence only. A non-significant result is not evidence of no benefit."]
        lines += ["", "Per-event gains:", ""]
        for _, c in comps.iterrows():
            g = json.loads(c["gains"])
            lines.append(f"- {c['label']}: " + ", ".join(f"{k} {v:+.3f}" for k, v in g.items()))
    lines += ["", "## Inputs", ""] + [f"- `{p}` sha256 `{h[:16]}…`" for p, h in man["inputs"].items()]
    path = d / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
