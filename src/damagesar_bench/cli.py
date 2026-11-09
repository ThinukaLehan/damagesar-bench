"""Command-line interface: ``bench <command>``.

Implemented now: ``validate``, ``audit``, ``status``.
Planned (see docs/preregistration.md and the next-steps guide): ``extract``, ``run``, ``report``,
``new-event``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from damagesar_bench import __version__
from damagesar_bench.config import REPO_ROOT, load_bench, load_events

PLANNED = {
    "extract": "extract and cache frozen-encoder features (Stage 2, from 24 Nov)",
    "run": "pooling, probes, recalibration, controls and statistics (Stage 2-5)",
    "report": "HTML report from saved results (Stage 2)",
    "new-event": "damaged-building estimate for an unlabelled event (Stage 5)",
}


def _find_config(name: str, given: str | None) -> Path:
    """Use an explicit path, else ./configs/<name>, else the repository's configs/<name>."""
    if given:
        return Path(given)
    local = Path.cwd() / "configs" / name
    return local if local.exists() else REPO_ROOT / "configs" / name


def _today(arg: str | None) -> date | None:
    return date.fromisoformat(arg) if arg else None


def cmd_validate(args: argparse.Namespace) -> int:
    from damagesar_bench.data.validate import validate_dataset, write_reports

    cfg = load_bench(_find_config("bench.yaml", args.config), root_override=args.root)
    events = load_events(_find_config("events.yaml", args.events))
    tiles, summary = validate_dataset(cfg, events, today=_today(args.today))
    t_path, s_path = write_reports(tiles, summary, args.out)

    n_bad = int((~tiles["ok"]).sum()) if len(tiles) else 0
    print(f"Validated {len(tiles)} tiles from {cfg.data.root}")
    cols = ["event", "role", "n_tiles", "n_problem_tiles", "n_excluded_nodata", "class_counts"]
    print(summary[cols].to_string(index=False))
    print(f"\nTile report:   {t_path}\nEvent summary: {s_path}")
    if n_bad:
        print(f"\n{n_bad} tile(s) have problems; see the 'problems' column.", file=sys.stderr)
        return 1
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from damagesar_bench.data.splits import make_folds, write_splits

    events = load_events(_find_config("events.yaml", args.events))
    today = _today(args.today)
    path = write_splits(events, Path(args.out) / "splits.json", today=today)
    folds = make_folds(events, today)
    print(f"Wrote {path} with {len(folds)} folds ({folds[0].phase if folds else 'none'} phase)")
    for f in folds:
        print(f"  hold out {f.held_out:<9} ({f.held_out_role}): train on {', '.join(f.train)}")
    print("Leakage and duplicate checks against pretraining corpora: planned (FR4).")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from damagesar_bench.data.events import labels_unlocked

    events = load_events(_find_config("events.yaml", args.events))
    state = "UNLOCKED" if labels_unlocked(events, _today(args.today)) else "LOCKED"
    print(f"damagesar-bench {__version__}")
    print(f"Freeze date: {events.freeze_date.isoformat()}  Confirmatory labels: {state}")
    for role in ("confirmatory", "development", "secondary"):
        print(f"  {role:<13} {', '.join(events.keys_with_role(role))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench", description="DamageSAR-Bench toolkit")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--events", help="path to events.yaml (default: configs/events.yaml)")
        sp.add_argument("--today", help=argparse.SUPPRESS)  # testing only: pretend the date is YYYY-MM-DD

    v = sub.add_parser("validate", help="check data: pairing, shape, CRS, label values, no-data")
    common(v)
    v.add_argument("--root", help="BRIGHT root folder holding post-event/ and target/")
    v.add_argument("--config", help="path to bench.yaml (default: configs/bench.yaml)")
    v.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    v.set_defaults(func=cmd_validate)

    a = sub.add_parser("audit", help="write splits.json (leave-one-event-out folds) and audit checks")
    common(a)
    a.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    a.set_defaults(func=cmd_audit)

    s = sub.add_parser("status", help="show event roles and whether confirmatory labels are locked")
    common(s)
    s.set_defaults(func=cmd_status)

    for name, desc in PLANNED.items():
        sp = sub.add_parser(name, help=f"[planned] {desc}")
        sp.set_defaults(func=lambda a, n=name, d=desc: _planned(n, d))
    return p


def _planned(name: str, desc: str) -> int:
    print(f"'bench {name}' is not implemented yet: {desc}.", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
