"""Flush run outputs from outputs/.

Dry run by default: it prints what would be deleted and touches nothing.
Pass --yes to actually delete.

Only folders whose names match the run-folder pattern <NAME><NNN>_<YYYYMMDD-HHMMSS>
(as created by src/saving.py) are ever removed, so other files in outputs/ are safe.

Examples (run from the repo root, inside the keystroke-gmm env):
    python scripts/flush_outputs.py                      # show what a full flush would delete
    python scripts/flush_outputs.py --yes                # delete all runs, runs_index.csv, report figures
    python scripts/flush_outputs.py --only bench_ --yes  # delete only runs whose name starts with bench_
    python scripts/flush_outputs.py --before 20261126-000000 --yes   # delete runs created before that time
    python scripts/flush_outputs.py --quick --yes        # delete outputs/_quick (KS_QUICK smoke runs)

With --only / --before, the matching rows are removed from runs_index.csv; the
index file itself and the report figures are kept (a figure can't be traced to one run).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = re.compile(r"^(?P<name>.+?)(?P<nnn>\d{3})_(?P<ts>\d{8}-\d{6})$")  # e.g. official_GMM001_20261125-182347


@dataclass
class Plan:
    """What a flush would delete."""
    runs: list[Path] = field(default_factory=list)       # run folders
    figures: list[Path] = field(default_factory=list)    # report_figures/*.png
    index_path: Path | None = None
    index_action: str = "keep"                           # "keep" | "delete" | "prune"
    index_rows_removed: int = 0
    extra_dirs: list[Path] = field(default_factory=list) # e.g. outputs/_quick


def make_plan(outputs: Path, only: str | None = None, before: str | None = None,
              quick: bool = False, keep_index: bool = False, keep_figures: bool = False) -> Plan:
    outputs = outputs.resolve()
    plan = Plan()
    if quick:  # the smoke-run folder only
        q = outputs / "_quick"
        if q.is_dir():
            plan.extra_dirs.append(q)
        return plan

    filtered = only is not None or before is not None
    for p in sorted(outputs.iterdir()) if outputs.is_dir() else []:
        m = RUN_DIR.match(p.name)
        if not (p.is_dir() and m):
            continue                                     # not a run folder: never touched
        if only is not None and not p.name.startswith(only):
            continue
        if before is not None and m.group("ts") >= before:
            continue
        plan.runs.append(p)

    index = outputs / "runs_index.csv"
    if index.exists() and not keep_index:
        plan.index_path = index
        if filtered:
            ids = {p.name for p in plan.runs}
            n = int(pd.read_csv(index)["run_id"].isin(ids).sum())
            plan.index_action, plan.index_rows_removed = ("prune", n) if n else ("keep", 0)
        else:
            plan.index_action = "delete"

    figs = outputs / "report_figures"
    if figs.is_dir() and not keep_figures and not filtered:
        plan.figures = sorted(figs.glob("*.png"))        # .gitkeep and anything else stay
    return plan


def describe(plan: Plan, outputs: Path) -> str:
    rel = lambda p: p.relative_to(outputs.resolve().parent) if outputs.resolve().parent in p.parents else p
    lines = [f"run folders: {len(plan.runs)}"]
    lines += [f"  {rel(p)}" for p in plan.runs]
    lines += [f"extra folder: {rel(p)}" for p in plan.extra_dirs]
    if plan.index_path is not None and plan.index_action != "keep":
        lines.append(f"runs_index.csv: {plan.index_action}"
                     + (f" ({plan.index_rows_removed} rows)" if plan.index_action == "prune" else ""))
    lines.append(f"report figures: {len(plan.figures)}")
    lines += [f"  {rel(p)}" for p in plan.figures]
    return "\n".join(lines)


def execute(plan: Plan, outputs: Path) -> None:
    root = outputs.resolve()
    for p in plan.runs + plan.extra_dirs:
        assert root in p.resolve().parents, f"refusing to delete outside {root}: {p}"
        shutil.rmtree(p)
    for p in plan.figures:
        p.unlink()
    if plan.index_path is not None:
        if plan.index_action == "delete":
            plan.index_path.unlink()
        elif plan.index_action == "prune":
            ids = {p.name for p in plan.runs}
            idx = pd.read_csv(plan.index_path)
            idx[~idx["run_id"].isin(ids)].to_csv(plan.index_path, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs", help="outputs folder (default: repo outputs/)")
    ap.add_argument("--only", metavar="PREFIX", help="only run folders whose name starts with PREFIX")
    ap.add_argument("--before", metavar="YYYYMMDD-HHMMSS", help="only runs created before this timestamp")
    ap.add_argument("--quick", action="store_true", help="only delete outputs/_quick (KS_QUICK smoke runs)")
    ap.add_argument("--keep-index", action="store_true", help="leave runs_index.csv untouched")
    ap.add_argument("--keep-figures", action="store_true", help="leave report_figures/ untouched")
    a = ap.parse_args(argv)
    if a.before and not re.fullmatch(r"\d{8}-\d{6}", a.before):
        ap.error("--before must look like 20261126-093000")

    plan = make_plan(a.outputs, a.only, a.before, a.quick, a.keep_index, a.keep_figures)
    print(describe(plan, a.outputs))
    nothing = not (plan.runs or plan.extra_dirs or plan.figures or plan.index_action != "keep")
    if nothing:
        print("nothing to delete")
    elif not a.yes:
        print("\ndry run: nothing deleted. Re-run with --yes to delete the above.")
    else:
        execute(plan, a.outputs)
        print("\ndeleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
