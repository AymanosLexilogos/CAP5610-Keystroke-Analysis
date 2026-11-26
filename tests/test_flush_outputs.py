"""scripts/flush_outputs.py: dry run by default, only run folders, index pruning."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

# scripts/ is not a package: load the script as a module (registered first, as dataclasses require)
spec = importlib.util.spec_from_file_location(
    "flush_outputs", Path(__file__).resolve().parents[1] / "scripts" / "flush_outputs.py")
FO = importlib.util.module_from_spec(spec)
sys.modules["flush_outputs"] = FO
spec.loader.exec_module(FO)

RUNS = ["official_GMM001_20261125-182347", "bench_GMM_raw31001_20261126-093000",
        "bench_GMM_raw31002_20261126-100000"]


def _make(tmp_path):
    out = tmp_path / "outputs"
    for r in RUNS:
        (out / r / "best_model").mkdir(parents=True)
        (out / r / "config.json").write_text("{}")
    (out / "report_figures").mkdir()
    (out / "report_figures" / ".gitkeep").write_text("")
    (out / "report_figures" / "fig.png").write_bytes(b"png")
    (out / "notes.txt").write_text("keep me")            # not a run folder
    (out / "my_folder").mkdir()                          # not a run folder
    (out / "_quick" / "x001_20261126-000000").mkdir(parents=True)
    pd.DataFrame({"run_id": RUNS, "mean_eer": [0.1, 0.2, 0.3]}).to_csv(out / "runs_index.csv", index=False)
    return out


def test_dry_run_deletes_nothing(tmp_path):
    out = _make(tmp_path)
    before = sorted(p.as_posix() for p in out.rglob("*"))
    FO.main(["--outputs", str(out)])
    assert sorted(p.as_posix() for p in out.rglob("*")) == before


def test_full_flush_keeps_non_run_files(tmp_path):
    out = _make(tmp_path)
    FO.main(["--outputs", str(out), "--yes"])
    assert not any((out / r).exists() for r in RUNS)
    assert not (out / "runs_index.csv").exists() and not (out / "report_figures" / "fig.png").exists()
    assert (out / "report_figures" / ".gitkeep").exists()
    assert (out / "notes.txt").exists() and (out / "my_folder").exists()
    assert (out / "_quick").exists()                     # only removed with --quick


def test_filtered_flush_prunes_index(tmp_path):
    out = _make(tmp_path)
    FO.main(["--outputs", str(out), "--only", "bench_", "--before", "20261126-095000", "--yes"])
    assert not (out / RUNS[1]).exists()
    assert (out / RUNS[0]).exists() and (out / RUNS[2]).exists()
    assert pd.read_csv(out / "runs_index.csv").run_id.tolist() == [RUNS[0], RUNS[2]]
    assert (out / "report_figures" / "fig.png").exists()   # figures kept when filtering


def test_quick_only(tmp_path):
    out = _make(tmp_path)
    FO.main(["--outputs", str(out), "--quick", "--yes"])
    assert not (out / "_quick").exists() and all((out / r).exists() for r in RUNS)
