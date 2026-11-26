"""Run folders, logging and saving for the notebooks.

Each notebook has ONE saving cell that creates a ``Saver`` and exposes its
functions; experiment cells only call them:

    start_run(name, config)  -> Run   creates outputs/<NAME><NNN>_<timestamp>/ and
                                      writes config.json + env.json BEFORE training
    run.capture()                     context manager: tees printed output and
                                      warnings into the run's log.txt
    run.add_figure(fig, name)         keeps a figure for the final save
    run.set_results(...)              keeps metrics / per-user table / scores / models
    save_model(run)                   best_model/ (best_config.json, parameters, checkpoints)
    save_figure(run, fig, name)       visualizations/<name>.png (+ report_figures/)
    save_run(run)                     metrics.json, per_user.csv, scores.npz, tables,
                                      and one row appended to runs_index.csv
    save_all()                        calls the three above for every run (last cell)

Folders are never overwritten: NNN increments per NAME and the folder is created
with exist_ok=False.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _slug(s) -> str:
    """Filesystem-safe name: 'Gaussian Bayes/MAP' -> 'Gaussian_Bayes_MAP'."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def _jsonable(o):
    """Recursively convert numpy / pandas objects into JSON-friendly Python types."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return _jsonable(o.item())
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return _jsonable(o.to_dict())
    if isinstance(o, float) and not np.isfinite(o):
        return None                      # NaN/inf are not valid JSON
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")


def env_info(repo: Path) -> dict:
    """Git commit + library versions, so every run can be traced and reproduced."""
    import matplotlib
    import scipy
    import sklearn
    import torch
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "notebooks"],
                                             cwd=repo, text=True).strip())
    except Exception as e:  # git missing (e.g. a zip download)
        commit, dirty = f"unavailable ({e})", None
    return {"git_commit": commit, "git_dirty_src_or_notebooks": dirty,
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__,
            "matplotlib": matplotlib.__version__, "conda_env": os.environ.get("CONDA_DEFAULT_ENV")}


class _Tee(io.TextIOBase):
    """Write to several streams at once (notebook output + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
@dataclass
class Run:
    """One output folder. Results are kept in memory until the final save cell."""
    name: str
    path: Path
    config: dict
    figures: list = field(default_factory=list)   # (fig, name, report)
    results: dict = field(default_factory=dict)
    saved: bool = False

    @property
    def id(self) -> str:
        return self.path.name

    @contextlib.contextmanager
    def capture(self):
        """Tee printed output into log.txt and append any warnings raised inside."""
        with open(self.path / "log.txt", "a", encoding="utf-8") as f, \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            old = sys.stdout
            sys.stdout = _Tee(old, f)
            t = time.time()
            try:
                yield self
            finally:
                sys.stdout = old
                for w in caught:
                    f.write(f"WARNING {w.category.__name__}: {w.message} ({w.filename}:{w.lineno})\n")
                f.write(f"[elapsed {time.time() - t:.1f}s]\n")

    def add_figure(self, fig, name: str, report: bool = False):
        """Show the figure now, keep it for the final save. report=True also copies it
        to outputs/report_figures/ (the only figures committed to git)."""
        import matplotlib.pyplot as plt
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            pass
        plt.close(fig)  # closed figures can still be saved; this just frees the pyplot manager
        self.figures.append((fig, name, report))
        return fig

    def set_results(self, per_user=None, metrics=None, scores=None, tables=None, models=None,
                    best_config=None, grid_table=None, extra=None, index=None):
        """Store everything save_run / save_model will write. Can be called repeatedly."""
        new = dict(per_user=per_user, metrics=metrics, scores=scores, tables=tables, models=models,
                   best_config=best_config, grid_table=grid_table, extra=extra, index=index)
        self.results.update({k: v for k, v in new.items() if v is not None})


# --------------------------------------------------------------------------
# Saver
# --------------------------------------------------------------------------
class Saver:
    def __init__(self, repo: Path, out_dir: Path | None = None):
        self.repo = Path(repo)
        self.out = Path(out_dir) if out_dir else self.repo / "outputs"
        self.report_dir = self.out / "report_figures"
        self.index_path = self.out / "runs_index.csv"
        self.out.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(exist_ok=True)
        self.runs: list[Run] = []

    # ---- before training --------------------------------------------------
    def start_run(self, name: str, config: dict) -> Run:
        """Create outputs/<NAME><NNN>_<timestamp>/ and write config.json + env.json now."""
        base = _slug(name)
        pattern = re.escape(base) + r"(\d{3})_\d{8}-\d{6}"
        nums = [int(m.group(1)) for p in self.out.iterdir() if (m := re.fullmatch(pattern, p.name))]
        path = self.out / f"{base}{max(nums, default=0) + 1:03d}_{datetime.now():%Y%m%d-%H%M%S}"
        path.mkdir(exist_ok=False)                     # never overwrite an existing run
        (path / "visualizations").mkdir()
        (path / "best_model").mkdir()
        _write_json(path / "config.json", {"run": path.name, "created": datetime.now().isoformat(), **config})
        _write_json(path / "env.json", env_info(self.repo))
        run = Run(name, path, dict(config))
        self.runs.append(run)
        print(f"[run] {path.relative_to(self.repo) if self.repo in path.parents else path}")
        return run

    # ---- final save ----------------------------------------------------------
    def save_figure(self, run: Run, fig, name: str, report: bool = False) -> Path:
        p = run.path / "visualizations" / f"{_slug(name)}.png"
        fig.savefig(p, bbox_inches="tight")
        if report:
            fig.savefig(self.report_dir / f"{_slug(name)}.png", bbox_inches="tight")
        return p

    def save_model(self, run: Run) -> Path:
        """best_model/: best_config.json (+ grid results), per-user Gaussian-family parameters
        (npz), sklearn GMMs (joblib), flow state_dicts with standardization stats (pt)."""
        import joblib
        import torch
        from . import flow as F
        from . import models as M

        r, bm = run.results, run.path / "best_model"
        cfg = dict(r.get("best_config") or {})
        if r.get("grid_table") is not None:
            r["grid_table"].to_csv(bm / "grid_results.csv", index=False)
            cfg["grid_results"] = r["grid_table"]
        _write_json(bm / "best_config.json", cfg)
        if r.get("extra"):
            _write_json(bm / "extra.json", r["extra"])

        models = r.get("models") or {}
        if not models:
            return bm
        # Flow: weights + standardization + architecture per user, plus the pretrained flows.
        flows = {u: m.get_state() for u, m in models.items() if hasattr(m, "get_state")}
        if flows:
            torch.save(flows, bm / "flow_per_user.pt")
            if any(getattr(m, "pretrained", None) is not None for m in models.values()):
                torch.save(F.pretrained_models(), bm / "flow_pretrained.pt")
        # sklearn GMM objects (reloadable with joblib.load) + their standardization.
        sk = {u: {"gmm": m.gmm_, "std_mean": m.std_.mean_, "std_scale": m.std_.scale_}
              for u, m in models.items() if isinstance(m, M.GMM)}
        if sk:
            joblib.dump(sk, bm / "gmm_per_user.joblib")
        # Everything else (and GMMs too, as plain arrays): one npz, keys "user07__mean".
        flat = {f"user{u:02d}__{k}": np.asarray(v)
                for u, m in models.items() if not hasattr(m, "get_state")
                for k, v in m.get_params().items()}
        if flat:
            np.savez_compressed(bm / "params_per_user.npz", **flat)
        return bm

    def save_run(self, run: Run) -> Path:
        """metrics.json, per_user.csv, scores.npz, extra tables; append to runs_index.csv."""
        r = run.results
        metrics = dict(r.get("metrics") or {})
        _write_json(run.path / "metrics.json", metrics)
        if r.get("per_user") is not None:
            r["per_user"].to_csv(run.path / "per_user.csv", index=False)
        if r.get("scores"):
            arrs = {}
            for k, (g, i) in r["scores"].items():
                key = "_".join(map(str, k)) if isinstance(k, tuple) else str(k)
                arrs[f"genuine_{key}"], arrs[f"impostor_{key}"] = g, i
            np.savez_compressed(run.path / "scores.npz", **arrs)
        for name, t in (r.get("tables") or {}).items():
            t.to_csv(run.path / f"{_slug(name)}.csv", index=False)

        # One summary row per run in outputs/runs_index.csv (committed to git).
        c = run.config
        row = {"run_id": run.id, "timestamp": datetime.now().isoformat(timespec="seconds"),
               "model": c.get("model"), "key_settings": json.dumps(_jsonable(c.get("key_settings", {}))),
               "features": c.get("features"), "split": c.get("split_short", c.get("protocol")),
               "n_train": c.get("n_train"), "sampling": c.get("sampling"),
               "mean_eer": metrics.get("mean_eer"), "ci95_low": metrics.get("ci95_low"),
               "ci95_high": metrics.get("ci95_high"), "test_ll_median": metrics.get("test_ll_median")}
        row.update(r.get("index") or {})
        row = {k: (json.dumps(_jsonable(v)) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
        pd.DataFrame([row]).to_csv(self.index_path, mode="a", header=not self.index_path.exists(), index=False)
        return run.path

    def save_all(self) -> pd.DataFrame:
        """Write models, figures and results of every run not saved yet. Returns a summary."""
        done = []
        for run in self.runs:
            if run.saved:
                continue
            self.save_model(run)
            for fig, name, report in run.figures:
                self.save_figure(run, fig, name, report)
            self.save_run(run)
            run.saved = True
            done.append({"run": run.id, "figures": len(run.figures),
                         "mean_eer": (run.results.get("metrics") or {}).get("mean_eer")})
        return pd.DataFrame(done)
