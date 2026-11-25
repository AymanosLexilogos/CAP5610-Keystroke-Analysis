"""Protocol runner: builds splits, fits one model per user, scores, checks leaks."""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import utils as U
from .models import Population


@dataclass
class ModelSpec:
    """A model family + hyperparameters. ``build(ctx)`` returns an unfitted model."""
    name: str
    cls: type
    params: dict = field(default_factory=dict)
    population: bool = False

    def build(self, ctx):
        if self.population:
            return self.cls(population=ctx["population"], **self.params)
        return self.cls(**self.params)

    def population_rows(self, split):
        """Rows this model uses as population (for the leak check)."""
        return split.population if self.population else np.array([], int)

    def label(self):
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({p})" if p else self.name


@dataclass
class EvalResult:
    per_user: pd.DataFrame
    scores: dict
    models: dict
    seconds: float

    @property
    def eers(self):
        return self.per_user["eer"].to_numpy()

    def summary(self):
        s = U.summarize(self.eers)
        if "ll_test" in self.per_user and self.per_user["ll_test"].notna().any():
            ll = self.per_user["ll_test"]
            s.update(test_ll_mean=float(ll.mean()), test_ll_median=float(ll.median()),
                     test_ll_min=float(ll.min()), test_ll_max=float(ll.max()))
        return s


def evaluate(X, spec, protocol="test", fold=None, users=None, train_fn=None,
             keep_models=False, folds=None) -> EvalResult:
    """Fit/score ``spec`` for each user under ``protocol``.

    train_fn(u) -> training rows overrides the default (enrollment experiments).
    Every fitted row and population row goes through ``utils.check_no_leak``.
    """
    t0 = time.time()
    folds = folds if folds is not None else U.openset_folds()
    if users is None:
        users = range(U.N_USERS)
        if protocol == "openset":
            users = [u for u in users if u not in folds[fold]]
    recs, scores, models = [], {}, {}
    for u in users:
        tr = train_fn(u) if train_fn is not None else None
        sp = U.make_split(u, protocol, fold=fold, train=tr, folds=folds)
        pop_rows = spec.population_rows(sp)
        U.check_no_leak(protocol, u, sp.train, pop_rows, fold=fold, folds=folds)
        ctx = {"user": u, "protocol": protocol, "fold": fold, "split": sp, "X": X}
        if spec.population:
            ctx["population"] = Population(X[sp.population], U.row_user(sp.population))
        if protocol == "val":
            ctx["X_val"] = X[sp.genuine]  # for early stopping (flow)
        m = spec.build(ctx).fit(X[sp.train])
        g, i = m.score(X[sp.genuine]), m.score(X[sp.impostor])
        rec = {"user": u, "eer": U.eer(g, i), "n_train": len(sp.train)}
        if getattr(m, "is_density", False):
            rec["ll_test"] = float(np.mean(g))       # mean log-lik of genuine held-out typings
            rec["ll_train"] = float(np.mean(m.score(X[sp.train])))
        for k in ("k_", "best_epoch_", "shrinkage_"):
            if hasattr(m, k):
                rec[k.rstrip("_")] = float(getattr(m, k))
        recs.append(rec)
        scores[u] = (g, i)
        if keep_models:
            models[u] = m
    return EvalResult(pd.DataFrame(recs), scores, models, time.time() - t0)


def evaluate_openset(X, spec, folds=None, keep_models=False) -> EvalResult:
    """3 folds of 17: held-out fold = impostors only. Per-user EER = mean over the
    two folds in which the user is enrolled, so results pair with other models."""
    folds = folds if folds is not None else U.openset_folds()
    t0 = time.time()
    parts, scores, models = [], {}, {}
    for f in range(len(folds)):
        r = evaluate(X, spec, "openset", fold=f, folds=folds, keep_models=keep_models)
        parts.append(r.per_user.assign(fold=f))
        for u, s in r.scores.items():
            scores[(u, f)] = s
        for u, m in r.models.items():
            models[(u, f)] = m
    allp = pd.concat(parts, ignore_index=True)
    num = allp.drop(columns=["fold"]).groupby("user").mean().reset_index()
    return EvalResult(num, scores, models, time.time() - t0)


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]


def tune(X, make_spec, param_grid, users=None, verbose=True):
    """Evaluate every grid point on the validation protocol only.

    Returns (best_params, table). Selection = lowest mean validation EER."""
    rows = []
    for p in param_grid:
        spec = make_spec(**p)
        r = evaluate(X, spec, "val", users=users)
        s = r.summary()
        row = {**p, "val_mean_eer": s["mean_eer"], "val_sd_eer": s["sd_eer"],
               "val_ll_median": s.get("test_ll_median", np.nan), "seconds": r.seconds}
        if "best_epoch" in r.per_user:
            row["median_best_epoch"] = float(r.per_user["best_epoch"].median())
        rows.append(row)
        if verbose:
            print(f"  {spec.label():60s} val EER {s['mean_eer']:.4f}")
    table = pd.DataFrame(rows).sort_values("val_mean_eer").reset_index(drop=True)
    best = {k: table.loc[0, k] for k in param_grid[0]}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    return best, table
