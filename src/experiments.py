"""Experiment runners used by notebooks/experiments.ipynb.

This module only *computes*: it returns DataFrames and fitted models. All
writing to disk is done by the notebook's saving cell.

Contents
--------
FAMILIES              model registry: family name -> (spec factory, validation grid)
tune_family/final_spec  tuning on validation, then the spec used for the single test run
enrollment            experiment 3 (first-n vs stratified enrollment size)
adaptation            experiment 4 (template adaptation with accepted logins)
gmm_session_alignment experiment 6 (do GMM components follow sessions?)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as M
from . import utils as U
from .evaluation import ModelSpec, evaluate, grid, tune
from .flow import FlowSpec

# Every paired test compares against this family on the 21 log features.
BASELINE = "Scaled Manhattan"

# Families that use other users' data; they also get the open-set evaluation.
POPULATION_FAMILIES = ("Gaussian tied", "Gaussian Bayes/MAP", "RealNVP (pretrained)")


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
def _spec(name, cls, population=False):
    """Factory: hyperparameters (as keyword args) -> ModelSpec."""
    return lambda **p: ModelSpec(name, cls, p, population)


def _flow_spec(pretrain=False, noise=0.0, lr=3e-3, hidden=32, epochs=None):
    """Factory for the flow. ``epochs`` is only set for the final refit (it comes
    from early stopping on validation); during tuning it stays None."""
    params = {"hidden": hidden, "noise": noise, "lr": lr, "weight_decay": 1e-3}
    if epochs is not None:
        params["epochs"] = int(epochs)
    return FlowSpec(params, pretrain=bool(pretrain),
                    pretrain_params={"hidden": hidden, "epochs": 150})


# family -> (factory, list of hyperparameter dicts tried on validation)
FAMILIES = {
    "Scaled Manhattan": (_spec("Scaled Manhattan", M.ScaledManhattan), [{}]),
    "Gaussian diag": (_spec("Gaussian diag", M.DiagGaussian), grid(reg=[0.0, 1e-3, 1e-2, 1e-1])),
    "Gaussian full": (_spec("Gaussian full", M.FullGaussian), grid(reg=[1e-3, 1e-2, 3e-2, 1e-1, 3e-1])),
    "Gaussian Ledoit-Wolf": (_spec("Gaussian Ledoit-Wolf", M.LedoitWolfGaussian),
                             grid(shrinkage=["auto", 0.05, 0.1, 0.2, 0.4])),
    "Gaussian tied": (_spec("Gaussian tied", M.TiedGaussian, True), grid(reg=[1e-3, 1e-2, 1e-1])),
    "Gaussian Bayes/MAP": (_spec("Gaussian Bayes/MAP", M.BayesGaussian, True),
                           grid(kappa0=[0.01, 1.0, 10.0], nu_extra=[5, 20, 100, 400])),
    "Student-t": (_spec("Student-t", M.StudentT), grid(nu=[2, 4, 8, 16, 32], reg=[1e-2, 1e-1])),
    "GMM": (_spec("GMM", M.GMM), grid(reg_covar=[1e-3, 3e-3, 1e-2])),
    "Laplace": (_spec("Laplace", M.Laplace), grid(loc=["median", "mean"])),
    "RealNVP": (_flow_spec, grid(pretrain=[False], noise=[0.0, 0.2, 0.5], lr=[1e-3, 3e-3])),
    "RealNVP (pretrained)": (_flow_spec, grid(pretrain=[True], noise=[0.0, 0.2, 0.5], lr=[1e-3, 3e-3])),
}


def final_spec(family, best, table=None):
    """Spec refit on all 200 training typings. For the flow, the number of epochs
    is the median best epoch found by early stopping on validation."""
    make, _ = FAMILIES[family]
    if family.startswith("RealNVP"):
        ep = max(1, int(round(table.loc[0, "median_best_epoch"])))
        return make(**best, epochs=ep)
    return make(**best)


def tune_family(X, family, verbose=True):
    """Run the family's grid on the validation protocol -> (best params, table)."""
    make, g = FAMILIES[family]
    if verbose:
        print(f"[tune] {family}: {len(g)} configs")
    return tune(X, make, g, verbose=verbose)


# --------------------------------------------------------------------------
# Enrollment size
# --------------------------------------------------------------------------
ENROLL_N = (8, 12, 24, 48, 100, 200)   # enrollment sizes (all divisible by 4)
STRAT_SEEDS = (0, 1, 2, 3, 4)           # seeds for the stratified draws


def enrollment(X, spec, ns=ENROLL_N, seeds=STRAT_SEEDS, samplings=("first", "stratified")):
    """Fit on n enrollment typings, test on sessions 5-8 vs others' typings 1-5.

    Returns a long table with columns sampling, seed, n, user, eer, ll_test, ...
    'first' is deterministic, so it runs once (seed 0); 'stratified' runs per seed.
    """
    out = []
    for sampling in samplings:
        for seed in (seeds if sampling == "stratified" else (0,)):
            for n in ns:
                # default args freeze n/seed for this loop iteration
                r = evaluate(X, spec, "test",
                             train_fn=lambda u, n=n, s=seed: U.enrollment_rows(u, n, sampling, s))
                out.append(r.per_user.assign(sampling=sampling, seed=seed, n=n))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Template adaptation
# --------------------------------------------------------------------------
def adaptation(X, spec, n0=24, n_imp_attempts=50, refit_every=10, seed=0):
    """Enroll on typings 1..n0 (session 1); stream sessions 2-4 (typings 51-200) in
    order, interleaved with impostor attempts (others' typings 11-200, never test
    typings). Accepted attempts (score >= threshold) are added to the template;
    the model is refit every ``refit_every`` acceptances and the threshold is
    re-chosen at the EER point on validation (session-1 typings n0+1..50 vs others'
    typings 6-10). Final model tested on sessions 5-8 vs others' 1-5.
    """
    n_genuine = U.TRAIN_END - U.REPS  # 150 login attempts from sessions 2-4
    recs, models = [], {}
    for u in range(U.N_USERS):
        rng = np.random.default_rng([seed, u])
        others = np.array([v for v in range(U.N_USERS) if v != u])

        # Validation set for the threshold: the rest of session 1 vs others' 6-10.
        val_g = U.rows(u, n0 + 1, U.REPS)
        val_i = U.rows(others, *U.IMP_VAL)

        # Impostor login attempts: others' typings 11-200 (never test typings 1-5,
        # never the validation impostors 6-10).
        imp_pool = U.rows(others, U.IMP_VAL[1] + 1, U.TRAIN_END)
        imp = rng.choice(imp_pool, n_imp_attempts, replace=False)

        # Build the login stream: genuine sessions 2-4 + impostor attempts, shuffled
        # so impostors land at random positions while the genuine typings keep
        # their chronological order.
        stream = np.concatenate([U.rows(u, U.REPS + 1, U.TRAIN_END), imp])
        is_imp = np.r_[np.zeros(n_genuine, bool), np.ones(n_imp_attempts, bool)]
        order = rng.permutation(len(stream))
        gpos = np.sort(order[:n_genuine])          # stream positions given to genuine typings
        ipos = order[n_genuine:]                   # stream positions given to impostors
        seq = np.empty(len(stream), int)
        seq[gpos] = np.arange(n_genuine)
        seq[ipos] = np.arange(n_genuine, len(stream))
        stream, is_imp = stream[seq], is_imp[seq]

        def refit(tr):
            """Fit on the template rows and re-pick the EER threshold on validation."""
            ctx = {"user": u, "protocol": "test", "fold": None, "X": X}
            if spec.population:
                sp = U.make_split(u, "test")
                ctx["population"] = M.Population(X[sp.population], U.row_user(sp.population))
            m = spec.build(ctx).fit(X[tr])
            th = U.eer_threshold(m.score(X[val_g]), m.score(X[val_i]))
            return m, th

        # Enroll on the first n0 typings, then replay the logins.
        template = list(U.rows(u, 1, n0))
        m, th = refit(template)
        static = m                                  # the never-updated model, for comparison
        acc_g = acc_i = pending = 0
        for r, bad in zip(stream, is_imp):
            if m.score(X[[r]])[0] >= th:            # login accepted -> absorbed into the template
                template.append(r)
                acc_i += bad
                acc_g += not bad
                pending += 1
                if pending >= refit_every:          # refit in batches, not after every login
                    m, th = refit(template)
                    pending = 0
        if pending:
            m, th = refit(template)
        U.check_no_leak("test", u, np.array(template))

        # Test once on sessions 5-8. "oracle" = perfect labels: all genuine logins added.
        te = U.make_split(u, "test")
        oracle = refit(list(U.rows(u, 1, n0)) + list(U.rows(u, U.REPS + 1, U.TRAIN_END)))[0]
        rec = {"user": u, "n0": n0,
               "eer_static": U.eer(static.score(X[te.genuine]), static.score(X[te.impostor])),
               "eer_adapted": U.eer(m.score(X[te.genuine]), m.score(X[te.impostor])),
               "eer_oracle": U.eer(oracle.score(X[te.genuine]), oracle.score(X[te.impostor])),
               "genuine_accepted": acc_g, "genuine_attempts": n_genuine,
               "impostors_absorbed": acc_i, "impostor_attempts": n_imp_attempts,
               "template_size": len(template)}
        if getattr(m, "is_density", False):
            rec["ll_test"] = float(np.mean(m.score(X[te.genuine])))
        recs.append(rec)
        models[u] = m
    return pd.DataFrame(recs), models


# --------------------------------------------------------------------------
# GMM diagnostics
# --------------------------------------------------------------------------
def gmm_session_alignment(X, gmm_models):
    """For users with K>=2: component vs training session (1-4) contingency,
    adjusted Rand index and component weights."""
    from sklearn.metrics import adjusted_rand_score
    rows_out, tables = [], {}
    for u, m in gmm_models.items():
        if m.k_ < 2:                                   # single Gaussian: nothing to align
            continue
        tr = U.rows(u, 1, U.TRAIN_END)
        comp = m.predict(X[tr])                        # most likely component per typing
        sess = (U.row_typing(tr) - 1) // U.REPS + 1    # session 1..4 per typing
        ct = pd.crosstab(pd.Series(sess, name="session"), pd.Series(comp, name="component"))
        tables[u] = ct
        # purity: share of typings in their component's majority session (0.25 = no alignment)
        purity = ct.max(axis=0).sum() / ct.values.sum()
        rows_out.append({"user": u, "k": m.k_, "weights": np.round(np.sort(m.gmm_.weights_)[::-1], 3).tolist(),
                         "ari_vs_session": adjusted_rand_score(sess, comp),
                         "ari_vs_early_late": adjusted_rand_score(sess > 2, comp),
                         "purity_by_session": float(purity)})
    return pd.DataFrame(rows_out), tables
