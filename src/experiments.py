"""Experiment runners used by notebooks/experiments.ipynb (no saving here:
all saving is done by the notebook's saving cell)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as M
from . import utils as U
from .evaluation import ModelSpec, evaluate, grid, tune
from .flow import FlowSpec

BASELINE = "Scaled Manhattan"
POPULATION_FAMILIES = ("Gaussian tied", "Gaussian Bayes/MAP", "RealNVP (pretrained)")


# --------------------------------------------------------------------------
# Model registry: family -> (spec factory, validation grid)
# --------------------------------------------------------------------------
def _spec(name, cls, population=False):
    return lambda **p: ModelSpec(name, cls, p, population)


def _flow_spec(pretrain=False, noise=0.0, lr=3e-3, hidden=32, epochs=None):
    params = {"hidden": hidden, "noise": noise, "lr": lr, "weight_decay": 1e-3}
    if epochs is not None:
        params["epochs"] = int(epochs)
    return FlowSpec(params, pretrain=bool(pretrain),
                    pretrain_params={"hidden": hidden, "epochs": 150})


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
    make, g = FAMILIES[family]
    if verbose:
        print(f"[tune] {family}: {len(g)} configs")
    return tune(X, make, g, verbose=verbose)


# --------------------------------------------------------------------------
# Enrollment size
# --------------------------------------------------------------------------
ENROLL_N = (8, 12, 24, 48, 100, 200)
STRAT_SEEDS = (0, 1, 2, 3, 4)


def enrollment(X, spec, ns=ENROLL_N, seeds=STRAT_SEEDS, samplings=("first", "stratified")):
    """Long table: sampling, seed, n, user, eer, ll_test. Test = sessions 5-8."""
    out = []
    for sampling in samplings:
        for seed in (seeds if sampling == "stratified" else (0,)):
            for n in ns:
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
    recs, models = [], {}
    for u in range(U.N_USERS):
        rng = np.random.default_rng([seed, u])
        others = np.array([v for v in range(U.N_USERS) if v != u])
        val_g = U.rows(u, n0 + 1, U.REPS)
        val_i = U.rows(others, *U.IMP_VAL)
        imp_pool = U.rows(others, U.IMP_VAL[1] + 1, U.TRAIN_END)
        imp = rng.choice(imp_pool, n_imp_attempts, replace=False)
        stream = np.concatenate([U.rows(u, U.REPS + 1, U.TRAIN_END), imp])
        is_imp = np.r_[np.zeros(U.TRAIN_END - U.REPS, bool), np.ones(n_imp_attempts, bool)]
        # random interleaving that keeps the genuine typings in chronological order
        order = rng.permutation(len(stream))
        gpos = np.sort(order[:len(stream) - n_imp_attempts])
        ipos = order[len(stream) - n_imp_attempts:]
        seq = np.empty(len(stream), int)
        seq[gpos] = np.arange(U.TRAIN_END - U.REPS)
        seq[ipos] = np.arange(U.TRAIN_END - U.REPS, len(stream))
        stream, is_imp = stream[seq], is_imp[seq]

        def refit(tr):
            ctx = {"user": u, "protocol": "test", "fold": None, "X": X}
            if spec.population:
                sp = U.make_split(u, "test")
                ctx["population"] = M.Population(X[sp.population], U.row_user(sp.population))
            m = spec.build(ctx).fit(X[tr])
            th = U.eer_threshold(m.score(X[val_g]), m.score(X[val_i]))
            return m, th

        template = list(U.rows(u, 1, n0))
        m, th = refit(template)
        static = m
        acc_g = acc_i = pending = 0
        for r, bad in zip(stream, is_imp):
            if m.score(X[[r]])[0] >= th:
                template.append(r)
                acc_i += bad
                acc_g += not bad
                pending += 1
                if pending >= refit_every:
                    m, th = refit(template)
                    pending = 0
        if pending:
            m, th = refit(template)
        U.check_no_leak("test", u, np.array(template))
        te = U.make_split(u, "test")
        oracle = refit(list(U.rows(u, 1, n0)) + list(U.rows(u, U.REPS + 1, U.TRAIN_END)))[0]
        rec = {"user": u, "n0": n0,
               "eer_static": U.eer(static.score(X[te.genuine]), static.score(X[te.impostor])),
               "eer_adapted": U.eer(m.score(X[te.genuine]), m.score(X[te.impostor])),
               "eer_oracle": U.eer(oracle.score(X[te.genuine]), oracle.score(X[te.impostor])),
               "genuine_accepted": acc_g, "genuine_attempts": U.TRAIN_END - U.REPS,
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
        if m.k_ < 2:
            continue
        tr = U.rows(u, 1, U.TRAIN_END)
        comp = m.predict(X[tr])
        sess = (U.row_typing(tr) - 1) // U.REPS + 1
        ct = pd.crosstab(pd.Series(sess, name="session"), pd.Series(comp, name="component"))
        tables[u] = ct
        purity = ct.max(axis=0).sum() / ct.values.sum()
        rows_out.append({"user": u, "k": m.k_, "weights": np.round(np.sort(m.gmm_.weights_)[::-1], 3).tolist(),
                         "ari_vs_session": adjusted_rand_score(sess, comp),
                         "ari_vs_early_late": adjusted_rand_score(sess > 2, comp),
                         "purity_by_session": float(purity)})
    return pd.DataFrame(rows_out), tables
