"""Figure builders. They only return matplotlib figures; saving is done by the
notebook's save_figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from . import utils as U
from .models import model_cov

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
C_REAL, C_MODEL = "#4C72B0", "#DD8452"


def raw_vs_log(raw, logx, names, idx=(0, 1, 13)):
    fig, ax = plt.subplots(2, len(idx), figsize=(4 * len(idx), 6))
    tr = U.rows(np.arange(U.N_USERS), 1, U.TRAIN_END)
    for j, k in enumerate(idx):
        ax[0, j].hist(raw[tr, k], bins=80, color=C_REAL)
        ax[0, j].set_title(f"raw {names[k]} (s)")
        ax[1, j].hist(logx[tr, k], bins=80, color=C_MODEL)
        ax[1, j].set_title(f"log {names[k]}")
    fig.suptitle("Training typings (sessions 1-4, all users): raw vs log timings")
    fig.tight_layout()
    return fig


def within_between(X, names):
    """Per feature: mean within-user SD vs SD of the user means (training data)."""
    within, means = [], []
    for u in range(U.N_USERS):
        Xu = X[U.rows(u, 1, U.TRAIN_END)]
        within.append(Xu.std(axis=0, ddof=1))
        means.append(Xu.mean(axis=0))
    w, b = np.mean(within, axis=0), np.std(means, axis=0, ddof=1)
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, w, 0.4, label="within-user SD", color=C_REAL)
    ax.bar(x + 0.2, b, 0.4, label="between-user SD (of user means)", color=C_MODEL)
    ax.set_xticks(x, [n.replace("log ", "") for n in names], rotation=70, fontsize=8)
    ax.set_ylabel("log-seconds")
    ax.set_title("Within- vs between-user spread (log features, training typings)")
    ax.legend()
    fig.tight_layout()
    return fig


def _ellipse(ax, mean, cov, i, j, color, nsd=2.0):
    S = cov[np.ix_([i, j], [i, j])]
    vals, vecs = np.linalg.eigh(S)
    ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w, h = 2 * nsd * np.sqrt(np.maximum(vals[::-1], 0))
    ax.add_patch(Ellipse((mean[i], mean[j]), w, h, angle=ang, fill=False, color=color, lw=2,
                         label=f"model {nsd:.0f}-SD ellipse"))


def user_samples(X, models, names, users, feats=(0, 1, 2), pair=(0, 11), n=200, title=""):
    """Real typings vs samples from p(x|u): marginals + 2-feature scatter with ellipse."""
    fig, ax = plt.subplots(len(users), len(feats) + 1, figsize=(4 * (len(feats) + 1), 3.2 * len(users)))
    ax = np.atleast_2d(ax)
    for r, u in enumerate(users):
        m = models[u]
        real = X[U.rows(u, 1, U.TRAIN_END)]
        smp = m.sample(n, rng=u)
        for c, k in enumerate(feats):
            bins = np.histogram_bin_edges(np.r_[real[:, k], smp[:, k]], 30)
            ax[r, c].hist(real[:, k], bins, alpha=0.6, density=True, color=C_REAL, label="real")
            ax[r, c].hist(smp[:, k], bins, alpha=0.6, density=True, color=C_MODEL, label="samples")
            ax[r, c].set_title(f"user {u}: {names[k]}", fontsize=9)
        i, j = pair
        a = ax[r, -1]
        a.scatter(real[:, i], real[:, j], s=8, color=C_REAL, label="real")
        a.scatter(smp[:, i], smp[:, j], s=8, color=C_MODEL, alpha=0.5, label="samples")
        cov = model_cov(m)
        mean = getattr(m, "mean_", real.mean(axis=0))
        if cov is not None:
            _ellipse(a, mean, cov, i, j, "k")
        a.set_xlabel(names[i], fontsize=8)
        a.set_ylabel(names[j], fontsize=8)
        a.set_title(f"user {u}", fontsize=9)
    ax[0, 0].legend(fontsize=8)
    ax[0, -1].legend(fontsize=8)
    fig.suptitle(title or "Real typings vs samples from p(x|u)")
    fig.tight_layout()
    return fig


def population_samples(X, user_models, names, feats=(0, 11), seed=0):
    """Pooled real training data vs (a) one pooled Gaussian, (b) mixture of user models."""
    rng = np.random.default_rng(seed)
    tr = U.rows(np.arange(U.N_USERS), 1, U.TRAIN_END)
    real = X[tr]
    mu, cov = real.mean(axis=0), np.cov(real, rowvar=False)
    pooled = rng.multivariate_normal(mu, cov, size=len(real))
    mix = np.vstack([user_models[u].sample(U.TRAIN_END, rng=int(rng.integers(1 << 30)))
                     for u in sorted(user_models)])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for c, k in enumerate(feats):
        bins = np.histogram_bin_edges(real[:, k], 60)
        ax[c].hist(real[:, k], bins, density=True, alpha=0.5, color=C_REAL, label="pooled real")
        ax[c].hist(pooled[:, k], bins, density=True, histtype="step", lw=2, color="k", label="pooled Gaussian")
        ax[c].hist(mix[:, k], bins, density=True, histtype="step", lw=2, color=C_MODEL, label="mixture of users")
        ax[c].set_title(names[k], fontsize=9)
    ax[0].legend(fontsize=8)
    i, j = feats
    for a, s, t in ((ax[2], pooled, "pooled Gaussian samples"), (ax[3], mix, "mixture-of-users samples")):
        a.scatter(real[:, i], real[:, j], s=1, alpha=0.3, color=C_REAL, label="real")
        a.scatter(s[:, i], s[:, j], s=1, alpha=0.3, color=C_MODEL, label="samples")
        a.set_title(t, fontsize=9)
        a.set_xlabel(names[i], fontsize=8)
        a.set_ylabel(names[j], fontsize=8)
    fig.suptitle("Overall (population) model: samples vs pooled real data")
    fig.tight_layout()
    return fig


def det_curves(results: dict, title="DET (vertical average over users)"):
    """results: label -> EvalResult. FRR averaged over users at fixed FAR."""
    from scipy.stats import norm
    grid_far = np.logspace(-3, 0, 200)
    fig, ax = plt.subplots(figsize=(6, 6))
    for lab, r in results.items():
        # Scores are not comparable across users (each user has its own model), so
        # pooling them would be misleading. Instead: one DET per user, read off at a
        # common FAR grid, then average the FRRs ("vertical averaging").
        frrs = []
        for g, i in r.scores.values():
            _, far, frr = U.det_curve(g, i)
            o = np.argsort(far)
            frrs.append(np.interp(grid_far, far[o], frr[o]))
        f = np.clip(np.mean(frrs, axis=0), 1e-4, 1 - 1e-4)  # clip: probit axes can't show 0 or 1
        ax.plot(norm.ppf(np.clip(grid_far, 1e-4, 1 - 1e-4)), norm.ppf(f),
                label=f"{lab} (EER {r.eers.mean():.3f})")
    ticks = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.4])
    ax.set_xticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
    ax.plot([-3.5, 0], [-3.5, 0], "k:", lw=1)
    ax.set_xlabel("false accept rate")
    ax.set_ylabel("false reject rate")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def per_user_bars(eer_a, eer_b, label_a, label_b):
    o = np.argsort(eer_b)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    x = np.arange(len(o))
    ax.bar(x - 0.2, np.asarray(eer_b)[o], 0.4, label=label_b, color=C_REAL)
    ax.bar(x + 0.2, np.asarray(eer_a)[o], 0.4, label=label_a, color=C_MODEL)
    ax.set_xlabel("user (sorted by baseline EER)")
    ax.set_ylabel("EER")
    ax.set_title(f"Per-user EER: {label_a} vs {label_b}")
    ax.legend()
    fig.tight_layout()
    return fig


def learning_curves(histories: dict, title="RealNVP learning curves (validation run)"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for u, h in list(histories.items())[:8]:
        ax[0].plot(h[:, 0], h[:, 1], lw=1, label=f"user {u}")
        ax[1].plot(h[:, 0], h[:, 2], lw=1, label=f"user {u}")
    ax[0].set_title("train NLL (standardized space)")
    ax[1].set_title("validation NLL (typings 161-200)")
    for a in ax:
        a.set_xlabel("epoch")
    ax[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def enrollment_plot(df, title="Enrollment size"):
    """df columns: model, sampling, n, eer (per user/seed)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a, s in zip(ax, ("first", "stratified")):
        d = df[df.sampling == s]
        for mname, g in d.groupby("model"):
            m = g.groupby("n")["eer"].mean()
            a.plot(m.index, m.values, marker="o", label=mname)
        a.set_xscale("log")
        a.set_xticks(sorted(d.n.unique()), [str(v) for v in sorted(d.n.unique())])
        a.set_title({"first": "first n (one sitting)", "stratified": "stratified (n/4 per session, 5 seeds)"}[s])
        a.set_xlabel("enrollment typings n")
    ax[0].set_ylabel("mean EER (test: sessions 5-8)")
    ax[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
