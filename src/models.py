"""Anomaly detectors and generative models for keystroke verification.

Every model implements ``fit(X) -> self`` and ``score(X) -> array`` (higher =
more genuine); density models also implement ``sample(n)`` and set
``is_density = True`` so that ``score`` is a log-density in the input space
(comparable test log-likelihoods). Models that need a population take it in
the constructor as a ``Population`` (never inside ``fit``), so the caller
controls exactly which rows are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture

from .utils import Standardizer

LOG2PI = np.log(2 * np.pi)


@dataclass
class Population:
    """Other users' training typings: X (m, d) and integer group labels."""
    X: np.ndarray
    groups: np.ndarray

    def mean(self):
        return self.X.mean(axis=0)

    def within_cov(self):
        """Pooled within-user covariance (each user centred on its own mean)."""
        d = self.X.shape[1]
        S, dof = np.zeros((d, d)), 0
        for g in np.unique(self.groups):
            Xg = self.X[self.groups == g]
            C = Xg - Xg.mean(axis=0)
            S += C.T @ C
            dof += len(Xg) - 1
        return S / dof


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ridge(cov, reg):
    """cov + reg * mean(diag) * I (scale-free ridge)."""
    d = cov.shape[0]
    return cov + reg * max(np.trace(cov) / d, 1e-12) * np.eye(d)


def mvn_logpdf(X, mu, cov):
    L = np.linalg.cholesky(cov)
    Z = np.linalg.solve(L, (X - mu).T)
    logdet = 2 * np.log(np.diag(L)).sum()
    return -0.5 * (Z ** 2).sum(axis=0) - 0.5 * (len(mu) * LOG2PI + logdet)


def mvt_logpdf(X, mu, scale, nu):
    d = len(mu)
    L = np.linalg.cholesky(scale)
    Z = np.linalg.solve(L, (X - mu).T)
    delta = (Z ** 2).sum(axis=0)
    logdet = 2 * np.log(np.diag(L)).sum()
    return (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet - 0.5 * (nu + d) * np.log1p(delta / nu))


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


class Detector:
    is_density = False
    needs_population = False
    name = "detector"

    def get_params(self) -> dict:
        """Arrays/scalars needed to reuse the fitted model (saved as npz)."""
        return {}


# --------------------------------------------------------------------------
# Published distance detectors (Killourhy & Maxion 2009)
# --------------------------------------------------------------------------
class Euclidean(Detector):
    name = "Euclidean"

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        return self

    def score(self, X):
        return -np.sum((X - self.mean_) ** 2, axis=1)

    def get_params(self):
        return {"mean": self.mean_}


class Manhattan(Euclidean):
    name = "Manhattan"

    def score(self, X):
        return -np.sum(np.abs(X - self.mean_), axis=1)


class ScaledManhattan(Detector):
    """Manhattan distance scaled by each feature's mean absolute deviation."""
    name = "Scaled Manhattan"

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.mad_ = np.abs(X - self.mean_).mean(axis=0) + 1e-12
        return self

    def score(self, X):
        return -np.sum(np.abs(X - self.mean_) / self.mad_, axis=1)

    def get_params(self):
        return {"mean": self.mean_, "mad": self.mad_}


class Mahalanobis(Detector):
    """Squared Mahalanobis distance with a pseudo-inverse (31 raw features are rank 21)."""
    name = "Mahalanobis"

    def __init__(self, rcond: float = 1e-8):
        self.rcond = rcond

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.prec_ = np.linalg.pinv(np.cov(X, rowvar=False), rcond=self.rcond, hermitian=True)
        return self

    def score(self, X):
        C = X - self.mean_
        return -np.einsum("ij,jk,ik->i", C, self.prec_, C)

    def get_params(self):
        return {"mean": self.mean_, "precision": self.prec_}


# --------------------------------------------------------------------------
# Gaussian family (densities on the log-feature space)
# --------------------------------------------------------------------------
class _GaussianBase(Detector):
    is_density = True

    def score(self, X):
        return mvn_logpdf(X, self.mean_, self.cov_)

    def sample(self, n, rng=None):
        return _rng(rng).multivariate_normal(self.mean_, self.cov_, size=n)

    def get_params(self):
        return {"mean": self.mean_, "cov": self.cov_}


class DiagGaussian(_GaussianBase):
    name = "Gaussian diag"

    def __init__(self, reg: float = 1e-3):
        self.reg = reg

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        v = X.var(axis=0, ddof=1) if len(X) > 1 else np.ones(X.shape[1])
        self.cov_ = np.diag(v + self.reg * v.mean())
        return self


class FullGaussian(_GaussianBase):
    name = "Gaussian full"

    def __init__(self, reg: float = 1e-2):
        self.reg = reg

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.cov_ = _ridge(np.cov(X, rowvar=False), self.reg)
        return self


class LedoitWolfGaussian(_GaussianBase):
    """Shrinkage covariance; shrinkage='auto' uses the Ledoit-Wolf estimate."""
    name = "Gaussian Ledoit-Wolf"

    def __init__(self, shrinkage="auto"):
        self.shrinkage = shrinkage

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        if self.shrinkage == "auto":
            lw = LedoitWolf().fit(X)
            self.cov_, self.shrinkage_ = lw.covariance_, lw.shrinkage_
        else:
            S = np.cov(X, rowvar=False, bias=True)
            mu = np.trace(S) / S.shape[0]
            a = float(self.shrinkage)
            self.cov_, self.shrinkage_ = (1 - a) * S + a * mu * np.eye(S.shape[0]), a
        return self


class TiedGaussian(_GaussianBase):
    """Per-user mean, covariance shared: pooled within-user cov of the population."""
    name = "Gaussian tied"
    needs_population = True

    def __init__(self, population: Population, reg: float = 1e-3):
        self.population, self.reg = population, reg
        self._cov = _ridge(population.within_cov(), reg)

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.cov_ = self._cov
        return self


class BayesGaussian(Detector):
    """Normal-inverse-Wishart prior from the population, posterior predictive (Student-t).

    Prior: mean m0 = population mean, E[Sigma] = pooled within-user covariance,
    strength kappa0 (on the mean) and nu0 = d + 1 + nu_extra (on the covariance).
    """
    name = "Gaussian Bayes/MAP"
    is_density = True
    needs_population = True

    def __init__(self, population: Population, kappa0: float = 0.1, nu_extra: float = 20):
        self.population, self.kappa0, self.nu_extra = population, kappa0, nu_extra
        self.m0_ = population.mean()
        self.Sw_ = population.within_cov()

    def fit(self, X):
        n, d = X.shape
        k0 = self.kappa0
        nu0 = d + 1 + self.nu_extra
        psi0 = (nu0 - d - 1) * self.Sw_
        xbar = X.mean(axis=0)
        C = X - xbar
        kn, nun = k0 + n, nu0 + n
        self.mean_ = (k0 * self.m0_ + n * xbar) / kn
        dm = (xbar - self.m0_)[:, None]
        psin = psi0 + C.T @ C + (k0 * n / kn) * (dm @ dm.T)
        self.df_ = nun - d + 1
        self.scale_ = psin * (kn + 1) / (kn * self.df_)
        self.map_cov_ = psin / (nun + d + 1)  # MAP covariance (for plots)
        return self

    def score(self, X):
        return mvt_logpdf(X, self.mean_, self.scale_, self.df_)

    def sample(self, n, rng=None):
        return _sample_t(_rng(rng), self.mean_, self.scale_, self.df_, n)

    def get_params(self):
        return {"mean": self.mean_, "scale": self.scale_, "df": self.df_, "m0": self.m0_,
                "within_cov": self.Sw_}


def _sample_t(rng, mu, scale, nu, n):
    z = rng.multivariate_normal(np.zeros(len(mu)), scale, size=n)
    g = rng.chisquare(nu, size=n) / nu
    return mu + z / np.sqrt(g)[:, None]


class StudentT(Detector):
    """Multivariate Student-t fitted by EM with fixed degrees of freedom."""
    name = "Student-t"
    is_density = True

    def __init__(self, nu: float = 5.0, reg: float = 1e-2, max_iter: int = 200, tol: float = 1e-6):
        self.nu, self.reg, self.max_iter, self.tol = nu, reg, max_iter, tol

    def fit(self, X):
        n, d = X.shape
        mu = X.mean(axis=0)
        S = _ridge(np.cov(X, rowvar=False), self.reg)
        base = self.reg * np.trace(np.cov(X, rowvar=False)) / d
        prev = -np.inf
        for it in range(self.max_iter):
            L = np.linalg.cholesky(S)
            delta = (np.linalg.solve(L, (X - mu).T) ** 2).sum(axis=0)
            w = (self.nu + d) / (self.nu + delta)
            mu = (w[:, None] * X).sum(axis=0) / w.sum()
            C = X - mu
            S = (w[:, None] * C).T @ C / n + base * np.eye(d)
            ll = mvt_logpdf(X, mu, S, self.nu).sum()
            if abs(ll - prev) < self.tol * max(1.0, abs(ll)):
                break
            prev = ll
        self.mean_, self.cov_, self.n_iter_ = mu, S, it + 1
        return self

    def score(self, X):
        return mvt_logpdf(X, self.mean_, self.cov_, self.nu)

    def sample(self, n, rng=None):
        return _sample_t(_rng(rng), self.mean_, self.cov_, self.nu, n)

    def get_params(self):
        return {"mean": self.mean_, "scale": self.cov_, "nu": self.nu}


class GMM(Detector):
    """Full-covariance GMM on per-user standardized data, K in 1..k_max by BIC.

    K is capped at 2 when fewer than 50 training typings. Densities are mapped
    back to the (log-)feature space with the standardization Jacobian.
    """
    name = "GMM"
    is_density = True

    def __init__(self, k_max: int = 3, reg_covar: float = 3e-3, n_init: int = 3, seed: int = 0):
        self.k_max, self.reg_covar, self.n_init, self.seed = k_max, reg_covar, n_init, seed

    def fit(self, X):
        self.std_ = Standardizer().fit(X)
        Z = self.std_.transform(X)
        kmax = min(self.k_max, 2) if len(X) < 50 else self.k_max
        self.bic_ = {}
        best = None
        for k in range(1, kmax + 1):
            g = GaussianMixture(k, covariance_type="full", reg_covar=self.reg_covar,
                                n_init=self.n_init, random_state=self.seed).fit(Z)
            self.bic_[k] = float(g.bic(Z))
            if best is None or self.bic_[k] < self.bic_[best.n_components]:
                best = g
        self.gmm_ = best
        self.k_ = best.n_components
        return self

    def predict(self, X):
        return self.gmm_.predict(self.std_.transform(X))

    def score(self, X):
        return self.gmm_.score_samples(self.std_.transform(X)) + self.std_.log_jacobian

    def sample(self, n, rng=None):
        self.gmm_.random_state = int(_rng(rng).integers(2 ** 31))
        Z, _ = self.gmm_.sample(n)
        return self.std_.inverse_transform(Z)

    def get_params(self):
        return {"weights": self.gmm_.weights_, "means_z": self.gmm_.means_,
                "covariances_z": self.gmm_.covariances_, "std_mean": self.std_.mean_,
                "std_scale": self.std_.scale_, "k": self.k_}


class Laplace(Detector):
    """Product of independent Laplace marginals: the density version of Scaled Manhattan.

    loc='median' is the MLE; loc='mean' matches Scaled Manhattan exactly
    (score = -sum|x-mu|/b + const), so the two rank typings identically.
    """
    name = "Laplace"
    is_density = True

    def __init__(self, loc: str = "median"):
        self.loc = loc

    def fit(self, X):
        self.mean_ = np.median(X, axis=0) if self.loc == "median" else X.mean(axis=0)
        self.b_ = np.abs(X - self.mean_).mean(axis=0) + 1e-6
        return self

    def score(self, X):
        return -np.sum(np.log(2 * self.b_) + np.abs(X - self.mean_) / self.b_, axis=1)

    def sample(self, n, rng=None):
        return _rng(rng).laplace(self.mean_, self.b_, size=(n, len(self.mean_)))

    def get_params(self):
        return {"loc": self.mean_, "scale": self.b_}


def model_cov(model):
    """A covariance-like matrix for drawing ellipses (None if not applicable)."""
    for attr in ("cov_", "map_cov_"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if isinstance(model, Laplace):
        return np.diag(2 * model.b_ ** 2)
    return None
