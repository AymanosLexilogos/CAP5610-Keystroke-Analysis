"""Test log-likelihoods are finite and in a sane range (|values| in the millions = collapsed fit)."""
import numpy as np
import pytest

from src import models as M
from src.evaluation import ModelSpec, evaluate

SPECS = [
    ModelSpec("diag", M.DiagGaussian, {"reg": 1e-3}),
    ModelSpec("full", M.FullGaussian, {"reg": 1e-2}),
    ModelSpec("lw", M.LedoitWolfGaussian, {"shrinkage": "auto"}),
    ModelSpec("tied", M.TiedGaussian, {"reg": 1e-3}, population=True),
    ModelSpec("bayes", M.BayesGaussian, {"kappa0": 1.0, "nu_extra": 20}, population=True),
    ModelSpec("t", M.StudentT, {"nu": 5, "reg": 1e-2}),
    ModelSpec("gmm", M.GMM, {"reg_covar": 3e-3}),
    ModelSpec("laplace", M.Laplace, {"loc": "median"}),
]
USERS = list(range(0, 51, 5))


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_test_loglik_sane(X21, spec):
    r = evaluate(X21, spec, "test", users=USERS)
    ll = r.per_user["ll_test"].to_numpy()
    assert np.isfinite(ll).all()
    assert (ll > -1e3).all() and (ll < 1e3).all(), ll
    assert np.median(ll) > -200
    assert r.eers.mean() < 0.3


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_sample_shape(X21, spec):
    r = evaluate(X21, spec, "test", users=[0], keep_models=True)
    s = r.models[0].sample(50, rng=0)
    assert s.shape == (50, X21.shape[1]) and np.isfinite(s).all()


def test_laplace_mean_equals_scaled_manhattan_ranking(X21):
    a = evaluate(X21, ModelSpec("l", M.Laplace, {"loc": "mean"}), "test").eers
    b = evaluate(X21, ModelSpec("sm", M.ScaledManhattan), "test").eers
    assert np.allclose(a, b, atol=1e-6)
