"""No test typings (sessions 5-8, impostors' typings 1-5) in training, tuning or population."""
import numpy as np
import pytest

from src import utils as U
from src import models as M
from src.evaluation import ModelSpec, evaluate

T = U.row_typing
USR = U.row_user


@pytest.mark.parametrize("u", range(U.N_USERS))
def test_official_and_val_splits(u):
    te = U.make_split(u, "test")
    assert set(USR(te.train)) == {u} and T(te.train).max() == 200
    assert set(USR(te.genuine)) == {u} and T(te.genuine).min() == 201
    assert u not in set(USR(te.impostor)) and T(te.impostor).max() == 5 and len(te.impostor) == 250
    assert np.intersect1d(te.population, te.impostor).size == 0
    assert T(te.population).min() == 6 and T(te.population).max() == 200
    U.check_no_leak("test", u, te.train, te.population)

    va = U.make_split(u, "val")
    every = np.concatenate([va.train, va.genuine, va.impostor, va.population])
    assert T(every).max() <= 200, "validation touches sessions 5-8"
    assert np.all(T(every[USR(every) != u]) > 5), "validation touches impostor test typings"
    assert np.intersect1d(va.population, va.impostor).size == 0
    assert np.intersect1d(va.train, va.genuine).size == 0
    U.check_no_leak("val", u, va.train, va.population)


def test_openset_folds():
    folds = U.openset_folds()
    assert sorted(np.concatenate(folds).tolist()) == list(range(51))
    assert all(len(f) == 17 for f in folds)
    for f, held in enumerate(folds):
        for u in np.setdiff1d(np.arange(51), held):
            sp = U.make_split(u, "openset", fold=f)
            assert set(USR(sp.impostor)) == set(held)
            assert not np.isin(USR(sp.population), held).any()
            assert u not in set(USR(sp.population))
            U.check_no_leak("openset", u, sp.train, sp.population, fold=f)
        pop = U.shared_population_rows("openset", f)
        assert not np.isin(USR(pop), held).any() and T(pop).max() <= 200


def test_shared_population():
    pt = U.shared_population_rows("test")
    assert T(pt).min() == 6 and T(pt).max() == 200
    pv = U.shared_population_rows("val")
    assert T(pv).min() == 11 and T(pv).max() == 160
    for u in range(51):
        U.check_no_leak("test", u, U.rows(u, 1, 200), pt)
        U.check_no_leak("val", u, U.rows(u, 1, 160), pv)


@pytest.mark.parametrize("n", [8, 12, 24, 48, 100, 200])
def test_enrollment_rows(n):
    for u in (0, 25, 50):
        f = U.enrollment_rows(u, n, "first")
        assert T(f).tolist() == list(range(1, n + 1))
        s = U.enrollment_rows(u, n, "stratified", seed=3)
        sess = (T(s) - 1) // 50 + 1
        assert sorted(np.bincount(sess)[1:].tolist()) == [n // 4] * 4
        assert np.array_equal(s, U.enrollment_rows(u, n, "stratified", seed=3)), "same draw for all models"
        U.check_no_leak("test", u, s, ())


def test_leak_check_catches_leaks():
    with pytest.raises(AssertionError):
        U.check_no_leak("test", 0, U.rows(0, 1, 201))                      # session 5
    with pytest.raises(AssertionError):
        U.check_no_leak("test", 0, U.rows(0, 1, 200), U.rows([1, 2], 1, 200))  # impostor test rows
    with pytest.raises(AssertionError):
        U.check_no_leak("val", 0, U.rows(0, 1, 170))                       # val genuine
    with pytest.raises(AssertionError):
        U.check_no_leak("val", 0, U.rows(0, 1, 160), U.rows([3], 6, 200))  # val impostors
    with pytest.raises(AssertionError):
        U.check_no_leak("openset", 1, U.rows(1, 1, 200), U.rows(U.openset_folds()[0], 6, 50), fold=0)


def test_population_model_runs_through_leak_check(X21):
    spec = ModelSpec("tied", M.TiedGaussian, {"reg": 1e-3}, population=True)
    for proto in ("val", "test"):
        r = evaluate(X21, spec, proto, users=[0, 1, 2])
        assert np.isfinite(r.eers).all()
