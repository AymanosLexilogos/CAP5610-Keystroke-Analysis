import numpy as np

from src import utils as U
from src.evaluation import evaluate
from src.flow import FlowSpec


def test_flow_scratch_sane_and_deterministic(X21):
    a = evaluate(X21, FlowSpec({"hidden": 16, "epochs": 30}), "test", users=[0, 7], keep_models=True)
    b = evaluate(X21, FlowSpec({"hidden": 16, "epochs": 30}), "test", users=[0, 7])
    ll = a.per_user["ll_test"].to_numpy()
    assert np.isfinite(ll).all() and (np.abs(ll) < 1e3).all()
    assert np.allclose(a.eers, b.eers) and np.allclose(ll, b.per_user["ll_test"])
    s = a.models[0].sample(20, rng=1)
    assert s.shape == (20, 21) and np.isfinite(s).all()


def test_flow_pretrained_passes_leak_check(X21):
    sp = FlowSpec({"hidden": 16, "lr": 1e-3}, pretrain=True,
                  pretrain_params={"hidden": 16, "epochs": 3})
    r = evaluate(X21, sp, "val", users=[0, 1])  # early stopping on validation
    assert np.isfinite(r.per_user["ll_test"]).all()
    enrolled = np.setdiff1d(np.arange(51), U.openset_folds()[0])[:2]
    r = evaluate(X21, sp, "openset", fold=0, users=enrolled)
    assert np.isfinite(r.eers).all()
