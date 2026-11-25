"""Killourhy & Maxion (2009) results on 31 raw features, within 0.002 EER."""
import numpy as np
import pytest

from src import models as M
from src.evaluation import ModelSpec, evaluate

PUBLISHED = {"Euclidean": (M.Euclidean, 0.1706), "Manhattan": (M.Manhattan, 0.1529),
             "Mahalanobis": (M.Mahalanobis, 0.1101), "Scaled Manhattan": (M.ScaledManhattan, 0.0962)}


@pytest.mark.parametrize("name", list(PUBLISHED))
def test_published_eer(X31, name):
    cls, target = PUBLISHED[name]
    r = evaluate(X31, ModelSpec(name, cls), "test")
    assert len(r.per_user) == 51
    assert abs(r.eers.mean() - target) <= 0.002, f"{name}: {r.eers.mean():.4f} vs {target}"


def test_baseline_log21(X21):
    r = evaluate(X21, ModelSpec("Scaled Manhattan", M.ScaledManhattan), "test")
    assert abs(r.eers.mean() - 0.085) < 0.005


def test_eer_edge_cases():
    from src.utils import eer
    assert eer([1, 2, 3], [-1, -2]) == 0.0
    assert eer([-1, -2], [1, 2, 3]) == 1.0
    assert abs(eer([0, 1, 2, 3], [0.5, 1.5, 2.5, 3.5]) - 0.5) < 0.26
