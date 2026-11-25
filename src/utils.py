"""Data loading, feature sets, evaluation splits and metrics for the CMU
keystroke benchmark (Killourhy & Maxion 2009, DSL-StrongPasswordData).

Conventions
-----------
* One typing = one row = one data point. Rows are sorted by
  (subject, sessionIndex, rep); ``typing`` = 1..400 is the chronological index
  inside a user (sessions are 50 typings each).
* Scores: higher = more genuine. Distances are negated.
* Every split returns *row positions* into the sorted DataFrame so that leakage
  can be checked mechanically (see ``check_no_leak``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILENAME = "DSL-StrongPasswordData.csv"
DATA_MD5 = "470235f96568f28f9ea0da62234ec857"
DATA_URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
DATA_DIRS = ("data", "datasets")

N_USERS, N_SESSIONS, REPS = 51, 8, 50
N_TYPINGS = N_SESSIONS * REPS  # 400
TRAIN_END = 200                # typings 1-200 = sessions 1-4
VAL_TRAIN_END = 160            # validation: fit 1-160, validate 161-200
N_IMP_TEST = 5                 # impostor test = other users' typings 1-5
IMP_VAL = (6, 10)              # impostor validation = other users' typings 6-10
N_FOLDS = 3                    # open-set folds of 17 users


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_data_file() -> Path:
    for d in DATA_DIRS:
        p = REPO_ROOT / d / DATA_FILENAME
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{DATA_FILENAME} not found in {[str(REPO_ROOT / d) for d in DATA_DIRS]}. "
        f"Download it from {DATA_URL} (MD5 {DATA_MD5}) into data/."
    )


def load_data(path: str | Path | None = None, check_md5: bool = True) -> pd.DataFrame:
    """Load the benchmark sorted by (subject, session, rep) with a ``typing`` column."""
    path = Path(path) if path is not None else find_data_file()
    if check_md5:
        got = md5sum(path)
        if got != DATA_MD5:
            raise ValueError(f"MD5 mismatch for {path}: {got} != {DATA_MD5}")
    df = pd.read_csv(path)
    df = df.sort_values(["subject", "sessionIndex", "rep"], kind="stable").reset_index(drop=True)
    df["typing"] = (df["sessionIndex"] - 1) * REPS + df["rep"]
    users = sorted(df["subject"].unique())
    assert len(users) == N_USERS and len(df) == N_USERS * N_TYPINGS, "unexpected dataset shape"
    expected = np.tile(np.arange(1, N_TYPINGS + 1), N_USERS)
    assert np.array_equal(df["typing"].to_numpy(), expected), "missing/duplicate typings"
    return df


def users_of(df: pd.DataFrame) -> list[str]:
    return sorted(df["subject"].unique())


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def timing_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.split(".")[0] in ("H", "DD", "UD")]


def columns_for(df: pd.DataFrame, kind: str) -> list[str]:
    cols = timing_columns(df)
    if kind == "31":
        return cols
    if kind == "21":  # 11 H + 10 DD, original order; UD = DD - H is redundant
        return [c for c in cols if c.startswith(("H.", "DD."))]
    raise ValueError(kind)


FEATURE_SETS = ("raw31", "log21", "rhythm20", "rhythm20+mean")


def make_features(df: pd.DataFrame, feature_set: str = "log21") -> tuple[np.ndarray, list[str]]:
    """Return (X, names).

    raw31          31 raw timings (H, DD, UD) - only for published reproduction
    raw21          11 H + 10 DD raw seconds
    log21          log of 11 H + 10 DD
    rhythm20       log21 minus each typing's mean log-time, last feature dropped
                   (the centred vector sums to zero -> singular covariance)
    rhythm20+mean  rhythm20 plus the mean log-time (overall speed) as a feature
    """
    if feature_set == "raw31":
        cols = columns_for(df, "31")
        return df[cols].to_numpy(float), cols
    cols = columns_for(df, "21")
    raw = df[cols].to_numpy(float)
    if feature_set == "raw21":
        return raw, cols
    if np.any(raw <= 0):
        raise ValueError("non-positive H/DD timing, cannot log-transform")
    L = np.log(raw)
    if feature_set == "log21":
        return L, [f"log {c}" for c in cols]
    m = L.mean(axis=1, keepdims=True)
    R = (L - m)[:, :-1]
    names = [f"rhythm {c}" for c in cols[:-1]]
    if feature_set == "rhythm20":
        return R, names
    if feature_set == "rhythm20+mean":
        return np.hstack([R, m]), names + ["mean log-time"]
    raise ValueError(feature_set)


class Standardizer:
    """Per-feature z-scoring. Fit on training data only."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, Z):
        return Z * self.scale_ + self.mean_

    @property
    def log_jacobian(self) -> float:
        """log|dz/dx|: add to a density in z-space to get the density in x-space."""
        return float(-np.sum(np.log(self.scale_)))


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def rows(user_idx, lo: int, hi: int) -> np.ndarray:
    """Row positions of typings lo..hi (1-based, inclusive) for one or many users."""
    user_idx = np.atleast_1d(np.asarray(user_idx, int))
    t = np.arange(lo, hi + 1)
    return (user_idx[:, None] * N_TYPINGS + (t - 1)[None, :]).ravel()


def row_typing(r):
    return np.asarray(r) % N_TYPINGS + 1


def row_user(r):
    return np.asarray(r) // N_TYPINGS


def openset_folds(seed: int = 0) -> list[np.ndarray]:
    """3 folds of 17 user indices (fixed random permutation)."""
    perm = np.random.default_rng(seed).permutation(N_USERS)
    return [np.sort(perm[i::N_FOLDS]) for i in range(N_FOLDS)]


@dataclass
class Split:
    """Everything one user's evaluation touches, as row positions."""
    user: int
    protocol: str
    train: np.ndarray
    genuine: np.ndarray
    impostor: np.ndarray
    population: np.ndarray = field(default_factory=lambda: np.array([], int))
    fold: int | None = None


def enrollment_rows(u: int, n: int | None = None, sampling: str = "first",
                    seed: int = 0, end: int = TRAIN_END) -> np.ndarray:
    """Training rows for user u.

    sampling='first'      : typings 1..n (one sitting when n <= 50)
    sampling='stratified' : n/4 random typings from each of sessions 1-4.
                            The draw depends only on (seed, user, n), so all
                            models see the same typings.
    """
    if n is None:
        n = end
    if sampling == "first":
        return rows(u, 1, n)
    if sampling == "stratified":
        assert n % 4 == 0 and n <= TRAIN_END
        rng = np.random.default_rng([seed, u, n])
        per = n // 4
        out = [rows(u, s * REPS + 1, (s + 1) * REPS)[np.sort(rng.choice(REPS, per, replace=False))]
               for s in range(4)]
        return np.concatenate(out)
    raise ValueError(sampling)


def make_split(u: int, protocol: str = "test", fold: int | None = None,
               train: np.ndarray | None = None, folds: list | None = None) -> Split:
    """Build the split for user index u.

    protocol='val'     : fit 1-160, genuine 161-200, impostors = others' 6-10,
                         population = others' 11-200.
    protocol='test'    : fit 1-200, genuine 201-400, impostors = others' 1-5,
                         population = others' 6-200 (impostor test typings and
                         sessions 5-8 excluded).
    protocol='openset' : held-out fold = impostors only (their typings 1-5);
                         population = the other non-held-out users' 1-200.
    ``train`` overrides the default training rows (enrollment experiments).
    """
    others = np.array([v for v in range(N_USERS) if v != u])
    if protocol == "val":
        tr = rows(u, 1, VAL_TRAIN_END) if train is None else train
        return Split(u, protocol, tr, rows(u, VAL_TRAIN_END + 1, TRAIN_END),
                     rows(others, *IMP_VAL), rows(others, IMP_VAL[1] + 1, TRAIN_END))
    if protocol == "test":
        tr = rows(u, 1, TRAIN_END) if train is None else train
        return Split(u, protocol, tr, rows(u, TRAIN_END + 1, N_TYPINGS),
                     rows(others, 1, N_IMP_TEST), rows(others, N_IMP_TEST + 1, TRAIN_END))
    if protocol == "openset":
        folds = folds if folds is not None else openset_folds()
        held = folds[fold]
        assert u not in held, "target user must not be in the held-out fold"
        pop_users = np.array([v for v in range(N_USERS) if v != u and v not in held])
        tr = rows(u, 1, TRAIN_END) if train is None else train
        return Split(u, protocol, tr, rows(u, TRAIN_END + 1, N_TYPINGS),
                     rows(held, 1, N_IMP_TEST), rows(pop_users, 1, TRAIN_END), fold)
    raise ValueError(protocol)


def fold_of(u: int, folds=None) -> int:
    folds = folds if folds is not None else openset_folds()
    return next(f for f, members in enumerate(folds) if u in members)


def pretrain_population_rows(protocol: str, u: int, fold: int | None = None,
                             folds=None) -> tuple[tuple, np.ndarray]:
    """Population for models pretrained once and shared by many users (the flow).

    Returns (cache_key, rows). To keep pretraining affordable (3 runs, not 51)
    but never include the target user:
      val     : users outside u's fold, typings 11-200 (no 1-5 test, no 6-10 val impostors)
      test    : users outside u's fold, typings 6-200  (no impostor test typings)
      openset : users outside the held-out fold, typings 1-200. This includes the
                target's own 200 training typings (legitimate training data).
    """
    folds = folds if folds is not None else openset_folds()
    allu = np.arange(N_USERS)
    if protocol in ("val", "test"):
        f = fold_of(u, folds)
        keep = np.setdiff1d(allu, folds[f])
        lo = IMP_VAL[1] + 1 if protocol == "val" else N_IMP_TEST + 1
        return (protocol, f), rows(keep, lo, TRAIN_END)
    if protocol == "openset":
        keep = np.setdiff1d(allu, folds[fold])
        return (protocol, fold), rows(keep, 1, TRAIN_END)
    raise ValueError(protocol)


def check_no_leak(protocol: str, user: int, train_rows, population_rows=(), fold=None,
                  folds=None) -> None:
    """Raise AssertionError if any fitted/population row is a test typing.

    Always forbidden: anyone's sessions 5-8 (typings 201-400).
    test/val: other users' impostor test typings 1-5 are forbidden in population.
    val:      also the validation genuine (161-200 of target) and validation
              impostors (others' 6-10) must not be fitted.
    openset:  nothing from the held-out fold may be used.
    """
    tr = np.asarray(train_rows, int)
    pop = np.asarray(population_rows, int)
    used = np.concatenate([tr, pop])
    assert np.all(row_typing(used) <= TRAIN_END), "sessions 5-8 used for fitting"
    if protocol in ("test", "val"):
        others = row_user(pop) != user
        assert not np.any(others & (row_typing(pop) <= N_IMP_TEST)), "impostor test typings in population"
        assert not np.any((row_user(tr) != user) & (row_typing(tr) <= N_IMP_TEST))
    if protocol == "val":
        assert not np.any((row_user(used) == user) & (row_typing(used) > VAL_TRAIN_END)), \
            "validation genuine typings fitted"
        t = row_typing(used)
        assert not np.any((row_user(used) != user) & (t >= IMP_VAL[0]) & (t <= IMP_VAL[1])), \
            "validation impostor typings fitted"
    if protocol == "openset":
        folds = folds if folds is not None else openset_folds()
        assert not np.any(np.isin(row_user(used), folds[fold])), "held-out fold used"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def det_curve(genuine, impostor):
    """FRR and FAR for every threshold (accept if score >= threshold)."""
    g = np.sort(np.asarray(genuine, float))
    i = np.sort(np.asarray(impostor, float))
    thr = np.unique(np.concatenate([g, i, [np.inf]]))
    frr = np.searchsorted(g, thr, side="left") / len(g)        # genuine below thr
    far = 1.0 - np.searchsorted(i, thr, side="left") / len(i)  # impostor >= thr
    return thr, far, frr


def eer(genuine, impostor) -> float:
    """Equal error rate, linearly interpolated at the FAR/FRR crossing."""
    _, far, frr = det_curve(genuine, impostor)
    d = far - frr                    # decreasing in threshold
    k = int(np.argmax(d <= 0))       # first threshold with FAR <= FRR
    if k == 0:
        return float((far[0] + frr[0]) / 2)
    d0, d1 = d[k - 1], d[k]
    w = d0 / (d0 - d1) if d0 != d1 else 0.5
    return float(far[k - 1] + w * (far[k] - far[k - 1]))


def eer_threshold(genuine, impostor) -> float:
    """Threshold at the EER point (accept if score >= threshold)."""
    thr, far, frr = det_curve(genuine, impostor)
    return float(thr[int(np.argmin(np.abs(far - frr)))])


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(per_user_eer) -> dict:
    v = np.asarray(per_user_eer, float)
    lo, hi = bootstrap_ci(v)
    return {"mean_eer": float(v.mean()), "sd_eer": float(v.std(ddof=1)),
            "ci95_low": lo, "ci95_high": hi, "n_users": int(len(v))}


def paired_test(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison over the same users: diff = a - b (negative = a better)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    return {"mean_diff": float(d.mean()), "diff_ci95_low": lo, "diff_ci95_high": hi,
            "wilcoxon_p": p, "n_better": int((d < 0).sum()), "n_worse": int((d > 0).sum())}
