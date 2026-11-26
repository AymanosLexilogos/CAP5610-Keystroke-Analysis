"""Small RealNVP normalizing flow (PyTorch, CPU) for per-user keystroke densities.

Per user: standardize -> affine coupling layers -> elementwise affine -> N(0, I).
Optionally pretrained once on a population (see utils.pretrain_population_rows)
and fine-tuned per user. Early stopping on validation NLL; when no validation set
is given (final refit), training runs for a fixed number of epochs chosen on
validation. Seeds are fixed per (seed, user-data) so runs are reproducible.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from . import utils as U
from .models import Detector

DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
LOG2PI = float(np.log(2 * np.pi))


class Coupling(nn.Module):
    """Affine coupling layer.

    Features where mask == 1 pass through unchanged and condition a small MLP;
    the MLP outputs a log-scale s and shift t for the other features:
        y = x_m + (1 - m) * (x * exp(s(x_m)) + t(x_m)),   log|det J| = sum(s).
    Because s, t depend only on the unchanged half, the inverse is exact and cheap.
    """

    def __init__(self, d: int, mask: torch.Tensor, hidden: int):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2 * d))
        nn.init.zeros_(self.net[-1].weight)  # start as the identity map
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, xm):
        s, t = self.net(xm).chunk(2, dim=1)
        inv = 1 - self.mask
        # 2*tanh(s/2) bounds the log-scale to (-2, 2): keeps training stable on ~200 points
        return 2.0 * torch.tanh(s / 2.0) * inv, t * inv

    def forward(self, x):
        xm = x * self.mask
        s, t = self._st(xm)
        return xm + (1 - self.mask) * (x * torch.exp(s) + t), s.sum(dim=1)

    def inverse(self, z):
        zm = z * self.mask
        s, t = self._st(zm)
        return zm + (1 - self.mask) * ((z - t) * torch.exp(-s))


class RealNVP(nn.Module):
    """Stack of coupling layers followed by a learnable elementwise affine map.

    x --coupling x n_layers--> h --(h * exp(log_a) + b)--> z ~ N(0, I).
    The final affine layer lets the flow represent a diagonal Gaussian from the
    start (the couplings are initialised to the identity).
    """

    def __init__(self, d: int, n_layers: int = 4, hidden: int = 32):
        super().__init__()
        idx = torch.arange(d)
        masks = []
        for i in range(n_layers):
            # Mask pattern per layer: even/odd, odd/even, first/second half, second/first half, ...
            # so every feature is transformed and every pair of features can interact.
            m = (idx % 2 == 0) if (i // 2) % 2 == 0 else (idx < d // 2)
            masks.append((m if i % 2 == 0 else ~m).float())
        self.layers = nn.ModuleList(Coupling(d, m, hidden) for m in masks)
        self.log_a = nn.Parameter(torch.zeros(d))  # final elementwise log-scale
        self.b = nn.Parameter(torch.zeros(d))      # final elementwise shift

    def forward(self, x):
        logdet = torch.zeros(x.shape[0], dtype=x.dtype)
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        z = x * torch.exp(self.log_a) + self.b
        return z, logdet + self.log_a.sum()

    def log_prob(self, x):
        """Change of variables: log p(x) = log N(z; 0, I) + log|det dz/dx|."""
        z, logdet = self(x)
        return -0.5 * (z ** 2).sum(dim=1) - 0.5 * z.shape[1] * LOG2PI + logdet

    @torch.no_grad()
    def sample(self, n, generator=None):
        d = self.b.shape[0]
        z = torch.randn(n, d, generator=generator, dtype=self.b.dtype)
        x = (z - self.b) * torch.exp(-self.log_a)
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


def _t(a):
    """numpy -> float32 CPU tensor."""
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)


def train_flow(model, Ztr, Zval=None, epochs=300, lr=3e-3, weight_decay=1e-3,
               batch_size=None, patience=40, noise=0.0, seed=0):
    """Adam on NLL. With Zval: early stopping, returns the best state.
    History rows: (epoch, train_nll, val_nll)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Ztr = _t(Ztr)
    Zv = _t(Zval) if Zval is not None else None
    n = len(Ztr)
    bs = n if batch_size is None or batch_size >= n else batch_size
    hist, best, best_state, best_ep, bad = [], np.inf, None, epochs, 0
    for ep in range(1, epochs + 1):
        # --- one pass over the training data (full batch for per-user fits) ---
        model.train()
        perm = torch.randperm(n, generator=g)
        for k in range(0, n, bs):
            xb = Ztr[perm[k:k + bs]]
            if noise > 0:  # Gaussian noise augmentation: a simple regulariser for tiny datasets
                xb = xb + noise * torch.randn(xb.shape, generator=g)
            loss = -model.log_prob(xb).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

        # --- monitor clean train / validation NLL ---
        model.eval()
        with torch.no_grad():
            tr = float(-model.log_prob(Ztr).mean())
            va = float(-model.log_prob(Zv).mean()) if Zv is not None else np.nan
        hist.append((ep, tr, va))

        # --- early stopping: keep the best validation state, stop after `patience` bad epochs ---
        if Zv is not None:
            if va < best - 1e-4:
                best, best_state, best_ep, bad = va, copy.deepcopy(model.state_dict()), ep, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return np.array(hist), best_ep


def pretrain(Xpop, n_layers=4, hidden=32, epochs=150, lr=2e-3, weight_decay=1e-4,
             batch_size=256, seed=0):
    """Pretrain on population data (standardized with population statistics).
    10% of the population rows are held out for early stopping."""
    torch.manual_seed(seed)
    std = U.Standardizer().fit(Xpop)
    Z = std.transform(Xpop)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Z))
    nv = max(1, len(Z) // 10)
    model = RealNVP(Z.shape[1], n_layers, hidden).to(DEVICE)
    hist, best_ep = train_flow(model, Z[perm[nv:]], Z[perm[:nv]], epochs=epochs, lr=lr,
                               weight_decay=weight_decay, batch_size=batch_size,
                               patience=15, seed=seed)
    return {"state_dict": copy.deepcopy(model.state_dict()), "std_mean": std.mean_,
            "std_scale": std.scale_, "n_layers": n_layers, "hidden": hidden,
            "history": hist, "best_epoch": best_ep, "n_rows": len(Xpop)}


class FlowDetector(Detector):
    """Per-user RealNVP. ``pretrained`` (from ``pretrain``) -> fine-tune."""
    name = "RealNVP"
    is_density = True

    def __init__(self, n_layers=4, hidden=32, lr=3e-3, weight_decay=1e-3, max_epochs=400,
                 epochs=None, patience=40, noise=0.0, seed=0, pretrained=None, X_val=None):
        self.n_layers, self.hidden, self.lr, self.weight_decay = n_layers, hidden, lr, weight_decay
        self.max_epochs, self.epochs, self.patience, self.noise = max_epochs, epochs, patience, noise
        self.seed, self.pretrained, self.X_val = seed, pretrained, X_val

    def fit(self, X):
        # Seed derived from the training data: different per user, identical across reruns.
        seed = self.seed + int(abs(X[:, 0].sum()) * 1e4) % 100000
        torch.manual_seed(seed)
        if self.pretrained is not None:
            # Fine-tune: start from the population flow and keep the population's
            # standardization, so the pretrained weights stay meaningful.
            p = self.pretrained
            self.std_ = U.Standardizer()
            self.std_.mean_, self.std_.scale_ = p["std_mean"], p["std_scale"]
            self.model_ = RealNVP(X.shape[1], p["n_layers"], p["hidden"]).to(DEVICE)
            self.model_.load_state_dict(p["state_dict"])
        else:
            # From scratch: standardize with this user's training statistics only.
            self.std_ = U.Standardizer().fit(X)
            self.model_ = RealNVP(X.shape[1], self.n_layers, self.hidden).to(DEVICE)
        # With validation data (tuning): train up to max_epochs and stop early.
        # Without (final refit): train exactly `epochs`, the value chosen on validation.
        Zv = self.std_.transform(self.X_val) if self.X_val is not None else None
        ep = self.max_epochs if Zv is not None or self.epochs is None else int(self.epochs)
        self.history_, self.best_epoch_ = train_flow(
            self.model_, self.std_.transform(X), Zv, epochs=ep, lr=self.lr,
            weight_decay=self.weight_decay, patience=self.patience, noise=self.noise, seed=seed)
        self.model_.eval()
        return self

    def score(self, X):
        """Log-density in the log-feature space (the Jacobian of the standardization is
        added), so values are comparable with the Gaussian models."""
        with torch.no_grad():
            lp = self.model_.log_prob(_t(self.std_.transform(X))).double().numpy()
        return lp + self.std_.log_jacobian

    def sample(self, n, rng=None):
        seed = rng if isinstance(rng, (int, np.integer)) else 0
        g = torch.Generator().manual_seed(int(seed))
        return self.std_.inverse_transform(self.model_.sample(n, g).double().numpy())

    def get_state(self):
        """Everything needed to reuse the model: weights + standardization + architecture."""
        return {"state_dict": self.model_.state_dict(), "std_mean": self.std_.mean_,
                "std_scale": self.std_.scale_, "n_layers": len(self.model_.layers),
                "hidden": self.model_.layers[0].net[0].out_features,
                "best_epoch": self.best_epoch_, "history": self.history_}


# Pretraining is the expensive part, and every pretrained grid config uses the same
# population, so results are shared across FlowSpec instances.
# key: (protocol/fold key, pretrain params, data fingerprint) -> output of pretrain()
_PRETRAIN_CACHE: dict = {}


def pretrained_models():
    """All pretrained population flows built so far (for saving)."""
    return {f"{k[0][0]}_fold{k[0][1]}": v for k, v in _PRETRAIN_CACHE.items()}


class FlowSpec:
    """Model spec for the flow (works with evaluation.evaluate).

    pretrain=True: one pretraining per (protocol, fold) on
    utils.pretrain_population_rows, cached, then per-user fine-tuning.
    """
    population = False

    def __init__(self, params=None, pretrain=False, pretrain_params=None, name="RealNVP"):
        self.params = dict(params or {})
        self.pretrain = pretrain
        self.pretrain_params = dict(pretrain_params or {})
        self.name = name + (" (pretrained)" if pretrain else "")

    def label(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"

    def population_rows(self, split):
        if not self.pretrain:
            return np.array([], int)
        return U.pretrain_population_rows(split.protocol, split.user, split.fold)[1]

    def build(self, ctx):
        pre = None
        if self.pretrain:
            key, rows = U.pretrain_population_rows(ctx["protocol"], ctx["user"], ctx["fold"])
            Xp = ctx["X"][rows]
            # The shape/sum fingerprint keeps caches separate across feature sets.
            ck = (key, tuple(sorted(self.pretrain_params.items())), Xp.shape, float(Xp.sum()))
            if ck not in _PRETRAIN_CACHE:
                _PRETRAIN_CACHE[ck] = pretrain(Xp, **self.pretrain_params)
            pre = _PRETRAIN_CACHE[ck]
        return FlowDetector(**self.params, pretrained=pre, X_val=ctx.get("X_val"))
