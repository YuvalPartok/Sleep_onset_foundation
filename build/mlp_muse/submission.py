"""Track 03 (Sleep Onset) submission: a basic MLP regressor.

Codabench only calls ``Solver.load_model(meta)`` and the returned model's
``predict(X)``; ``fit`` and ``save_model`` are local-training hooks used by
``benchopt run ... -o "Sleep-onset[training=True]"`` to produce ``weights.pt``.

Contract (see competition/tracks/sleep_onset/README.md):
- input  X: float tensor (B, C, T) on meta["device"], raw microvolts
- output  : (B,) float latencies in seconds to the first N2 epoch, in [0, 600]

This file must stay self-contained: only ``torch`` and packages from the
competition's requirements.txt may be imported, and no other local modules.
"""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

CAP_S = 600.0
WEIGHTS_FILE = "weights.pt"

HIDDEN = (256, 64)
DROPOUT = 0.2
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4


class SleepOnsetMLP(nn.Module):
    """(B, C, T) -> per-channel standardisation -> flatten -> MLP -> (B,).

    The per-channel mean/std are fitted on the training set and stored as
    buffers, so they ship inside ``weights.pt`` with the rest of the model.
    """

    def __init__(self, n_chans, n_times, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.register_buffer("chan_mean", torch.zeros(1, n_chans, 1))
        self.register_buffer("chan_std", torch.ones(1, n_chans, 1))
        layers, in_dim = [], n_chans * n_times
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        X = (X - self.chan_mean) / self.chan_std
        # Fraction of CAP_S in [0, 1].
        return torch.sigmoid(self.net(X.flatten(1))).squeeze(-1)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X.float()) * CAP_S  # (B,) seconds

    def reset_state(self):
        # Stateless: nothing to reset between streamed recordings.
        pass


class Solver(CompetSolver):

    name = "Sleep-MLP"

    def load_model(self, meta):
        model = SleepOnsetMLP(meta["n_chans"], meta["n_times"])
        weights = meta["submission_dir"] / WEIGHTS_FILE
        # Inference (Codabench): load the shipped weights. A local training
        # run (train_loader set) starts from scratch instead.
        if weights.exists() and self.train_loader is None:
            model.load_state_dict(torch.load(
                weights, map_location=meta["device"], weights_only=True))
        return model.to(meta["device"]).eval()

    def fit(self, model, train_loader):
        # Per-channel normalisation stats over the whole training set.
        total, total_sq, count = 0.0, 0.0, 0
        for X, _y, _info in train_loader:
            X = X.float()
            total = total + X.sum(dim=(0, 2))
            total_sq = total_sq + (X ** 2).sum(dim=(0, 2))
            count += X.shape[0] * X.shape[2]
        mean = total / count
        std = (total_sq / count - mean ** 2).clamp_min(1e-12).sqrt()
        model.chan_mean.copy_(mean.view(1, -1, 1))
        model.chan_std.copy_(std.view(1, -1, 1))

        opt = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # L1 matches the MAE-based track metric.
        loss_fn = nn.L1Loss()
        model.train()
        for epoch in range(EPOCHS):
            running, n = 0.0, 0
            for X, y, _info in train_loader:
                opt.zero_grad()
                loss = loss_fn(model(X.float()), y.float().clamp(0, CAP_S) / CAP_S)
                loss.backward()
                opt.step()
                running += loss.item() * len(y)
                n += len(y)
            print(f"[Sleep-MLP] epoch {epoch + 1}/{EPOCHS} "
                  f"train MAE {running / n * CAP_S:.1f} s")
        model.eval()

    def save_model(self, model, path):
        torch.save(model.state_dict(), path / WEIGHTS_FILE)
