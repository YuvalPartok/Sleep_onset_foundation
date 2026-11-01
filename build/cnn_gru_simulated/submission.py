"""Track 03 (Sleep Onset) submission: CNN + band powers -> GRU -> time-to-N2
distribution.

Each 5 s window goes through two encoders:
- a 1-D CNN on the raw signal, to learn waveform shapes (alpha/theta
  rhythms, slow eye movements on AF7/AF8, spindles, K-complexes);
- log band powers (delta, theta, alpha, sigma, beta, high/EMG), absolute and
  relative, the classic sleep-scoring markers.
Their features, plus the time elapsed since the recording started, feed a
GRU that keeps its state across the windows of a recording (streamed
evaluation: one window per ``predict`` call, ``reset_state`` per recording).

The head outputs a distribution over the time to the first N2 epoch, in
5 s bins (0, 5, ..., 595, 600 = capped). ``p[0]`` is the hazard (probability
that N2 starts within the next window) and the prediction is the
``q``-quantile of the distribution, with ``q`` tuned on held-out subjects.

Contract (see competition/tracks/sleep_onset/README.md):
- input  X: float tensor (B, C, T) on meta["device"], raw microvolts
- output  : (B,) float latencies in seconds to the first N2 epoch, in [0, 600]

This file must stay self-contained: only ``torch`` and packages from the
competition's requirements.txt may be imported, and no other local modules.
"""

import math
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver

CAP_S = 600.0
WEIGHTS_FILE = "weights.pt"

BIN_S = 5.0
N_BINS = int(CAP_S / BIN_S) + 1          # 0, 5, ..., 600 s
CLIP_UV = 500.0
BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 15.0), (15.0, 30.0),
         (30.0, 60.0))
# Metric ranges and severity weights (objective.py).
RANGE_EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)
RANGE_WEIGHTS = (10.0, 5.0, 3.0, 1.0)

CNN_CHANNELS = (32, 64, 96, 128)
BP_DIM = 64
GRU_HIDDEN = 128
GRU_LAYERS = 2
DROPOUT = 0.2

EPOCHS = 30
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
RECS_PER_BATCH = 16
CHUNK = 120                              # TBPTT length, in windows (10 min)
VAL_FRACTION = 0.15
LABEL_SIGMA = 1.5                        # soft-label width, in bins
L1_WEIGHT = 0.1
QUANTILES = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
SEED = 0


def preprocess(X):
    """Raw microvolts (B, C, T) -> per-window DC removed, artifacts clipped."""
    X = X - X.mean(dim=-1, keepdim=True)
    return X.clamp(-CLIP_UV, CLIP_UV)


class SleepOnsetCNNGRU(nn.Module):

    def __init__(self, n_chans, n_times, sfreq):
        super().__init__()
        self.n_chans, self.n_times, self.sfreq = n_chans, n_times, sfreq
        self.window_s = n_times / sfreq
        self.register_buffer("chan_std", torch.ones(1, n_chans, 1))
        n_bp = 2 * len(BANDS) * n_chans + n_chans
        self.register_buffer("bp_mean", torch.zeros(n_bp))
        self.register_buffer("bp_std", torch.ones(n_bp))
        self.register_buffer("quantile", torch.tensor(0.5))
        self.register_buffer(
            "bin_values", torch.arange(N_BINS, dtype=torch.float32) * BIN_S)

        # Band masks over the rFFT bins, from the actual sampling rate.
        freqs = torch.fft.rfftfreq(n_times, d=1.0 / sfreq)
        nyq = sfreq / 2
        masks = torch.stack([((freqs >= lo) & (freqs < min(hi, nyq))).float()
                             for lo, hi in BANDS])
        self.register_buffer("band_masks", masks, persistent=False)
        self.register_buffer("hann", torch.hann_window(n_times),
                             persistent=False)

        layers, in_ch = [], n_chans
        kernels = (max(3, int(sfreq // 2)) | 1, 9, 7, 5)
        for out_ch, k in zip(CNN_CHANNELS, kernels):
            layers += [nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False),
                       nn.BatchNorm1d(out_ch), nn.GELU(), nn.MaxPool1d(2)]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.bp_proj = nn.Sequential(nn.Linear(n_bp, BP_DIM), nn.GELU())
        self.in_proj = nn.Sequential(
            nn.Linear(2 * CNN_CHANNELS[-1] + BP_DIM + 1, GRU_HIDDEN),
            nn.GELU(), nn.Dropout(DROPOUT))
        self.gru = nn.GRU(GRU_HIDDEN, GRU_HIDDEN, GRU_LAYERS, batch_first=True,
                          dropout=DROPOUT)
        self.head = nn.Sequential(nn.Dropout(DROPOUT),
                                  nn.Linear(GRU_HIDDEN, N_BINS))
        self.reset_state()

    # --- per-window encoder ---------------------------------------------

    def band_powers(self, X):
        """Preprocessed (N, C, T) -> raw log band-power features (N, F)."""
        spec = torch.fft.rfft(X * self.hann, dim=-1).abs() ** 2
        power = torch.einsum("nct,bt->ncb", spec, self.band_masks)
        log_abs = torch.log(power + 1e-6)
        log_rel = torch.log(power / (power.sum(-1, keepdim=True) + 1e-6)
                            + 1e-6)
        theta_alpha = log_abs[..., 1] - log_abs[..., 2]
        return torch.cat([log_abs.flatten(1), log_rel.flatten(1),
                          theta_alpha], dim=1)

    def encode(self, X, elapsed_min):
        """Windows (N, C, T) + elapsed minutes (N,) -> GRU inputs (N, H)."""
        X = preprocess(X)
        bp = (self.band_powers(X) - self.bp_mean) / self.bp_std
        h = self.cnn(X / self.chan_std)
        cnn = torch.cat([h.mean(-1), h.std(-1)], dim=1)
        t = torch.log1p(elapsed_min).unsqueeze(1)
        return self.in_proj(torch.cat([cnn, self.bp_proj(bp), t], dim=1))

    # --- sequences ------------------------------------------------------

    def forward(self, X, start_index=0, h=None):
        """Sequences (B, L, C, T) of consecutive windows -> logits (B, L, K).

        ``start_index`` is the index of the first window in its recording
        (int or (B,) tensor), so chunks of a long recording get the right
        elapsed time; ``h`` is the GRU state carried from the previous chunk.
        """
        B, L = X.shape[:2]
        idx = torch.arange(L, device=X.device).float().unsqueeze(0)
        idx = idx + torch.as_tensor(start_index, device=X.device,
                                    dtype=torch.float32).reshape(-1, 1)
        elapsed = (idx.expand(B, L) * self.window_s / 60.0).reshape(-1)
        z = self.encode(X.reshape(B * L, *X.shape[2:]), elapsed)
        out, h = self.gru(z.reshape(B, L, -1), h)
        return self.head(out), h

    def to_seconds(self, logits):
        """Logits (..., K) -> the ``quantile`` of the distribution, seconds."""
        cdf = logits.softmax(-1).cumsum(-1)
        k = torch.searchsorted(cdf.contiguous(),
                               self.quantile.expand(cdf.shape[:-1] + (1,))
                               .contiguous())
        return self.bin_values[k.clamp_max(N_BINS - 1)].squeeze(-1)

    # --- streamed inference ---------------------------------------------

    def _step_logits(self, X):
        # The B windows of a call are consecutive windows of one recording
        # (B = 1 in streamed evaluation).
        logits, self._h = self(X.float().unsqueeze(0), self._n, self._h)
        self._n += X.shape[0]
        return logits[0]

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self.to_seconds(self._step_logits(X)).clamp(0.0, CAP_S)

    @torch.inference_mode()
    def predict_dist(self, X):
        """(seconds, p over 5 s bins, hazard) for analysis; advances state."""
        self.eval()
        logits = self._step_logits(X)
        p = logits.softmax(-1)
        return self.to_seconds(logits).clamp(0.0, CAP_S), p, p[..., 0]

    def reset_state(self):
        self._h = None
        self._n = 0


# --- training helpers (local only) ------------------------------------------


def _load_windows(dataset):
    """Read every window once, in dataset order, preprocessed, as float16."""
    from torch.utils.data import DataLoader
    Xs, ys, recs, onsets, subs = [], [], [], [], []
    t0 = time.time()
    for X, y, info in DataLoader(dataset, batch_size=512, shuffle=False):
        Xs.append(preprocess(X.float().cpu()).half())
        ys.append(y.float().cpu().reshape(-1))
        recs.append(np.asarray(info["record_id"]).reshape(-1))
        onsets.append(np.asarray(info["onset"], dtype=np.float64).reshape(-1))
        sid = info.get("subject_id", info["record_id"])
        subs.append(np.asarray(sid).reshape(-1))
    print(f"[CNN-GRU] loaded {sum(len(x) for x in Xs)} windows in "
          f"{time.time() - t0:.0f} s")
    return (torch.cat(Xs), torch.cat(ys), np.concatenate(recs),
            np.concatenate(onsets), np.concatenate(subs))


def _recordings(y, rec, onset, sub, rng):
    """Window indices of each recording, in time order, plus its subject.

    Data without real recordings (the Simulated set: one record of random
    windows) is cut into pseudo-recordings sorted by decreasing latency.
    """
    out = []
    if len(np.unique(rec)) < 4:
        perm = rng.permutation(len(y))
        for i, part in enumerate(np.array_split(perm, 10)):
            part = part[np.argsort(-y[part].numpy(), kind="stable")]
            out.append((part, i))
        return out
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        out.append((idx[np.argsort(onset[idx], kind="stable")],
                    int(sub[idx[0]])))
    return out


def _window_weights(y):
    """Per-window loss weights of one recording, mimicking its W-bMAE.

    Each non-empty metric range gets its severity weight split over its
    windows; the recording's weights sum to 1.
    """
    edges = np.asarray(RANGE_EDGES)
    r = np.clip(np.digitize(y, edges[1:-1]), 0, len(RANGE_WEIGHTS) - 1)
    w = np.zeros(len(y))
    present = np.unique(r)
    total = sum(RANGE_WEIGHTS[b] for b in present)
    for b in present:
        mask = r == b
        w[mask] = RANGE_WEIGHTS[b] / total / mask.sum()
    return torch.as_tensor(w, dtype=torch.float32)


def _soft_labels(y, device):
    """Targets (N,) seconds -> Gaussian-smoothed bin distributions (N, K)."""
    centers = torch.arange(N_BINS, device=device).float()
    pos = (y.clamp(0, CAP_S) / BIN_S).unsqueeze(-1)
    logits = -0.5 * ((centers - pos) / LABEL_SIGMA) ** 2
    return logits.softmax(-1)


def _augment(X):
    """Per-recording amplitude scaling, noise and channel dropout.

    X: (B, L, C, T), preprocessed microvolts.
    """
    B, _, C, _ = X.shape
    scale = torch.empty(B, 1, 1, 1, device=X.device).uniform_(0.7, 1.3)
    X = X * scale + torch.randn_like(X) * 2.0
    drop = torch.rand(B, device=X.device) < 0.1
    if drop.any():
        ch = torch.randint(0, C, (B,), device=X.device)
        mask = torch.ones(B, 1, C, 1, device=X.device)
        mask[drop, 0, ch[drop], 0] = 0.0
        X = X * mask
    return X


def _batch_tensor(X_all, recs, start, stop, device):
    """Pad the windows [start, stop) of each recording into (B, L, C, T)."""
    L = min(stop, max(len(idx) for idx in recs)) - start
    X = torch.zeros(len(recs), L, *X_all.shape[1:], dtype=torch.float16)
    mask = torch.zeros(len(recs), L, dtype=torch.bool)
    for b, idx in enumerate(recs):
        part = idx[start:stop]
        X[b, :len(part)] = X_all[part]
        mask[b, :len(part)] = True
    return X.to(device).float(), mask.to(device)


def _sequence_logits(model, X_all, idx, device):
    """Logits (L, K) of one whole recording, chunked, in eval mode."""
    out, h = [], None
    for s in range(0, len(idx), 4 * CHUNK):
        X, _ = _batch_tensor(X_all, [idx], s, s + 4 * CHUNK, device)
        logits, h = model(X, s, h)
        out.append(logits[0, :min(4 * CHUNK, len(idx) - s)])
    return torch.cat(out)


@torch.no_grad()
def _validate(model, X_all, y_all, recs, device):
    """W-bMAE per recording, averaged, for every candidate quantile."""
    from benchmark_utils.metrics import binned_mae
    model.eval()
    saved_q = model.quantile.clone()
    scores = {q: [] for q in QUANTILES}
    for idx, _sub in recs:
        logits = _sequence_logits(model, X_all, idx, device)
        y = y_all[idx].numpy()
        for q in QUANTILES:
            model.quantile.fill_(q)
            pred = model.to_seconds(logits).clamp(0, CAP_S).cpu().numpy()
            scores[q].append(binned_mae(y, pred, RANGE_EDGES, RANGE_WEIGHTS))
    model.quantile.copy_(saved_q)
    return {q: float(np.mean(s)) for q, s in scores.items()}


def _train_device(default):
    forced = os.environ.get("SLEEP_TRAIN_DEVICE")
    if forced:
        return forced
    if default == "cpu" and torch.backends.mps.is_available():
        return "mps"
    return default


class Solver(CompetSolver):

    name = "CNN-GRU-Hazard"

    def load_model(self, meta):
        model = SleepOnsetCNNGRU(meta["n_chans"], meta["n_times"],
                                 float(meta["sfreq"]))
        weights = meta["submission_dir"] / WEIGHTS_FILE
        # Inference (Codabench): load the shipped weights. A local training
        # run (train_loader set) starts from scratch instead.
        if weights.exists() and self.train_loader is None:
            model.load_state_dict(torch.load(
                weights, map_location=meta["device"], weights_only=True))
        return model.to(meta["device"]).eval()

    def fit(self, model, train_loader):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        X_all, y_all, rec, onset, sub = _load_windows(train_loader.dataset)
        recs = _recordings(y_all, rec, onset, sub, rng)

        # Hold out whole subjects for early stopping and the quantile.
        subjects = np.unique([s for _, s in recs])
        n_val = max(1, int(round(VAL_FRACTION * len(subjects))))
        val_subjects = set(rng.choice(subjects, n_val, replace=False))
        train_recs = [r for r in recs if r[1] not in val_subjects]
        val_recs = [r for r in recs if r[1] in val_subjects]
        print(f"[CNN-GRU] {len(train_recs)} train / {len(val_recs)} val "
              f"recordings ({len(subjects)} subjects)")
        weights = {id(idx): _window_weights(y_all[idx].numpy())
                   for idx, _ in recs}

        # Normalisation stats from the training windows.
        train_idx = np.concatenate([idx for idx, _ in train_recs])
        sample = X_all[rng.choice(train_idx, min(20000, len(train_idx)),
                                  replace=False)].float()
        model.chan_std.copy_(sample.std(dim=(0, 2)).clamp_min(1e-3)
                             .view(1, -1, 1))
        feats = model.band_powers(sample.to(model.chan_std.device))
        model.bp_mean.copy_(feats.mean(0))
        model.bp_std.copy_(feats.std(0).clamp_min(1e-3))

        out_device = model.chan_std.device
        device = _train_device(str(out_device))
        model.to(device)
        print(f"[CNN-GRU] training on {device}")

        opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(EPOCHS):
            t0 = time.time()
            model.train()
            order = rng.permutation(len(train_recs))
            # Batch recordings of similar length to limit padding.
            order = sorted(order, key=lambda i: len(train_recs[i][0])
                           + rng.uniform(0, 60))
            batches = [order[i:i + RECS_PER_BATCH]
                       for i in range(0, len(order), RECS_PER_BATCH)]
            rng.shuffle(batches)
            running, n_rec = 0.0, 0
            for batch in batches:
                idxs = [train_recs[i][0] for i in batch]
                max_len = max(len(idx) for idx in idxs)
                y = torch.zeros(len(idxs), max_len)
                w = torch.zeros(len(idxs), max_len)
                for b, idx in enumerate(idxs):
                    y[b, :len(idx)] = y_all[idx]
                    w[b, :len(idx)] = weights[id(idx)]
                y, w = y.to(device), w.to(device)
                h = None
                for s in range(0, max_len, CHUNK):
                    X, mask = _batch_tensor(X_all, idxs, s, s + CHUNK, device)
                    logits, h = model(_augment(X), s, h)
                    h = h.detach()
                    yc, wc = y[:, s:s + CHUNK], w[:, s:s + CHUNK] * mask
                    logp = logits.log_softmax(-1)
                    ce = -(_soft_labels(yc, device) * logp).sum(-1)
                    mean = (logp.exp() * model.bin_values).sum(-1)
                    l1 = (mean - yc.clamp(0, CAP_S)).abs() / CAP_S
                    loss = ((ce + L1_WEIGHT * l1) * wc).sum() / len(idxs)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    running += loss.item() * len(idxs)
                n_rec += len(idxs)
            sched.step()

            scores = _validate(model, X_all, y_all, val_recs, device)
            q, score = min(scores.items(), key=lambda kv: kv[1])
            print(f"[CNN-GRU] epoch {epoch + 1}/{EPOCHS} "
                  f"loss {running / n_rec:.3f} val W-bMAE {score:.1f} s "
                  f"(q={q}) {time.time() - t0:.0f} s")
            if score < best:
                best, bad = score, 0
                model.quantile.fill_(q)
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.to(out_device).eval()
        print(f"[CNN-GRU] best val W-bMAE {best:.1f} s, "
              f"q={model.quantile.item():.2f}")
        self._check_streaming(model, X_all, val_recs[0][0], out_device)

    @staticmethod
    @torch.no_grad()
    def _check_streaming(model, X_all, idx, device):
        """Window-by-window predict must match a whole-sequence forward."""
        idx = idx[:200]
        model.eval()
        seq = model.to_seconds(_sequence_logits(model, X_all, idx, device))
        model.reset_state()
        stream = torch.cat([model.predict(X_all[i:i + 1].float().to(device))
                            for i in idx])
        model.reset_state()
        diff = (seq.clamp(0, CAP_S) - stream).abs().max().item()
        print(f"[CNN-GRU] streaming check: max |seq - stream| = {diff:.4f} s")
        if diff > BIN_S:
            raise RuntimeError("streamed predictions differ from the "
                               "sequence forward pass")

    def save_model(self, model, path):
        torch.save(model.state_dict(), path / WEIGHTS_FILE)
