"""Track 03 (Sleep Onset) submission: an ensemble of spectral-feature -> GRU
models that each output a time-to-N2 distribution.

Each 5 s window is summarised by hand-crafted spectral features:
- log band powers (delta, theta, alpha, sigma, beta, high/EMG), absolute and
  relative, and the theta/alpha ratio: the classic sleep-scoring markers;
- with ``feats=2`` (default) also drowsiness markers: slow (< 0.6 Hz) power
  per channel and on the AF7-AF8 derivation (slow eye movements), 0.6-4 Hz
  power on AF7-AF8, spectral entropy and 95 % spectral edge per channel;
- with ``feats=3`` also movement / arousal markers: fraction of clipped
  samples, log peak amplitude and log line length (movements and arousals
  pull a drowsy sleeper back to wake).
Optionally (``use_cnn``, off by default since the bottleneck experiments
showed it does not help with this much data) a 1-D CNN on the raw signal
adds learned waveform features.
These, plus the time elapsed since the recording started, feed a GRU that
keeps its state across the windows of a recording (streamed evaluation: one
window per ``predict`` call, ``reset_state`` per recording).

Per-sleeper adaptation (``adapt``): the features are also given relative to
the recording's running mean, so each window is judged against the same
sleeper's history; with ``feats=2`` also relative to a ~1 min EMA (the
short-term change) and the EMA relative to the running mean (the trend);
with ``feats=3`` also a decaying peak of each movement marker's deviation
from the running mean over ~1 and ~5 min, so a recent movement stays
visible for a few minutes.
All statistics are causal, over the windows seen so far.

Preprocessing v2 (``SLEEP_PREP=2``): the band powers skip 48-52 Hz (the
recordings carry 50 Hz mains noise, which otherwise leaks into the 30-60 Hz
band and every relative power) and use a Welch estimate (2 s Hann segments,
1 s hop) instead of one 5 s periodogram, for less variance.

Autoencoder features (``SLEEP_AE``): a denoising MLP autoencoder is first
trained, without labels, on the training windows' log spectra (32
log-spaced bins, 0.4-47 Hz, on the 4 channels and AF7-AF8); its frozen
encoder's 32-dim code replaces the hand-crafted features (``ae=1``) or is
added to them (``ae=2``). The per-recording adaptation applies to it as to
the hand-crafted features. Only the encoder ships with the weights.

Staging output (``SLEEP_STAGE_WEIGHT`` > 0, training only): a second head on
the GRU predicts YASA's sleep stage of each window's 30 s epoch (wake /
drowsy = N1 + REM / N2+), soft labels distilled from YASA run offline on the
full recordings (scripts/yasa_check.py -> scripts/make_yasa_teacher.py ->
data/yasa_teacher.npz). It only shapes the GRU's state during training;
predictions come from the time-to-N2 head alone.

The head outputs a distribution over the time to the first N2 epoch, in
5 s bins (0, 5, ..., 595, 600 = capped). The ensemble averages the members'
distributions; ``p[0]`` is the hazard (probability that N2 starts within the
next window) and the prediction is the ``q``-quantile of the distribution,
with ``q`` tuned on held-out subjects.

Contract (see competition/tracks/sleep_onset/README.md):
- input  X: float tensor (B, C, T) on meta["device"], raw microvolts
- output  : (B,) float latencies in seconds to the first N2 epoch, in [0, 600]

This file must stay self-contained: only ``torch`` and packages from the
competition's requirements.txt may be imported, and no other local modules.
"""

import os
import time

import numpy as np
import torch
from torch import nn

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

# Training-time choices; inference reads them back from weights.pt.
N_SEEDS = int(os.environ.get("SLEEP_N_SEEDS", 5))
ADAPT = os.environ.get("SLEEP_ADAPT", "1") == "1"
# Raw-signal CNN branch (off: it did not help in the bottleneck experiments)
# and the feature set (1 = band powers only, 2 = + drowsiness and trend).
USE_CNN = os.environ.get("SLEEP_USE_CNN", "0") == "1"
FEATS = int(os.environ.get("SLEEP_FEATS", 2))
# Also train on NeuralBench's own validation sleepers of the public Muse
# training release, which the harness's train loader leaves out.
EXTRA_DATA = os.environ.get("SLEEP_EXTRA_DATA", "1") == "1"
# Diagnostics (see colab notebook): train on a fraction of the training
# sleepers, change the members' seed. The validation sleepers do not depend
# on any of these settings, so runs stay comparable.
TRAIN_FRAC = float(os.environ.get("SLEEP_TRAIN_FRAC", 1.0))
MEMBER_SEED = int(os.environ.get("SLEEP_SEED", 0))
EMA_S = 60.0                             # short-term EMA time constant
N_MOVE = 3                               # movement markers (feats >= 3)
PEAK_S = (60.0, 300.0)                   # decaying-peak horizons (feats >= 3)
# Staging output: loss weight (0 = off) and the YASA teacher labels.
STAGE_WEIGHT = float(os.environ.get("SLEEP_STAGE_WEIGHT", 0.0))
PREP = int(os.environ.get("SLEEP_PREP", 1))     # 2: notch + Welch
AE_MODE = int(os.environ.get("SLEEP_AE", 0))     # 1 replace, 2 add
AE_DIM = 32                              # autoencoder code size
AE_HIDDEN = 128
AE_EPOCHS = int(os.environ.get("SLEEP_AE_EPOCHS", 20))
LINE_HZ = (48.0, 52.0)                   # mains band skipped with PREP >= 2
TEACHER_FILE = os.environ.get("SLEEP_TEACHER", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
    "yasa_teacher.npz"))
N_STAGES = 3                             # wake, drowsy (N1 + REM), N2+
# Cross-validated all-data training (final builds): with SLEEP_CV_FOLDS = K
# > 0 the ensemble has K members; member k trains on every recording except
# the sleepers of fold k, early-stops on fold k, and the quantile is chosen
# on the out-of-fold predictions of all recordings. SLEEP_USE_TEST=1 also
# trains on the 40 public warm-up test recordings (the warm-up is over; the
# local test score is then meaningless).
CV_FOLDS = int(os.environ.get("SLEEP_CV_FOLDS", 0))
USE_TEST = os.environ.get("SLEEP_USE_TEST", "0") == "1"

EPOCHS = int(os.environ.get("SLEEP_EPOCHS", 30))
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
RECS_PER_BATCH = 16
CHUNK = 120                              # TBPTT length, in windows (10 min)
VAL_FRACTION = 0.15
LABEL_SIGMA = 1.5                        # soft-label width, in bins
L1_WEIGHT = 0.1
QUANTILES = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
SEED = 0                                 # validation split seed (fixed)


def preprocess(X):
    """Raw microvolts (..., C, T) -> per-window DC removed, artifacts clipped."""
    X = X - X.mean(dim=-1, keepdim=True)
    return X.clamp(-CLIP_UV, CLIP_UV)


def quantile_seconds(p, q):
    """Distributions (..., K) over the 5 s bins -> their q-quantile, seconds."""
    cdf = p.cumsum(-1).contiguous()
    k = torch.searchsorted(cdf, torch.full(cdf.shape[:-1] + (1,), float(q),
                                           device=cdf.device))
    return (k.clamp_max(N_BINS - 1).squeeze(-1) * BIN_S).float()


class SleepOnsetCNNGRU(nn.Module):
    """One member: (B, L, C, T) sequences -> logits (B, L, K) over bins."""

    def __init__(self, n_chans, n_times, sfreq, adapt=ADAPT, use_cnn=USE_CNN,
                 feats=FEATS, stage=STAGE_WEIGHT > 0, prep=PREP, ae=AE_MODE):
        super().__init__()
        self.prep, self.ae = int(prep), int(ae)
        self.register_buffer("prep_v", torch.tensor(self.prep))
        self.register_buffer("ae_mode", torch.tensor(self.ae))
        self.n_chans, self.n_times, self.sfreq = n_chans, n_times, sfreq
        self.window_s = n_times / sfreq
        self.use_adapt = bool(adapt)
        self.use_cnn = bool(use_cnn)
        self.n_feats = int(feats)
        self.ema_alpha = min(1.0, self.window_s / EMA_S)
        self.register_buffer("adapt", torch.tensor(self.use_adapt))
        self.register_buffer("cnn_on", torch.tensor(self.use_cnn))
        self.register_buffer("feats", torch.tensor(self.n_feats))
        self.register_buffer("chan_std", torch.ones(n_chans))
        self.register_buffer(
            "bin_values", torch.arange(N_BINS, dtype=torch.float32) * BIN_S)

        # Band masks over the rFFT bins, from the actual sampling rate.
        freqs = torch.fft.rfftfreq(n_times, d=1.0 / sfreq)
        nyq = sfreq / 2
        # Band powers: one 5 s periodogram (prep 1) or Welch over 2 s Hann
        # segments with a 1 s hop, skipping the mains band (prep 2).
        self.seg = int(round(2 * sfreq)) if self.prep >= 2 else n_times
        self.hop = int(round(sfreq))
        if self.seg > n_times:
            self.seg = n_times
        bfreqs = torch.fft.rfftfreq(self.seg, d=1.0 / sfreq)
        keep = torch.ones_like(bfreqs, dtype=torch.bool)
        if self.prep >= 2:
            keep = (bfreqs < LINE_HZ[0]) | (bfreqs > LINE_HZ[1])
        masks = torch.stack([((bfreqs >= lo) & (bfreqs < min(hi, nyq))
                              & keep).float() for lo, hi in BANDS])
        self.register_buffer("band_masks", masks, persistent=False)
        self.register_buffer("seg_hann", torch.hann_window(self.seg),
                             persistent=False)
        self.register_buffer("hann", torch.hann_window(n_times),
                             persistent=False)
        self.register_buffer("freqs", freqs, persistent=False)
        self.register_buffer("slow_mask", ((freqs > 0) & (freqs < 0.6)).float(),
                             persistent=False)
        self.register_buffer("eog_mask", ((freqs >= 0.6) & (freqs < 4.0))
                             .float(), persistent=False)
        self.register_buffer("broad_mask", ((freqs >= 0.5)
                                            & (freqs < min(30.0, nyq))),
                             persistent=False)
        if self.ae:
            # Log-spaced spectrum bins, at least 0.4 Hz wide, up to 47 Hz.
            edges, e = [0.4], 0.4
            while e < 47.0:
                e = min(47.0, max(e * 1.15, e + 0.4))
                edges.append(e)
            bins = torch.stack([((freqs >= lo) & (freqs < hi)).float()
                                for lo, hi in zip(edges[:-1], edges[1:])], 1)
            self.register_buffer("ae_bins", bins / bins.sum(0).clamp_min(1),
                                 persistent=False)
            ae_in = (n_chans + 1) * bins.shape[1]
            self.register_buffer("spec_mean", torch.zeros(ae_in))
            self.register_buffer("spec_std", torch.ones(ae_in))
            self.ae_enc = nn.Sequential(nn.Linear(ae_in, AE_HIDDEN), nn.GELU(),
                                        nn.Linear(AE_HIDDEN, AE_DIM))
            self.ae_enc.requires_grad_(False)   # trained beforehand, frozen
        else:
            self.ae_enc = None
        n_bp = self.band_powers(torch.zeros(1, n_chans, n_times)).shape[1]
        self.register_buffer("bp_mean", torch.zeros(n_bp))
        self.register_buffer("bp_std", torch.ones(n_bp))

        layers, in_ch = [], n_chans
        kernels = (max(3, int(sfreq // 2)) | 1, 9, 7, 5)
        for out_ch, k in zip(CNN_CHANNELS, kernels):
            layers += [nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False),
                       nn.BatchNorm1d(out_ch), nn.GELU(), nn.MaxPool1d(2)]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers) if self.use_cnn else None
        self.use_ema = self.n_feats >= 2 or self.ae > 0
        self.use_peaks = self.n_feats >= 3 and self.ae != 1
        bp_in = n_bp * ((4 if self.use_ema else 2) if self.use_adapt
                        else 1)
        if self.use_adapt and self.use_peaks:
            bp_in += N_MOVE * len(PEAK_S)
        # Decaying peaks lose 1 normalised unit per PEAK_S seconds.
        self.peak_decay = tuple(self.window_s / h for h in PEAK_S)
        self.bp_proj = nn.Sequential(nn.Linear(bp_in, BP_DIM), nn.GELU())
        cnn_dim = 2 * CNN_CHANNELS[-1] if self.use_cnn else 0
        self.in_proj = nn.Sequential(
            nn.Linear(cnn_dim + BP_DIM + 1, GRU_HIDDEN),
            nn.GELU(), nn.Dropout(DROPOUT))
        self.gru = nn.GRU(GRU_HIDDEN, GRU_HIDDEN, GRU_LAYERS, batch_first=True,
                          dropout=DROPOUT)
        self.head = nn.Sequential(nn.Dropout(DROPOUT),
                                  nn.Linear(GRU_HIDDEN, N_BINS))
        # Training-only auxiliary head (see module docstring).
        self.stage_head = (nn.Sequential(nn.Dropout(DROPOUT),
                                         nn.Linear(GRU_HIDDEN, N_STAGES))
                           if stage else None)
        self.reset_state()

    def ae_input(self, X):
        """Preprocessed (N, C, T) -> log binned spectra (N, (C + 1) * nb)."""
        a, b = (1, 2) if X.shape[1] >= 3 else (0, X.shape[1] - 1)
        X = torch.cat([X, (X[:, a] - X[:, b]).unsqueeze(1)], dim=1)
        spec = torch.fft.rfft(X * self.hann, dim=-1).abs() ** 2
        return torch.log(spec @ self.ae_bins + 1e-6).flatten(1)

    def band_powers(self, X):
        """Preprocessed (N, C, T) -> raw features (N, F): the autoencoder
        code and/or the hand-crafted features."""
        if not self.ae:
            return self.hand_features(X)
        code = self.ae_enc((self.ae_input(X) - self.spec_mean)
                           / self.spec_std)
        if self.ae == 1:
            return code
        return torch.cat([code, self.hand_features(X)], dim=1)

    def hand_features(self, X):
        """Preprocessed (N, C, T) -> hand-crafted log band-power features."""
        spec = torch.fft.rfft(X * self.hann, dim=-1).abs() ** 2
        if self.seg < X.shape[-1]:
            segs = X.unfold(-1, self.seg, self.hop)
            bspec = (torch.fft.rfft(segs * self.seg_hann, dim=-1).abs() ** 2
                     ).mean(-2)
        else:
            bspec = spec
        power = torch.einsum("nct,bt->ncb", bspec, self.band_masks)
        log_abs = torch.log(power + 1e-6)
        log_rel = torch.log(power / (power.sum(-1, keepdim=True) + 1e-6)
                            + 1e-6)
        theta_alpha = log_abs[..., 1] - log_abs[..., 2]
        out = [log_abs.flatten(1), log_rel.flatten(1), theta_alpha]
        if self.n_feats >= 2:
            # Drowsiness: slow eye movements (< 0.6 Hz, strongest on the
            # AF7-AF8 derivation, where horizontal eye movements are of
            # opposite polarity), spectral flattening (entropy, edge).
            out.append(torch.log(spec @ self.slow_mask + 1e-6))
            a, b = (1, 2) if X.shape[1] >= 3 else (0, X.shape[1] - 1)
            eog = torch.fft.rfft((X[:, a] - X[:, b]) * self.hann,
                                 dim=-1).abs() ** 2
            out.append(torch.stack([torch.log(eog @ self.slow_mask + 1e-6),
                                    torch.log(eog @ self.eog_mask + 1e-6)],
                                   dim=1))
            p = spec[..., self.broad_mask]
            p = p / (p.sum(-1, keepdim=True) + 1e-12)
            out.append(-(p * torch.log(p + 1e-12)).sum(-1)
                       / np.log(p.shape[-1]))
            edge = (p.cumsum(-1) >= 0.95).float().argmax(-1)
            out.append(self.freqs[self.broad_mask][edge] / 30.0)
        if self.n_feats >= 3:
            # Movement / arousal markers; the last N_MOVE features. X is
            # already clipped at CLIP_UV, so "clipped" = at that limit.
            a = X.abs()
            out.append(torch.stack([
                (a >= 0.99 * CLIP_UV).float().mean((1, 2)),
                torch.log(a.amax((1, 2)) + 1e-3),
                torch.log((X[..., 1:] - X[..., :-1]).abs().sum(-1).mean(-1)
                          + 1e-3)], dim=1))
        return torch.cat(out, dim=1)

    def encode(self, X, start_index, sums):
        """Sequences (B, L, C, T) -> GRU inputs (B, L, H), new running sums.

        ``sums`` = (sum of band powers (B, F), sum of channel variances
        (B, C), window count (B, 1), short-term EMA of the band powers
        (B, F), decaying movement peaks (list of (B, N_MOVE) or None)) over
        the earlier windows of each recording, or None at its start.
        """
        B, L, C, T = X.shape
        Xf = preprocess(X.reshape(B * L, C, T))
        bp = ((self.band_powers(Xf) - self.bp_mean) / self.bp_std
              ).reshape(B, L, -1)
        if self.use_adapt:
            var = Xf.var(-1).reshape(B, L, C)
            if sums is None:
                sums = (bp.new_zeros(B, bp.shape[-1]), var.new_zeros(B, C),
                        var.new_zeros(B, 1), None, None)
            count = sums[2].unsqueeze(1) + torch.arange(
                1, L + 1, device=X.device).float().view(1, L, 1)
            cum_bp = sums[0].unsqueeze(1) + bp.cumsum(1)
            cum_var = sums[1].unsqueeze(1) + var.cumsum(1)
            mean = cum_bp / count
            rel = bp - mean
            scale = (cum_var / count).sqrt().clamp_min(
                0.25 * self.chan_std.view(1, 1, C))
            parts_bp = [bp, rel]
            ema = sums[3]
            if self.use_ema:
                emas = []
                for i in range(L):
                    ema = (bp[:, i] if ema is None
                           else ema + self.ema_alpha * (bp[:, i] - ema))
                    emas.append(ema)
                emas = torch.stack(emas, dim=1)
                parts_bp += [bp - emas, emas - mean]
            peaks = sums[4] if len(sums) > 4 else None
            if self.use_peaks:
                # Decaying peak of each movement marker's deviation from the
                # recording's running mean: jumps to a new movement, then
                # fades by 1 normalised unit per horizon.
                move = rel[..., -N_MOVE:]
                out_peaks = [[] for _ in self.peak_decay]
                for i in range(L):
                    x = move[:, i]
                    peaks = [x if peaks is None else
                             torch.maximum(x, pk - d)
                             for pk, d in zip(peaks or [None] * len(
                                 self.peak_decay), self.peak_decay)]
                    for j, pk in enumerate(peaks):
                        out_peaks[j].append(pk)
                parts_bp += [torch.stack(o, dim=1) for o in out_peaks]
            sums = (cum_bp[:, -1], cum_var[:, -1], count[:, -1], ema, peaks)
            bp_in = torch.cat(parts_bp, dim=-1)
        else:
            scale = self.chan_std.view(1, 1, C).expand(B, L, C)
            bp_in = bp
        parts = []
        if self.use_cnn:
            h = self.cnn(Xf / scale.reshape(B * L, C, 1))
            parts.append(torch.cat([h.mean(-1), h.std(-1)], dim=1
                                   ).reshape(B, L, -1))

        idx = torch.arange(L, device=X.device).float().unsqueeze(0)
        idx = idx + torch.as_tensor(start_index, device=X.device,
                                    dtype=torch.float32).reshape(-1, 1)
        t = torch.log1p(idx.expand(B, L) * self.window_s / 60.0).unsqueeze(-1)
        z = self.in_proj(torch.cat(parts + [self.bp_proj(bp_in), t], dim=-1))
        return z, sums

    def forward(self, X, start_index=0, state=None, return_stage=False):
        """Sequences (B, L, C, T) of consecutive windows -> logits (B, L, K).

        ``start_index`` is the index of the first window in its recording
        (int or (B,) tensor) and ``state`` = (GRU state, running sums) is
        carried from the previous chunk of the same recordings. With
        ``return_stage`` the staging logits (B, L, 3) come second.
        """
        h, sums = state if state is not None else (None, None)
        z, sums = self.encode(X, start_index, sums)
        out, h = self.gru(z, h)
        if return_stage:
            return self.head(out), self.stage_head(out), (h, sums)
        return self.head(out), (h, sums)

    # --- streamed inference ---------------------------------------------

    def step_proba(self, X):
        """Next windows (B, C, T) of the current recording -> p (B, K)."""
        # The B windows of a call are consecutive windows of one recording
        # (B = 1 in streamed evaluation).
        logits, self._state = self(X.float().unsqueeze(0), self._n,
                                   self._state)
        self._n += X.shape[0]
        return logits[0].softmax(-1)

    def reset_state(self):
        self._state = None
        self._n = 0


class SleepOnsetEnsemble(nn.Module):
    """Averages the members' time-to-N2 distributions; predicts a quantile."""

    def __init__(self, n_chans, n_times, sfreq, n_members=N_SEEDS,
                 adapt=ADAPT, use_cnn=USE_CNN, feats=FEATS,
                 stage=STAGE_WEIGHT > 0, prep=PREP, ae=AE_MODE):
        super().__init__()
        self.members = nn.ModuleList(
            SleepOnsetCNNGRU(n_chans, n_times, sfreq, adapt, use_cnn, feats,
                             stage, prep, ae)
            for _ in range(n_members))
        self.register_buffer("quantile", torch.tensor(0.5))

    def _step_proba(self, X):
        return torch.stack([m.step_proba(X) for m in self.members]).mean(0)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        p = self._step_proba(X)
        return quantile_seconds(p, self.quantile.item()).clamp(0.0, CAP_S)

    @torch.inference_mode()
    def predict_dist(self, X):
        """(seconds, p over 5 s bins, hazard) for analysis; advances state."""
        self.eval()
        p = self._step_proba(X)
        seconds = quantile_seconds(p, self.quantile.item()).clamp(0.0, CAP_S)
        return seconds, p, p[..., 0]

    def reset_state(self):
        for m in self.members:
            m.reset_state()


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


def _load_splits(train_dataset, splits):
    """Other splits of the public Muse release -> {split: (windows, dataset)}.

    The harness builds NeuralBench's train/val/test loaders but hands the
    solver only the train one. Its ``val`` split (20 % of the sleepers
    without a test recording, fixed by ``valid_random_state``) and ``test``
    split (the 40 public warm-up test recordings) are public training data
    too, so they are loaded here, for training only. Local training only:
    never runs on Codabench. Returns {} when unavailable (e.g. the Simulated
    dataset); a split that would overlap another one is skipped.
    """
    if not hasattr(train_dataset, "seg_ds") or not splits:
        return {}
    try:
        from benchopt.config import get_data_path
        from benchmark_utils.nb_task import load_task
        t0 = time.time()
        loaders, _ = load_task(
            "eeg", "_sleep_onset_stream",
            data_dir=get_data_path("neural_compet"), device="cpu",
            batch_size=512, target_transform=lambda y: y.reshape(()))
    except Exception as err:          # noqa: BLE001 - optional extra data
        print(f"[CNN-GRU] extra data unavailable ({err}); skipping")
        return {}
    seen = {seg.timeline for seg in train_dataset.seg_ds.segments}
    out = {}
    for split in splits:
        ds = loaders[split].dataset
        tl = {seg.timeline for seg in ds.seg_ds.segments}
        if seen & tl:
            print(f"[CNN-GRU] split {split} overlaps the data already "
                  f"loaded ({len(seen & tl)} recordings); skipping")
            continue
        seen |= tl
        print(f"[CNN-GRU] extra split {split}: {len(tl)} recordings "
              f"(set up in {time.time() - t0:.0f} s)")
        out[split] = (_load_windows(ds), ds)
    return out


def _extra_windows(train_dataset):
    """NeuralBench's validation split -> (windows, dataset), or None."""
    return _load_splits(train_dataset, ["val"]).get("val")


def _timeline_fields(dataset, n):
    """{subject, session, ...} of each of the first ``n`` windows, or None."""
    if not hasattr(dataset, "seg_ds"):
        return None
    segs = dataset.seg_ds.segments
    order = getattr(dataset, "index", np.arange(len(segs)))
    return [dict(kv.split("=") for kv in
                 segs[int(order[i])].timeline.split(":", 1)[1].split(","))
            for i in range(n)]


def _teacher_targets(dataset, onset):
    """YASA stage probabilities (N, 3) of each window + a found mask (N,).

    Window i of ``dataset`` starts ``onset[i]`` s into its recording (the
    harness windows tile each recording from 0 in 5 s steps), so it lies in
    the recording's 30 s YASA epoch ``floor(onset / 30)``. Recordings are
    matched by subject and session, parsed from the window's timeline.
    """
    n = len(onset)
    target = torch.zeros(n, N_STAGES)
    found = torch.zeros(n, dtype=torch.bool)
    all_fields = _timeline_fields(dataset, n)
    if all_fields is None:
        return target, found
    t = np.load(TEACHER_FILE)
    where = {k: (int(a), int(b)) for k, a, b in zip(t["keys"], t["start"],
                                                    t["n"])}
    proba = torch.as_tensor(t["proba"].astype(np.float32))
    for i, fields in enumerate(all_fields):
        key = f"{fields['subject']}/{fields['session']}"
        if key not in where:
            continue
        start, count = where[key]
        k = int(onset[i] // 30.0)
        if k < count:
            target[i] = proba[start + k]
            found[i] = True
    return target, found


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


def _detach(state):
    h, sums = state

    def det(s):
        if s is None:
            return None
        if isinstance(s, list):
            return [x.detach() for x in s]
        return s.detach()
    return h.detach(), None if sums is None else tuple(det(s) for s in sums)


@torch.no_grad()
def _sequence_proba(member, X_all, idx, device):
    """Distributions (L, K) of one whole recording, chunked, in eval mode."""
    member.eval()
    out, state = [], None
    for s in range(0, len(idx), 4 * CHUNK):
        X, _ = _batch_tensor(X_all, [idx], s, s + 4 * CHUNK, device)
        logits, state = member(X, s, state)
        out.append(logits[0].softmax(-1))
    return torch.cat(out)


def _scores(probas, y_all, recs):
    """W-bMAE averaged over recordings, for every candidate quantile.

    ``probas[i]`` is the (L, K) distribution of recording ``recs[i]``.
    """
    from benchmark_utils.metrics import binned_mae
    scores = {q: [] for q in QUANTILES}
    for p, (idx, _sub) in zip(probas, recs):
        y = y_all[idx].numpy()
        for q in QUANTILES:
            pred = quantile_seconds(p, q).cpu().numpy()
            scores[q].append(binned_mae(y, pred, RANGE_EDGES, RANGE_WEIGHTS))
    return {q: float(np.mean(s)) for q, s in scores.items()}


def _train_device(default):
    forced = os.environ.get("SLEEP_TRAIN_DEVICE")
    if forced:
        return forced
    if default == "cpu" and torch.backends.mps.is_available():
        return "mps"
    return default


def _train_member(model, seed, X_all, y_all, train_recs, val_recs, weights,
                  device, tag, stage_all=None):
    """Train one member in place; keep its best epoch on validation."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW([p for p in model.parameters()
                             if p.requires_grad], lr=LR,
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
        use_stage = stage_all is not None and model.stage_head is not None
        running, running_stage, n_rec = 0.0, 0.0, 0
        for batch in batches:
            idxs = [train_recs[i][0] for i in batch]
            max_len = max(len(idx) for idx in idxs)
            y = torch.zeros(len(idxs), max_len)
            w = torch.zeros(len(idxs), max_len)
            st = torch.zeros(len(idxs), max_len, N_STAGES)
            sm = torch.zeros(len(idxs), max_len)
            for b, idx in enumerate(idxs):
                y[b, :len(idx)] = y_all[idx]
                w[b, :len(idx)] = weights[id(idx)]
                if use_stage:
                    st[b, :len(idx)] = stage_all[0][idx]
                    sm[b, :len(idx)] = stage_all[1][idx].float()
            y, w = y.to(device), w.to(device)
            st, sm = st.to(device), sm.to(device)
            state = None
            for s in range(0, max_len, CHUNK):
                X, mask = _batch_tensor(X_all, idxs, s, s + CHUNK, device)
                if use_stage:
                    logits, stage_logits, state = model(_augment(X), s, state,
                                                        return_stage=True)
                else:
                    logits, state = model(_augment(X), s, state)
                state = _detach(state)
                yc, wc = y[:, s:s + CHUNK], w[:, s:s + CHUNK] * mask
                logp = logits.log_softmax(-1)
                ce = -(_soft_labels(yc, device) * logp).sum(-1)
                mean = (logp.exp() * model.bin_values).sum(-1)
                l1 = (mean - yc.clamp(0, CAP_S)).abs() / CAP_S
                loss = ((ce + L1_WEIGHT * l1) * wc).sum() / len(idxs)
                if use_stage:
                    # Distil YASA's stage, weighted like the main loss (so
                    # near-onset windows, where N1 matters, count most).
                    stage_ce = -(st[:, s:s + CHUNK]
                                 * stage_logits.log_softmax(-1)).sum(-1)
                    stage_loss = (stage_ce * wc * sm[:, s:s + CHUNK]
                                  ).sum() / len(idxs)
                    loss = loss + STAGE_WEIGHT * stage_loss
                    running_stage += stage_loss.item() * len(idxs)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running += loss.item() * len(idxs)
            n_rec += len(idxs)
        sched.step()

        probas = [_sequence_proba(model, X_all, idx, device)
                  for idx, _ in val_recs]
        q, score = min(_scores(probas, y_all, val_recs).items(),
                       key=lambda kv: kv[1])
        stage_msg = (f" (stage CE {running_stage / n_rec:.3f})"
                     if use_stage else "")
        print(f"[CNN-GRU {tag}] epoch {epoch + 1}/{EPOCHS} "
              f"loss {running / n_rec:.3f}{stage_msg} val W-bMAE {score:.1f} s "
              f"(q={q}) {time.time() - t0:.0f} s")
        if score < best:
            best, bad = score, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return best


def _train_autoencoder(member, X_all, idx, device, tag, max_windows=80000):
    """Fit ``member``'s spectral autoencoder on the windows ``idx`` (no
    labels): sets spec_mean / spec_std and the frozen encoder ``ae_enc``."""
    t0 = time.time()
    gen = np.random.default_rng(len(idx))
    idx = gen.choice(idx, min(max_windows, len(idx)), replace=False)
    home = member.spec_mean.device
    member.to(device)
    with torch.no_grad():
        S = torch.cat([member.ae_input(X_all[idx[i:i + 2048]].float()
                                       .to(device))
                       for i in range(0, len(idx), 2048)])
    mean, std = S.mean(0), S.std(0).clamp_min(1e-3)
    S = (S - mean) / std
    n_hold = max(1, len(S) // 20)
    S_val, S_tr = S[:n_hold], S[n_hold:]

    torch.manual_seed(1000 * MEMBER_SEED + len(idx))
    enc = nn.Sequential(nn.Linear(S.shape[1], AE_HIDDEN), nn.GELU(),
                        nn.Linear(AE_HIDDEN, AE_DIM)).to(device)
    dec = nn.Sequential(nn.Linear(AE_DIM, AE_HIDDEN), nn.GELU(),
                        nn.Linear(AE_HIDDEN, S.shape[1])).to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, AE_EPOCHS)
    for epoch in range(AE_EPOCHS):
        perm = torch.randperm(len(S_tr), device=device)
        total = 0.0
        for i in range(0, len(S_tr), 512):
            x = S_tr[perm[i:i + 512]]
            # Denoising: Gaussian noise and 10 % of the inputs dropped.
            noisy = (x + 0.1 * torch.randn_like(x)) * (
                torch.rand_like(x) > 0.1).float()
            loss = ((dec(enc(noisy)) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        sched.step()
    with torch.no_grad():
        val = ((dec(enc(S_val)) - S_val) ** 2).mean().item()
    print(f"[CNN-GRU AE {tag}] {len(S_tr)} windows, {S.shape[1]} inputs -> "
          f"{AE_DIM}: train MSE {total / len(S_tr):.3f}, held-out MSE "
          f"{val:.3f} (1.0 = predicting the mean) {time.time() - t0:.0f} s")
    member.spec_mean.copy_(mean)
    member.spec_std.copy_(std)
    member.ae_enc.load_state_dict(enc.state_dict())
    member.ae_enc.requires_grad_(False)
    member.to(home)


def _subsample_subjects(recs, frac):
    """Keep the recordings of a random ``frac`` of their subjects."""
    if frac >= 1.0:
        return recs
    subjects = np.unique([s for _, s in recs])
    rng = np.random.default_rng(12345)
    keep = set(rng.choice(subjects, max(1, int(round(frac * len(subjects)))),
                          replace=False))
    return [r for r in recs if r[1] in keep]


def _save_val_predictions(val_recs, rec, sub, y_all, preds):
    """Validation-sleeper predictions -> build/<label>_valpred.npz.

    Training-time analysis only (blend, error per range); written next to
    the exported folder, not inside it, so it never ships in the ZIP.
    """
    sub_dir = os.environ.get("COMPET_SUBMISSION_DIR")
    if not sub_dir:
        return
    sub_dir = os.path.abspath(sub_dir)
    out = os.path.join(os.path.dirname(sub_dir),
                       os.path.basename(sub_dir) + "_valpred.npz")
    idx = np.concatenate([i for i, _ in val_recs])
    # ``record`` = index of the validation recording (identical order in
    # cnn_gru.py and gbm.py), ``record_id`` = the dataset's own id.
    np.savez(out, record=np.concatenate([np.full(len(i), k) for k, (i, _)
                                         in enumerate(val_recs)]),
             record_id=np.asarray(rec[idx]).astype(str),
             subject=np.asarray(sub[idx]).astype(str),
             position=np.concatenate([np.arange(len(i)) for i, _ in val_recs]),
             y=y_all[idx].numpy(), pred=np.concatenate(preds))
    print(f"[CNN-GRU] validation predictions -> {out}")


class Solver(CompetSolver):

    name = "CNN-GRU-Hazard"

    def load_model(self, meta):
        weights = meta["submission_dir"] / WEIGHTS_FILE
        state = None
        # Inference (Codabench): load the shipped weights, which also fix the
        # ensemble size and the adaptation flag. A local training run
        # (train_loader set) starts from scratch instead.
        if weights.exists() and self.train_loader is None:
            state = torch.load(weights, map_location=meta["device"],
                               weights_only=True)
            n_members = len({k.split(".")[1] for k in state
                             if k.startswith("members.")})
            adapt = bool(state["members.0.adapt"])
            # Older checkpoints predate these switches: CNN on, feats 1.
            use_cnn = bool(state.get("members.0.cnn_on", True))
            feats = int(state.get("members.0.feats", 1))
            prep = int(state.get("members.0.prep_v", 1))
            ae = int(state.get("members.0.ae_mode", 0))
            for i in range(n_members):
                state.setdefault(f"members.{i}.cnn_on", torch.tensor(True))
                state.setdefault(f"members.{i}.feats", torch.tensor(1))
                state.setdefault(f"members.{i}.prep_v", torch.tensor(1))
                state.setdefault(f"members.{i}.ae_mode", torch.tensor(0))
            # The staging head is training-only but saved with the weights.
            stage = any(".stage_head." in k for k in state)
        else:
            n_members, adapt, use_cnn, feats = N_SEEDS, ADAPT, USE_CNN, FEATS
            prep, ae = PREP, AE_MODE
            if CV_FOLDS > 0:
                n_members = CV_FOLDS           # one member per fold
            stage = STAGE_WEIGHT > 0
        model = SleepOnsetEnsemble(meta["n_chans"], meta["n_times"],
                                   float(meta["sfreq"]), n_members, adapt,
                                   use_cnn, feats, stage, prep, ae)
        if state is not None:
            model.load_state_dict(state)
        return model.to(meta["device"]).eval()

    def fit(self, model, train_loader):
        if CV_FOLDS > 0:
            return self._fit_cv(model, train_loader)
        rng = np.random.default_rng(SEED)
        X_all, y_all, rec, onset, sub = _load_windows(train_loader.dataset)
        recs = _recordings(y_all, rec, onset, sub, rng)
        use_stage = STAGE_WEIGHT > 0 and hasattr(train_loader.dataset,
                                                 "seg_ds")
        if STAGE_WEIGHT > 0 and not os.path.exists(TEACHER_FILE):
            raise FileNotFoundError(
                f"SLEEP_STAGE_WEIGHT={STAGE_WEIGHT} needs the YASA teacher "
                f"labels at {TEACHER_FILE} (scripts/make_yasa_teacher.py)")
        if use_stage:
            stage_t, stage_m = _teacher_targets(train_loader.dataset, onset)

        # Hold out whole subjects for early stopping and the quantile.
        subjects = np.unique([s for _, s in recs])
        n_val = max(1, int(round(VAL_FRACTION * len(subjects))))
        val_subjects = set(rng.choice(subjects, n_val, replace=False))
        train_recs = [r for r in recs if r[1] not in val_subjects]
        val_recs = [r for r in recs if r[1] in val_subjects]
        train_recs = _subsample_subjects(train_recs, TRAIN_FRAC)

        # Extra training-only recordings; the validation sleepers above are
        # drawn before they are added, so validation stays comparable.
        extra = _extra_windows(train_loader.dataset) if EXTRA_DATA else None
        if extra is not None:
            n0, rec_off = len(y_all), int(rec.max()) + 1
            sub_off = int(np.max(sub)) + 1
            (X_e, y_e, rec_e, onset_e, sub_e), extra_ds = extra
            if use_stage:
                t_e, m_e = _teacher_targets(extra_ds, onset_e)
                stage_t = torch.cat([stage_t, t_e])
                stage_m = torch.cat([stage_m, m_e])
            extra_recs = [(idx + n0, s + sub_off) for idx, s in
                          _recordings(y_e, rec_e, onset_e, sub_e, rng)]
            X_all = torch.cat([X_all, X_e])
            y_all = torch.cat([y_all, y_e])
            rec = np.concatenate([rec, rec_e + rec_off])
            onset = np.concatenate([onset, onset_e])
            sub = np.concatenate([sub, sub_e + sub_off])
            train_recs = train_recs + extra_recs
            recs = recs + extra_recs
            print(f"[CNN-GRU] + {len(extra_recs)} extra training recordings")

        print(f"[CNN-GRU] {len(train_recs)} train / {len(val_recs)} val "
              f"recordings ({len(subjects)} subjects); "
              f"{len(model.members)} members, "
              f"adapt={model.members[0].use_adapt}, "
              f"cnn={model.members[0].use_cnn}, "
              f"feats={model.members[0].n_feats}, extra={extra is not None}, "
              f"train_frac={TRAIN_FRAC}, seed={MEMBER_SEED}, "
              f"stage_weight={STAGE_WEIGHT if use_stage else 0}, "
              f"prep={model.members[0].prep}, ae={model.members[0].ae}")
        if use_stage:
            train_idx = np.concatenate([idx for idx, _ in train_recs])
            m = stage_m[train_idx]
            mean_t = stage_t[train_idx][m].mean(0).tolist()
            print(f"[CNN-GRU] YASA labels for {m.float().mean():.1%} of "
                  f"training windows; mean wake/drowsy/N2+ = "
                  + "/".join(f"{v:.2f}" for v in mean_t))
        weights = {id(idx): _window_weights(y_all[idx].numpy())
                   for idx, _ in recs}

        # Normalisation stats from the training windows, shared by members.
        train_idx = np.concatenate([idx for idx, _ in train_recs])
        sample = X_all[rng.choice(train_idx, min(20000, len(train_idx)),
                                  replace=False)].float()
        chan_std = sample.std(dim=(0, 2)).clamp_min(1e-3)
        first = model.members[0]
        out_device = model.quantile.device
        device = _train_device(str(out_device))
        if first.ae:
            # One autoencoder on the training windows, shared by members.
            _train_autoencoder(first, X_all, train_idx, device, "shared")
            for m in model.members[1:]:
                m.load_state_dict(first.state_dict())
        feats = first.band_powers(sample.to(first.chan_std.device))
        for m in model.members:
            m.chan_std.copy_(chan_std)
            m.bp_mean.copy_(feats.mean(0))
            m.bp_std.copy_(feats.std(0).clamp_min(1e-3))

        model.to(device)
        print(f"[CNN-GRU] training on {device}")

        member_probas = []
        for i, member in enumerate(model.members):
            best = _train_member(member, 1000 * MEMBER_SEED + i, X_all, y_all,
                                 train_recs,
                                 val_recs, weights, device,
                                 f"{i + 1}/{len(model.members)}",
                                 (stage_t, stage_m) if use_stage else None)
            print(f"[CNN-GRU] member {i + 1}: best val W-bMAE {best:.1f} s")
            member_probas.append([_sequence_proba(member, X_all, idx, device)
                                  for idx, _ in val_recs])

        ensemble = [torch.stack(ps).mean(0) for ps in zip(*member_probas)]
        q, score = min(_scores(ensemble, y_all, val_recs).items(),
                       key=lambda kv: kv[1])
        model.quantile.fill_(q)
        model.to(out_device).eval()
        print(f"[CNN-GRU] ensemble val W-bMAE {score:.1f} s, q={q:.2f}")
        _save_val_predictions(
            val_recs, rec, sub, y_all,
            [quantile_seconds(p, q).clamp(0, CAP_S).cpu().numpy()
             for p in ensemble])
        self._check_streaming(model, X_all, val_recs[0][0], out_device)

    def _fit_cv(self, model, train_loader):
        """Cross-validated all-data training (see CV_FOLDS)."""
        rng = np.random.default_rng(SEED)
        train_ds = train_loader.dataset
        X_all, y_all, rec, onset, sub = _load_windows(train_ds)
        n_train = len(y_all)

        # The usual validation recordings (drawn exactly as in ``fit``), only
        # to report a score comparable with the experiments' val W-bMAE.
        base_recs = _recordings(y_all, rec, onset, sub, rng)
        subjects = np.unique([s for _, s in base_recs])
        n_val = max(1, int(round(VAL_FRACTION * len(subjects))))
        classic_val = set(rng.choice(subjects, n_val, replace=False))
        classic_rec = {int(rec[idx[0]]) for idx, s in base_recs
                       if s in classic_val}

        # Sleeper of every window, by name, so a sleeper's recordings from
        # different splits land in the same fold.
        fields = _timeline_fields(train_ds, n_train)
        names = (np.array([f["subject"] for f in fields]) if fields
                 else np.asarray(sub).astype(str))
        use_stage = STAGE_WEIGHT > 0 and fields is not None
        if STAGE_WEIGHT > 0 and not os.path.exists(TEACHER_FILE):
            raise FileNotFoundError(
                f"SLEEP_STAGE_WEIGHT={STAGE_WEIGHT} needs the YASA teacher "
                f"labels at {TEACHER_FILE} (scripts/make_yasa_teacher.py)")
        if use_stage:
            stage_t, stage_m = _teacher_targets(train_ds, onset)

        split_of = np.zeros(n_train, dtype=object)
        split_of[:] = "train"
        wanted = (["val"] if EXTRA_DATA else []) + (["test"] if USE_TEST
                                                     else [])
        for split, ((X_e, y_e, rec_e, onset_e, _), ds) in _load_splits(
                train_ds, wanted).items():
            f_e = _timeline_fields(ds, len(y_e))
            rec_off = int(rec.max()) + 1
            X_all = torch.cat([X_all, X_e])
            y_all = torch.cat([y_all, y_e])
            rec = np.concatenate([rec, rec_e + rec_off])
            onset = np.concatenate([onset, onset_e])
            names = np.concatenate([names, [f["subject"] for f in f_e]])
            split_of = np.concatenate([split_of, np.full(len(y_e), split,
                                                         dtype=object)])
            if use_stage:
                t_e, m_e = _teacher_targets(ds, onset_e)
                stage_t = torch.cat([stage_t, t_e])
                stage_m = torch.cat([stage_m, m_e])

        _, sub_int = np.unique(names, return_inverse=True)
        recs = _recordings(y_all, rec, onset, sub_int, rng)
        rec_split = [split_of[idx[0]] for idx, _ in recs]
        is_classic = [split_of[idx[0]] == "train"
                      and int(rec[idx[0]]) in classic_rec for idx, _ in recs]

        # Folds over sleepers.
        fold_subjects = np.random.default_rng(SEED + 1).permutation(
            np.unique([s for _, s in recs]))
        K = len(model.members)
        fold_of = {s: k for k, part in enumerate(np.array_split(
            fold_subjects, K)) for s in part}
        rec_fold = np.array([fold_of[s] for _, s in recs])
        counts = {sp: rec_split.count(sp) for sp in ("train", "val", "test")}
        print(f"[CNN-GRU] CV: {len(recs)} recordings "
              f"({counts['train']} train + {counts['val']} extra + "
              f"{counts['test']} warm-up test) from {len(fold_subjects)} "
              f"sleepers, {K} folds of "
              f"{'/'.join(str((rec_fold == k).sum()) for k in range(K))} "
              f"recordings; cnn={model.members[0].use_cnn}, "
              f"feats={model.members[0].n_feats}, "
              f"stage_weight={STAGE_WEIGHT if use_stage else 0}, "
              f"prep={model.members[0].prep}, ae={model.members[0].ae}")
        if counts["test"]:
            print("[CNN-GRU] note: the warm-up test recordings are in "
                  "training, so this build's test W-bMAE is meaningless")
        weights = {id(idx): _window_weights(y_all[idx].numpy())
                   for idx, _ in recs}

        # Normalisation stats from all windows, shared by members.
        sample = X_all[rng.choice(len(y_all), min(20000, len(y_all)),
                                  replace=False)].float()
        chan_std = sample.std(dim=(0, 2)).clamp_min(1e-3)
        first = model.members[0]
        out_device = model.quantile.device
        device = _train_device(str(out_device))
        if not first.ae:
            feats = first.band_powers(sample.to(first.chan_std.device))
        for m in model.members:
            m.chan_std.copy_(chan_std)
            if not first.ae:
                m.bp_mean.copy_(feats.mean(0))
                m.bp_std.copy_(feats.std(0).clamp_min(1e-3))

        print(f"[CNN-GRU] training on {device}")

        oof = [None] * len(recs)
        for k, member in enumerate(model.members):
            tr = [r for r, f in zip(recs, rec_fold) if f != k]
            va_i = [i for i, f in enumerate(rec_fold) if f == k]
            va = [recs[i] for i in va_i]
            if member.ae:
                # This fold's autoencoder and feature stats, from its
                # training sleepers only (keeps the out-of-fold honest).
                member.to(out_device)
                tr_idx = np.concatenate([idx for idx, _ in tr])
                _train_autoencoder(member, X_all, tr_idx, device,
                                   f"fold {k + 1}/{K}")
                s = X_all[rng.choice(tr_idx, min(20000, len(tr_idx)),
                                     replace=False)].float()
                with torch.no_grad():
                    f_k = member.band_powers(s.to(member.chan_std.device))
                member.bp_mean.copy_(f_k.mean(0))
                member.bp_std.copy_(f_k.std(0).clamp_min(1e-3))
            member.to(device)
            best = _train_member(member, 1000 * MEMBER_SEED + k, X_all, y_all,
                                 tr, va, weights, device,
                                 f"fold {k + 1}/{K}",
                                 (stage_t, stage_m) if use_stage else None)
            print(f"[CNN-GRU] fold {k + 1}: {len(tr)} train / {len(va)} "
                  f"held-out recordings, best held-out W-bMAE {best:.1f} s")
            for i in va_i:
                oof[i] = _sequence_proba(member, X_all, recs[i][0], device)

        all_scores = _scores(oof, y_all, recs)
        q, score = min(all_scores.items(), key=lambda kv: kv[1])
        model.quantile.fill_(q)
        model.to(out_device).eval()

        def subset(mask):
            pairs = [(p, r) for p, r, m in zip(oof, recs, mask) if m]
            if not pairs:
                return float("nan")
            return _scores([p for p, _ in pairs], y_all,
                           [r for _, r in pairs])[q]

        msg = (f"out-of-fold W-bMAE {score:.1f} s on all {len(recs)} "
               f"recordings, q={q:.2f}")
        if counts["test"]:
            msg += (f"; warm-up test recordings "
                    f"{subset([sp == 'test' for sp in rec_split]):.1f} s")
        print(f"[CNN-GRU] {msg}")
        # Comparable with the experiments' val W-bMAE (same 60 recordings,
        # each predicted by the member that did not train on its sleeper).
        print(f"[CNN-GRU] ensemble val W-bMAE {subset(is_classic):.1f} s "
              f"(out-of-fold, the usual validation recordings)")
        classic = [r for r, m in zip(recs, is_classic) if m]
        _save_val_predictions(
            classic, rec, sub_int, y_all,
            [quantile_seconds(p, q).clamp(0, CAP_S).cpu().numpy()
             for p, m in zip(oof, is_classic) if m])
        self._check_streaming(model, X_all, recs[0][0], out_device)

    @staticmethod
    @torch.no_grad()
    def _check_streaming(model, X_all, idx, device):
        """Window-by-window predict must match a whole-sequence forward."""
        idx = idx[:200]
        p = torch.stack([_sequence_proba(m, X_all, idx, device)
                         for m in model.members]).mean(0)
        seq = quantile_seconds(p, model.quantile.item())
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
