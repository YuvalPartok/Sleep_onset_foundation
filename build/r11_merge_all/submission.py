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
GRU_HIDDEN = int(os.environ.get("SLEEP_GRU_HIDDEN", 128))
GRU_LAYERS = int(os.environ.get("SLEEP_GRU_LAYERS", 2))
DROPOUT = float(os.environ.get("SLEEP_DROPOUT", 0.2))
RNN = os.environ.get("SLEEP_RNN", "gru").lower()   # "gru" or "lstm"

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
EMA_S = float(os.environ.get("SLEEP_EMA_S", 60.0))  # short-term EMA, s
N_MOVE = 3                               # movement markers (feats >= 3)
PEAK_S = (60.0, 300.0)                   # decaying-peak horizons (feats >= 3)
# Staging output: loss weight (0 = off) and the YASA teacher labels.
STAGE_WEIGHT = float(os.environ.get("SLEEP_STAGE_WEIGHT", 0.0))
PREP = int(os.environ.get("SLEEP_PREP", 1))     # 2: notch + Welch
AE_MODE = int(os.environ.get("SLEEP_AE", 0))     # 1 replace, 2 add
AE_DIM = int(os.environ.get("SLEEP_AE_DIM", 32))  # autoencoder code size
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
# Members per fold (repeats with seeds SLEEP_SEED, SLEEP_SEED + 1, ...): the
# ensemble then has CV_FOLDS * CV_SEEDS members, and each recording's
# out-of-fold distribution is the mean over its fold's members.
CV_SEEDS = int(os.environ.get("SLEEP_CV_SEEDS", 1))
USE_TEST = os.environ.get("SLEEP_USE_TEST", "0") == "1"

EPOCHS = int(os.environ.get("SLEEP_EPOCHS", 30))
PATIENCE = 8
LR = float(os.environ.get("SLEEP_LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("SLEEP_WEIGHT_DECAY", 1e-4))
RECS_PER_BATCH = 16
CHUNK = int(os.environ.get("SLEEP_CHUNK", 120))  # TBPTT length, windows
VAL_FRACTION = 0.15
LABEL_SIGMA = float(os.environ.get("SLEEP_LABEL_SIGMA", 1.5))  # in bins
L1_WEIGHT = float(os.environ.get("SLEEP_L1", 0.1))
# External pseudo-Muse PSG windows (scripts/prep_external.py), CV mode only:
# added to every fold's training set with their loss weight scaled by
# EXT_WEIGHT; a domain flag input tells them apart (always 0 at inference).
# SLEEP_EXTERNAL is one file or a comma-separated list; ``path@w`` gives that
# file its own loss weight instead of EXT_WEIGHT.
EXTERNAL = os.environ.get("SLEEP_EXTERNAL", "")
EXT_WEIGHT = float(os.environ.get("SLEEP_EXT_WEIGHT", 0.5))


def _external_files(spec=None, default_w=None):
    """SLEEP_EXTERNAL -> [(path, loss weight)]."""
    spec = EXTERNAL if spec is None else spec
    default_w = EXT_WEIGHT if default_w is None else default_w
    out = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        path, _, w = part.rpartition("@") if "@" in part else (part, "", "")
        out.append((path, float(w) if w else default_w))
    return out
# Elapsed-time buckets (minutes) of the per-bucket quantile ``q_time``.
TIME_EDGES_MIN = (2.0, 5.0, 10.0, 20.0)
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
                 feats=FEATS, stage=STAGE_WEIGHT > 0, prep=PREP, ae=AE_MODE,
                 hidden=GRU_HIDDEN, ae_dim=AE_DIM, ema_s=EMA_S, dom=False,
                 rnn=RNN, layers=GRU_LAYERS):
        super().__init__()
        self.rnn_type, self.layers = str(rnn), int(layers)
        self.prep, self.ae = int(prep), int(ae)
        self.hidden, self.dom = int(hidden), bool(dom)
        self.register_buffer("ema_s", torch.tensor(float(ema_s)))
        self.register_buffer("dom_on", torch.tensor(self.dom))
        self.register_buffer("prep_v", torch.tensor(self.prep))
        self.register_buffer("ae_mode", torch.tensor(self.ae))
        self.n_chans, self.n_times, self.sfreq = n_chans, n_times, sfreq
        self.window_s = n_times / sfreq
        self.use_adapt = bool(adapt)
        self.use_cnn = bool(use_cnn)
        self.n_feats = int(feats)
        self.ema_alpha = min(1.0, self.window_s / float(ema_s))
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
                                        nn.Linear(AE_HIDDEN, int(ae_dim)))
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
            nn.Linear(cnn_dim + BP_DIM + 1 + int(self.dom), self.hidden),
            nn.GELU(), nn.Dropout(DROPOUT))
        # Recurrent core (attribute keeps its name for old weights.pt files).
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.gru = rnn_cls(self.hidden, self.hidden, self.layers,
                           batch_first=True,
                           dropout=DROPOUT if self.layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(DROPOUT),
                                  nn.Linear(self.hidden, N_BINS))
        # Training-only auxiliary head (see module docstring).
        self.stage_head = (nn.Sequential(nn.Dropout(DROPOUT),
                                         nn.Linear(self.hidden, N_STAGES))
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

    def encode(self, X, start_index, sums, dom=None):
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
        extra = [t]
        if self.dom:
            # Domain flag: 1 for external PSG recordings (training only).
            d = (torch.zeros(B, device=X.device) if dom is None
                 else dom.to(X.device).float())
            extra.append(d.view(B, 1, 1).expand(B, L, 1))
        z = self.in_proj(torch.cat(parts + [self.bp_proj(bp_in)] + extra,
                                   dim=-1))
        return z, sums

    def forward(self, X, start_index=0, state=None, return_stage=False,
                dom=None):
        """Sequences (B, L, C, T) of consecutive windows -> logits (B, L, K).

        ``start_index`` is the index of the first window in its recording
        (int or (B,) tensor) and ``state`` = (GRU state, running sums) is
        carried from the previous chunk of the same recordings. With
        ``return_stage`` the staging logits (B, L, 3) come second.
        """
        h, sums = state if state is not None else (None, None)
        z, sums = self.encode(X, start_index, sums, dom)
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
                 stage=STAGE_WEIGHT > 0, prep=PREP, ae=AE_MODE,
                 configs=None, dom=False):
        super().__init__()
        # ``configs``: one dict of member settings per member (mixed
        # ensembles, see scripts/merge_builds.py); else n_members alike.
        if configs is None:
            configs = [dict(adapt=adapt, use_cnn=use_cnn, feats=feats,
                            stage=stage, prep=prep, ae=ae, dom=dom)
                       for _ in range(n_members)]
        self.members = nn.ModuleList(
            SleepOnsetCNNGRU(n_chans, n_times, sfreq, **c) for c in configs)
        self.register_buffer("quantile", torch.tensor(0.5))
        # Averaging weights of the members, and the quantile per elapsed-time
        # bucket (TIME_EDGES_MIN); both default to plain averaging / q.
        self.register_buffer("member_w", torch.ones(len(configs)))
        self.register_buffer("q_time", torch.full(
            (len(TIME_EDGES_MIN) + 1,), 0.5))

    def set_quantile(self, q):
        self.quantile.fill_(float(q))
        self.q_time.fill_(float(q))

    def q_for(self, start, n):
        """Quantiles (n,) of windows start .. start + n - 1 of a recording."""
        window_s = self.members[0].window_s
        minutes = (start + torch.arange(n, device=self.q_time.device)
                   ) * window_s / 60.0
        edges = torch.tensor(TIME_EDGES_MIN, device=self.q_time.device)
        return self.q_time[torch.bucketize(minutes, edges, right=True)]

    def _step_proba(self, X):
        w = self.member_w / self.member_w.sum()
        if not torch.is_grad_enabled():
            if not hasattr(self, "_fast"):
                self._fast = _FastEnsemble.build(self)
            if self._fast is not None:
                return self._fast.step(X, w)
        return (torch.stack([m.step_proba(X) for m in self.members])
                * w.view(-1, 1, 1)).sum(0)

    def _seconds(self, p, start):
        q = self.q_for(start, p.shape[0])
        cdf = p.cumsum(-1).contiguous()
        k = torch.searchsorted(cdf, q.view(-1, 1).to(cdf.dtype))
        return (k.clamp_max(N_BINS - 1).squeeze(-1) * BIN_S).float(
        ).clamp(0.0, CAP_S)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        start = self.members[0]._n
        p = self._step_proba(X)
        return self._seconds(p, start)

    @torch.inference_mode()
    def predict_dist(self, X):
        """(seconds, p over 5 s bins, hazard) for analysis; advances state."""
        self.eval()
        start = self.members[0]._n
        p = self._step_proba(X)
        return self._seconds(p, start), p, p[..., 0]

    def reset_state(self):
        for m in self.members:
            m.reset_state()
        if getattr(self, "_fast", None) is not None:
            self._fast.reset()

    def _apply(self, fn, *args, **kwargs):
        # A device / dtype move invalidates the stacked weights.
        self.__dict__.pop("_fast", None)
        return super()._apply(fn, *args, **kwargs)


class _FastEnsemble:
    """Streamed inference with the members stacked: members that share their
    settings run as one batched computation (einsum over stacked weights, a
    hand-written GRU), and the spectral features are computed once per
    window for all members. Same outputs as the per-member path up to
    float rounding; used for inference only, rebuilt after weight changes.
    """

    @staticmethod
    def build(ens):
        ms = list(ens.members)
        if any(m.use_cnn or not m.use_adapt or m.use_peaks for m in ms):
            return None                 # not covered: per-member path
        groups = {}
        for i, m in enumerate(ms):
            key = (m.n_feats, m.prep, m.ae, m.hidden, m.ema_alpha, m.dom,
                   m.rnn_type, m.layers,
                   m.use_ema, m.bp_mean.numel(),
                   m.ae_enc[2].weight.shape[0] if m.ae else 0)
            groups.setdefault(key, []).append(i)
        return _FastEnsemble(ens, [(k, idx) for k, idx in groups.items()])

    def __init__(self, ens, groups):
        self.ens = ens
        self.groups = []
        for key, idx in groups:
            ms = [ens.members[i] for i in idx]
            st = lambda f: torch.stack([f(m) for m in ms]).detach()
            g = dict(idx=torch.tensor(idx), m0=ms[0], ae=ms[0].ae,
                     feat_key=(ms[0].n_feats, ms[0].prep, ms[0].ae),
                     alpha=ms[0].ema_alpha, dom=ms[0].dom,
                     use_ema=ms[0].use_ema, H=ms[0].hidden,
                     lstm=ms[0].rnn_type == "lstm",
                     bp_mean=st(lambda m: m.bp_mean),
                     bp_std=st(lambda m: m.bp_std),
                     P1w=st(lambda m: m.bp_proj[0].weight),
                     P1b=st(lambda m: m.bp_proj[0].bias),
                     I1w=st(lambda m: m.in_proj[0].weight),
                     I1b=st(lambda m: m.in_proj[0].bias),
                     Hw=st(lambda m: m.head[1].weight),
                     Hb=st(lambda m: m.head[1].bias),
                     gru=[tuple(st(lambda m, n=n: getattr(m.gru, f"{n}_l{l}"))
                                for n in ("weight_ih", "weight_hh",
                                          "bias_ih", "bias_hh"))
                          for l in range(ms[0].gru.num_layers)])
            if g["ae"]:
                g.update(spec_mean=st(lambda m: m.spec_mean),
                         spec_std=st(lambda m: m.spec_std),
                         A1w=st(lambda m: m.ae_enc[0].weight),
                         A1b=st(lambda m: m.ae_enc[0].bias),
                         A2w=st(lambda m: m.ae_enc[2].weight),
                         A2b=st(lambda m: m.ae_enc[2].bias))
            self.groups.append(g)
        # Member order of the concatenated group outputs -> ensemble order.
        self.order = torch.argsort(torch.cat([g["idx"] for g in self.groups]))
        self.reset()

    def reset(self):
        self.n = 0
        for g in self.groups:
            g.update(cum=None, ema=None, h=None)

    @staticmethod
    def _lin(x, w, b):
        """x (k, i), stacked weights w (k, o, i), b (k, o) -> (k, o)."""
        return torch.baddbmm(b.unsqueeze(-1), w, x.unsqueeze(-1)).squeeze(-1)

    def _window(self, x):
        """One raw window (C, T) -> mixed-in-later member probas (M, K)."""
        Xf = preprocess(x.float().unsqueeze(0))
        feats, out = {}, []
        t = float(np.log1p(self.n * self.groups[0]["m0"].window_s / 60.0))
        for g in self.groups:
            m0, k = g["m0"], len(g["idx"])
            fk = g["feat_key"]
            if fk not in feats:
                hand = m0.hand_features(Xf)[0] if g["ae"] != 1 else None
                aein = m0.ae_input(Xf)[0] if g["ae"] else None
                feats[fk] = (hand, aein)
            hand, aein = feats[fk]
            parts = []
            if g["ae"]:
                a = (aein.unsqueeze(0) - g["spec_mean"]) / g["spec_std"]
                a = nn.functional.gelu(self._lin(a, g["A1w"], g["A1b"]))
                parts.append(self._lin(a, g["A2w"], g["A2b"]))
            if g["ae"] != 1:
                parts.append(hand.unsqueeze(0).expand(k, -1))
            bp = (torch.cat(parts, 1) - g["bp_mean"]) / g["bp_std"]
            cum = bp if g["cum"] is None else g["cum"] + bp
            mean = cum / (self.n + 1)
            pieces = [bp, bp - mean]
            if g["use_ema"]:
                ema = (bp if g["ema"] is None
                       else g["ema"] + g["alpha"] * (bp - g["ema"]))
                pieces += [bp - ema, ema - mean]
                g["ema"] = ema
            g["cum"] = cum
            z = nn.functional.gelu(self._lin(torch.cat(pieces, 1),
                                             g["P1w"], g["P1b"]))
            extra = [z.new_full((k, 1), t)]
            if g["dom"]:
                extra.append(z.new_zeros(k, 1))
            z = nn.functional.gelu(self._lin(torch.cat([z] + extra, 1),
                                             g["I1w"], g["I1b"]))
            H = g["H"]
            hs = (g["h"] if g["h"] is not None
                  else [(z.new_zeros(k, H), z.new_zeros(k, H))
                        for _ in g["gru"]])
            new_h = []
            for (wi, wh, bi, bh), (h, c) in zip(g["gru"], hs):
                gi, gh = self._lin(z, wi, bi), self._lin(h, wh, bh)
                if g["lstm"]:           # gates i, f, g, o (PyTorch order)
                    a = gi + gh
                    c = (torch.sigmoid(a[:, H:2 * H]) * c
                         + torch.sigmoid(a[:, :H]) * torch.tanh(a[:, 2 * H:3 * H]))
                    z = torch.sigmoid(a[:, 3 * H:]) * torch.tanh(c)
                else:
                    r = torch.sigmoid(gi[:, :H] + gh[:, :H])
                    u = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
                    n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
                    z = (1 - u) * n + u * h
                new_h.append((z, c))
            g["h"] = new_h
            out.append(self._lin(z, g["Hw"], g["Hb"]).softmax(-1))
        self.n += 1
        return torch.cat(out)[self.order.to(z.device)]

    def step(self, X, w):
        """Next windows (B, C, T) of the current recording -> p (B, K)."""
        ps = torch.stack([(self._window(x) * w.view(-1, 1)).sum(0)
                          for x in X])
        for m in self.ens.members:          # keep the members' counters in step
            m._n += X.shape[0]
        return ps


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


def _recordings(y, rec, onset, sub, rng, pseudo=True):
    """Window indices of each recording, in time order, plus its subject.

    Data without real recordings (the Simulated set: one record of random
    windows) is cut into pseudo-recordings sorted by decreasing latency.
    """
    out = []
    if pseudo and len(np.unique(rec)) < 4:
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
    h = tuple(x.detach() for x in h) if isinstance(h, tuple) else h.detach()
    return h, None if sums is None else tuple(det(s) for s in sums)


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
                  device, tag, stage_all=None, ext_mask=None):
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
            dom = (torch.tensor([bool(ext_mask[idx[0]]) for idx in idxs])
                   if ext_mask is not None and model.dom else None)
            st, sm = st.to(device), sm.to(device)
            state = None
            for s in range(0, max_len, CHUNK):
                X, mask = _batch_tensor(X_all, idxs, s, s + CHUNK, device)
                if use_stage:
                    logits, stage_logits, state = model(_augment(X), s, state,
                                                        return_stage=True,
                                                        dom=dom)
                else:
                    logits, state = model(_augment(X), s, state, dom=dom)
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


def _train_autoencoder(member, X_all, idx, device, tag, max_windows=80000,
                       seed=None):
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

    torch.manual_seed(1000 * (MEMBER_SEED if seed is None else seed)
                      + len(idx))
    ae_dim = member.ae_enc[2].out_features
    enc = nn.Sequential(nn.Linear(S.shape[1], AE_HIDDEN), nn.GELU(),
                        nn.Linear(AE_HIDDEN, ae_dim)).to(device)
    dec = nn.Sequential(nn.Linear(ae_dim, AE_HIDDEN), nn.GELU(),
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
          f"{ae_dim}: train MSE {total / len(S_tr):.3f}, held-out MSE "
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


def member_configs(state):
    """Per-member settings of a saved ensemble state dict.

    Older checkpoints predate some switches: CNN on, feats 1, prep 1, no
    autoencoder, 60 s EMA, no domain flag; the missing buffers are added to
    ``state`` with those values so it loads strictly.
    """
    n = len({k.split(".")[1] for k in state if k.startswith("members.")})
    configs = []
    for i in range(n):
        pre = f"members.{i}."
        for key, val in (("cnn_on", True), ("feats", 1), ("prep_v", 1),
                         ("ae_mode", 0), ("ema_s", 60.0), ("dom_on", False)):
            state.setdefault(pre + key, torch.tensor(val))
        ae = int(state[pre + "ae_mode"])
        configs.append(dict(
            adapt=bool(state[pre + "adapt"]),
            use_cnn=bool(state[pre + "cnn_on"]),
            feats=int(state[pre + "feats"]),
            stage=any(k.startswith(pre + "stage_head.") for k in state),
            prep=int(state[pre + "prep_v"]), ae=ae,
            hidden=int(state[pre + "gru.weight_hh_l0"].shape[1]),
            rnn="lstm" if state[pre + "gru.weight_hh_l0"].shape[0]
            == 4 * state[pre + "gru.weight_hh_l0"].shape[1] else "gru",
            layers=sum(k.startswith(pre + "gru.weight_hh_l") for k in state),
            ae_dim=int(state[pre + "ae_enc.2.weight"].shape[0]) if ae
            else AE_DIM,
            ema_s=float(state[pre + "ema_s"]),
            dom=bool(state[pre + "dom_on"])))
    return configs


def ensemble_from_state(state, n_chans, n_times, sfreq):
    """Saved state dict -> the SleepOnsetEnsemble it came from."""
    configs = member_configs(state)
    q = state["quantile"]
    state.setdefault("member_w", torch.ones(len(configs)))
    state.setdefault("q_time", torch.full((len(TIME_EDGES_MIN) + 1,),
                                          float(q)))
    model = SleepOnsetEnsemble(n_chans, n_times, sfreq, configs=configs)
    model.load_state_dict(state)
    return model


def _load_external(path):
    """scripts/prep_external.py output -> (windows, y, rec, onset, subject).

    Windows are raw microvolts (float16); they get the Muse preprocessing.
    """
    if os.path.isdir(path):     # Kaggle unzips an uploaded .npz into a folder
        d = {k: np.load(os.path.join(path, k + ".npy"),
                        mmap_mode="r" if k == "X" else None, allow_pickle=False)
             for k in ("X", "y", "rec", "onset", "subject", "dataset")}
    else:
        d = np.load(path)
    X_raw = d["X"]
    X = torch.empty(X_raw.shape, dtype=torch.float16)
    for i in range(0, len(X_raw), 20000):        # keeps the float32 copy small
        X[i:i + 20000] = preprocess(
            torch.from_numpy(np.array(X_raw[i:i + 20000])).float()).half()
    del X_raw
    return (X, torch.from_numpy(d["y"]).float(), d["rec"].astype(np.int64),
            d["onset"].astype(np.float64), d["subject"].astype(str),
            d["dataset"].astype(str))


def _save_oof(recs, rec_split, rec_fold, names, y_all, probas, q):
    """Out-of-fold distributions of a CV build -> build/<label>_oof.npz.

    For offline analysis and scripts/merge_builds.py: per window the
    recording index (into the build's recordings, same order in every CV
    build), position, target and the (K,) distribution (float16); per
    recording its split, fold and sleeper. Never shipped in the ZIP.
    """
    sub_dir = os.environ.get("COMPET_SUBMISSION_DIR")
    if not sub_dir:
        return
    sub_dir = os.path.abspath(sub_dir)
    out = os.path.join(os.path.dirname(sub_dir),
                       os.path.basename(sub_dir) + "_oof.npz")
    np.savez_compressed(
        out,
        record=np.concatenate([np.full(len(i), k) for k, (i, _)
                               in enumerate(recs)]),
        position=np.concatenate([np.arange(len(i)) for i, _ in recs]),
        y=np.concatenate([y_all[i].numpy() for i, _ in recs]),
        proba=torch.cat([p.float().cpu() for p in probas]).half().numpy(),
        rec_split=np.asarray(rec_split).astype(str),
        rec_fold=np.asarray(rec_fold),
        rec_subject=np.asarray([names[i[0]] for i, _ in recs]).astype(str),
        quantile=np.float32(q))
    print(f"[CNN-GRU] out-of-fold distributions -> {out}")


class Solver(CompetSolver):

    name = "CNN-GRU-Hazard"

    def load_model(self, meta):
        weights = meta["submission_dir"] / WEIGHTS_FILE
        # Inference (Codabench): load the shipped weights, which fix every
        # member's own settings (a merged ensemble may mix them). A local
        # training run (train_loader set) starts from scratch instead.
        if weights.exists() and self.train_loader is None:
            state = torch.load(weights, map_location=meta["device"],
                               weights_only=True)
            model = ensemble_from_state(state, meta["n_chans"],
                                        meta["n_times"], float(meta["sfreq"]))
            return model.to(meta["device"]).eval()
        n_members = CV_FOLDS * CV_SEEDS if CV_FOLDS > 0 else N_SEEDS
        model = SleepOnsetEnsemble(
            meta["n_chans"], meta["n_times"], float(meta["sfreq"]),
            n_members, ADAPT, USE_CNN, FEATS, STAGE_WEIGHT > 0, PREP,
            AE_MODE, dom=bool(EXTERNAL) and CV_FOLDS > 0)
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
        model.set_quantile(q)
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
        K = len(model.members) // max(1, CV_SEEDS)
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

        # External pseudo-Muse PSG recordings: training-only, every fold.
        ext_recs, ext_mask = [], None
        if EXTERNAL and model.members[0].dom:
            ext_mask = np.zeros(len(y_all), dtype=bool)
            n_sub = int(sub_int.max()) + 1
            for path, w in _external_files():
                X_e, y_e, rec_e, onset_e, sub_e, ds_e = _load_external(path)
                n0, rec_off = len(y_all), int(rec.max()) + 1
                _, sub_e_int = np.unique(sub_e, return_inverse=True)
                sub_e_int = sub_e_int + n_sub
                n_sub = int(sub_e_int.max()) + 1
                new = [(idx + n0, s) for idx, s in _recordings(
                    y_e, rec_e, onset_e, sub_e_int, rng, pseudo=False)]
                ext_recs += new
                X_all = torch.cat([X_all, X_e])
                y_all = torch.cat([y_all, y_e])
                rec = np.concatenate([rec, rec_e + rec_off])
                ext_mask = np.concatenate(
                    [ext_mask, np.ones(len(y_e), dtype=bool)])
                for idx, _ in new:
                    weights[id(idx)] = w * _window_weights(y_all[idx].numpy())
                names_e, counts_e = np.unique(
                    ds_e[[i[0] - n0 for i, _ in new]], return_counts=True)
                print(f"[CNN-GRU] + {len(new)} external recordings "
                      f"({', '.join(f'{n} {c}' for n, c in zip(names_e, counts_e))}"
                      f") from {os.path.basename(path)}, {len(y_e)} windows, "
                      f"loss weight x{w}")
                del X_e

        R = len(model.members) // K
        oof_rep = [[None] * len(recs) for _ in range(R)]
        for j, member in enumerate(model.members):
            k, rep_i = j % K, j // K
            tag = f"fold {k + 1}/{K}" + (f" seed {rep_i + 1}/{R}" if R > 1
                                          else "")
            tr = [r for r, f in zip(recs, rec_fold) if f != k] + ext_recs
            va_i = [i for i, f in enumerate(rec_fold) if f == k]
            va = [recs[i] for i in va_i]
            if member.ae:
                # This fold's autoencoder and feature stats, from its
                # training sleepers only (keeps the out-of-fold honest).
                member.to(out_device)
                # Muse windows only: the autoencoder and stats ignore the
                # external recordings.
                tr_idx = np.concatenate([idx for idx, _ in tr
                                         if ext_mask is None
                                         or not ext_mask[idx[0]]])
                _train_autoencoder(member, X_all, tr_idx, device, tag,
                                   seed=MEMBER_SEED + rep_i)
                s = X_all[rng.choice(tr_idx, min(20000, len(tr_idx)),
                                     replace=False)].float()
                with torch.no_grad():
                    f_k = member.band_powers(s.to(member.chan_std.device))
                member.bp_mean.copy_(f_k.mean(0))
                member.bp_std.copy_(f_k.std(0).clamp_min(1e-3))
            member.to(device)
            best = _train_member(member, 1000 * (MEMBER_SEED + rep_i) + k,
                                 X_all, y_all, tr, va, weights, device, tag,
                                 (stage_t, stage_m) if use_stage else None,
                                 ext_mask)
            print(f"[CNN-GRU] {tag.replace('/' + str(K), '', 1)}: {len(tr)} "
                  f"train / {len(va)} held-out recordings, best held-out "
                  f"W-bMAE {best:.1f} s")
            for i in va_i:
                oof_rep[rep_i][i] = _sequence_proba(member, X_all,
                                                    recs[i][0], device)

        if R > 1:
            for r_i, o in enumerate(oof_rep):
                q_r, s_r = min(_scores(o, y_all, recs).items(),
                               key=lambda kv: kv[1])
                print(f"[CNN-GRU] seed {MEMBER_SEED + r_i} alone: out-of-fold "
                      f"W-bMAE {s_r:.1f} s, q={q_r:.2f}")
        oof = [torch.stack(ps).mean(0) for ps in zip(*oof_rep)]

        all_scores = _scores(oof, y_all, recs)
        q, score = min(all_scores.items(), key=lambda kv: kv[1])
        model.set_quantile(q)
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
        _save_oof(recs, rec_split, rec_fold, names, y_all, oof, q)
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
        w = (model.member_w / model.member_w.sum()).to(device)
        p = (torch.stack([_sequence_proba(m, X_all, idx, device)
                          for m in model.members]) * w.view(-1, 1, 1)).sum(0)
        seq = model._seconds(p, 0)
        model.reset_state()
        t0 = time.time()
        stream = torch.cat([model.predict(X_all[i:i + 1].float().to(device))
                            for i in idx])
        per_window = (time.time() - t0) / max(1, len(idx))
        model.reset_state()
        diff = (seq.clamp(0, CAP_S) - stream).abs().max().item()
        print(f"[CNN-GRU] streaming check: max |seq - stream| = {diff:.4f} s; "
              f"{len(model.members)} members, {1000 * per_window:.1f} ms per "
              f"window on {device}")
        if diff > BIN_S:
            raise RuntimeError("streamed predictions differ from the "
                               "sequence forward pass")

    def save_model(self, model, path):
        torch.save(model.state_dict(), path / WEIGHTS_FILE)
