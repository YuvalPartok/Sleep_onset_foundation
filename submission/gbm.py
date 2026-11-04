"""Track 03 (Sleep Onset) submission: classical ML baseline.

Hand-crafted spectral features per 5 s window + causal context features over
the recording so far -> scikit-learn HistGradientBoostingRegressor trained
with an absolute-error loss and W-bMAE-like sample weights.

Per-window features (per channel unless noted): log absolute and relative
band powers (delta, theta, alpha, sigma, beta, high/EMG), theta/alpha ratio,
spectral entropy, 95 % spectral edge, log variance, and one AF7-AF8 slow
(< 1 Hz) power as an eye-movement proxy.

Context features, computed causally from the windows already seen in the
recording (the streaming state, cleared by ``reset_state``): log elapsed
time, the expanding mean of each feature and the current window's deviation
from it, ~1 min and ~5 min EMAs, and the change since the first 2 minutes.

``SLEEP_GBM_FEATURES=time`` keeps only the elapsed time (a control that
shows how much the EEG adds). ``SLEEP_TRAIN_FRAC`` trains on a fraction of
the training sleepers. The validation sleepers are drawn exactly as in
``cnn_gru.py``, so validation scores are comparable across the two solvers.

Self-contained: numpy, scikit-learn, joblib and torch only.
"""

import os
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

CAP_S = 600.0
WEIGHTS_FILE = "weights.joblib"
CLIP_UV = 500.0
BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 15.0), (15.0, 30.0),
         (30.0, 60.0))
RANGE_EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)
RANGE_WEIGHTS = (10.0, 5.0, 3.0, 1.0)

FEATURES = os.environ.get("SLEEP_GBM_FEATURES", "all")      # all | time
TRAIN_FRAC = float(os.environ.get("SLEEP_TRAIN_FRAC", 1.0))
MEMBER_SEED = int(os.environ.get("SLEEP_SEED", 0))
SEED = 0                                 # validation split seed (fixed)
VAL_FRACTION = 0.15                      # same as cnn_gru.py

EMA_FAST_S = 60.0
EMA_SLOW_S = 300.0
BASELINE_S = 120.0

GBM_PARAMS = dict(loss="absolute_error", learning_rate=0.05, max_iter=600,
                  max_leaf_nodes=31, min_samples_leaf=200,
                  l2_regularization=1.0, max_features=0.5)


# --- features ----------------------------------------------------------------


def window_features(X, sfreq):
    """Raw windows (N, C, T) microvolts -> per-window features (N, F)."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=-1, keepdims=True)
    X = np.clip(X, -CLIP_UV, CLIP_UV)
    N, C, T = X.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    spec = np.abs(np.fft.rfft(X * np.hanning(T), axis=-1)) ** 2   # (N, C, K)

    nyq = sfreq / 2
    power = np.stack([spec[..., (freqs >= lo) & (freqs < min(hi, nyq))]
                      .sum(-1) for lo, hi in BANDS], axis=-1)       # (N, C, B)
    log_abs = np.log(power + 1e-6)
    log_rel = np.log(power / (power.sum(-1, keepdims=True) + 1e-6) + 1e-6)
    theta_alpha = log_abs[..., 1] - log_abs[..., 2]

    band = (freqs >= 0.5) & (freqs < min(60.0, nyq))
    p = spec[..., band]
    p = p / (p.sum(-1, keepdims=True) + 1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum(-1) / np.log(p.shape[-1])
    edge = freqs[band][np.argmax(p.cumsum(-1) >= 0.95, axis=-1)]
    log_var = np.log(X.var(-1) + 1e-6)

    a, b = (1, 2) if C >= 3 else (0, min(1, C - 1))   # AF7, AF8 on Muse
    eog = X[:, a] - X[:, b]
    eog_spec = np.abs(np.fft.rfft(eog * np.hanning(T), axis=-1)) ** 2
    slow = np.log(eog_spec[:, (freqs > 0) & (freqs < 1.0)].sum(-1) + 1e-6)

    return np.concatenate([
        log_abs.reshape(N, -1), log_rel.reshape(N, -1), theta_alpha, entropy,
        edge, log_var, slow[:, None]], axis=1).astype(np.float32)


class ContextState:
    """Causal per-recording summaries of the window features."""

    def __init__(self, window_s):
        self.window_s = window_s
        self.a_fast = min(1.0, window_s / EMA_FAST_S)
        self.a_slow = min(1.0, window_s / EMA_SLOW_S)
        self.n_base = max(1, int(round(BASELINE_S / window_s)))
        self.reset()

    def reset(self):
        self.n = 0
        self.total = self.fast = self.slow = self.base_sum = None

    def step(self, f):
        """Features of the next window (F,) -> context features (D,)."""
        f = f.astype(np.float64)
        if self.n == 0:
            self.total = np.zeros_like(f)
            self.base_sum = np.zeros_like(f)
            self.fast = f.copy()
            self.slow = f.copy()
        else:
            self.fast += self.a_fast * (f - self.fast)
            self.slow += self.a_slow * (f - self.slow)
        self.n += 1
        self.total += f
        if self.n <= self.n_base:
            self.base_sum += f
        mean = self.total / self.n
        base = self.base_sum / min(self.n, self.n_base)
        elapsed_min = (self.n - 1) * self.window_s / 60.0
        t = np.array([np.log1p(elapsed_min), elapsed_min])
        if FEATURES == "time":
            return t
        return np.concatenate([t, f, f - mean, self.fast, self.slow,
                               self.fast - base, self.slow - base,
                               self.fast - self.slow])

    def run(self, feats):
        """Features (L, F) of consecutive windows -> context (L, D)."""
        return np.stack([self.step(f) for f in feats]).astype(np.float32)


# --- model -------------------------------------------------------------------


class SleepOnsetGBM:
    """Streaming wrapper: window features -> context -> gradient boosting."""

    def __init__(self, sfreq, n_times, estimator=None):
        self.sfreq = float(sfreq)
        self.window_s = n_times / self.sfreq
        self.clf = estimator
        self.state = ContextState(self.window_s)

    def reset_state(self):
        self.state.reset()

    def predict(self, X):
        """Consecutive windows (B, C, T) of one recording -> (B,) seconds."""
        ctx = self.state.run(window_features(to_numpy(X), self.sfreq))
        if self.clf is None:
            return np.full(len(ctx), CAP_S / 2)
        return np.clip(self.clf.predict(ctx), 0.0, CAP_S)


# --- training helpers (local only) -------------------------------------------


def _window_weights(y):
    """Per-window weights of one recording, mimicking its W-bMAE (sum 1)."""
    edges = np.asarray(RANGE_EDGES)
    r = np.clip(np.digitize(y, edges[1:-1]), 0, len(RANGE_WEIGHTS) - 1)
    w = np.zeros(len(y))
    present = np.unique(r)
    total = sum(RANGE_WEIGHTS[b] for b in present)
    for b in present:
        mask = r == b
        w[mask] = RANGE_WEIGHTS[b] / total / mask.sum()
    return w


def _load(dataset, sfreq):
    """Every window once -> features (N, F), y, record, onset, subject."""
    import torch
    from torch.utils.data import DataLoader
    F, ys, recs, onsets, subs = [], [], [], [], []
    t0 = time.time()
    for X, y, info in DataLoader(dataset, batch_size=512, shuffle=False):
        F.append(window_features(X.float().cpu().numpy(), sfreq))
        ys.append(torch.as_tensor(y).float().cpu().reshape(-1).numpy())
        recs.append(np.asarray(info["record_id"]).reshape(-1))
        onsets.append(np.asarray(info["onset"], dtype=np.float64).reshape(-1))
        subs.append(np.asarray(info.get("subject_id", info["record_id"]))
                    .reshape(-1))
    print(f"[GBM] features of {sum(len(f) for f in F)} windows in "
          f"{time.time() - t0:.0f} s")
    return (np.concatenate(F), np.concatenate(ys), np.concatenate(recs),
            np.concatenate(onsets), np.concatenate(subs))


def _recordings(y, rec, onset, sub, rng):
    """Same as cnn_gru._recordings: time-ordered indices + subject."""
    out = []
    if len(np.unique(rec)) < 4:
        perm = rng.permutation(len(y))
        for i, part in enumerate(np.array_split(perm, 10)):
            out.append((part[np.argsort(-y[part], kind="stable")], i))
        return out
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        out.append((idx[np.argsort(onset[idx], kind="stable")],
                    int(sub[idx[0]])))
    return out


def _subsample_subjects(recs, frac):
    """Same as cnn_gru._subsample_subjects."""
    if frac >= 1.0:
        return recs
    subjects = np.unique([s for _, s in recs])
    rng = np.random.default_rng(12345)
    keep = set(rng.choice(subjects, max(1, int(round(frac * len(subjects)))),
                          replace=False))
    return [r for r in recs if r[1] in keep]


def _stack(recs, feats, y_all, window_s):
    X, y, w = [], [], []
    for idx, _ in recs:
        X.append(ContextState(window_s).run(feats[idx]))
        y.append(np.clip(y_all[idx], 0, CAP_S))
        w.append(_window_weights(y_all[idx]))
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


def _wbmae(clf, recs, feats, y_all, window_s):
    from benchmark_utils.metrics import binned_mae
    scores, preds = [], []
    for idx, _ in recs:
        p = np.clip(clf.predict(ContextState(window_s).run(feats[idx])),
                    0, CAP_S)
        preds.append(p)
        scores.append(binned_mae(y_all[idx], p, RANGE_EDGES, RANGE_WEIGHTS))
    return float(np.mean(scores)), preds


def _save_val_predictions(val_recs, rec, sub, y_all, preds):
    """Validation predictions -> build/<label>_valpred.npz (not zipped)."""
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
             y=y_all[idx], pred=np.concatenate(preds))
    print(f"[GBM] validation predictions -> {out}")


class Solver(CompetSolver):

    name = "GBM-Context"

    def load_model(self, meta):
        weights = meta["submission_dir"] / WEIGHTS_FILE
        estimator = None
        if weights.exists() and self.train_loader is None:
            estimator = joblib.load(weights)
        return SleepOnsetGBM(meta["sfreq"], meta["n_times"], estimator)

    def fit(self, model, train_loader):
        rng = np.random.default_rng(SEED)
        feats, y_all, rec, onset, sub = _load(train_loader.dataset,
                                              model.sfreq)
        recs = _recordings(y_all, rec, onset, sub, rng)
        subjects = np.unique([s for _, s in recs])
        n_val = max(1, int(round(VAL_FRACTION * len(subjects))))
        val_subjects = set(rng.choice(subjects, n_val, replace=False))
        train_recs = [r for r in recs if r[1] not in val_subjects]
        val_recs = [r for r in recs if r[1] in val_subjects]
        train_recs = _subsample_subjects(train_recs, TRAIN_FRAC)
        print(f"[GBM] {len(train_recs)} train / {len(val_recs)} val "
              f"recordings ({len(subjects)} subjects); features={FEATURES}, "
              f"train_frac={TRAIN_FRAC}, seed={MEMBER_SEED}")

        Xtr, ytr, wtr = _stack(train_recs, feats, y_all, model.window_s)
        print(f"[GBM] training on {Xtr.shape[0]} windows x "
              f"{Xtr.shape[1]} features")

        # Pick the number of boosting rounds on the validation sleepers'
        # W-bMAE, by growing the ensemble with warm_start.
        params = {**GBM_PARAMS, "min_samples_leaf": max(
            5, min(GBM_PARAMS["min_samples_leaf"], len(ytr) // 50))}
        clf = HistGradientBoostingRegressor(
            **params, early_stopping=False, warm_start=True,
            random_state=MEMBER_SEED)
        best, best_iter, t0 = float("inf"), None, time.time()
        for n_iter in range(50, GBM_PARAMS["max_iter"] + 1, 50):
            clf.set_params(max_iter=n_iter)
            clf.fit(Xtr, ytr, sample_weight=wtr)
            score, _ = _wbmae(clf, val_recs, feats, y_all, model.window_s)
            print(f"[GBM] {n_iter} trees: val W-bMAE {score:.1f} s "
                  f"({time.time() - t0:.0f} s)")
            if score < best:
                best, best_iter = score, n_iter

        clf = HistGradientBoostingRegressor(
            **{**params, "max_iter": best_iter}, early_stopping=False,
            random_state=MEMBER_SEED)
        clf.fit(Xtr, ytr, sample_weight=wtr)
        score, preds = _wbmae(clf, val_recs, feats, y_all, model.window_s)
        model.clf = clf
        print(f"[GBM] ensemble val W-bMAE {score:.1f} s, {best_iter} trees")
        _save_val_predictions(val_recs, rec, sub, y_all, preds)

    def save_model(self, model, path):
        joblib.dump(model.clf, path / WEIGHTS_FILE)
