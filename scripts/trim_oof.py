#!/usr/bin/env python3
"""Out-of-fold distributions on start-cut streams (the sealed setup since
November 16), for merging with ``merge_builds.py --oof-suffix _oofT``.

Since the Nov 16 update the sealed streams no longer include the opening
minutes of each recording: the first window a model receives comes some
minutes after lights-out, and elapsed time and running statistics start
there. This script replays every public recording through the fold members
that never saw its sleeper (member j belongs to fold j % K), starting the
stream after a cut of TRIM_EVAL_MIN minutes (one cut per recording, fixed
seed, the same for every build), and writes build/<label>_oofT.npz in the
layout of <label>_oof.npz:

* windows before the cut get y = -1 (ignored by the scorer) and zero
  probabilities; recordings whose onset falls inside the cut have no
  scored window left;
* ``position`` counts windows from the first delivered window (negative
  before the cut), so the elapsed-time quantile buckets match the stream.

Needs the build's weights.pt and the public Muse data (the windows are
cached in build/muse_aligned.npz on first use). The window layout comes from
the build's _oof.npz, or from any other build's (all CV builds share it), so
builds without one (e.g. split from a merge) work too.

Usage:
    python scripts/trim_oof.py r8_ext1_s0 r9_all05_s0 ...
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "competition" / "tracks" / "sleep_onset"
SOLVER = ROOT / "submission" / "cnn_gru.py"
CACHE = ROOT / "build" / "muse_aligned.npz"
TRIM_SEED = 2026


def load_solver():
    sys.path.insert(0, str(TRACK))
    spec = importlib.util.spec_from_file_location("cnn_gru", SOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def aligned_windows(mod, ref_y):
    """Preprocessed public windows (N, C, T) in the out-of-fold order."""
    import torch
    if CACHE.exists():
        d = np.load(CACHE)
        X, y = torch.from_numpy(d["X"]), d["y"]
    else:
        from benchmark_utils.nb_task import load_task
        try:
            from benchopt.config import get_data_path
            data_dir = get_data_path("neural_compet")
        except Exception:                      # noqa: BLE001 - outside benchopt
            data_dir = TRACK / "data" / "neural_compet"
        cwd = os.getcwd()
        os.chdir(TRACK)
        try:
            loaders, _ = load_task("eeg", "_sleep_onset_stream",
                                   data_dir=os.path.abspath(data_dir),
                                   device="cpu", batch_size=512,
                                   target_transform=lambda y: y.reshape(()))
        finally:
            os.chdir(cwd)
        Xs, ys, recs, onsets, off = [], [], [], [], 0
        for split in ("train", "val", "test"):
            Xi, yi, ri, oi, _ = mod._load_windows(loaders[split].dataset)
            ri = ri.astype(np.int64) + off
            Xs.append(Xi), ys.append(yi), recs.append(ri), onsets.append(oi)
            off = int(ri.max()) + 1
        X, y_t = torch.cat(Xs), torch.cat(ys)
        rec, onset = np.concatenate(recs), np.concatenate(onsets)
        order = np.concatenate([i for i, _ in mod._recordings(
            y_t, rec, onset, np.zeros(len(y_t), int),
            np.random.default_rng(0), pseudo=False)])
        X, y = X[order], y_t[order].numpy()
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, X=X.numpy(), y=y)
    if not np.array_equal(y, ref_y):
        raise SystemExit("cached windows do not match the out-of-fold order; "
                         f"delete {CACHE} and rerun")
    return X


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("builds", nargs="+")
    ap.add_argument("--force", action="store_true", help="recompute existing")
    args = ap.parse_args()
    import torch
    mod = load_solver()
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    X = None
    for label in args.builds:
        src, dst = ROOT / "build" / f"{label}_oof.npz", ROOT / "build" / f"{label}_oofT.npz"
        wfile = ROOT / "build" / label / "weights.pt"
        if dst.exists() and not args.force:
            print(f"{label}: {dst.name} exists")
            continue
        if not src.exists():
            others = sorted(p for p in (ROOT / "build").glob("*_oof.npz"))
            src = others[0] if others else src
        if not (src.exists() and wfile.exists()):
            print(f"{label}: no weights or no _oof.npz layout; skipped")
            continue
        t0 = time.time()
        oof = dict(np.load(src))
        if X is None:
            X = aligned_windows(mod, oof["y"])
        rec, y, rf = oof["record"], oof["y"], oof["rec_fold"]
        K = int(rf.max()) + 1
        starts = np.r_[0, np.flatnonzero(np.diff(rec)) + 1]
        ends = np.r_[starts[1:], len(rec)]
        trims = np.random.default_rng(TRIM_SEED).choice(
            mod.TRIM_EVAL_MIN, len(starts))
        st = torch.load(wfile, map_location="cpu", weights_only=True)
        ens = mod.ensemble_from_state(st, X.shape[1], X.shape[2], 128.0)
        ens = ens.to(device).eval()
        M = len(ens.members)
        if M % K:
            print(f"{label}: {M} members is not a multiple of {K} folds; skipped")
            continue
        w = (ens.member_w / ens.member_w.sum()).cpu().numpy()
        window_s = ens.members[0].window_s
        proba = np.zeros_like(oof["proba"], dtype=np.float32)
        y_t, pos_t = np.full_like(y, -1.0), np.zeros_like(oof["position"])
        kept = 0
        with torch.inference_mode():
            for r, (a, b) in enumerate(zip(starts, ends)):
                k = a + int(round(trims[r] * 60 / window_s))
                pos_t[a:b] = np.arange(b - a) - (k - a)
                if k >= b or y[k] <= 0:
                    continue                   # onset inside the cut
                kept += 1
                x = X[k:b].float().unsqueeze(0).to(device)
                ms = [j for j in range(M) if j % K == rf[r]]
                p = sum(float(w[j]) * ens.members[j](x)[0][0].softmax(-1)
                        for j in ms) / float(w[ms].sum())
                proba[k:b] = p.cpu().numpy()
                y_t[k:b] = y[k:b]
        np.savez(dst, **{**oof, "proba": proba.astype(oof["proba"].dtype),
                         "y": y_t, "position": pos_t, "trim_min": trims})
        print(f"{label}: {kept}/{len(starts)} recordings after the cut "
              f"({sorted(set(trims.tolist()))} min) -> {dst.name} "
              f"({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
