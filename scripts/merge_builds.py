#!/usr/bin/env python3
"""Merge cross-validated CNN-GRU builds into one ensemble submission.

Every CV build (SLEEP_CV_FOLDS > 0) writes build/<label>_oof.npz: the
out-of-fold time-to-N2 distribution of every window of the 540 public
recordings. All CV builds use the same folds and recording order, so their
out-of-fold distributions can be mixed honestly: each recording is still
predicted only by models that never saw its sleeper.

1. Greedy ensemble selection with replacement (Caruana et al. 2004) picks
   build weights that minimise the out-of-fold W-bMAE, and coordinate descent
   picks the quantile per elapsed-time bucket (``q_time``). To show how much
   of the gain is selection luck, both are first fitted on folds 1-4 and
   scored on fold 5 (honest), then refitted on all folds.
2. The selected builds' members are concatenated into one SleepOnsetEnsemble
   (members of a build share its weight), saved as build/<label>/weights.pt
   with the solver as submission.py.
3. Unless --no-eval, the folder is reloaded inference-only with benchopt on
   the Muse data (as Codabench does, like scripts/build_submission.py), then
   zipped into dist/<label>.zip.

Usage:
    python scripts/merge_builds.py --label r8_merged r8_combo_s0 r8_ae64_s0 ...
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "competition" / "tracks" / "sleep_onset"
SOLVER = ROOT / "submission" / "cnn_gru.py"
EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0])
WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])
QUANTILES = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
TIME_EDGES_MIN = (2.0, 5.0, 10.0, 20.0)
BIN_S = 5.0


def load_oof(label, suffix="_oof"):
    d = np.load(ROOT / "build" / f"{label}{suffix}.npz")
    return {k: d[k] for k in d.files}


class Scorer:
    """W-bMAE per recording (benchmark_utils.metrics.binned_mae), averaged."""

    def __init__(self, oof, window_s=5.0):
        self.rec = oof["record"]
        self.y = oof["y"].astype(np.float64)
        self.n_rec = int(self.rec.max()) + 1
        keep = (self.y >= EDGES[0]) & (self.y <= EDGES[-1])
        rng = np.clip(np.digitize(self.y, EDGES[1:-1]), 0, 3)
        self.cell = np.where(keep, self.rec * 4 + rng, -1)
        counts = np.bincount(self.cell[keep], minlength=self.n_rec * 4)
        self.counts = counts.reshape(self.n_rec, 4)
        present = self.counts > 0
        self.norm = (present * WEIGHTS).sum(1)
        self.present = present
        # Recordings with a scored window (start-cut oofs drop some).
        self.valid = self.norm > 0
        self.keep = keep
        minutes = oof["position"] * window_s / 60.0
        self.bucket = np.digitize(minutes, TIME_EDGES_MIN, right=False)

    def per_recording(self, pred):
        err = np.abs(pred - self.y)
        sums = np.bincount(self.cell[self.keep], weights=err[self.keep],
                           minlength=self.n_rec * 4).reshape(self.n_rec, 4)
        means = np.where(self.present, sums / np.maximum(self.counts, 1), 0)
        return (means * WEIGHTS).sum(1) / np.maximum(self.norm, 1e-9)

    def score(self, pred, recs):
        recs = recs[self.valid[recs]]
        return float(self.per_recording(pred)[recs].mean())


def quantile_pred(cdf, q_per_window):
    k = (cdf < q_per_window[:, None]).sum(1)
    return np.minimum(k, cdf.shape[1] - 1) * BIN_S


def best_q(scorer, cdf, recs):
    """Best single quantile -> (score, q)."""
    return min((scorer.score(quantile_pred(cdf, np.full(len(cdf), q)), recs),
                q) for q in QUANTILES)


def tune_q_time(scorer, cdf, recs, q0):
    """Coordinate descent over the elapsed-time buckets' quantiles."""
    q_time = np.full(len(TIME_EDGES_MIN) + 1, q0)
    best = scorer.score(quantile_pred(cdf, q_time[scorer.bucket]), recs)
    for _ in range(2):
        for b in range(len(q_time)):
            for q in QUANTILES:
                trial = q_time.copy()
                trial[b] = q
                s = scorer.score(quantile_pred(cdf, trial[scorer.bucket]),
                                 recs)
                if s < best - 1e-6:
                    best, q_time = s, trial
    return best, q_time


def greedy(scorer, probas, recs, steps, sizes=None, max_members=None,
           init=None):
    """Caruana selection with replacement -> (counts per build, score, q).

    ``sizes`` (members per build) and ``max_members`` cap the merged
    ensemble: a build not yet picked is skipped once it would not fit.
    ``init`` (counts per build) starts the selection from a given ensemble,
    so it only grows from there when a pick improves it.
    """
    counts = np.zeros(len(probas), dtype=int)
    total = np.zeros_like(probas[0], dtype=np.float32)
    best_score, best_q_ = np.inf, 0.5
    if init is not None and np.any(init):
        counts = np.asarray(init, dtype=int).copy()
        total = sum(c * p for c, p in zip(counts, probas) if c).astype(np.float32)
        best_score, best_q_ = best_q(scorer, np.cumsum(total / counts.sum(), 1),
                                     recs)
    for _ in range(steps):
        trials = []
        used = sum(n for n, c in zip(sizes or [], counts) if c)
        for b, p in enumerate(probas):
            if (max_members and sizes and not counts[b]
                    and used + sizes[b] > max_members):
                continue
            mix = (total + p) / (counts.sum() + 1)
            s, q = best_q(scorer, np.cumsum(mix, 1), recs)
            trials.append((s, q, b))
        if not trials:
            break
        s, q, b = min(trials)
        if s >= best_score - 0.01:
            break
        best_score, best_q_ = s, q
        counts[b] += 1
        total += probas[b]
    return counts, best_score, best_q_


def mix_cdf(probas, counts):
    total = sum(c * p for c, p in zip(counts, probas) if c)
    return np.cumsum(total / counts.sum(), 1)


def load_solver():
    sys.path.insert(0, str(TRACK))
    spec = importlib.util.spec_from_file_location("cnn_gru", SOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge_states(labels, counts, q_time, mod):
    import torch
    merged, weights, j = {}, [], 0
    for label, c in zip(labels, counts):
        if not c:
            continue
        state = torch.load(ROOT / "build" / label / "weights.pt",
                           map_location="cpu", weights_only=True)
        mod.member_configs(state)            # fills old checkpoints' buffers
        n = len({k.split(".")[1] for k in state if k.startswith("members.")})
        w_old = state.get("member_w", torch.ones(n)).float()
        for i in range(n):
            pre = f"members.{i}."
            for k, v in state.items():
                if k.startswith(pre):
                    merged[f"members.{j}." + k[len(pre):]] = v
            weights.append(float(c) * float(w_old[i] / w_old.sum()))
            j += 1
    merged["member_w"] = torch.tensor(weights)
    merged["q_time"] = torch.tensor(q_time, dtype=torch.float32)
    merged["quantile"] = torch.tensor(float(np.median(q_time)))
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("builds", nargs="+", help="CV build labels (build/<label>)")
    ap.add_argument("--label", required=True, help="output build name")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-members", type=int, default=30,
                    help="cap on the merged ensemble's size (inference time)")
    ap.add_argument("--oof-suffix", default="_oof",
                    help="_oofT: select on start-cut streams (scripts/trim_oof.py)")
    ap.add_argument("--init", default="",
                    help="start the selection from these builds, e.g. "
                         "r15_tf_s0:1,r15_mingru_s1:2 (counts = weights)")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the benchopt inference-only check (no Muse data)")
    args = ap.parse_args()

    labels = [b for b in args.builds
              if (ROOT / "build" / f"{b}{args.oof_suffix}.npz").exists()
              and (ROOT / "build" / b / "weights.pt").exists()]
    missing = sorted(set(args.builds) - set(labels))
    if missing:
        print(f"skipping (no {args.oof_suffix}.npz or weights): {', '.join(missing)}")
    if not labels:
        raise SystemExit("no usable builds")
    oofs = [load_oof(b, args.oof_suffix) for b in labels]
    for b, o in zip(labels[1:], oofs[1:]):
        for key in ("record", "position", "y"):
            if not np.array_equal(o[key], oofs[0][key]):
                raise SystemExit(f"{b}: recordings differ from {labels[0]} "
                                 f"({key}); not the same CV setup")
    import torch
    sizes = []
    for b in labels:
        st = torch.load(ROOT / "build" / b / "weights.pt", map_location="cpu",
                        weights_only=True)
        sizes.append(len({k.split(".")[1] for k in st
                          if k.startswith("members.")}))
    scorer = Scorer(oofs[0])
    probas = [o["proba"].astype(np.float32) for o in oofs]
    init = None
    if args.init:
        want = dict((kv.split(":")[0], int(kv.split(":")[1]) if ":" in kv else 1)
                    for kv in args.init.split(","))
        init = np.array([want.get(b, 0) for b in labels])
        lost = sorted(set(want) - set(labels))
        if lost:
            print(f"--init builds not available, left out: {', '.join(lost)}")
        print("selection starts from: " + ", ".join(
            f"{b} x{c}" for b, c in zip(labels, init) if c))
    rec_fold = oofs[0]["rec_fold"]
    all_recs = np.arange(scorer.n_rec)
    print(f"{len(labels)} builds, {scorer.n_rec} recordings, "
          f"{len(scorer.y)} windows")

    print("\nsingle builds (out-of-fold W-bMAE, all recordings):")
    singles = []
    for b, p in zip(labels, probas):
        s, q = best_q(scorer, np.cumsum(p, 1), all_recs)
        singles.append(s)
        print(f"  {b:28s} {s:6.2f} s  (q={q}, {sizes[len(singles) - 1]} members)")

    # Honest check: select and tune on folds 1-4, score fold 5.
    last = int(rec_fold.max())
    fit_r, hold_r = all_recs[rec_fold != last], all_recs[rec_fold == last]
    counts, _, q = greedy(scorer, probas, fit_r, args.steps, sizes,
                          args.max_members, init)
    _, q_time = tune_q_time(scorer, mix_cdf(probas, counts), fit_r, q)
    cdf = mix_cdf(probas, counts)
    hold_ens = scorer.score(quantile_pred(cdf, q_time[scorer.bucket]), hold_r)
    hold_single = min(
        (best_q(scorer, np.cumsum(p, 1), fit_r)[0],
         scorer.score(quantile_pred(np.cumsum(p, 1), np.full(
             len(p), best_q(scorer, np.cumsum(p, 1), fit_r)[1])), hold_r))
        for p in probas)[1]
    print(f"\nhonest check, fitted on folds 1-{last}, scored on fold "
          f"{last + 1}: ensemble {hold_ens:.2f} s vs best single build "
          f"{hold_single:.2f} s")

    # Final: fit on all recordings.
    counts, s_sel, q = greedy(scorer, probas, all_recs, args.steps, sizes,
                              args.max_members, init)
    s_time, q_time = tune_q_time(scorer, mix_cdf(probas, counts), all_recs, q)
    print("\nselected (weight = times picked):")
    for b, c in zip(labels, counts):
        if c:
            print(f"  {b:28s} x{c}")
    print(f"out-of-fold W-bMAE: best single {min(singles):.2f} s, "
          f"ensemble {s_sel:.2f} s (q={q}), + time quantiles {s_time:.2f} s "
          f"(q_time={q_time.tolist()} for <{TIME_EDGES_MIN} min)")

    mod = load_solver()
    state = merge_states(labels, counts, q_time, mod)
    model = mod.ensemble_from_state(dict(state), 4, 640, 128.0)
    out_dir = ROOT / "build" / args.label
    zip_path = ROOT / "dist" / f"{args.label}.zip"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    shutil.copy(SOLVER, out_dir / "submission.py")
    print(f"\n{len(model.members)} members -> {out_dir}")

    if not args.no_eval:
        benchopt = shutil.which("benchopt") or str(
            Path(sys.executable).with_name("benchopt"))
        cmd = [benchopt, "run", str(TRACK), "-d", "Interaxon2026Muse",
               "-r", "1", "--no-cache", "--no-plot", "--no-html",
               "--no-display", "--local", "-o", "Sleep-onset[training=False]",
               "-s", str(out_dir / "submission.py"),
               "--output", str(ROOT / "build" / f"{args.label}_eval.parquet")]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT,
                       env={**os.environ, "COMPET_SUBMISSION_DIR": str(out_dir)})

    zip_path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.iterdir()):
            if path.is_file():
                zf.write(path, path.name)
    print(f"\nReady to upload: {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
