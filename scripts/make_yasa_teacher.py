#!/usr/bin/env python3
"""Compact YASA teacher labels for the staging output of submission/cnn_gru.py.

Reads build/yasa_check/epochs.parquet (written by scripts/yasa_check.py) and
writes data/yasa_teacher.npz: for every recording, YASA's per-30 s-epoch
probabilities of three classes, averaged over the channel configs:

  0 wake    = p(WAKE)
  1 drowsy  = p(N1) + p(REM)   (REM before sleep onset is almost certainly
                                slow eye movements mistaken for REM)
  2 sleep   = p(N2) + p(N3)

Layout: ``keys`` (R,) "sub-XXX/ses-YYY", ``start`` (R,) offset of each
recording's first epoch in ``proba``, ``n`` (R,) its epoch count, ``proba``
(E, 3) float16. Epoch k of a recording covers [30 k, 30 k + 30) s from its
start, the same time origin as the harness windows.

Usage (any env with pandas + pyarrow):
    python scripts/make_yasa_teacher.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "build" / "yasa_check" / "epochs.parquet"
OUT = ROOT / "data" / "yasa_teacher.npz"


def main() -> int:
    ep = pd.read_parquet(SRC)
    ep["wake"] = ep.p_WAKE
    ep["drowsy"] = ep.p_N1 + ep.p_REM
    ep["sleep"] = ep.p_N2 + ep.p_N3
    avg = (ep.groupby(["subject", "session", "epoch_start_s"])
           [["wake", "drowsy", "sleep"]].mean().reset_index()
           .sort_values(["subject", "session", "epoch_start_s"]))
    keys, start, n, probas = [], [], [], []
    offset = 0
    for (sub, ses), g in avg.groupby(["subject", "session"], sort=True):
        p = g[["wake", "drowsy", "sleep"]].to_numpy()
        p = p / p.sum(1, keepdims=True)
        keys.append(f"{sub}/{ses}")
        start.append(offset)
        n.append(len(p))
        probas.append(p)
        offset += len(p)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, keys=np.array(keys), start=np.array(start),
                        n=np.array(n),
                        proba=np.concatenate(probas).astype(np.float16))
    print(f"wrote {OUT}: {len(keys)} recordings, {offset} epochs, "
          f"{OUT.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
