#!/usr/bin/env python3
"""External PSG sleep onsets as pseudo-Muse training windows.

Downloads two open PhysioNet PSG datasets from the anonymous S3 mirror
(``physionet-open``), one recording at a time (download, cut, delete, so the
disk never holds more than a few recordings), and writes their pre-N2
windows in the Muse task's format to one compressed file:

* Sleep-EDF Expanded, sleep cassette (Kemp et al. 2000; 153 nights, 78
  sleepers, 100 Hz; ~7.6 GB): Fpz-Cz, Pz-Oz, horizontal EOG. R&K stage 2 is
  taken as N2.
* HMC sleep staging (Alvarez-Estevez & Rijsman 2021/2022; 151 nights, 256 Hz;
  ~17 GB): F4-M1, C3-M2, E1-M2, E2-M2 (plus C4, O2), AASM stages.

Pseudo-Muse channels (Muse records TP9, AF7, AF8, TP10 against Fpz; a
mastoid is close to TP9/TP10, and Fpz to the frontal sites), in microvolts:

    HMC        TP9 = -(F4-M1)   AF7 = (E1-M2) - (F4-M1)
               TP10 = -(C3-M2)  AF8 = (E2-M2) - (F4-M1)   (AF7-AF8 = E1-E2)
    Sleep-EDF  TP9 = -(Fpz-Cz)  AF7 = +EOG / 2
               TP10 = Pz-Oz     AF8 = -EOG / 2            (AF7-AF8 = EOG)

These only approximate Muse; the solver gives them a domain flag and a lower
loss weight, and never scores them.

Each stream starts U seconds before the first N2 epoch. The Muse recordings
start when the sleeper starts the recording, so U is HMC's own lights-off
time when it is annotated (and 60-1800 s before N2); otherwise U is drawn
from the public Muse onset latencies (MUSE_ONSET_PCT), so the elapsed-time
input means the same as on Muse. As in the Muse task, the stream is cut into
5 s windows at 128 Hz from its start; a window ending at time t has target
``min(first_N2 - t, 600)`` s.

Output (``--out``, default external_windows.npz): X (N, 4, 640) float16 raw
microvolts; y float32; rec int (recording index); onset float (window start
within its stream, s); subject str; dataset str.

Usage:
    python scripts/prep_external.py                        # both, all nights
    python scripts/prep_external.py --limit 2 --out /tmp/ext_test.npz
"""

import argparse
import concurrent.futures as cf
import os
import tempfile
import time
from pathlib import Path

import numpy as np

BUCKET = "physionet-open"
SFREQ = 128.0
WIN_S = 5.0
CAP_S = 600.0
MAX_U_S = 1800.0
# Percentiles 0, 5, ..., 100 of the 540 public Muse onset latencies (s).
MUSE_ONSET_PCT = [60, 268, 330, 390, 420, 450, 510, 540, 600, 630, 690, 750,
                  810, 870, 960, 1001, 1080, 1197, 1270, 1378, 1500]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f} s] {msg}", flush=True)


def s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                            max_pool_connections=32))


def list_keys(client, prefix):
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys


def recordings(client, dataset):
    """[(subject, psg key, annotation key)] of one dataset."""
    if dataset == "sleepedf":
        keys = list_keys(client, "sleep-edfx/1.0.0/sleep-cassette/")
        psg = {Path(k).name[:6]: k for k in keys if k.endswith("-PSG.edf")}
        hyp = {Path(k).name[:6]: k for k in keys
               if k.endswith("-Hypnogram.edf")}
        # SC4ssN: ss = subject, N = night.
        return [(f"sleepedf-{n[3:5]}", psg[n], hyp[n])
                for n in sorted(psg) if n in hyp]
    keys = list_keys(client, "hmc-sleep-staging/1.1/recordings/")
    names = sorted(Path(k).stem for k in keys
                   if k.endswith(".edf") and "_sleepscoring" not in k)
    pre = "hmc-sleep-staging/1.1/recordings/"
    return [(f"hmc-{n}", pre + f"{n}.edf", pre + f"{n}_sleepscoring.edf")
            for n in names if pre + f"{n}_sleepscoring.edf" in keys]


def draw_u(rng):
    pct = rng.uniform(0, 100)
    return float(np.interp(pct, np.arange(0, 101, 5), MUSE_ONSET_PCT))


def process(dataset, subject, psg_key, ann_key, seed, tmp):
    import mne
    mne.set_log_level("ERROR")
    client = s3()
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(dir=tmp) as d:
        psg = os.path.join(d, "psg.edf")
        ann = os.path.join(d, "ann.edf")
        client.download_file(BUCKET, ann_key, ann)
        annots = mne.read_annotations(ann)
        desc = [str(s) for s in annots.description]
        n2 = annots.onset[np.array([s in ("Sleep stage 2", "Sleep stage N2")
                                    for s in desc], dtype=bool)]
        if not len(n2):
            return None, "no N2"
        first_n2 = float(n2.min())
        u = None
        lights = annots.onset[np.array(["lights off" in s.lower()
                                        for s in desc], dtype=bool)]
        if len(lights):
            cand = first_n2 - float(lights.min())
            if 60.0 <= cand <= MAX_U_S:
                u = cand
        if u is None:
            u = draw_u(rng)
        u = min(u, first_n2)
        if u < 2 * WIN_S:
            return None, "N2 too close to the start"
        client.download_file(BUCKET, psg_key, psg)
        raw = mne.io.read_raw_edf(psg, preload=False)
        start = first_n2 - u
        raw.crop(tmin=max(0.0, start - 2.0),
                 tmax=min(raw.times[-1], first_n2 + 2.0))
        raw.load_data()
        names = raw.ch_names
        if dataset == "sleepedf":
            need = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
        else:
            need = ["EEG F4-M1", "EEG C3-M2", "EOG E1-M2", "EOG E2-M2"]
        if any(n not in names for n in need):
            return None, f"missing channels ({names})"
        raw.pick(need)
        if raw.info["sfreq"] > SFREQ:                      # anti-alias
            raw.filter(None, 0.45 * SFREQ, verbose="ERROR")
        raw.resample(SFREQ)
        x = raw.get_data() * 1e6                           # V -> uV
        t0 = raw.first_time if hasattr(raw, "first_time") else 0.0
        offset = int(round((start - max(0.0, start - 2.0)) * SFREQ))
        if dataset == "sleepedf":
            fpz_cz, pz_oz, eog = x
            muse = np.stack([-fpz_cz, eog / 2, -eog / 2, pz_oz])
        else:
            f4, c3, e1, e2 = x
            muse = np.stack([-f4, e1 - f4, e2 - f4, -c3])
        n_win = int(u // WIN_S)
        n_t = int(WIN_S * SFREQ)
        seg = muse[:, offset:offset + n_win * n_t]
        n_win = seg.shape[1] // n_t
        if n_win < 2:
            return None, "too short after cropping"
        seg = seg[:, :n_win * n_t]
        X = seg.reshape(4, n_win, n_t).transpose(1, 0, 2)
        ends = (np.arange(n_win) + 1) * WIN_S
        y = np.minimum(u - ends, CAP_S).clip(0, None)
        del t0
        return dict(X=np.clip(X, -6e4, 6e4).astype(np.float16),
                    y=y.astype(np.float32),
                    onset=(np.arange(n_win) * WIN_S).astype(np.float64),
                    subject=subject, dataset=dataset, u=u), "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="sleepedf,hmc")
    ap.add_argument("--out", default="external_windows.npz")
    ap.add_argument("--limit", type=int, default=0,
                    help="recordings per dataset (0 = all); for testing")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tmp", default=None, help="scratch folder for downloads")
    args = ap.parse_args()

    client = s3()
    jobs = []
    for ds in args.datasets.split(","):
        recs = recordings(client, ds)
        if args.limit:
            recs = recs[:args.limit]
        log(f"{ds}: {len(recs)} recordings")
        jobs += [(ds, *r) for r in recs]

    results, skipped = [], {}
    with cf.ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(process, ds, sub, pk, ak, i, args.tmp): (ds, pk)
                for i, (ds, sub, pk, ak) in enumerate(jobs)}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            ds, pk = futs[fut]
            try:
                res, why = fut.result()
            except Exception as err:  # noqa: BLE001 - keep going
                res, why = None, f"error {err}"
            if res is None:
                skipped[Path(pk).name] = why
            else:
                results.append(res)
            if n % 10 == 0 or n == len(jobs):
                log(f"{n}/{len(jobs)} done, {len(results)} kept, "
                    f"{len(skipped)} skipped")

    if not results:
        raise SystemExit(f"nothing kept; skipped: {skipped}")
    results.sort(key=lambda r: (r["dataset"], r["subject"], r["u"]))
    X = np.concatenate([r["X"] for r in results])
    np.savez_compressed(
        args.out, X=X,
        y=np.concatenate([r["y"] for r in results]),
        rec=np.concatenate([np.full(len(r["y"]), i)
                            for i, r in enumerate(results)]),
        onset=np.concatenate([r["onset"] for r in results]),
        subject=np.concatenate([np.full(len(r["y"]), r["subject"])
                                for r in results]),
        dataset=np.concatenate([np.full(len(r["y"]), r["dataset"])
                                for r in results]))
    for ds in sorted({r["dataset"] for r in results}):
        rs = [r for r in results if r["dataset"] == ds]
        log(f"{ds}: {len(rs)} recordings, {sum(len(r['y']) for r in rs)} "
            f"windows, median stream {np.median([r['u'] for r in rs]):.0f} s")
    for name, why in sorted(skipped.items())[:20]:
        log(f"skipped {name}: {why}")
    log(f"wrote {args.out}: {X.shape[0]} windows, "
        f"{os.path.getsize(args.out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
