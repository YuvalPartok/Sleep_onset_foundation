#!/usr/bin/env python3
"""External PSG sleep onsets as pseudo-Muse training windows.

Streams open sleep datasets from anonymous S3 (PhysioNet's ``physionet-open``
mirror and OpenNeuro's ``openneuro.org`` bucket), one recording at a time
(download, cut, delete, so the disk never holds more than a few recordings),
and writes their pre-N2 windows in the Muse task's format:

* ``sleepedf``: Sleep-EDF Expanded, sleep cassette (Kemp et al. 2000; 153
  nights, 78 sleepers, 100 Hz; ~7.6 GB). Fpz-Cz, Pz-Oz, horizontal EOG; R&K
  stage 2 is taken as N2.
* ``sleepedf_st``: Sleep-EDF Expanded, sleep telemetry (44 nights, 22
  sleepers, same channels; ~1.1 GB).
* ``hmc``: HMC sleep staging (Alvarez-Estevez & Rijsman 2021/2022; 151
  nights, 256 Hz; ~17 GB). F4-M1, C3-M2, E1-M2, E2-M2, AASM stages.
* ``p2018``: PhysioNet/CinC Challenge 2018 "You Snooze, You Win" training
  set (994 scored nights, 200 Hz, AASM stages in the ``.arousal``
  annotations). The ``.mat`` files are frame-interleaved int16 at a 24-byte
  offset, so only the bytes up to the first N2 are fetched with an HTTP range
  request (~20 MB of the ~135 MB per night).
* ``boas``: Bitbrain Open Access Sleep, OpenNeuro ds005555 (128 nights, 100
  people). Only the wearable forehead headband (HB_1, HB_2, 256 Hz,
  referenced to the left mastoid M1, ~28 MB per night) is downloaded; the
  stages are the 3-scorer consensus of the simultaneous PSG (same epoch
  grid). Nights whose PSG and headband AI stagings disagree (< 60% of
  epochs) are skipped as misaligned.

Pseudo-Muse channels (Muse records TP9, AF7, AF8, TP10 against Fpz; a
mastoid is close to TP9/TP10, and Fpz to the frontal sites), in microvolts:

    HMC        TP9 = -(F4-M1)   AF7 = (E1-M2) - (F4-M1)
               TP10 = -(C3-M2)  AF8 = (E2-M2) - (F4-M1)   (AF7-AF8 = E1-E2)
    p2018      as HMC, but there is no E2: AF8 = (F3-M2) - (F4-M1)
    Sleep-EDF  TP9 = -(Fpz-Cz)  AF7 = +EOG / 2
               TP10 = Pz-Oz     AF8 = -EOG / 2            (AF7-AF8 = EOG)
    BOAS       AF7 = (HB_1 - HB_2) / 2, AF8 = -AF7,
               TP9 = TP10 = -(HB_1 + HB_2) / 2  (M1 - forehead ~ TP - Fpz)

These only approximate Muse; the solver gives them a domain flag and its own
loss weight, and never scores them.

A night is skipped if its scoring starts asleep, or if N3 or REM comes before
its first N2 (a truncated or mislabelled start). Each stream starts U seconds
before the first N2 epoch. The Muse recordings start when the sleeper starts
the recording, so U is the time from the dataset's own start marker (HMC
lights-off, p2018 first scored epoch, BOAS recording start) when that is
60-1800 s before N2; otherwise U is drawn from the public Muse onset
latencies (MUSE_ONSET_PCT), so the elapsed-time input means the same as on
Muse. As in the Muse task, the stream is cut into 5 s windows at 128 Hz from
its start; a window ending at time t has target ``min(first_N2 - t, 600)``.

Every recording's result is written to ``--shard-dir/<dataset>/<name>.npz``
(or ``<name>.skip`` with the reason) as soon as it is done, and a re-run
skips the recordings that already have one: an interrupted run (Kaggle or
Colab disconnect) resumes where it stopped. Put the shard dir on Google Drive
on Colab.

Output: one file per dataset, ``--out-dir/external_<dataset>.npz``, or all
datasets in one file with ``--out``. Arrays: X (N, 4, 640) float16 raw
microvolts; y float32; rec int (recording index within the file); onset
float (window start within its stream, s); subject str; dataset str.

Usage:
    python scripts/prep_external.py --datasets p2018,boas,sleepedf_st \\
        --out-dir /kaggle/working --shard-dir /kaggle/working/ext_shards
    python scripts/prep_external.py --datasets p2018 --limit 3 --out-dir /tmp
    python scripts/prep_external.py --datasets sleepedf,hmc \\
        --out external_windows.npz          # the round 8 file
"""

import argparse
import concurrent.futures as cf
import io
import os
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

BUCKET = "physionet-open"
OPENNEURO = "openneuro.org"
BOAS = "ds005555"
P2018 = "challenge-2018/1.0.0/training/"
SFREQ = 128.0
WIN_S = 5.0
CAP_S = 600.0
MAX_U_S = 1800.0
DATASETS = ("sleepedf", "sleepedf_st", "hmc", "p2018", "boas")
# Percentiles 0, 5, ..., 100 of the 540 public Muse onset latencies (s).
MUSE_ONSET_PCT = [60, 268, 330, 390, 420, 450, 510, 540, 600, 630, 690, 750,
                  810, 870, 960, 1001, 1080, 1197, 1270, 1378, 1500]
# Stage labels -> W, N1, N2, N3, R (R&K 3 and 4 are N3).
MNE_STAGES = {"Sleep stage W": "W", "Sleep stage 1": "N1",
              "Sleep stage N1": "N1", "Sleep stage 2": "N2",
              "Sleep stage N2": "N2", "Sleep stage 3": "N3",
              "Sleep stage 4": "N3", "Sleep stage N3": "N3",
              "Sleep stage R": "R"}
BOAS_STAGES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f} s] {msg}", flush=True)


def s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                            max_pool_connections=32,
                                            retries={"max_attempts": 8}))


def list_keys(client, prefix, bucket=BUCKET):
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys


def get_bytes(client, key, bucket=BUCKET, end=None):
    kw = {} if end is None else {"Range": f"bytes=0-{end - 1}"}
    return client.get_object(Bucket=bucket, Key=key, **kw)["Body"].read()


def recordings(client, dataset):
    """[dict(name, subject, ...)] of one dataset."""
    if dataset in ("sleepedf", "sleepedf_st"):
        folder = "sleep-cassette" if dataset == "sleepedf" else "sleep-telemetry"
        keys = list_keys(client, f"sleep-edfx/1.0.0/{folder}/")
        psg = {Path(k).name[:6]: k for k in keys if k.endswith("-PSG.edf")}
        hyp = {Path(k).name[:6]: k for k in keys
               if k.endswith("-Hypnogram.edf")}
        # SC4ssN / ST7ssN: ss = subject, N = night.
        return [dict(name=n, subject=f"{dataset}-{n[3:5]}", psg=psg[n],
                     ann=hyp[n]) for n in sorted(psg) if n in hyp]
    if dataset == "hmc":
        pre = "hmc-sleep-staging/1.1/recordings/"
        keys = set(list_keys(client, pre))
        names = sorted(Path(k).stem for k in keys
                       if k.endswith(".edf") and "_sleepscoring" not in k)
        return [dict(name=n, subject=f"hmc-{n}", psg=pre + f"{n}.edf",
                     ann=pre + f"{n}_sleepscoring.edf")
                for n in names if pre + f"{n}_sleepscoring.edf" in keys]
    if dataset == "p2018":
        keys = set(list_keys(client, P2018))
        names = sorted({Path(k).stem for k in keys if k.endswith(".arousal")})
        return [dict(name=n, subject=f"p2018-{n}", pre=P2018 + f"{n}/{n}")
                for n in names if P2018 + f"{n}/{n}.mat" in keys]
    if dataset == "boas":
        import pandas as pd
        part = pd.read_csv(io.BytesIO(get_bytes(
            client, f"{BOAS}/participants.tsv", OPENNEURO)), sep="\t")
        pid = dict(zip(part.participant_id, part.pid.astype(str)))
        subs = sorted(pid, key=lambda s: int(s.split("-")[1]))
        return [dict(name=s, subject=f"boas-{pid[s]}",
                     pre=f"{BOAS}/{s}/eeg/{s}_task-Sleep_acq-") for s in subs]
    raise SystemExit(f"unknown dataset {dataset}; known: {DATASETS}")


def draw_u(rng):
    pct = rng.uniform(0, 100)
    return float(np.interp(pct, np.arange(0, 101, 5), MUSE_ONSET_PCT))


def check_stages(stages):
    """[(onset s, label)] -> (first N2 onset, None) or (None, reason)."""
    stages = sorted(stages)
    n2 = [t for t, s in stages if s == "N2"]
    if not n2:
        return None, "no N2"
    first_n2 = n2[0]
    if stages[0][1] in ("N2", "N3", "R"):
        return None, "scoring starts asleep"
    if any(s in ("N3", "R") for t, s in stages if t < first_n2):
        return None, "N3/REM before the first N2"
    return float(first_n2), None


def choose_u(first_n2, start_marker, rng):
    """Stream length before N2: the dataset's start marker if plausible."""
    u = None
    if start_marker is not None:
        cand = first_n2 - float(start_marker)
        if 60.0 <= cand <= MAX_U_S:
            u = cand
    if u is None:
        u = draw_u(rng)
    return min(u, first_n2)


def to_windows(muse, sfreq, start, u):
    """(4, n) uV at ``sfreq`` whose sample 0 is at time 0 -> windows."""
    from scipy.signal import resample_poly
    i0 = int(round(max(0.0, start - 2.0) * sfreq))
    i1 = int(round((start + u + 2.0) * sfreq))
    seg = muse[:, i0:i1].astype(np.float64)
    ratio = Fraction(SFREQ / sfreq).limit_denominator(1000)
    seg = resample_poly(seg, ratio.numerator, ratio.denominator, axis=1)
    offset = int(round((start - i0 / sfreq) * SFREQ))
    return windows_from(seg, offset, u)


def windows_from(seg, offset, u):
    n_t = int(WIN_S * SFREQ)
    n_win = int(u // WIN_S)
    seg = seg[:, offset:offset + n_win * n_t]
    n_win = seg.shape[1] // n_t
    if n_win < 2:
        return None
    X = seg[:, :n_win * n_t].reshape(4, n_win, n_t).transpose(1, 0, 2)
    if not np.isfinite(X).all():
        return None
    ends = (np.arange(n_win) + 1) * WIN_S
    y = np.minimum(u - ends, CAP_S).clip(0, None)
    return dict(X=np.clip(X, -6e4, 6e4).astype(np.float16),
                y=y.astype(np.float32),
                onset=(np.arange(n_win) * WIN_S).astype(np.float64))


def process_edf(dataset, job, rng, client, d):
    """sleepedf, sleepedf_st, hmc: EDF + EDF+ hypnogram (MNE)."""
    import mne
    mne.set_log_level("ERROR")
    ann = os.path.join(d, "ann.edf")
    client.download_file(BUCKET, job["ann"], ann)
    annots = mne.read_annotations(ann)
    desc = [str(s) for s in annots.description]
    stages = [(float(t), MNE_STAGES[s]) for t, s in zip(annots.onset, desc)
              if s in MNE_STAGES]
    first_n2, why = check_stages(stages)
    if why:
        return None, why
    lights = [float(t) for t, s in zip(annots.onset, desc)
              if "lights off" in s.lower()]
    u = choose_u(first_n2, min(lights) if lights else None, rng)
    if u < 2 * WIN_S:
        return None, "N2 too close to the start"
    psg = os.path.join(d, "psg.edf")
    client.download_file(BUCKET, job["psg"], psg)
    raw = mne.io.read_raw_edf(psg, preload=False)
    start = first_n2 - u
    raw.crop(tmin=max(0.0, start - 2.0),
             tmax=min(raw.times[-1], first_n2 + 2.0))
    raw.load_data()
    if dataset == "hmc":
        need = ["EEG F4-M1", "EEG C3-M2", "EOG E1-M2", "EOG E2-M2"]
    else:
        need = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
    if any(n not in raw.ch_names for n in need):
        return None, f"missing channels ({raw.ch_names})"
    raw.pick(need)
    if raw.info["sfreq"] > SFREQ:                      # anti-alias
        raw.filter(None, 0.45 * SFREQ, verbose="ERROR")
    raw.resample(SFREQ)
    x = raw.get_data() * 1e6                           # V -> uV
    offset = int(round((start - max(0.0, start - 2.0)) * SFREQ))
    if dataset == "hmc":
        f4, c3, e1, e2 = x
        muse = np.stack([-f4, e1 - f4, e2 - f4, -c3])
    else:
        fpz_cz, pz_oz, eog = x
        muse = np.stack([-fpz_cz, eog / 2, -eog / 2, pz_oz])
    res = windows_from(muse, offset, u)
    return (None, "too short after cropping") if res is None else (
        dict(res, u=u), "ok")


def process_p2018(job, rng, client, d):
    """PhysioNet 2018: .hea + .arousal, then a range read of the .mat."""
    import wfdb
    base = os.path.join(d, "r")
    for ext in ("hea", "arousal"):
        with open(f"{base}.{ext}", "wb") as f:
            f.write(get_bytes(client, f"{job['pre']}.{ext}"))
    hdr = wfdb.rdheader(base)
    ann = wfdb.rdann(base, "arousal")
    fs = float(hdr.fs)
    stages = [(s / fs, n) for s, n in zip(ann.sample, ann.aux_note)
              if n in ("W", "N1", "N2", "N3", "R")]
    first_n2, why = check_stages(stages)
    if why:
        return None, why
    u = choose_u(first_n2, min(t for t, _ in stages), rng)
    if u < 2 * WIN_S:
        return None, "N2 too close to the start"
    names = list(hdr.sig_name)
    need = ["F3-M2", "F4-M1", "C3-M2", "E1-M2"]
    if any(n not in names for n in need) or set(hdr.fmt) != {"16"}:
        return None, f"unexpected header ({names}, {set(hdr.fmt)})"
    n_sig, off = hdr.n_sig, int(hdr.byte_offset[0] or 0)
    n_samp = min(int((first_n2 + 3.0) * fs), hdr.sig_len)
    body = get_bytes(client, f"{job['pre']}.mat", end=off + 2 * n_sig * n_samp)
    raw = np.frombuffer(body[off:], dtype="<i2")
    raw = raw[:len(raw) // n_sig * n_sig].reshape(-1, n_sig).T
    gain = np.asarray(hdr.adc_gain, dtype=np.float64)[:, None]
    base_ = np.asarray(hdr.baseline, dtype=np.float64)[:, None]
    sig = (raw - base_) / gain                         # uV for the EEG/EOG
    f3, f4, c3, e1 = (sig[names.index(n)] for n in need)
    muse = np.stack([-f4, e1 - f4, f3 - f4, -c3])
    res = to_windows(muse, fs, first_n2 - u, u)
    return (None, "too short after cropping") if res is None else (
        dict(res, u=u), "ok")


def process_boas(job, rng, client, d):
    """BOAS: headband EDF + the PSG's consensus stages on the same grid."""
    import mne
    import pandas as pd
    mne.set_log_level("ERROR")
    tsv = lambda kind: pd.read_csv(io.BytesIO(get_bytes(
        client, f"{job['pre']}{kind}_events.tsv", OPENNEURO)), sep="\t")
    psg, hb = tsv("psg"), tsv("headband")
    n = min(len(psg), len(hb))
    if n < 20 or (psg.stage_ai.values[:n] == hb.stage_ai.values[:n]).mean() < 0.6:
        return None, "PSG and headband stagings disagree (misaligned?)"
    stages = [(float(t), BOAS_STAGES[int(s)])
              for t, s in zip(psg.onset, psg.stage_hum) if int(s) in BOAS_STAGES]
    first_n2, why = check_stages(stages)
    if why:
        return None, why
    u = choose_u(first_n2, 0.0, rng)
    if u < 2 * WIN_S:
        return None, "N2 too close to the start"
    path = os.path.join(d, "hb.edf")
    client.download_file(OPENNEURO, f"{job['pre']}headband_eeg.edf", path)
    raw = mne.io.read_raw_edf(path, preload=False)
    if not {"HB_1", "HB_2"} <= set(raw.ch_names):
        return None, f"missing channels ({raw.ch_names})"
    fs = raw.info["sfreq"]
    stop = min(raw.n_times, int((first_n2 + 3.0) * fs))
    hb1, hb2 = raw.get_data(picks=["HB_1", "HB_2"], start=0, stop=stop) * 1e6
    af7 = (hb1 - hb2) / 2
    tp = -(hb1 + hb2) / 2
    muse = np.stack([tp, af7, -af7, tp])
    res = to_windows(muse, fs, first_n2 - u, u)
    return (None, "too short after cropping") if res is None else (
        dict(res, u=u), "ok")


def process(dataset, job, seed, tmp):
    rng = np.random.default_rng(seed)
    client = s3()
    with tempfile.TemporaryDirectory(dir=tmp) as d:
        if dataset == "p2018":
            return process_p2018(job, rng, client, d)
        if dataset == "boas":
            return process_boas(job, rng, client, d)
        return process_edf(dataset, job, rng, client, d)


def run_dataset(client, dataset, args):
    """Process (or resume) one dataset -> list of per-recording dicts."""
    recs = recordings(client, dataset)
    if args.limit:
        recs = recs[:args.limit]
    shard_dir = Path(args.shard_dir) / dataset
    shard_dir.mkdir(parents=True, exist_ok=True)
    todo = [(i, r) for i, r in enumerate(recs)
            if not (shard_dir / f"{r['name']}.npz").exists()
            and not (shard_dir / f"{r['name']}.skip").exists()]
    log(f"{dataset}: {len(recs)} recordings, {len(recs) - len(todo)} "
        f"already done, {len(todo)} to go")
    seed0 = DATASETS.index(dataset) * 100000
    with cf.ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(process, dataset, r, seed0 + i, args.tmp): r
                for i, r in todo}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            r = futs[fut]
            try:
                res, why = fut.result()
            except Exception as err:  # noqa: BLE001 - keep going
                res, why = None, f"error {type(err).__name__}: {err}"
                if "error" in why and any(k in why for k in (
                        "Timeout", "Connection", "Throttl", "SlowDown")):
                    log(f"{r['name']}: {why} (will retry on a re-run)")
                    continue
            if res is None:
                (shard_dir / f"{r['name']}.skip").write_text(why)
            else:
                np.savez(shard_dir / f"{r['name']}.npz", subject=r["subject"],
                         **res)
            if n % 20 == 0 or n == len(todo):
                log(f"{dataset}: {n}/{len(todo)} processed")

    kept, skipped = [], {}
    for r in recs:
        p = shard_dir / f"{r['name']}.npz"
        if p.exists():
            z = np.load(p)
            kept.append(dict(X=z["X"], y=z["y"], onset=z["onset"],
                             u=float(z["u"]), subject=str(z["subject"]),
                             dataset=dataset))
        elif (shard_dir / f"{r['name']}.skip").exists():
            skipped[r["name"]] = (shard_dir / f"{r['name']}.skip").read_text()
    reasons = {}
    for why in skipped.values():
        key = why.split("(")[0].strip()[:60]
        reasons[key] = reasons.get(key, 0) + 1
    log(f"{dataset}: kept {len(kept)}, skipped {len(skipped)} {reasons}")
    return kept


def write(results, out):
    if not results:
        log(f"nothing kept for {out}; not written")
        return
    results = sorted(results, key=lambda r: (r["dataset"], r["subject"], r["u"]))
    X = np.concatenate([r["X"] for r in results])
    np.savez_compressed(
        out, X=X,
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
        log(f"{out}: {ds} {len(rs)} recordings, "
            f"{sum(len(r['y']) for r in rs)} windows, "
            f"{len({r['subject'] for r in rs})} sleepers, median stream "
            f"{np.median([r['u'] for r in rs]):.0f} s")
    log(f"wrote {out}: {X.shape[0]} windows, "
        f"{os.path.getsize(out) / 1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="p2018,boas,sleepedf_st",
                    help=f"comma-separated, from {', '.join(DATASETS)}")
    ap.add_argument("--out", default=None,
                    help="one file for all datasets (else one per dataset)")
    ap.add_argument("--out-dir", default=".",
                    help="folder for external_<dataset>.npz")
    ap.add_argument("--shard-dir", default="ext_shards",
                    help="per-recording results; a re-run resumes from them")
    ap.add_argument("--limit", type=int, default=0,
                    help="recordings per dataset (0 = all); for testing")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tmp", default=None, help="scratch folder for downloads")
    args = ap.parse_args()

    client = s3()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    everything = []
    for ds in datasets:
        kept = run_dataset(client, ds, args)
        if args.out:
            everything += kept
        else:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            write(kept, os.path.join(args.out_dir, f"external_{ds}.npz"))
    if args.out:
        write(everything, args.out)


if __name__ == "__main__":
    main()
