#!/usr/bin/env python3
"""Can YASA's sleep staging serve as a teacher on the Muse recordings?

Runs YASA's pretrained sleep-stage classifier (30 s epochs) on every full
Muse recording (start -> annotated first N2 + 5 min) and compares it with the
one label we have, the annotated first N2 epoch:

* timing: YASA's first N2 (or N3) epoch vs the annotated onset, in seconds;
* plausibility: the stages and probabilities YASA gives as a function of the
  time to the annotated onset (it should move W -> N1 -> N2 near onset, and
  say N2 after it).

YASA expects a central EEG channel (+ optional EOG); Muse only has TP9, AF7,
AF8 and TP10, so several Muse channel combinations are tried (``CONFIGS``).

Outputs (``--out``, default build/yasa_check/):
  recordings.csv  one row per recording x config (timing metrics)
  epochs.parquet  one row per 30 s epoch x config (stage + probabilities),
                  reusable as distillation targets if the check passes
  summary.txt     the tables printed at the end
  run.log         the progress log (also printed)

Runs in its own environment, never in the submission one:
    conda create -n sleep_yasa --clone sleep_foundation
    conda activate sleep_yasa
    pip install yasa
    conda install -c conda-forge llvm-openmp     # LightGBM needs it on macOS
    python scripts/yasa_check.py                 # all 540 recordings
    python scripts/yasa_check.py --limit 5       # quick test
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "competition/tracks/sleep_onset/data/neural_compet/"
        "Interaxon2026Muse/download/nm000287")
EPOCH_S = 30.0
STAGES = ("WAKE", "N1", "N2", "N3", "REM")

# (name, EEG channel, EOG channel or None). "EOG" = the AF7-AF8 derivation.
CONFIGS = (
    ("TP10+AF7", "TP10", "AF7"),
    ("TP9+AF8", "TP9", "AF8"),
    ("TP10+EOG", "TP10", "EOG"),
)
# Time-to-onset bins (s) for the plausibility tables; negative = after onset.
TTO_BINS = [-300, -150, 0, 30, 90, 180, 300, 600, 1e9]
TTO_LABELS = ["after 2.5-5 min", "after 0-2.5 min", "0-30 s before",
              "30-90 s before", "90-180 s before", "3-5 min before",
              "5-10 min before", ">10 min before"]

LOG_FILE: Path | None = None
T0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


# --- data ----------------------------------------------------------------------


def index_recordings(root: Path) -> pd.DataFrame:
    """One row per recording: paths, annotated onset, split."""
    rows = []
    for vhdr in sorted(root.glob("sub-*/ses-*/eeg/*_eeg.vhdr")):
        sub, ses = vhdr.parts[-4], vhdr.parts[-3]
        events = pd.read_csv(vhdr.with_name(vhdr.name.replace(
            "_eeg.vhdr", "_events.tsv")), sep="\t")
        onset = events.loc[events.trial_type == "n2_onset", "onset"]
        sessions = pd.read_csv(root / sub / f"{sub}_sessions.tsv", sep="\t")
        split = sessions.set_index("session_id").loc[ses, "split"]
        rows.append({"subject": sub, "session": ses, "vhdr": str(vhdr),
                     "onset_s": float(onset.iloc[0]), "split": split})
    return pd.DataFrame(rows)


def stage_recording(rec: dict) -> tuple[list[dict], list[pd.DataFrame], float]:
    """Run every config on one recording -> (metric rows, epoch tables, s)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    warnings.filterwarnings("ignore")
    import mne
    import yasa
    mne.set_log_level("ERROR")
    logging.getLogger("yasa").setLevel(logging.ERROR)

    t0 = time.time()
    raw = mne.io.read_raw_brainvision(rec["vhdr"], preload=True,
                                      verbose="ERROR")
    raw.set_channel_types({c: "eeg" for c in raw.ch_names})
    raw = mne.set_bipolar_reference(raw, "AF7", "AF8", ch_name="EOG",
                                    drop_refs=False, verbose="ERROR")
    duration = raw.n_times / raw.info["sfreq"]
    onset = rec["onset_s"]

    rows, tables = [], []
    for name, eeg, eog in CONFIGS:
        hyp = yasa.SleepStaging(raw, eeg_name=eeg, eog_name=eog).predict()
        stage = np.asarray(hyp.hypno.values).astype(str)
        proba = hyp.proba.reindex(columns=list(STAGES), fill_value=0.0)
        start = np.arange(len(stage)) * EPOCH_S
        tto = onset - start                      # > 0: before the onset
        sleep = np.isin(stage, ["N2", "N3"])

        first = np.flatnonzero(sleep)
        sustained = np.flatnonzero(sleep[:-1] & sleep[1:])
        before = (start + EPOCH_S) <= onset + 1e-6
        after = start >= onset - 1e-6
        near = before & (tto <= 180)
        far = before & (tto > 180)
        rows.append({
            **{k: rec[k] for k in ("subject", "session", "split", "onset_s")},
            "config": name, "duration_s": duration, "n_epochs": len(stage),
            "first_n2_s": start[first[0]] if len(first) else np.nan,
            "first_sustained_n2_s": (start[sustained[0]] if len(sustained)
                                     else np.nan),
            "frac_n2_before": sleep[before].mean() if before.any() else np.nan,
            "frac_n2_after": sleep[after].mean() if after.any() else np.nan,
            "frac_n1_last3min": ((stage[near] == "N1").mean() if near.any()
                                 else np.nan),
            "frac_n1_earlier": ((stage[far] == "N1").mean() if far.any()
                                else np.nan),
        })
        tables.append(pd.DataFrame({
            "subject": rec["subject"], "session": rec["session"],
            "split": rec["split"], "config": name, "epoch_start_s": start,
            "time_to_onset_s": tto, "stage": stage,
            **{f"p_{s}": proba[s].to_numpy() for s in STAGES}}))
    return rows, tables, time.time() - t0


# --- summary -------------------------------------------------------------------


def summarise(recs: pd.DataFrame, epochs: pd.DataFrame) -> str:
    out = []
    recs = recs.assign(err=recs.first_n2_s - recs.onset_s,
                       err_sus=recs.first_sustained_n2_s - recs.onset_s)

    def timing(err: pd.Series) -> dict:
        found = err.notna()
        a = err[found].abs()
        return {"found N2": f"{found.mean():.0%}",
                "median |err| s": round(a.median(), 0),
                "median err s": round(err[found].median(), 0),
                "within 30 s": f"{(a <= 30).mean():.0%}",
                "within 60 s": f"{(a <= 60).mean():.0%}",
                "within 120 s": f"{(a <= 120).mean():.0%}",
                "early (>60 s)": f"{(err[found] < -60).mean():.0%}",
                "late (>60 s)": f"{(err[found] > 60).mean():.0%}"}

    for col, title in (("err", "first N2/N3 epoch"),
                       ("err_sus", "first 2 consecutive N2/N3 epochs")):
        t = pd.DataFrame({c: timing(g[col]) for c, g in recs.groupby("config")})
        out.append(f"Timing of YASA's {title} vs the annotated onset "
                   f"(err = YASA - annotation; negative = too early)\n"
                   f"{t.to_string()}\n")

    t = recs.groupby("config")[["frac_n2_before", "frac_n2_after",
                                 "frac_n1_last3min", "frac_n1_earlier"]
                                ].mean().T.round(2)
    out.append("Stage fractions (mean over recordings). Ideal: N2 before "
               "onset ~0, N2 after high, N1 higher in the last 3 min than "
               f"earlier\n{t.to_string()}\n")

    epochs = epochs.assign(bin=pd.cut(epochs.time_to_onset_s, TTO_BINS,
                                      labels=TTO_LABELS, right=False))
    for name, g in epochs.groupby("config"):
        stage_pct = (pd.crosstab(g.bin, g.stage, normalize="index") * 100
                     ).reindex(columns=list(STAGES), fill_value=0).round(0)
        mean_p = g.groupby("bin", observed=True)[[f"p_{s}" for s in STAGES]
                                                 ].mean().round(2)
        n = g.groupby("bin", observed=True).size().rename("epochs")
        table = pd.concat([n, stage_pct.add_prefix("% "), mean_p], axis=1)
        out.append(f"[{name}] stages (% of epochs) and mean probabilities by "
                   f"time to the annotated onset\n"
                   f"{table.iloc[::-1].to_string()}\n")
    return "\n".join(out)


# --- main ----------------------------------------------------------------------


def main() -> int:
    global LOG_FILE
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "yasa_check")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N recordings (quick test)")
    parser.add_argument("--jobs", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    LOG_FILE = args.out / "run.log"
    LOG_FILE.write_text("")

    import yasa
    log(f"YASA {yasa.__version__} | data {args.root} | out {args.out} | "
        f"{args.jobs} parallel jobs | configs: "
        + ", ".join(f"{n} (EEG {e}, EOG {o})" for n, e, o in CONFIGS))

    log("Stage 1/3: indexing recordings")
    recs = index_recordings(args.root)
    if args.limit:
        recs = recs.head(args.limit)
    log(f"  {len(recs)} recordings from {recs.subject.nunique()} subjects "
        f"({(recs.split == 'train').sum()} train / "
        f"{(recs.split == 'test').sum()} test); onset median "
        f"{recs.onset_s.median():.0f} s")

    log(f"Stage 2/3: YASA staging, {len(recs)} recordings x {len(CONFIGS)} "
        "configs")
    from joblib import Parallel, delayed
    rows, tables, t_start, done = [], [], time.time(), 0
    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(_safe_stage)(r) for r in recs.to_dict("records"))
    for rec, res in results:
        done += 1
        elapsed = time.time() - t_start
        eta = elapsed / done * (len(recs) - done)
        tag = f"  [{done}/{len(recs)}] {rec['subject']} {rec['session']}"
        if isinstance(res, str):
            log(f"{tag}: FAILED ({res})")
            continue
        r_rows, r_tables, secs = res
        rows += r_rows
        tables += r_tables
        first = {r["config"]: r["first_n2_s"] for r in r_rows}
        log(f"{tag}: onset {rec['onset_s']:.0f} s, YASA first N2 "
            + ", ".join(f"{c} {'-' if np.isnan(v) else f'{v:.0f}'}"
                        for c, v in first.items())
            + f" | {secs:.1f} s | ETA {eta / 60:.1f} min")

    log("Stage 3/3: saving and summarising")
    recs_df, epochs_df = pd.DataFrame(rows), pd.concat(tables, ignore_index=True)
    recs_df.to_csv(args.out / "recordings.csv", index=False)
    epochs_df.to_parquet(args.out / "epochs.parquet", index=False)
    summary = summarise(recs_df, epochs_df)
    (args.out / "summary.txt").write_text(summary)
    log(f"  wrote recordings.csv ({len(recs_df)} rows), epochs.parquet "
        f"({len(epochs_df)} rows), summary.txt")
    print("\n" + summary, flush=True)
    log(f"Done in {(time.time() - T0) / 60:.1f} min")
    return 0


def _safe_stage(rec: dict):
    try:
        return rec, stage_recording(rec)
    except Exception as err:            # noqa: BLE001 - report and continue
        return rec, f"{type(err).__name__}: {err}"


if __name__ == "__main__":
    raise SystemExit(main())
