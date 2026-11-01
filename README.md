# EEG sleep foundation — Neural Interfaces 2026, Track 03 (Sleep Onset)

Predict the seconds until the first N2 sleep epoch from a 5 s EEG window
(Muse: TP9, AF7, AF8, TP10 at 128 Hz → `X` of shape `(B, 4, 640)`, raw µV).
`predict(X)` returns `(B,)` floats in `[0, 600]`. Evaluation streams one window
at a time (`B = 1`) per recording, on a `copy.deepcopy` of the model.

Submission guide: `competition/codabench/pages/participate.md`
Track contract: `competition/tracks/sleep_onset/README.md`

## Layout

```text
submission/submission.py     # the solver that goes into the ZIP (MLP for now)
scripts/build_submission.py  # train → reload check → ZIP
competition/                 # clone of neural-interfaces26/2026-competition (benchmark harness)
build/<label>/               # exported submission folder (submission.py + weights.pt)
dist/<label>.zip             # upload this on Codabench → My Submissions
```

`submission.py` must stay self-contained: one file, `class Solver(CompetSolver)`,
imports limited to the packages in `competition/requirements.txt`, weights read
from `meta["submission_dir"]`.

## Setup

```bash
conda activate sleep_foundation          # Python 3.12
pip install -r competition/requirements.txt
```

## Build a submission

Pipeline check on synthetic data with the real warm-up shapes (no download;
the score is meaningless):

```bash
python scripts/build_submission.py
```

Train on the public Muse warm-up data:

```bash
benchopt prepare competition/tracks/sleep_onset -d Interaxon2026Muse
python scripts/build_submission.py --data muse
```

Each run trains via `Solver.fit`, exports to `build/<label>/`, re-runs the exported
folder inference-only (as Codabench does) and writes `dist/<label>.zip`.
Add `--force` to overwrite a previous build.


## Check the local score
The build script already scores the model on the Muse test split. After the build finishes, run:

```bash
python -c "import pandas as pd; d=pd.read_parquet('build/mlp_muse_eval.parquet'); print(d.filter(like='objective_').iloc[-1])"
```

`objective_wbmae_recording_mean` is the track's metric.
## Where the model lives and what you can change

Everything you edit is in `submission/`. Everything under `competition/` is the
official harness (data loading, the train/test split, streaming, the metric). It is
the same code Codabench runs, so **don't edit it**: a change there would not exist on
the server, and your local score would stop matching the leaderboard.

### Map of `submission/submission.py`

| Lines (approx.) | Part | What it controls | Runs on Codabench? |
|---|---|---|---|
| 15–17 | Imports | Only `torch` and packages in `competition/requirements.txt` (e.g. `numpy`, `scipy`, `mne`, `braindecode`, `scikit-learn`) | yes |
| 20–27 | Hyperparameters | `HIDDEN`, `DROPOUT`, `EPOCHS`, `LR`, `WEIGHT_DECAY` | partly (architecture sizes) |
| 30–46 | `SleepOnsetMLP.__init__` | **Architecture**: layers, plus buffers that are saved in `weights.pt` | yes |
| 48–51 | `SleepOnsetMLP.forward` | **Preprocessing + forward pass**: raw `(B, C, T)` → fraction of 600 s | yes |
| 53–56 | `predict` | Converts output to seconds; the final output must be `(B,)` in `[0, 600]` | yes |
| 58–60 | `reset_state` | Called at the start of each recording; reset any streaming state here | yes |
| 63–75 | `Solver.load_model` | Builds the model from `meta` and loads `weights.pt` | yes |
| 77–106 | `Solver.fit` | **Training**: normalization stats, optimizer, loss, epochs | no (local only) |
| 108–109 | `Solver.save_model` | Writes the weights that `load_model` reads back | no (local only) |

### The rule that decides where code goes

Codabench calls only `load_model(meta)` and then `model.predict(X)`, with **raw** data:
5 s windows of unfiltered µV, shape `(1, 4, 640)` (TP9, AF7, AF8, TP10 at 128 Hz),
one window at a time, in time order within each recording.
So anything applied to the input **at inference** must live inside the model
(`forward`/`predict`), not in `fit` or in a separate script.

### Where to add each kind of change

**Preprocessing** (filtering, re-referencing, scaling, clipping artifacts, spectral features)
- Put it at the top of `SleepOnsetMLP.forward` (where the per-channel normalization is
  now), or in its own `nn.Module` that `forward` calls first. During training, `fit` calls
  `model(X)`, so train and inference use exactly the same preprocessing.
- Anything learned from data (means/stds, PCA, filter coefficients) goes in
  `self.register_buffer(...)` in `__init__` and is filled in `fit`; buffers are saved in
  `weights.pt` automatically. `chan_mean`/`chan_std` show the pattern.
- Use torch ops (e.g. `torch.fft.rfft` for band powers, `torchaudio`-style FIR via
  `F.conv1d`) so it runs on `meta["device"]`. NumPy/SciPy/MNE work too, but move data to
  CPU and back.
- Example: replace flattening with log band powers (delta 0.5–4, theta 4–8, alpha 8–12,
  sigma 12–15, beta 15–30 Hz) per channel → 4 × 5 = 20 features into the MLP.
  Use `meta["sfreq"]` (pass it into `__init__`) instead of hard-coding 128.

**Architecture** (CNN, EEGNet, transformer, foundation model)
- Write a new `nn.Module` class in the same file and construct it in `load_model`.
  Keep `predict(X) -> (B,)` seconds. Additional `.py` files are **not** allowed in the
  submission, so all model code must be in this one file.
- Build sizes from `meta["n_chans"]`, `meta["n_times"]`, `meta["sfreq"]`, never fixed numbers.
- `braindecode` models are available (e.g. `from braindecode.models import EEGNetv4`);
  see `competition/tracks/sleep_onset/solvers/eegnet_reg.py` and `reve_probe.py` for
  working examples, including the pretrained REVE encoder.

**Using the time context of a recording** (sequence models, smoothing)
- Windows arrive in order and the model is a fresh `copy.deepcopy` per recording, so you
  can keep history on `self` (e.g. a buffer of past features, or an exponentially
  smoothed prediction) and clear it in `reset_state`. Predictions should get smaller as
  sleep approaches, so smoothing over time can help a lot.

**Training** (loss, optimizer, epochs, augmentation, validation split)
- All in `Solver.fit`. `train_loader` yields `(X, y, info)`: `X` `(B, C, T)` and `y` `(B,)`
  seconds, already on the device. `info` is a dict with keys such as `record_id` and
  `subject_id`.
- There is no separate validation loader. To choose epochs or do early stopping, hold
  out some recordings yourself, splitting by `info["subject_id"]` so the same person
  isn't in both parts.
- The metric weights errors near onset most heavily (bins `0–40`, `40–90`, `90–300`,
  `300–600` s with weights 10, 5, 3, 1), so weighting the loss the same way, or sampling
  short-latency windows more often, should help.
- Augmentation (noise, time shifts, channel dropout) is applied to `X` inside `fit` only.

**Weights and other files**
- `save_model(model, path)` can write any files (e.g. `weights.pt`, `config.json`,
  a `joblib` head); read them in `load_model` from `meta["submission_dir"]`.
- If you rename or replace `weights.pt`, also update the check in
  `scripts/build_submission.py` (`for name in ("submission.py", "weights.pt")`).

### Trying several models side by side

Copy `submission/submission.py` to a new file, give its `Solver` a unique `name`, and build it:

```bash
cp submission/submission.py submission/bandpower_mlp.py   # then edit it, set name = "Bandpower-MLP"
python scripts/build_submission.py --data muse --solver submission/bandpower_mlp.py
python -c "import pandas as pd; d=pd.read_parquet('build/bandpower_mlp_muse_eval.parquet'); print(d.filter(like='objective_').iloc[-1])"
```

Each solver gets its own `build/<file>_<data>/` folder and `dist/<file>_<data>.zip`.
The exported file is always renamed to `submission.py` inside the ZIP, as Codabench requires.
Test a new model on `--data simulated` first; it takes seconds and catches most shape bugs.
