# EEG sleep foundation: Neural Interfaces 2026, Track 03 (Sleep Onset)

The task: predict the seconds until the first N2 sleep epoch, streaming one 5 s EEG window at a time from a Muse headband.
- **Input:** TP9, AF7, AF8, TP10 at 128 Hz, so `X` has shape `(1, 4, 640)` in raw µV.
- **Output:** `predict(X)` returns a latency in `[0, 600]` s.
- **Metric:** W-bMAE. The mean absolute error is weighted by the true latency:

  | True latency | 0–40 s | 40–90 s | 90–300 s | 300–600 s |
  |---|---|---|---|---|
  | Weight | ×10 | ×5 | ×3 | ×1 |

## Final result

| | |
|---|---|
| Final submission | `r20_merge_free40` (`dist/r20_merge_free40.zip`) |
| Sealed W-bMAE | **34.43 s** |
| Codabench run time | 1,695 s |
| Leaderboard place | **61** |
| Local cut-stream out-of-fold / honest fold-5 check | 35.96 s / 35.94 s |

A full write-up of the model is in [`docs/r18_merge_free_model.md`](docs/r18_merge_free_model.md). It covers `r18_merge_free`, which is the final submission minus two builds.

---

## 1. The final model in one picture

The final model is an **ensemble of 40 models**: **8 builds × 5 cross-validation fold models**.
- Every fold model is trained without one fifth of the sleepers.
- All 40 are weighted equally.
- The ensemble averages their predicted distributions over the time to N2 and outputs a quantile of the average.

```
raw 5 s window (4 × 640 µV)
 ├─ remove DC, clip ±500 µV
 ├─ spectral models (35): Welch band powers → 66 hand-crafted features
 │                        log spectrum (27 bins × 5 signals) → frozen autoencoder / ICA code
 │                        + comparison with the same sleeper (running mean, 30 s EMA, trend)
 │                        + log elapsed time → GRU or minGRU (2 × 128)
 └─ raw-signal model (5): 50 Hz notch → CNN (4 conv blocks) → GRU (2 × 128)
                    ↓
     each model: distribution over the time to N2, in 121 bins of 5 s
                    ↓
     average of 40 → quantile q (depends on elapsed time) → seconds
```

### The 8 builds

| Build | Sequence model | Input features | Loss weight power *a* | External data | Score alone (cut-stream OOF) |
|---|---|---|---|---|---|
| `r15_mingru_s1` | minGRU | hand + AE 64 (all channels) | 1 | 4 PSG sets ×0.5 | 37.88 |
| `r13_t_ae64_all05_s1` | GRU | hand + AE 64 | 1 | 4 sets ×0.5 | 37.76 |
| `r13_t_ema30_s0` | GRU | hand + AE 32 | 1 | none | 38.06 |
| `r17_gru_w0_s1` | GRU | hand + AE 64 | 0 (flat) | 4 sets ×0.5 | 37.76 |
| `r18_ae_temporal_s0` | GRU | hand + AE 64 on **TP9/TP10 only** | 0.25 | 4 sets ×0.5 | 37.72 |
| `r18_ae_temporal_s1` | GRU | the same, seed 1 | 0.25 | 4 sets ×0.5 | 37.98 |
| `r18_mingru_ica_bag_s0` | minGRU | **ICA 32** + 46 of 66 hand features (random per fold) | 0.25 | 4 sets ×0.5 | 37.93 |
| `r20_cnn_gru_raw_s0` | GRU | **CNN on the notched raw signal only** | 0.25 | 4 sets ×0.5 | 41.34 |

Each build alone scores 37.7–41.3 s; the ensemble scores 35.96 s.

The raw-signal CNN is the weakest model alone, yet every merge in round 20 picked it. It's the only model that doesn't look at spectra, so its errors are the least correlated with the others'. Adding it gave the final 0.35 s sealed gain.

### Preprocessing (inside the model, at inference)
- **DC removal** per channel per window, then **clipping** at ±500 µV.
- **Welch band powers:** 2 s Hann segments with a 1 s hop, **skipping 48–52 Hz** (the recordings carry 50 Hz mains noise).
- **The AF7 − AF8 derivation** stands in for an EOG channel: horizontal eye movements have opposite polarity on the two sites.
- **For the raw-signal model:** a zero-phase 48–52 Hz notch per window (FFT), then division by the recording's running channel standard deviation.

### Features (spectral models)
- **66 hand-crafted features:**
  - absolute and relative log power in 6 bands per channel: δ 0.5–4, θ 4–8, α 8–12, σ 12–15, β 15–30, and 30–60 Hz for EMG;
  - θ/α ratio;
  - slow power below 0.6 Hz;
  - AF7−AF8 slow-eye-movement and eye-movement power;
  - spectral entropy;
  - 95 % spectral edge.
- **A learned spectral code** from a frozen encoder, fitted without labels per fold on the log spectrum (27 log-spaced bins, 0.4–47 Hz, × 5 signals). Three kinds:
  - a denoising autoencoder on all channels;
  - one on the temporal channels only;
  - ICA (linear, written exactly into the same encoder).
- **Comparison with the same sleeper:** each window's features are also given relative to the recording's running mean, the change against a 30 s EMA, and the EMA's trend against the mean. All of it is causal and carried between calls.
- **Elapsed time:** `log(1 + minutes since the stream started)`.

### Training (per fold model)
- **Data:**
  - 540 public Muse recordings, in 5 folds by sleeper;
  - plus external PSG recordings converted to pseudo-Muse windows (`scripts/prep_external.py`): Sleep-EDF (cassette and telemetry), HMC, PhysioNet 2018 and BOAS. They get loss weight 0.5 and a domain-flag input.
- **Loss:**
  - cross-entropy against a Gaussian-smoothed target (σ = 7.5 s), plus a small L1 term on the predicted mean;
  - each window is weighted by the metric's weights raised to the power *a*. With *a* < 1 the weights are flatter, and the missing tilt is folded into the output bias, so inference is unchanged.
- **Start cuts:** with probability 0.9, each training recording starts 0–10 min late. The sealed streams start a few minutes after lights-out, and the elapsed-time input has to match that.
- **Optimiser:** AdamW, lr 1e-3, cosine schedule over 30 epochs, truncated backpropagation through 10-minute chunks.
- **Early stopping:** patience of 8 epochs on the held-out fold, scored with the same start cuts.
- **Augmentation:** amplitude scaling, Gaussian noise, random channel dropout.

### Inference
- **Streaming:** one window per `predict` call. Every model keeps its own state: recurrent state, running mean, EMA and window count.
- **Fast path:** models of the same shape are stacked and run as one batched computation. The CNN models run separately.
- **Output:** the averaged distribution's quantile, with *q* depending on elapsed time:

  | Elapsed | < 2 min | 2–5 min | 5–10 min | 10–20 min | ≥ 20 min |
  |---|---|---|---|---|---|
  | q | 0.55 | 0.65 | 0.65 | 0.60 | 0.40 |

---

## 2. What we did: project history

Every step was judged on 5-fold out-of-fold predictions by sleeper. After the Nov 16 change to the sealed streams, the out-of-fold streams were cut the same way (`scripts/trim_oof.py`). Ensembles were chosen by greedy (Caruana) selection (`scripts/merge_builds.py`), with an honest check: fitted on folds 1–4, scored on fold 5.

| Stage | What changed | Sealed |
|---|---|---|
| Start | MLP on the raw window | – |
| CNN-GRU v1/v2 | CNN on the raw signal + GRU across the recording's windows | – |
| GRU, hand features | spectral hand features, per-sleeper adaptation (running mean, EMA), a 5-seed ensemble; the CNN dropped | – |
| CV on all data | 5-fold CV by sleeper over every public recording; quantile output tuned out-of-fold | – |
| r7 | 50 Hz-aware Welch features + autoencoder code alongside hand features | 40.18 |
| r9_combined | 30 s EMA + external PSG data (Sleep-EDF, HMC, then PhysioNet 2018, BOAS, Sleep-EDF telemetry) | 38.14 |
| *Nov 16* | *sealed streams now start minutes after lights-out; r9 was rescored to 38.32* | |
| r11 | LSTM and wider models, but trained on uncut streams | 39.47 |
| r13 | **start-cut training** and selection on cut streams | 35.61 |
| r15 | other sequence models (minGRU, LRU, Transformer, TCN) | 35.48 |
| r17 | **flattened loss weights** (*a* = 0, 0.5) for variety | 34.92 |
| r18 | **different feature views per model**: temporal-only AE, ICA/PCA codes, hand-feature bagging | 34.78 |
| **r20** | **raw-signal CNN-only model** (notched raw → CNN → GRU) added to the ensemble | **34.43** |

### What did not help
- **CNN on top of the spectral features** (round 10): 0.7–1.5 s worse.
- **YASA sleep-staging distillation** as an auxiliary head.
- **Spindle and K-complex features:** no gain in a linear probe.
- **Per-model-type hyperparameter tuning** (round 16), and more seeds, combinations, hyperparameters and external-data weights after round 18 (round 19). The free merge picked the same ensemble again.
- **Round 20's other ideas:**
  - spectral slope (aperiodic 1/f) features: 38.3 alone, never selected;
  - AE fitted with external data: 38.67;
  - loss weights more extreme than the metric (*a* = 1.5, 2): 39.5 and 40.8.
- **Decoding and calibration tricks:** Bayes/tilt decoding and temperature scaling.

### Lessons
- **Matching the evaluation** mattered most. Training and selecting on cut streams gained about 4 s once the sealed streams changed.
- **Variety beats single-model accuracy late on.** Flattened loss weights, different feature views and the raw CNN were all worse alone, but they improved the ensemble.
- **Cut-stream out-of-fold scores tracked the sealed ranking.** Sealed came out about 1–1.5 s better than local throughout.

---

## 3. Layout

```text
submission/cnn_gru.py          # the solver (becomes submission.py in the ZIP); all model code
scripts/build_submission.py    # train → reload check → ZIP (local or Kaggle)
scripts/make_colab_notebook.py # generates colab/train_on_colab.ipynb (runs, merges per round)
scripts/merge_builds.py        # greedy ensemble selection, q_time tuning, honest check, merged weights
scripts/trim_oof.py            # out-of-fold predictions on start-cut streams (_oofT.npz)
scripts/split_merge.py         # split a merged weights.pt back into its builds
scripts/prep_external.py       # external PSG datasets → pseudo-Muse windows
scripts/yasa_check.py, make_yasa_teacher.py   # YASA staging teacher (not used in the final model)
docs/r18_merge_free_model.md   # detailed description of the model
competition/                   # clone of the official harness (do not edit)
build/<label>/                 # exported builds (submission.py + weights.pt)
dist/<label>.zip               # upload candidates
```

`submission.py` must stay self-contained:
- one file, with `class Solver(CompetSolver)`;
- imports limited to `competition/requirements.txt`;
- weights read from `meta["submission_dir"]`.

Codabench only calls `load_model` and `predict`, on raw windows, so **all preprocessing lives inside the model**. Training-only switches are environment variables (`SLEEP_*`) read by `submission/cnn_gru.py`. Every inference setting is saved in `weights.pt`, so mixed ensembles load correctly.

## 4. Setup and reproduction

```bash
conda activate sleep_foundation          # Python 3.12
pip install -r competition/requirements.txt
benchopt prepare competition/tracks/sleep_onset -d Interaxon2026Muse
```

**Local build:** run a quick pipeline check on synthetic data, or train on Muse:

```bash
python scripts/build_submission.py                                   # synthetic smoke test
SLEEP_CV_FOLDS=5 python scripts/build_submission.py --data muse --solver submission/cnn_gru.py
```

**Kaggle/Colab training:**
1. `python scripts/make_colab_notebook.py` writes `colab/train_on_colab.ipynb`.
2. Set `ROUND` in it.
3. Attach the external-data dataset and the previous round's notebook output.
4. Run all.

The notebook trains each run as a 5-fold build, writes the cut-stream out-of-fold predictions and builds the round's merges into `dist/`.

**Building the final ensemble:**
1. `scripts/trim_oof.py <builds>`
2. `scripts/merge_builds.py --oof-suffix _oofT <builds>`, optionally with `--init label:count,...` to start from an existing ensemble.

**ZIPs for Codabench:** they must hold `submission.py` and `weights.pt` at the top level. A Finder-made zip nests them in a folder and fails, so build them with:

```bash
COPYFILE_DISABLE=1 zip -X -j -9 dist/<name>.zip build/<name>/submission.py build/<name>/weights.pt
```
