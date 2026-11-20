# `r18_merge_free`: how our best submission is built

Track 03 (sleep onset). Sealed score **34.78 s** W-bMAE, 944 s on Codabench (uploaded 2026-11-20).
Local cut-stream out-of-fold score 36.01 s; honest check (fitted on folds 1–4, scored on fold 5) 36.42 s.

Shipped as `dist/r18_merge_free.zip`, which holds two files:

- `submission.py`: a copy of `submission/cnn_gru.py`. It differs only in comments and in the training-only channel options added in round 19.
- `weights.pt`: 30 models plus the ensemble settings.

Everything below was checked against `weights.pt` itself, not only against the training scripts.

---

## 1. The task in one paragraph

A Muse headband records 4 EEG channels: **TP9, AF7, AF8, TP10**, against Fpz, at 128 Hz.

- **Input:** the recording is streamed to the model one **5 s window** (4 × 640 samples) at a time.
- **Output:** after each window, the model predicts the **time left until the first N2 epoch**, in seconds, capped at 600 s.
- **Scoring:** W-bMAE. The mean absolute error is computed per range of the true latency, and each range is weighted:

  | True latency | 0–40 s | 40–90 s | 90–300 s | 300–600 s |
  |---|---|---|---|---|
  | Weight | ×10 | ×5 | ×3 | ×1 |

  Errors close to sleep onset matter most.
- **Sealed streams:** since Nov 16 they start a few minutes **after** lights-out. The opening minutes are cut, and elapsed time counts from the first window delivered.

---

## 2. The ensemble at a glance

The ensemble has **6 builds × 5 fold models = 30 models**, all weighted equally. Every fold model is trained without one fifth of the sleepers (5-fold CV by sleeper), so each build is a 5-model average.

| # | Build | Sequence model | Learned spectral code | Hand features | Loss weight power *a* | External PSG data | Score alone (cut-stream OOF) |
|---|---|---|---|---|---|---|---|
| 1 | `r15_mingru_s1` | minGRU, 2 layers, 128 | AE 64, all channels | all 66 | 1 (metric weights) | all 4 sets, ×0.5 | 37.88 s |
| 2 | `r13_t_ae64_all05_s1` | GRU, 2 layers, 128 | AE 64, all channels | all 66 | 1 | all 4 sets, ×0.5 | 37.76 s |
| 3 | `r13_t_ema30_s0` | GRU, 2 layers, 128 | AE 32, all channels | all 66 | 1 | **none** | 38.06 s |
| 4 | `r17_gru_w0_s1` | GRU, 2 layers, 128 | AE 64, all channels | all 66 | **0** (flat) | all 4 sets, ×0.5 | 37.76 s |
| 5 | `r18_ae_temporal_s0` | GRU, 2 layers, 128 | AE 64, **TP9 + TP10 only** | all 66 | 0.25 | all 4 sets, ×0.5 | 37.72 s |
| 6 | `r18_mingru_ica_bag_s0` | minGRU, 2 layers, 128 | **ICA 32**, all channels | **46 of 66** (random per fold) | 0.25 | all 4 sets, ×0.5 | 37.93 s |

Shared by every build:
- preprocessing v2 (notch + Welch);
- a 30 s EMA;
- start-cut training (up to 10 min);
- 30 epochs, early stopping;
- about 270k–280k parameters per model.

The six builds differ along four axes:
- **Sequence model:** GRU or minGRU.
- **Spectral code:** a nonlinear autoencoder on all channels, one on the temporal channels only, or a linear ICA code.
- **Hand features:** all of them, or a random subset.
- **Training:** loss weighting, and whether external PSG data was used.

That variety is what the ensemble gains from. Each build alone scores 37.7–38.1 s, the ensemble 36.0 s.

The selection was a **greedy (Caruana) merge** on the cut-stream out-of-fold predictions of 16 candidate builds from rounds 13–18 (`scripts/merge_builds.py --oof-suffix _oofT`). It kept adding the build that most improved the score and stopped when nothing helped. Every build was picked once.

---

## 3. Data used for training

### 3.1 Muse recordings (scored)

The data is the full public release: **540 recordings**, made up of:
- the training split;
- NeuralBench's own validation sleepers, which the harness's train loader leaves out;
- the 40 warm-up test recordings.

**Folds:** recordings are grouped by **sleeper name** and split into 5 folds (fixed seed). A sleeper's recordings are never split between training and held-out.

**Labels:** for every window, the time from the window's end to the first N2 epoch, capped at 600 s.

### 3.2 External PSG recordings (training only, never scored)

`scripts/prep_external.py` turns open sleep datasets into **pseudo-Muse** windows: 4 channels, 128 Hz, 5 s, cut from the start of the recording up to the first N2. The channels are rebuilt from re-referenced PSG leads. A mastoid stands in for TP9/TP10, and Fpz or the forehead for the frontal sites.

| Key | Dataset | Nights |
|---|---|---|
| `v1` | Sleep-EDF Expanded (sleep cassette) + HMC sleep staging | ~300 |
| `p2018` | PhysioNet/CinC 2018 "You Snooze, You Win" (MGH) | 994 |
| `boas` | Bitbrain Open Access Sleep, forehead headband only | 128 |
| `sleepedf_st` | Sleep-EDF Expanded (sleep telemetry) | 44 |

How the external data is used:
- **Every fold:** the external recordings are added to the training set of every fold model.
- **Loss weight:** their loss is scaled by **0.5**.
- **Domain flag:** a domain-flag input marks them as 1 during training. At inference the flag is always 0, meaning "this is Muse".
- **Excluded from fitting:** they are not used to fit the autoencoder or the feature normalisation, which use Muse windows only.

Build 3 (`r13_t_ema30`) uses no external data at all, and has no domain-flag input.

---

## 4. Preprocessing (every window, inside the model)

All of this runs inside `submission.py` at inference time, on the raw microvolts the harness delivers.

1. **Remove the DC offset** of each channel within the window: subtract the mean.
2. **Clip artifacts** to ±500 µV.
3. **Band powers via Welch** (preprocessing v2):
   - 2 s Hann segments with a 1 s hop, averaged. This has less variance than a single 5 s periodogram.
   - The **48–52 Hz mains band is skipped**. The recordings carry 50 Hz line noise, which would otherwise leak into the 30–60 Hz band and into every relative power.
4. **Single-window spectra:** a 5 s Hann-window spectrum is also computed. It is used for the slow and eye-movement features, entropy, spectral edge and the autoencoder input.
5. **The eye-movement derivation AF7 − AF8.** Horizontal eye movements have opposite polarity on AF7 and AF8, so their difference acts as a crude EOG channel.

---

## 5. Features of a window

Each window gives `bp` = **[learned code, hand features]**: 130 values with a 64-dim code, 98 with a 32-dim code. These are standardised with a mean and std fitted **per fold** on 20,000 of that fold's own Muse training windows.

### 5.1 Hand-crafted features (66)

| Group | Count | What |
|---|---|---|
| Absolute log band power | 4 ch × 6 = 24 | delta 0.5–4, theta 4–8, alpha 8–12, sigma 12–15, beta 15–30, high/EMG 30–60 Hz (without 48–52) |
| Relative log band power | 4 × 6 = 24 | each band ÷ total power |
| Theta/alpha ratio (log) | 4 | the classic drowsiness marker: alpha fades and theta rises at sleep onset |
| Slow power < 0.6 Hz | 4 | per channel: slow eye movements and drifts |
| AF7−AF8 slow power < 0.6 Hz and 0.6–4 Hz | 2 | slow eye movements (drowsiness) and faster eye movements (wake) |
| Spectral entropy, 0.5–30 Hz | 4 | flattening of the spectrum |
| 95 % spectral edge frequency | 4 | the spectrum shifts towards low frequencies |

**Build 6 uses feature bagging.** Each of its 5 fold models keeps a random 46 of the 66 (70 %). The other 20 get a normalisation std of 10⁶, which turns them into ~0. The kept subset differs per fold model, so the 5 models of the build see different features.

### 5.2 Learned spectral code

- **The input** (135 values) is the **log power spectrum in 27 log-spaced bins** from 0.4 to 47 Hz. Each bin is at least 0.4 Hz wide and 15 % wider than the one before. The spectrum is taken for 5 signals: TP9, AF7, AF8, TP10 and AF7−AF8.
- **Standardisation:** each value is standardised per fold.
- **Encoder:** the code is computed by a frozen encoder `Linear(135→128) → GELU → Linear(128→code)`. The encoder is fitted **without labels** before the sequence model is trained, separately for each fold, on that fold's Muse training windows (up to 80,000).

| Code | Builds | How it is fitted |
|---|---|---|
| **Denoising autoencoder** (64 or 32 dims) | 1–4 | A decoder mirrors the encoder. It is trained for 20 epochs to rebuild the clean spectrum from a noisy copy (Gaussian noise 0.1 and 10 % of the inputs dropped). Only the encoder is kept. |
| **Temporal-only autoencoder** (64) | 5 | As above, but the AF7, AF8 and AF7−AF8 blocks are zeroed. The encoder's weights for them are set to 0 and their std to 10⁶, so the code sees only **TP9 and TP10** (54 of 135 inputs). It turned out to be the best single model. The temporal sites pick up the occipital/posterior alpha that fades at sleep onset, with fewer eye artifacts. |
| **ICA** (32) | 6 | Whitened PCA via SVD, then FastICA on those components, scaled to unit variance. The linear map is written into the same Linear–GELU–Linear encoder with a +50/−50 shift, so GELU acts as the identity and the encoder computes the ICA projection exactly. |

### 5.3 Per-recording adaptation: comparing each sleeper with themselves

People's EEG differs a lot in absolute level. So each window's `bp` vector also gets three causal comparisons, using only the windows seen so far in this recording:

1. `bp − running mean`: the window compared with the recording so far;
2. `bp − EMA`: the short-term change, with an exponential moving average of ~30 s (α = 5/30);
3. `EMA − running mean`: the trend.

The input to the model is `[bp, bp − mean, bp − EMA, EMA − mean]` = **4 × 130 = 520 values** (392 for the 32-dim code). The running sums and the EMA are carried between calls. Streaming a recording one window at a time gives exactly the same result as processing it all at once; this is checked after every build.

### 5.4 Elapsed time

`log(1 + minutes since the stream started)`. On the sealed streams this counts from the first window delivered, after the cut, which is why every model is trained with random start cuts (§7).

---

## 6. Model architecture (one fold model)

```
5 s window (4 × 640 µV)
  │ preprocess: DC removal, ±500 µV clip
  ├─ hand features (66) ──┐
  └─ log spectrum 27 bins × 5 signals ─ frozen encoder ─ code (64 / 32)
                          │
          bp = [code, hand]  (standardised per fold)
                          │
   [bp, bp − running mean, bp − EMA30, EMA30 − mean]   (520 / 392)
                          │
   Linear → 64, GELU                       ("bp_proj")
                          │
   + log(1 + elapsed minutes) (+ domain flag, 0 at inference)
                          │
   Linear → 128, GELU, dropout 0.2         ("in_proj")
                          │
   sequence core: GRU or minGRU, 2 layers, hidden 128, state carried across windows
                          │
   dropout 0.2, Linear 128 → 121           ("head")
                          │
   softmax: distribution over the time to N2 in 5 s bins: 0, 5, …, 595, 600 (= "600 s or more")
```

**Sequence cores used here:**

- **GRU** (builds 2–5): PyTorch `nn.GRU`, 2 layers, with dropout between the layers.
- **minGRU** (builds 1 and 6), from Feng et al. 2024:
  - The gate and the candidate state depend only on the input, not on the previous state: `h = (1 − z)·h + z·h̃`.
  - Each layer is pre-LayerNorm, with a residual connection and a feed-forward block.
  - Training runs it as a parallel scan in log space.
  - It reacts differently from the GRU, which makes its errors less correlated with the GRU models'.

---

## 7. Training (one fold model)

| Setting | Value |
|---|---|
| Training set | the 4 other folds' Muse recordings + external recordings (×0.5 loss weight) |
| Early-stopping set | the held-out fold, **with the start cut** of the sealed streams (2–7 min, fixed per recording) |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4, gradient clipping 1.0 |
| Schedule | cosine decay over 30 epochs; stop after 8 epochs without improvement; keep the best epoch |
| Batching | 16 recordings per batch, sorted by length; truncated backpropagation through time (TBPTT) in chunks of 120 windows (10 min), with the state carried between chunks |
| Augmentation, per recording | amplitude × U(0.7, 1.3), Gaussian noise of 2 µV, one channel zeroed with probability 0.1 |
| **Start cuts** | with probability 0.9, each training recording starts 0–10 min later (drawn again each epoch), so the elapsed-time input and running statistics match the sealed streams |

**Loss for each window:**
- **Main term:** cross-entropy against a **Gaussian-smoothed target**: the true bin, spread with σ = 1.5 bins (7.5 s).
- **Small extra term:** 0.1 × |mean of the predicted distribution − truth| / 600.

**Window weights.** Each window's loss is weighted like the metric: each latency range's weight (10/5/3/1) is split over that range's windows in the recording.

The **loss-weight power *a*** (round 17) controls how strongly this weighting is applied:
- **a = 1 (builds 1–3):** exactly the metric weights. A window near onset counts ~75× as much as one 5–10 min before it.
- **a = 0 (build 4) and a = 0.25 (builds 5–6):** flatter weights, so windows far from onset also train the model.

The missing tilt `(1 − a)·log(metric weight per bin)` is folded into the head's output bias. The saved model therefore makes the same kind of decision as a metric-weighted one, and inference is unchanged. These flatter models were less good alone but added useful variety. Round 17's `r17_merge_new` gained 0.56 s on the sealed set.

---

## 8. From 30 distributions to one number (inference)

1. **Streaming:** the harness calls `predict` once per window. Every model updates its own state (core state, running mean, EMA, window count) and gives a 121-bin distribution. `reset_state` starts each new recording.
2. **Average:** the 30 distributions are averaged with equal weights. A fast path stacks models of the same shape and runs them as one batched computation. It matches the model-by-model result to about 1e-8, and it is why 30 models fit in 944 s.
3. **Quantile by elapsed time.** The prediction is the **q-quantile** of the averaged distribution, in seconds, clipped to [0, 600]. *q* depends on how long the stream has been running. It was tuned on the cut-stream out-of-fold predictions:

   | Elapsed | < 2 min | 2–5 min | 5–10 min | 10–20 min | ≥ 20 min |
   |---|---|---|---|---|---|
   | q | 0.55 | 0.65 | 0.65 | 0.55 | 0.40 |

   A quantile, not the mean, is the right choice for an absolute-error metric. Shifting *q* by elapsed time corrects for how the onset risk changes during the night: early on the model hedges later, and after 20 min it predicts sooner.

---

## 9. How we got here (what mattered)

| Step | Sealed |
|---|---|
| r7: first CV ensemble | 40.18 |
| r9_combined: + 30 s EMA, external PSG data, autoencoder | 38.14 |
| (Nov 16: sealed streams start cut; models trained on uncut streams fell behind) | |
| r13_merge_all: **start-cut training** + selection on cut streams | 35.61 |
| r15_merge_all: + other sequence models (minGRU, LRU, Transformer, LSTM) | 35.48 |
| r17_merge_new: + **flattened loss weights** (a = 0, 0.5) | 34.92 |
| **r18_merge_free: + different feature views per model (temporal-only AE, ICA, feature bagging)** | **34.78** |

**What did *not* help:**
- a CNN on the raw signal;
- spindle and K-complex features;
- YASA-staging distillation;
- tuning each model type separately (round 16);
- after round 18, more seeds, combinations, hyperparameters and external-data weights (round 19). The free merge picked exactly this same ensemble again.

---

## 10. Reproducing it

The builds were trained on Kaggle with `colab/train_on_colab.ipynb`, which is generated by `scripts/make_colab_notebook.py`. Each build is a run of `submission/cnn_gru.py` with environment switches:

| Build | Settings (on top of `SLEEP_CV_FOLDS=5 SLEEP_USE_TEST=1 SLEEP_FEATS=2 SLEEP_AE=2 SLEEP_PREP=2 SLEEP_EMA_S=30 SLEEP_TRIM_MAX=10`) |
|---|---|
| `r15_mingru_s1` | all external ×0.5, `SLEEP_AE_DIM=64 SLEEP_RNN=mingru SLEEP_SEED=1` |
| `r13_t_ae64_all05_s1` | all external ×0.5, `SLEEP_AE_DIM=64 SLEEP_SEED=1` |
| `r13_t_ema30_s0` | (no external data) |
| `r17_gru_w0_s1` | all external ×0.5, `SLEEP_AE_DIM=64 SLEEP_WEIGHT_POW=0 SLEEP_SEED=1` |
| `r18_ae_temporal_s0` | all external ×0.5, `SLEEP_AE_DIM=64 SLEEP_WEIGHT_POW=0.25 SLEEP_AE_CHANNELS=temporal` |
| `r18_mingru_ica_bag_s0` | all external ×0.5, `SLEEP_AE_DIM=32 SLEEP_RNN=mingru SLEEP_WEIGHT_POW=0.25 SLEEP_AE_KIND=ica SLEEP_FEAT_KEEP=0.7` |

**"All external ×0.5"** means `SLEEP_EXTERNAL=` the four files (v1, p2018, boas, sleepedf_st) and `SLEEP_EXT_WEIGHT=0.5`.

**Merging:**
1. `scripts/trim_oof.py` writes the cut-stream out-of-fold predictions (`*_oofT.npz`).
2. `scripts/merge_builds.py --oof-suffix _oofT` selects the builds, tunes `q_time` and writes the combined `weights.pt`.
3. `scripts/split_merge.py` recovers single builds from a merged `weights.pt`.
