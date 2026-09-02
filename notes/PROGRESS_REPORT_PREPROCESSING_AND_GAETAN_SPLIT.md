# SIMBA progress report — preprocessing audit + Gaetan split (items 7a & 1)

## 7a — preprocessing audit (train vs val vs retrieval)

Traced the full spectrum path end-to-end for train, val, and retrieval, with code references at each step.

Found that `tools/simba_retrieval.py` diverged from train/val: it skipped precursor-peak removal and a protonated-adduct filter that both training and validation apply. Fixed it to reuse the same `Preprocessor.preprocess_spectrum` step as train/val (spectrum *loading* stays unfiltered by design — every spectrum in the fold is still evaluated; only the *preprocessing* step was misaligned and is now fixed).

## 1 — train on Gaetan's split

### What was built

- `tools/prepare_msg_gaetan_split_max_lb_hdf5.py` — new dataset using Gaetan's train/val/test split TSV instead of the official MassSpecGym split. MCES ground truth = `max(lb_matrix, HDF5)`, no separate exact-recompute pass for the [10,20] MCES band (same recipe used for all prior datasets).
- While validating this script against the reference (`tools/prepare_msg_max_lb_hdf5.py`), found a real bug: the HDF5 lookup used the wrong triangular-index formula (it reused the `lb_matrix` indexing convention, but the HDF5 file uses scipy's condensed-distance-matrix convention). Verified against independently-computed exact-MCES ground truth on real pairs — confirmed the bug produces silently wrong values (e.g. true MCES=1.0 read back as 45.0).
- Confirmed empirically that this same bug affected the **old official-split training data** used in experiments 001–004.
- Built a second, parallel dataset — `tools/prepare_msg_official_split_max_lb_hdf5.py` — using the same official split as before, with the bug fixed. This isolates "does the split matter" from "does the bug fix matter."
- Trained two new experiments:
  - **Experiment 005**: Gaetan split + fixed MCES (single train/val/test, no scaffold/official split).
  - **Experiment 006**: official split + fixed MCES — directly comparable to the old **Experiment 004** (official split + buggy MCES), same hyperparameters (8 epochs, bf16, DDP 4×H200, MCES==20 excluded, identity pairs added).

### Validation metrics: 004 (buggy) vs 006 (fixed), same official split

**Scaffold val (in-distribution):**

| | final (epoch 7) MAE | final spearman | loss trajectory |
|---|---|---|---|
| 004 | 4.256 | 0.861 | bottoms out at epoch 4, then flat/noisy |
| 006 | 4.278 | **0.869** | still improving at epoch 7 |

**Official val (out-of-distribution scaffolds):**

| | peak MAE | peak spearman | final (epoch 7) MAE | final (epoch 7) spearman |
|---|---|---|---|---|
| 004 | 6.24 (epoch 1) | 0.730 (epoch 1) | 7.55 | 0.608 |
| 006 | 6.21 (epoch 3) | 0.731 (epoch 3) | **7.06** | **0.645** |

Peak achievable accuracy is nearly identical between buggy and fixed data — the bug wasn't severe enough to cap what the model could learn. The difference shows up in **overfitting behavior**: 004 (buggy) starts degrading on official val after epoch 1 and its checkpoint-selection metric (scaffold val loss) plateaus by epoch 4; 006 (fixed) keeps improving through all 8 epochs and degrades less on official val by the end.

### Retrieval hit-rates (official test fold, n=17,556, formula-matched candidates)

| Model | checkpoint | hit@1 | hit@5 | hit@20 |
|---|---|---|---|---|
| 004 | best (epoch 4, scaffold-loss-selected) | 0.0583 | 0.1625 | 0.3360 |
| 006 | best (epoch 7, scaffold-loss-selected) | 0.0568 | 0.1612 | **0.3405** |
| 005 (Gaetan split) | best | **0.0648** | **0.1668** | **0.3510** |

005 (Gaetan split) scores highest on all three metrics when evaluated against the official test fold — but with a caveat: Gaetan's split doesn't align with the official split, so some official-test molecules likely appeared in 005's training set (as train pairs under Gaetan's split). This is accepted/known leakage, not yet a clean apples-to-apples generalization result.

### Bottom line

- The MCES-labeling bug fix (004 → 006) is a real, measurable improvement, but modest: less overfitting, milder end-of-training degradation on official val, comparable-or-better retrieval hit-rates. It did not change the ceiling of what the model can learn.
- Gaetan's split (005) currently looks best on retrieval, but the leakage caveat means this isn't confirmed as a genuine generalization gain yet — would need a leakage-free eval to be sure.

## Precursor-mass noise (Wout & Gaetan's thread)

Checked the codebase: the ±1% uniform precursor-mass noise Wout described already exists — `add_false_precursor_masses_positives` in `simba/core/data/augmentation.py`, train-only, `max_noise=0.01`, applied to ~10% of samples (`prob_aug=0.5` dataset-level gate × `p_augmentation=0.2` function-level gate).

What Wout/Gaetan are proposing is an upgrade: replace the uniform ±1% noise with a truncated-Gaussian resample scaled to real instrument ppm tolerance (15/10/5 ppm for Ion Trap/Q-ToF/Orbitrap, per BUDDY), following MIST-CF — because library precursor masses are unrealistically exact. Gaetan has this implemented already in `spectrawl` (`extract_test_mgf.py:91-96`) and suggests factoring it into a shared `metabo-depthcharge` utility.

This is adjacent to, but distinct from, roadmap item 7b ("encode precursor mass properly instead of raw float"): 7b is about *representation* (how mass is fed into the model), this proposal is about *augmentation realism* (how it's perturbed during training). Suggest treating it as its own small, self-contained item — cheap to do, independent of whatever encoding scheme 7b lands on.
