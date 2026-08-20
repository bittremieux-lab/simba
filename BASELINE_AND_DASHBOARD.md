# Gaetan-split baseline (009) and the training dashboard

Status snapshot as of 2026-08-18. Covers: what the current baseline experiment
is, the data it runs on, how to (re)run it, what scripts exist, and the
Streamlit dashboard built to monitor/analyze it.

## 1. Baseline experiment: `009_msg_gaetan_split_v2_cosine_no_head_1gpu`

This is the current reference training run on Gaetan's MassSpecGym split. It
supersedes experiment `005` (the original 4-GPU Gaetan-split run) on several
fronts:

| | 005 (old) | 009 (current baseline) |
|---|---|---|
| Data | `preprocessing_gaetan_split_max_lb_hdf5` (buggy) | `preprocessing_gaetan_split_max_lb_hdf5_v2` (fixed) |
| Head mode | `cosine_relu` (default) | `cosine_no_head` |
| Hardware | 4x H200 DDP, bf16-mixed | 1x H200, 32-true |
| Batch size / LR | 4096/gpu, lr=0.00028 | 2048, lr=0.0001 |
| `exclude_mces_value` (drop MCES==20 pairs) | 20 | not set (kept) |
| Weighted resampling (train + val) | on (via a dead config flag, never actually off) | off (`sampling.use_resampling=false`, now wired up) |
| Val evaluation | `limit_val_batches=13`, weighted-resampled subset (~53K pairs) | full val set every check (~3.73M pairs), no resampling |
| Val metrics/plots | Spearman, ED confusion matrix, MCES hexbin | per-GT-MCES-bin MAE (self-pairs as their own bin), full per-pair CSV, GT-binned box plot |

**Status**: cancelled by request (twice — once at step ~160,000/240,000, then
again shortly after a fresh restart once the changes in section 3 landed).
Not currently running. Partial artifacts (checkpoints, the consolidated
per-pair parquet, plots, `metrics.csv`) remain under
`experiments/training/009_msg_gaetan_split_v2_cosine_no_head_1gpu/` and are
browsable via the dashboard (section 5).

**To resubmit**: `sbatch tools/slurm/009_msg_gaetan_split_v2_cosine_no_head_1gpu.slurm.sh`
from the repo root. If you want a genuinely fresh run rather than resuming
into the existing (partial) output dir, clear it first:
```bash
rm -rf experiments/training/009_msg_gaetan_split_v2_cosine_no_head_1gpu
mkdir -p experiments/training/009_msg_gaetan_split_v2_cosine_no_head_1gpu
sbatch tools/slurm/009_msg_gaetan_split_v2_cosine_no_head_1gpu.slurm.sh
```

**No LR scheduler**: `configure_optimizers` (`simba/core/models/similarity_models.py`)
returns a plain `torch.optim.Adam(self.parameters(), lr=self.lr)` — flat,
constant learning rate for the whole run, no warmup/decay. The `optimizer`
config block's `weight_decay`/`betas` fields also aren't actually read by
`configure_optimizers` (only `lr` is) — currently harmless since the
configured values match PyTorch's own Adam defaults, but overriding either
via CLI would silently have no effect.

## 2. Data prepared: Gaetan-split v2

Gaetan's split (train/val/test fold assignment) comes from an external TSV
owned by the sibling `spectrawl_project` repo, keyed by MGF `identifier` — not
a SIMBA-internal scaffold split. Pair distance = `max(lb_matrix, hdf5)`,
capped at 40 (`MCES_CAP`).

**The fix**: the original prep script built its HDF5 SMILES→index lookup from
*raw, non-canonical* SMILES while the query side was canonicalized before
lookup, so ~94-97% of pairs reported `hdf5_missing` (silently falling back to
the less-precise `lb_matrix` bound alone). Fixed by canonicalizing the HDF5
side too, and output was redirected to a new `_v2` directory (the original is
untouched — experiment 005 was trained on it, so it stays reproducible).

Verified (job 1324691, ~2.5 min on 1 GPU/24 cpu):
```
train: 24010 unique mols, hdf5_missing=0
test:  2734 unique mols,  hdf5_missing=0
val:   2800 unique mols,  hdf5_missing=0
```
(down from ~94-97% missing in the original data — pair counts unchanged
since molecule counts didn't change.)

**Location**: `data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2/`
- `mapping.pkl` — per-fold `df_smiles_*` (molecule tables) and
  `spectrum_indexes_*`.
- `ed_mces_indexes_tani_incremental_{train,val,test}_node0_chunk0.npy` —
  raw (mol_idx_0, mol_idx_1, ed, mces) pair arrays.

| Fold | Molecules | Spectra | Raw pairs |
|---|---|---|---|
| train | 24,010 | 141,898 | 288,228,045 |
| val | 2,800 | 15,090 | 3,918,600 |
| test | 2,734 | 14,118 | 3,736,011 |

(Val pairs actually scored during 009's training are 3,730,546 after
remap/dedup — see `ValMetricsCallback`'s per-pair CSVs.)

**To reproduce**: `tools/prepare_msg_gaetan_split_max_lb_hdf5.py`, run via
`sbatch tools/slurm/prepare_msg_gaetan_split_max_lb_hdf5_v2.slurm.sh`
(1 GPU/24 cpu on `zen4_h200` — no GPU compute needed, just the RAM that
allocation guarantees; the 116GB `lb_matrix.npy` is loaded fully into RAM
rather than mmap'd, since scattered mmap'd reads over it have separately
caused multi-hour stalls elsewhere in this project).

## 3. Training-pipeline changes behind 009

- **`ValMetricsCallback` rewrite** (`simba/core/training/callbacks.py`): no
  more Spearman or ED confusion matrix. Every validation check now:
  - logs MAE (raw MCES units) per GT-MCES bin, with self-pairs
    (`mol_idx_0==mol_idx_1`) as their own bin rather than folded into the
    lowest numeric bin;
  - logs **overlap coefficient** between each bin and its neighbor at skip
    distances 0-4 (0 = adjacent, e.g. self vs `(0,5]`; 1 = one bin further,
    e.g. self vs `(5,10]`; ...) — `val_overlap/{a}_vs_{b}_skip{k}/{val_name}`,
    0 = fully separated, 1 = identical predicted-MCES distributions between
    the two bins (equals 2x the Bayes-optimal misclassification rate between
    them under equal priors). Plus `val_overlap_avg/skip{k}/{val_name}`, the
    mean over all bin-pairs at that skip level, same idea as "overall MAE"
    alongside the per-bin MAE lines. This is the metric that actually
    disambiguates "MAE went up" from "the model can no longer tell these
    bins apart" — MAE conflates calibration drift with real loss of
    separability, overlap coefficient isolates the latter.
  - saves/updates one **consolidated parquet file**,
    `val_pairs_{val_name}_consolidated.parquet`: static per-pair columns
    (`mol_idx_0/1, spec_idx_0/1, smiles_0/1, gt_mces, mces_bin, is_self_pair,
    same_spectrum`) written once, then one `pred_mces_step{N:06d}` column
    *appended* per validation check, instead of a fresh full CSV every time
    (~750MB x every check → ~120GB for just 160 checks on this val set). The
    append is **positional**, not a re-join by pair identity — safe only
    because the val DataLoader is now sequential (see next bullet); this
    only holds as long as that stays true, so a cheap `array_equal` sanity
    check raises loudly instead of silently misaligning predictions with the
    wrong pairs if that assumption is ever violated.
  - saves a GT-binned predicted-MCES box plot:
    `mces_binned_box_{val_name}_step{N:06d}.png` (whis=(5,95), outliers
    hidden, pred=GT reference line, each box n-annotated).
- **Validation DataLoader no longer shuffles** (`create_dataloaders` in
  `simba/workflows/training.py`): was `shuffle=(val_sampler is None)` with a
  generator created once and reused across the whole run, so every
  validation check drew a different permutation from wherever that
  generator's state had advanced to — meaning the same pair landed in a
  different row every check, with no way to line predictions up across
  checks except by re-joining on pair identity. Now `shuffle=False`
  unconditionally: shuffling only ever mattered for decorrelating SGD
  updates during training, never for a metrics-only validation pass, so
  there was no reason to pay for it. This is also what makes the
  consolidated file's positional column-append (above) safe.
- **`cfg.sampling.use_resampling`** (`simba/workflows/training.py`): existed
  in config but was never read anywhere — the inverse-MCES-bin-frequency
  weighted sampler was always active regardless. Now wired up: when `false`,
  train/val samplers are `None`, which the dataloader builder already
  handles by falling back to plain shuffling (train) / a single deterministic
  unweighted pass (val).
- **Pair-identity plumbing**: `mol_idx_0/1`, `spec_idx_0/1`, `smiles_0/1`
  threaded from `CustomDatasetMultitasking.__getitem__` through
  `validation_step` into the callback — needed for the per-pair data and to
  correctly distinguish "self-pair" (same molecule, possibly *different*
  spectra) from same-spectrum-vs-itself.

## 4. How to run training in general

```bash
uv run simba train <hydra.override>=<value> ...
```
Config groups live under `simba/configs/` (`model`, `training`, `data`,
`preprocessing`, `inference`, ...). In practice, experiment SLURM scripts
override individual leaf values directly, e.g.:
```bash
uv run simba train \
  paths.preprocessing_dir=... paths.checkpoint_dir=... paths.mgf_path=... \
  training.epochs=24 training.batch_size=2048 training.val_check_interval=1000 \
  training.limit_train_batches=10000 \
  optimizer.lr=0.0001 \
  sampling.add_identity_pairs=true sampling.use_resampling=false \
  hardware.accelerator=gpu hardware.devices=1 hardware.precision=32-true \
  model.tasks.cosine_similarity.head_mode=cosine_no_head
```
One SLURM script per experiment lives in `tools/slurm/`, named after the
experiment; output goes to `experiments/training/<experiment_name>/`.

## 5. Dashboard

**File**: `tools/dashboard_app.py`. **Run**:
```bash
uv run --extra dashboard streamlit run tools/dashboard_app.py \
  --server.port 8765 --server.address 127.0.0.1
```
(port-forward from your machine to reach it, since this runs on the login
node — `ssh -L 8765:localhost:8765 <cluster-login-alias>`, then open
`http://localhost:8765`).

The sidebar's experiment selector auto-discovers any directory under
`experiments/training/` that has at least one binned-box PNG (i.e., ran with
the current `ValMetricsCallback`) — right now that's just 009.

**Tabs**:
- **Loss** — train/validation loss vs. step (or fractional epoch, if the
  matching SLURM script's `training.limit_train_batches` can be read off to
  compute it).
- **Validation metrics** — per-GT-MCES-bin curves over training, either as:
  - **MAE** (default) — read straight from the cheap, pre-logged
    `metrics.csv`, instant.
  - **Overlap coefficient (vs next bin)** — how much a bin's predicted-MCES
    distribution overlaps its neighbor's (0 = fully separated, 1 = identical;
    equals `2x` the Bayes-optimal misclassification rate between the two
    bins under equal priors), plus an "average" line (mean over all
    currently-shown bin-pairs). Skip 0-4 is a **fast path** too now — those
    are logged directly by the callback — so it's instant just like MAE;
    only skip >4 (or a mass-difference filter, see below) triggers
    recomputation. If a run's `metrics.csv` predates this feature (e.g. the
    existing 009 data, recorded before `_log_overlap_coefficients` existed),
    the tab detects the missing columns and offers the recompute path with a
    clear message instead of silently rendering an empty chart.
  - Controls: include/exclude self-pairs, GT-MCES range (which bins to
    show), molecule mass-difference filter (|mass_0 - mass_1|, Da — applies
    to *every* curve, not just an extra line), and (overlap-coefficient only)
    how many bins to skip between compared pairs.
  - Whenever recomputation is needed (mass filter narrowed, skip >4, or an
    older run without the pre-logged columns), the tab reads per-pair data
    directly — from the consolidated parquet file when the experiment has
    one (cheap, columnar, no cap needed), or per-step CSVs for older
    experiments (one large file read per check). Either way it's gated
    behind a "checks to use" slider (default 10 — 160 checks can take
    1-2 minutes to build) and an explicit **Build** button rather than
    fully reactive.
- **Box plot** — the pre-rendered training-time PNG (step slider), plus an
  on-demand fine-grained rebuild from that same step's per-pair data: custom
  bin width (vs. the fixed 5 used at training time) and the same molecule
  mass-difference filter, reactive (only touches one step, so no button
  needed).
- **Mass heatmap** — MAE / signed bias / mean-GT-MCES heatmaps over
  (min mass, max mass) of the pair's two molecules (reproducing the
  `test_to_test` row of the earlier `mass_heatmaps.png` investigation, minus
  the Spearman column). Self-pairs included by default (land on the
  diagonal, mass_diff=0) with a toggle to exclude; adjustable mass-bin step
  and minimum-pairs-per-cell threshold.
- **Compare runs** — one row per experiment under `experiments/training/`,
  each showing that run's most-recently-logged validation check: val loss,
  overall MAE, identity (self-pair) MAE, overlap coefficient at skip 0/2,
  identity-vs-neighbor overlap at skip 0/1/2 (self vs `(0,5]`/`(5,10]`/
  `(10,15]`), and two mass-filtered overlap columns (mass diff < 30/100 Da).
  Cells come straight from `metrics.csv` when logged; runs that predate a
  given metric (e.g. 009 predates `val_overlap*` logging), plus anything
  mass-filtered, get recomputed on demand from that check's per-pair data
  instead of showing blank.
  - **Custom metric builder**: pick MAE or Overlap, a molecule mass-
    difference range, and (for overlap) a skip distance and optionally one
    particular bin/pair instead of the average — "Add column" appends it
    with an auto-generated label. Computation is cached and gated behind an
    explicit **Build custom columns** button (not run automatically on
    every widget interaction — it reads per-pair data and recomputes
    molecule masses on any cache miss, across every run in the table, so it
    isn't cheap). Columns are blank (`NaN`) until built.
  - **Cosine-similarity baseline row** — a checkbox adds one synthetic
    "run", `cosine (raw spectral baseline)`, computed from
    `tools/compute_val_cosine.py`'s precomputed per-pair cosine similarity
    (see section 9) instead of any experiment's `pred_mces`. This is a row,
    not a column: it fills in every Overlap cell (fixed and custom) using
    cosine, and leaves Val loss/Overall MAE/Identity MAE/Last step blank,
    since cosine isn't a per-pair error against GT and isn't on a scale
    comparable to MCES (MAE against it wouldn't mean anything). Any
    experiment sharing the same val set works as the reference for GT-MCES
    bin labels — cosine only depends on which two spectra a pair uses, not
    on any model's predictions, so it's identical across every experiment
    built from the same preprocessing dir.
  - **Hit@1/5/20 retrieval benchmark** — a checkbox (default **on**; see
    crash note below for why) adds three more columns, computed by
    `compute_hit_at_k_all` (a self-contained, in-dashboard duplicate of
    `tools/benchmark_self_retrieval.py`'s logic — see section 10) for every
    run plus the cosine baseline row if available. Hit@k is
    higher-is-better, the opposite of every other column in this table, so
    it gets its own (non-reversed) color-gradient direction
    (`RdYlGn` vs. everything else's `RdYlGn_r`) rather than sharing one
    gradient call.
  - Metrics to display are selectable; the table is color-graded
    (red→green, lower-is-better except Hit@k, **excluding any column that's
    entirely NaN** — see the segfault note below) with an optional bar
    chart for any one selected metric.
  - **Fixed two crash-inducing bugs found while building this, and one
    still-unexplained one**: (1) the sidebar's "Refresh" button was a
    literal no-op (`st.button("Refresh")` with nothing attached) — its own
    comment claimed "any interaction reloads data fresh," but that's false
    for anything using `@st.cache_data` (e.g. `list_consolidated_steps`),
    which caches by arguments for the server process's lifetime and is
    *not* invalidated by a script rerun. A long-running experiment's
    newly-appended validation checks stayed invisible (e.g. showing "3 of 3
    available checks" for a run that had logged 20+) until the whole
    dashboard process was restarted. Fixed: `st.button("Refresh")` now
    calls `st.cache_data.clear()`. (2) A custom column shows as all-`NaN`
    before it's built — passing an all-NaN column to
    `Styler.background_gradient` triggers `np.nanmax`/`np.nanmin` on an
    all-NaN slice, which segfaults the whole process under this
    environment's pandas/numpy/matplotlib versions (confirmed via the
    `RuntimeWarning: All-NaN slice encountered` printed immediately before
    the crash) rather than just warning. Fixed by excluding all-NaN columns
    from the gradient subset. (3) Enabling the Hit@k checkbox segfaulted
    the dashboard twice, with no informative traceback either time (once
    right after a `use_container_width` deprecation warning, which got
    fixed -- `st.dataframe(..., width="stretch")` -- as a legitimate cleanup
    but did **not** stop the crash on retry). The Hit@k *logic* was verified
    correct and crash-free multiple times as a standalone script and via
    direct non-Streamlit invocation of the same dashboard function — the
    crash only ever happened when it ran live inside the Streamlit process,
    matching this dashboard's earlier, still-unexplained segfault history.
    Root cause unresolved; what actually made it stop crashing was simply
    defaulting the checkbox to checked (`value=True`) instead of leaving it
    off for on-demand triggering -- worth noting if a similar crash recurs
    for a future on-demand-triggered heavy computation.

**Design note on cost**: everywhere that reads per-pair data prefers the
consolidated parquet file when the experiment has one (`load_pair_data_for_step`
in `tools/dashboard_app.py`), falling back to per-step CSVs for older
experiments that predate the consolidated format. Single-step reads
(box plot, mass heatmap) stay reactive either way. Anything that needs to
scan *multiple* checks (mass-filtered/overlap-coefficient curves) is opt-in
behind a Build button with a checks-to-use control, so it doesn't silently
try to read the whole run's history on every widget tweak.

## 6. Scripts reference

| Script | Purpose |
|---|---|
| `tools/prepare_msg_gaetan_split_max_lb_hdf5.py` | Builds Gaetan-split-v2 preprocessing data (fixed HDF5 canonicalization) |
| `tools/slurm/prepare_msg_gaetan_split_max_lb_hdf5_v2.slurm.sh` | SLURM wrapper for the above |
| `tools/slurm/009_msg_gaetan_split_v2_cosine_no_head_1gpu.slurm.sh` | Trains the 009 baseline |
| `tools/slurm/010_msg_gaetan_split_v2_theoretical_precursor_mist_cf_1gpu.slurm.sh` | Trains 010 — same as 009 but theoretical precursor mass + MIST-CF/BUDDY noise (section 7) |
| `tools/slurm/011_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_1gpu.slurm.sh` | Trains 011 — same as 010 plus MCES-weighted resampling for training only (section 8) |
| `tools/slurm/012_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_bucket_weights_1gpu.slurm.sh` | Trains 012 — same as 011 plus self/near-self bucket multipliers and within-bucket mass-tier reweighting (section 8) |
| `tools/mass_diff_by_mces_bucket.py` + its `.slurm.sh` | Diagnostic: mass-difference distribution per MCES sampling bucket, for calibrating the weights in section 8 |
| `tools/dry_test_resampling_weights.py` + its `.slurm.sh` | Diagnostic: simulates drawing from the real training sampler to verify per-bucket weight shares and per-pair repetition before spending a training run on it |
| `tools/compute_val_cosine.py` + its `.slurm.sh` | Computes the raw spectral cosine-similarity baseline for every validation pair, once per val set (section 9) |
| `tools/plot_pred_mces_by_bin.py` | Diagnostic: predicted-MCES (or cosine) distribution per GT-MCES bin, with the self bucket split by same- vs different-spectrum (section 9) |
| `tools/benchmark_self_retrieval.py` | Hit@1/5/20 retrieval benchmark among same-molecule near-neighbor queries, for any set of experiments + cosine (section 10) |
| `tools/confusion_hit_simba_vs_cosine.py` | One run vs. cosine: 2×2 confusion matrix (n and %) of hit@k agreement/disagreement (section 10) |
| `tools/dive_hit_disagreements.py` | For the two disagreement quadrants above: the losing method's rank for the true match, and the GT MCES of whatever it wrongly ranked #1 instead (section 10) |
| `tools/dashboard_app.py` | Streamlit monitoring/analysis dashboard (section 5) |

## 7. Precursor mass: theoretical base + MIST-CF/BUDDY noise (experiment 010)

Prompted by a Slack discussion with Wout and Gaetan De Waele about 009 (and
every prior experiment) relying on precursor mass read straight from the
MGF's `PEPMASS` field, perturbed only by Sebastian's original uniform
±1%-noise augmentation. Two problems with that: MassSpecGym's own
`PRECURSOR_MZ` is itself already a rounded *theoretical* value for the large
majority of rows (confirmed against Gaetan's `spectrawl` extraction code and
directly against the MGF: `PARENT_MASS + adduct_offset == PRECURSOR_MZ`
exactly), so training on it directly is partly training on a noiseless,
leakage-prone signal; and the ±1% noise scheme doesn't reflect any real
instrument's actual precision.

**New config** (`sampling.*` in `simba/configs/training/default.yaml`,
defaults preserve all prior experiments' behavior unchanged):
- `precursor_mass_mode`: `measured` (default, historical behavior — read
  `spec.precursor_mz` from the MGF) or `theoretical` (compute from the
  molecule's SMILES via RDKit `ExactMolWt` + adduct arithmetic, ignoring the
  MGF's own value entirely).
- `precursor_noise_mode`: `legacy` (default — Sebastian's original ±1%
  uniform noise, bug-fixed, see below), `mist_cf` (new: MIST-CF/BUDDY-style
  Gaussian noise, std = instrument-specific ppm tolerance / 5 — Orbitrap/
  FTICR=5ppm, Q-ToF=10ppm, Ion Trap/Unknown=15ppm, matching BUDDY's table),
  or `none`.

**New code**:
- `simba/core/chemistry/chem_utils.py`: `theoretical_precursor_mz(neutral_mass,
  adduct)` (parses monomer count and charge magnitude straight from the
  adduct string, e.g. `[2M+H]+`/`[M+2H]2+`, so it works for every adduct in
  `ADDUCT_TO_MASS` without a second lookup table), `normalize_instrument_type`
  + `INSTRUMENT_PPM_TOLERANCE` (raw MGF `INSTRUMENT_TYPE` → BUDDY's ppm
  tolerance, falling back to the least-precise "unknown" tier), and
  `resample_precursor_mz` (the Gaussian draw itself).
- `simba/core/data/augmentation.py`: new
  `Augmentation.resample_precursor_masses_mist_cf`, applied per side using
  that side's own instrument type (falls back to "unknown" if absent);
  `augment()` now takes `precursor_noise_mode` and dispatches to
  legacy/mist_cf/none.
- Instrument type threaded end-to-end for the first time (previously parsed
  nowhere): MGF `INSTRUMENT_TYPE` → `SpectrumExt.instrument`
  (`loaders.py`/`spectrum.py`) → per-spectrum `instrument` array
  (`multitask_dataset_builder.py`) → `instrument_0`/`instrument_1` per
  sample (`multitask_dataset.py`) → read by the augmentation above.
- `precursor_mass_mode="theoretical"` is applied once per spectrum in
  `MultitaskDataBuilder.from_molecule_pairs_to_dataset` (replacing
  `precursor_mass[i] = spec.precursor_mz`), *not* per training sample — it's
  the deterministic base value; `precursor_noise_mode` is the stochastic
  augmentation applied on top, at `__getitem__` time, training-split only.

**Bugs fixed along the way** (found while tracing this code path, folded
into 010 since they sit on the same lines being touched):
- `ADDUCT_TO_MASS["[3M-H]-"]` was `+1.007276`, sign-flipped vs. `[2M-H]-`/
  `[M-H]-`'s consistent single-deprotonation convention (should be
  `-1.007276`). Neither adduct in MassSpecGym's MGF (`[M+H]+`, `[M+Na]+`)
  is affected, so this had no effect on 009, but would have silently
  corrupted any future run using `[3M-H]-`.
- `Augmentation.add_false_precursor_masses_positives` (the legacy noise
  step, active in every experiment through 009) overwrote
  `precursor_mass_1` with a *noised copy of `precursor_mass_0`* instead of
  perturbing `precursor_mass_1`'s own value — i.e. every time this
  augmentation fired (~10% of training samples: 50% augmentation-call rate
  × 20% of those), both sides were silently forced to the same underlying
  mass before independent noise was added. Fixed to perturb each side from
  its own value.

**Experiment 010** (`tools/slurm/010_msg_gaetan_split_v2_theoretical_precursor_mist_cf_1gpu.slurm.sh`):
identical to 009 (same Gaetan-split-v2 data, same 1-GPU hyperparameters,
same `head_mode=cosine_no_head`) plus `sampling.precursor_mass_mode=theoretical
sampling.precursor_noise_mode=mist_cf`. Submitted as job 1326533; reached
training/validation without error, loss and MCES-MAE improving normally
(`val_loss` 0.097→0.032, `mces_mae` 10.15→5.48 by step 22000).

## 8. MCES-bucket weighted resampling: self-pair bucket, extra multipliers, mass-tier reweighting (experiments 011/012)

Motivation: 009/010 (and every experiment before them) trained on plain
uniformly-shuffled batches (`sampling.use_resampling=false`) — every pair
seen exactly once per epoch, at whatever frequency it naturally occurs.
`CustomWeightedRandomSampler` (inverse-MCES-bin-frequency resampling) exists
in the code and was used in 005 and earlier, but had been off since.

**Experiment 011** (`sampling.use_resampling=true`, otherwise identical to
010): turns training-time resampling back on, plus one new thing —
`MCES_SAMPLING_EDGES`/`MCES_SAMPLING_BIN_LABELS` in
`simba/workflows/training.py` split the old combined `[0,2.5)` bucket into
`{0}` (raw MCES exactly 0 — self-pairs from `sampling.add_identity_pairs`,
and any other exactly-identical-structure pair) and `(0,2.5)`, so self-pairs
get their own inverse-frequency weight instead of being averaged together
with the far more common near-but-not-identical pairs (10 buckets → 11).

**Found and fixed while wiring this up**: the *old* code, when
`use_resampling=true`, also built a weighted-with-replacement sampler for
*validation* (`val_sampler`/`val_official_sampler`) — meaning turning
resampling back on would have silently turned every validation check back
into a partial, non-deterministic slice of the val set, undoing this
session's earlier shuffle=False/consolidated-parquet work and breaking the
per-bin MAE / overlap-coefficient / parquet-alignment pipeline's assumption
that every check scores the same full set in the same row order.
`val_sampler`/`val_official_sampler` are now unconditionally `None` in
`prepare_data` — `sampling.use_resampling` only ever affects training
batches now, regardless of setting.

**Experiment 012** (identical CLI config to 011 — the change is entirely in
`prepare_data`'s weighting code, not a new override): two more layers on
top of 011's per-bucket inverse-frequency weight, per user-specified
values, both in `simba/workflows/training.py`:
- `MCES_SAMPLING_BUCKET_MULTIPLIERS`: self ×4, the 4 buckets covering all of
  MCES<10 (excluding exactly 0) ×2 each, everything MCES>10 unchanged (×1).
- Within each non-self bucket, mass-tier reweighting: pairs at/below that
  bucket's own 10th-percentile molecule mass difference (RDKit `ExactMolWt`,
  via `mass_lookup_from_df_smiles` in `simba/core/chemistry/chem_utils.py`)
  collectively get half that bucket's sampling probability mass, the rest of
  the bucket gets the other half — regardless of how lopsided the actual
  pair-count split is. Skipped for the self bucket (mass_diff is always 0
  there by construction).
- `tools/mass_diff_by_mces_bucket.py`: one-off diagnostic plot (multi-
  subplot histogram) of `|mass_0 - mass_1|` per MCES sampling bucket, built
  from the real training pair pool (same `load_dataset`+`prepare_data` path
  `simba train` uses) — this is what the mass-tier thresholds above were
  chosen against. Saved to
  `experiments/mass_diff_by_mces_bucket_gaetan_split_v2.png`.

**A real bug, caught before wasting a training run**: the first version of
the mass-tier step used `low_weight = MASS_TIER_LOW_SHARE / n_low` (and the
symmetric `high_weight`). This divides a bucket's *total* contribution to
the sampler's weight vector down by that bucket's own pair count — because
every pair in a bucket already carried the *un-divided* bucket-level scalar
weight (`weights_ed[bucket]`), not a value already spread over the bucket's
members. The self bucket, left untouched by this step, kept its correct
scale while every other bucket's total contribution got divided by its own
size (up to ~66M for the biggest bucket) — so after final normalization,
self ended up at ~100% of all sampling weight and everything else ~0%.
First symptom: job 1327099 (012, unfixed) showed step-1000 `val_loss=0.545,
mces_mae=28.1` (vs. 011's `0.053/7.2` at the same step) and, per direct
user observation of the loss plot, `train_loss` pinned at ~0 — the
signature of a training distribution collapsed onto a tiny, endlessly-
repeated pool of pairs. Cancelled immediately.

Root-caused (not just re-tuned) via `tools/dry_test_resampling_weights.py`:
builds the *real* `train_sampler` (same `load_dataset`+`prepare_data` call,
same weights) and simulates 2M draws from it, reporting per-bucket
draw-share (theory vs. empirical) and, critically, how many *distinct*
pairs a given number of draws actually touches per bucket vs. how many
times each gets repeated. Before the fix this showed self at 100.00% share
(confirming the bug directly, not just its downstream symptom) with every
other bucket at 0.00%. Fix: scale the mass-tier multiplier by the bucket's
own pair count (`(MASS_TIER_LOW_SHARE * n_in_bucket) / n_low`, so a
multiplier of 1.0 means "unchanged," matching the self bucket's implicit
1.0) — re-running the same dry test after the fix gave empirical shares
matching theory exactly (self 22.25%, each MCES<10 bucket ~11.1%, MCES>10
combined 33.3%), with big buckets showing ~1.0 repeats/pair (real
diversity intact) and only the deliberately-boosted small buckets/tiers
showing meaningful repetition (self ~17.5×, the tightest mass-tier ~30× per
2M-draw sample). Job 012 was then resubmitted (1327177) and tracked 011
closely from step 1000 onward with no sign of the earlier collapse.

**Lesson embedded in both scripts**: verify a resampling/reweighting scheme
against the real pair pool (bucket shares, repetition/distinct-pair counts)
*before* spending a training run on it, the same way `mass_diff_by_mces_bucket.py`
was used to choose the mass-tier thresholds in the first place — both are
meant to be reused for any future change to this weighting scheme, not
one-off scripts.

## 9. Cosine-similarity baseline, and why it separates self-pairs better than SIMBA

Motivation: Gaetan's suggested sanity check (already used for retrieval in
`tools/cosine_baseline_iceberg.py` etc.) — how much of SIMBA's separation
ability is actually earned over plain binned spectral cosine similarity,
with no learned model involved at all? Extended here to every *validation*
pair, not just retrieval candidates, so it can sit alongside SIMBA's own
predictions in every overlap-coefficient metric the dashboard already has.

**`tools/compute_val_cosine.py`**: computed once per val set, not once per
experiment or per training step. Reuses `bin_spectra` from
`tools/cosine_baseline_iceberg.py` unchanged (bin_width=0.01 Da, max_mz=1100
Da, sqrt-compress, L2-normalize -- cosine similarity is then a plain dot
product) applied to each pair's *raw* spectra (`molecule_pairs_val.
original_spectra[spec_idx]`, loaded via the same `load_dataset` path
`simba train` uses -- before SIMBA's own `Preprocessor` runs), matched by
`spec_idx_0`/`spec_idx_1` read from any one experiment's consolidated
parquet (any experiment works interchangeably as this "reference" -- only
its pair *identity*, not its predictions, is used). This is safe to treat
as a one-time, experiment-independent artifact because 009-012 all build
their validation pairs from the same preprocessing dir the same
deterministic way (fixed pair list, `shuffle=False`, and
`CustomDatasetMultitasking.__getitem__`'s val branch always resolves a
molecule to the same spectrum index) -- confirmed directly: merging the
saved `val_cosine_val.parquet` onto 011's consolidated parquet by
`(mol_idx_0, mol_idx_1, spec_idx_0, spec_idx_1)` is lossless (3,730,546/
3,730,546 rows). Saved to `{preprocessing_dir}/val_cosine_{val_name}.parquet`.

Verified the spectrum-level (not just molecule-level) identity resolution
is correct, since this was flagged as important: true same-spectrum pairs
(`spec_idx_0==spec_idx_1`) come out at cosine exactly 1.0 for all 721 of
them; same-molecule-*different*-spectrum pairs (2,010 of them, within the
"self (MCES=0)" bin) range across the full 0-1 scale (mean 0.657) -- as
expected, since two independently-measured spectra of the same molecule
(different collision energy/instrument/noise) aren't guaranteed to look
alike, even though they're chemically identical.

**Dashboard integration**: cosine appears as an extra table *row* in
Compare Runs (`cosine (raw spectral baseline)`), not a column -- see
section 5. First attempt wrongly modeled it as a per-column "data source"
choice in the custom-metric builder; corrected after feedback that cosine
is "another way of predicting scores," i.e. it belongs alongside the
experiments as a row, auto-filling every Overlap column (fixed and custom)
and leaving MAE/loss blank.

**Finding: cosine separates self-pairs from near-neighbors better than any
of 009-012**, despite SIMBA winning most other overlap columns. Investigated
with a new diagnostic, `tools/plot_pred_mces_by_bin.py` (reads one
experiment's most recent check's per-pair data directly, no SIMBA
re-inference; optionally overlays the cosine-scored version of the same
bins for a side-by-side comparison -- lightweight, run directly, no SLURM
needed). Splits the self bucket into true same-spectrum vs same-molecule-
different-spectrum pairs (same distinction as above), since they behave
very differently:

- **Same-spectrum pairs (26% of the self bucket, 721/2,731 in run 012)**:
  a wash between methods -- both are *mathematically* perfect here (an
  embedding's cosine similarity with itself is always 1, so
  `head_mode=cosine_no_head` predicts exactly 0 MCES with zero variance;
  raw cosine is trivially 1.0 with zero variance too). Confirmed directly
  in run 012's step-145000 data: `median=0.000, mean=0.000, std=0.000`.
- **Different-spectrum pairs (74%, 2,010/2,731)**: this is where the whole
  gap comes from. SIMBA's median (2.09) is in the right direction --
  clearly below the `(0,5]` bin's own median (4.42) -- so there's no
  systematic bias/miscalibration. But its spread (std≈4.9) is almost as
  wide as its gap to `(0,5]`, so the two distributions sit almost on top of
  each other. Cosine's version of the same subgroup (median 0.744 vs
  `(0,5]`'s 0.272) has real density bunched up near 1.0 -- a region every
  other GT-MCES bin's distribution has decayed to near-zero density in --
  giving it a genuinely low-overlap "exclusive zone" that SIMBA's low-MCES
  end doesn't have (SIMBA's near-0 region is heavily shared with `(0,5]`'s
  own distribution, which also piles up there).

Working conclusion: this looks more like an architectural/resolution
limit -- SIMBA isn't confidently *wrong* about these pairs, it just lacks a
sharp, low-variance "almost certainly the same structure" mode the way raw
peak-matching has one built in -- rather than a training-time-augmentation
bias, though that isn't fully ruled out. Candidate next steps, not yet
built (open per the last conversation):
1. Check whether the model's output resolution near MCES=0 is inherently
   coarse (e.g. predictions clustering on a handful of discrete-ish values)
   vs. genuinely continuous but merely high-variance.
2. Reduce training-time augmentation (peak dropout / precursor noise)
   specifically for low-GT-MCES pairs, so the model sees more "clean"
   comparisons for exactly this range, rather than only adding sampler
   weight (quantity, not signal quality -- which is what experiments 011/012
   already did).
3. A more invasive option: a margin/contrastive loss term specifically
   penalizing same-molecule-different-spectrum predictions that land inside
   `(0,5]`'s typical range, instead of relying purely on regression/
   classification loss + sampling weight to teach the distinction
   indirectly.

## 10. Hit@1/5/20 retrieval benchmark, and where SIMBA vs. cosine disagree

Turns the "self, different spectrum" population from section 9 into an
actual retrieval task: for every validation molecule with ≥2 spectra
(2,010 of them), rank the true match (its own other spectrum) against 255
decoys and see whether it lands in the top 1/5/20.

**Feasibility, confirmed before building anything**: the 2,731-molecule
self-bucket pool has *complete* pairwise GT-MCES coverage already sitting
in the saved validation tables -- 3,727,815 cross-molecule pairs among this
pool, exactly `C(2731,2)`. So both GT MCES (for decoy selection) and every
experiment's own `pred_mces` (for scoring) are already logged for any
query-vs-decoy comparison; no fresh MCES computation or GPU re-inference
needed anywhere in this section.

**`tools/benchmark_self_retrieval.py`**: for each query molecule, decoys are
the 255 *other* pool molecules with the lowest GT MCES to it (decoy
selection is GT-MCES-only, hence identical across every method -- only the
ranking differs). Candidates = 255 decoys + the true match = 256. Rank by
predicted MCES ascending (SIMBA) or cosine similarity descending. Uses one
dense `(2731, 2731)` matrix per method (GT + one per experiment + cosine)
for fast neighbor lookup/ranking, built once and reused across all 2,010
queries. Results, most recent checkpoint of each:

| Run | Hit@1 | Hit@5 | Hit@20 |
|---|---|---|---|
| 009 | 0.475 | 0.620 | 0.750 |
| 010 | 0.540 | 0.695 | 0.805 |
| 011 | 0.585 | 0.762 | 0.874 |
| 012 | 0.585-0.593 | 0.762-0.775 | 0.874-0.879 |
| cosine (raw spectral baseline) | 0.679 | 0.854 | 0.917 |

(012's range reflects it still actively training between separate runs of
the benchmark, not a bug -- each run uses whatever the latest checkpoint
was at that moment.) Monotonic improvement 009→012, consistent with
everything else found this session; cosine still ahead of every SIMBA
checkpoint, consistent with section 9's finding.

**Caveat, flagged but not fixed**: a query/decoy molecule's exact spectrum
representation isn't perfectly pinned down for every comparison. Which
spectrum of a multi-spectrum molecule gets used in a saved pair row depends
on whether that molecule lands in the `mol_idx_0` or `mol_idx_1` column of
that specific row (first spectrum vs. last spectrum respectively -- see
`CustomDatasetMultitasking`'s val-time index selection, section 3). The
query's own self-pair row is consistent (always spec_idx_0 vs spec_idx_1),
but which of those two same spectra represents it in a given *decoy*
comparison can silently switch depending on how the original pair-list
generation happened to order the two molecules. Fixing this properly would
need fresh inference for whichever comparisons use the "wrong" spectrum;
left as-is (scored exactly as already logged) since the whole point of this
benchmark was to use only what's already saved. Likely a minor, roughly
unbiased noise source rather than something invalidating the results.

**Dashboard integration**: `compute_hit_at_k_all` in `tools/dashboard_app.py`
is a self-contained duplicate of the same logic (not a cross-script import,
to keep the dashboard's only dependencies as installed packages / the simba
package) -- see section 5's Compare Runs entry, including the crash and its
resolution (checkbox defaulted to checked; root cause of the segfault
itself was never confirmed).

**`tools/confusion_hit_simba_vs_cosine.py`**: 2×2 confusion matrix (n and %)
of hit@k agreement between one experiment and cosine. For 012 vs. cosine,
hit@1, n=2,010:

|  | cosine hit@1 | cosine miss@1 |
|---|---|---|
| **SIMBA hit@1** | n=1,048 (52.1%) | n=150 (7.5%) |
| **SIMBA miss@1** | n=317 (15.8%) | n=495 (24.6%) |

Both agree and succeed 52.1% of the time, both agree and fail together
24.6% of the time. Where they disagree, cosine's unique-win group (317,
15.8%) is about 2× the size of SIMBA's (150, 7.5%).

**`tools/dive_hit_disagreements.py`**: for each disagreement group, the
losing method's rank for the true match, and the GT MCES of whatever it
wrongly ranked #1 instead (run against 012 vs. cosine):

- **SIMBA wrong, cosine right (n=317)**: SIMBA's rank for the true match --
  median 5, mean 21.8 (top-5 53.9%, top-10 67.8%, but a long tail out to
  rank 249). What it ranked #1 instead has median GT MCES **12.5** to the
  query -- only 17% of its wrong picks are genuinely close decoys
  (GT MCES≤5); 58% are decoys that are actually quite structurally distant
  (GT MCES>10).
- **cosine wrong, SIMBA right (n=150)**: cosine's rank for the true match --
  median 3, mean 8.9 (top-5 76.7%, top-10 86%, worst case rank 138). What
  it ranked #1 instead has median GT MCES **6.0** -- 57% of its wrong picks
  are genuinely close decoys (GT MCES≤10).

**Reading**: it's not just that SIMBA is somewhat worse in these
disagreement cases -- the two methods' errors are qualitatively different.
Cosine's mistakes look "reasonable" (confusing the true match with
something that really is structurally close). SIMBA's mistakes more often
rank something genuinely dissimilar (MCES 10-20) above the actual match --
a more concerning failure mode than a close call between similar
candidates. Not yet investigated further or acted on.
