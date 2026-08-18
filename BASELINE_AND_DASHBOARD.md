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
| `tools/dashboard_app.py` | Streamlit monitoring/analysis dashboard (section 5) |
