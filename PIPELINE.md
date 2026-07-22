# SIMBA MCES Pipeline

Step-by-step guide: preprocessing → training → validation → retrieval evaluation.

---

## 0. Data locations

| Path | Description |
|---|---|
| `/mnt/data2/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf` | Raw spectra |
| `/mnt/data2/gdewaele/lb_matrix.npy` | MCES lower-bound matrix |
| `/mnt/data2/gdewaele/lb_matrix.smiles.txt` | SMILES list for lb_matrix rows/cols |
| `/mnt/data2/nkubrakov/massspecgym/data/auxiliary/all_smiles_mces.hdf5` | Pairwise MCES (exact for MCES < 10) |
| `/mnt/data/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json` | Retrieval candidates |

---

## Step 1 · Preprocessing

**Script:** `tools/prepare_msg_max_lb_hdf5.py`

Builds pair npy files using `max(lb_matrix, HDF5 MCES)` as the distance source.
This gives exact values for similar pairs (MCES < 10 from HDF5) and tighter lower
bounds for dissimilar pairs (MCES ≥ 10 from lb_matrix).
Performs a Murcko scaffold split on the official train split to get additional validation from it: 80% train / 10% scaffold val / 10% official val / test set.

**Inputs** (paths hardcoded at top of script):
- `MassSpecGym.mgf` — raw spectra
- `lb_matrix.npy` + `lb_matrix.smiles.txt` — MCES lower-bound matrix
- `all_smiles_mces.hdf5` — pairwise MCES

**Outputs:** `/mnt/data/nkubrakov/massspecgym/preprocessing_msg_max_lb_hdf5/`
```
mapping.pkl                                          ← molecule/spectrum metadata, split indices
ed_mces_indexes_tani_incremental_train_node0_chunk0.npy   ← (N_train_pairs, 4): [mol_a, mol_b, 0, mces]
ed_mces_indexes_tani_incremental_val_node0_chunk0.npy
ed_mces_indexes_tani_incremental_val_official_node0_chunk0.npy
ed_mces_indexes_tani_incremental_test_node0_chunk0.npy
```

**Note:** Reads from `/mnt/data2`. Paths in the script are hardcoded; edit `MGF_PATH`, `LB_MATRIX`, etc. before running.

**Command:**
```bash
uv run python tools/prepare_msg_max_lb_hdf5.py
```

---

## Step 2 · Exact MCES for [10, 20] pairs

Pairs with distance in [10, 20] got only a weak lower bound in step 1.
This step improves them via ILP solving (MCES threshold 20).
It has two sub-steps: compute, then apply.

### 2a. Compute exact MCES

**Script:** `tools/compute_mces_exact_1020.py`

Splits the train pairs with lb ∈ [10, 20] into 200 blocks and solves them in parallel.
Each SLURM array task handles one block (~313k pairs).

**Inputs:**
- `preprocessing_msg_max_lb_hdf5/ed_mces_indexes_tani_incremental_train_node0_chunk0.npy`
- `lb_matrix.smiles.txt`

**Outputs:** `/mnt/data2/nkubrakov/mces_exact_1020/blocks/block_NNN.npy` + `.done` sentinels

**Check status:**
```bash
uv run python tools/compute_mces_exact_1020.py status
# or
watch -n 60 'ls /mnt/data2/nkubrakov/mces_exact_1020/blocks/*.done | wc -l'
```

**Combine completed blocks:**
```bash
uv run python tools/compute_mces_exact_1020.py combine \
    --output_dir /mnt/data2/nkubrakov/mces_exact_1020
```
This writes `mces_exact_10_20.npy` — a (M, 3) array of `[mol_a, mol_b, exact_mces]`.

**SLURM (200-task array):**
```bash
sbatch --array=0-199 tools/slurm/mces_exact_msg_official.slurm.sh
# Benchmark first: sbatch --array=0 tools/slurm/mces_exact_msg_official.slurm.sh
```

### 2b. Apply exact MCES

**Script:** `tools/apply_exact_mces_1020.py`

Replaces the distance column in all split npy files where exact MCES was computed.
Pairs where the ILP solver timed out keep the original lower bound.

**Inputs:**
- `preprocessing_msg_max_lb_hdf5/` — original pairs (all splits)
- `mces_exact_1020/mces_exact_10_20.npy` — computed exact values

**Outputs:** `/mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020/`
```
ed_mces_indexes_tani_incremental_train_node0_chunk0.npy   ← updated distances
ed_mces_indexes_tani_incremental_val_node0_chunk0.npy
ed_mces_indexes_tani_incremental_val_official_node0_chunk0.npy
ed_mces_indexes_tani_incremental_test_node0_chunk0.npy
mapping.pkl                                               ← symlink to original
```

**Dry-run (stats only, no writes):**
```bash
uv run python tools/apply_exact_mces_1020.py --dry_run
```

**Apply:**
```bash
uv run python tools/apply_exact_mces_1020.py
```

---

## Step 3 · Training

**Command:** `uv run simba train ...`

Trains the SIMBA transformer on MCES similarity only (no edit distance head).
Validation runs on both scaffold and official splits simultaneously.

**Inputs:**
- `preprocessing_msg_exact_mces_1020/` — pairs + mapping.pkl (from step 2b)

**Outputs:** `/mnt/data/nkubrakov/experiments_3_dataset/training/<experiment_name>/`
```
checkpoint-epoch=XX-step=YYYYYY.ckpt
```

**Key config flags:**
- `model.tasks.edit_distance.enabled=false` — MCES-only training (no classification head)
- `model.tasks.edit_distance.n_classes=11` — still required for model init
- `model.features.use_adduct=false model.features.use_ce=false model.features.use_ion_mode=false` — no metadata
- `model.multitasking.learnable=false` — fixed loss weights
- `training.early_stopping_patience=0` — no early stopping, run full epochs

**MGF path override (lightweight mapping.pkl only):**
If `mapping.pkl` stores a stale MGF path, pass the correct one explicitly — it takes precedence:
```
paths.mgf_path=/mnt/data/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf
```

**SLURM:**
```bash
sbatch tools/slurm/train_msg_official.slurm.sh
```

**Full command (for reference):**
```bash
PREPRO_DIR=/mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020
OUTPUT_DIR=/mnt/data/nkubrakov/experiments_3_dataset/training/my_experiment
MGF=/mnt/data/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf

export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run simba train \
  paths.preprocessing_dir="${PREPRO_DIR}" \
  paths.preprocessing_dir_train="${PREPRO_DIR}" \
  paths.preprocessing_pickle_file=mapping.pkl \
  paths.checkpoint_dir="${OUTPUT_DIR}" \
  paths.mgf_path="${MGF}" \
  training.epochs=8 \
  training.batch_size=2048 \
  training.val_check_interval=1000 \
  training.limit_train_batches=10000 \
  training.limit_val_batches=100 \
  training.early_stopping_patience=0 \
  optimizer.lr=0.0001 \
  hardware.accelerator=gpu \
  hardware.devices=1 \
  hardware.num_workers=14 \
  logging.enable_progress_bar=false \
  logging.log_every_n_steps=10 \
  model.features.use_adduct=false \
  model.features.use_ce=false \
  model.features.use_ion_mode=false \
  model.multitasking.learnable=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11
```

---

## Step 4 · Validation hexbin

Embeds the validation spectra with a checkpoint and plots predicted vs GT MCES.
The balanced panel (1.5-rule) is the primary quality metric.

**Scripts:**
- `tools/run_val_hexbin.py` — embeds spectra and saves predictions
- `tools/plot_val_hexbin_balanced.py` — plots balanced + unbalanced hexbins

**Inputs:**
- `checkpoint-epoch=XX-step=YYYYYY.ckpt`
- `preprocessing_msg_exact_mces_1020/` — pairs + mapping.pkl

**Outputs:** `<checkpoint_dir>/val_hexbin_step_XX/`
```
val_predictions_scaffold.csv    ← (gt_mces, pred_mces) for scaffold val pairs
val_predictions_official.csv    ← same for official val split
mces_hexbin.png
mces_hexbin_balanced.png
mces_hexbin_max20.png           ← zoomed to MCES ≤ 20
mces_hexbin_balanced_max20.png
```

**SLURM:**
```bash
sbatch tools/slurm/val_hexbin_msg_official.slurm.sh
```

**Full command:**
```bash
CHECKPOINT=/mnt/data/nkubrakov/experiments_3_dataset/training/my_experiment/checkpoint-epoch=04-step=44000.ckpt
OUTPUT_DIR=/mnt/data/nkubrakov/experiments_3_dataset/training/my_experiment/val_hexbin_step44k

uv run python tools/run_val_hexbin.py \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --prepro_dir /mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020 \
  --batch_size 3072 \
  --num_workers 8 \
  model.features.use_adduct=false \
  model.features.use_ce=false \
  model.features.use_ion_mode=false \
  model.multitasking.learnable=false \
  model.tasks.edit_distance.enabled=false \
  model.tasks.edit_distance.n_classes=11

uv run python tools/plot_val_hexbin_balanced.py --val_dir "${OUTPUT_DIR}"
uv run python tools/plot_val_hexbin_balanced.py --val_dir "${OUTPUT_DIR}" --mces_max 20
```

---

## Step 5 · Retrieval

Evaluates the model on the MassSpecGym retrieval benchmark:
embed all test spectra → nearest-neighbor search to train embeddings → transfer Morgan FP → rank candidates by Tanimoto.

**Script:** `tools/simba_retrieval.py`

**Key preprocessing details (must match training):**
- `n_layers=5` (not 8 — the default CLI value)
- Peak selection: top-100 by intensity (matching `filter_intensity` in training)
- Normalization: `sqrt` then L2 (matching `Augmentation.normalize_intensities`)

**Inputs:**
- Checkpoint (`.ckpt`)
- `MassSpecGym.mgf` — raw spectra for test set
- `MassSpecGym_retrieval_candidates_mass.json` — candidate molecules per query

**Outputs:** `<intermediates_dir>/`
```
test_embeddings.pt          ← (N_test, D) float32 tensor
test_smiles.json            ← SMILES list matching embedding rows
train_embeddings.pt         ← (N_train, D) float32 tensor
train_fps.npy               ← (N_train, 2048) Morgan fingerprints
retrieval_results.tsv       ← hit@1 / hit@5 / hit@10 / hit@20 per query, plus summary
```

**SLURM:**
```bash
sbatch tools/slurm/retrieval_msg_official.slurm.sh
```

**Full command:**
```bash
CHECKPOINT=/mnt/data/nkubrakov/experiments_3_dataset/training/my_experiment/checkpoint-epoch=04-step=44000.ckpt
INTERMEDIATES_DIR=/mnt/data/nkubrakov/experiments_3_dataset/retrieval/my_run

uv run python tools/simba_retrieval.py \
    --checkpoint "${CHECKPOINT}" \
    --mgf /mnt/data/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf \
    --candidates /mnt/data/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json \
    --split test \
    --batch_size 512 \
    --intermediates_dir "${INTERMEDIATES_DIR}" \
    --output_tsv "${INTERMEDIATES_DIR}/retrieval_results.tsv"
```

**Reading results:**
```
hit@1   hit@5   hit@10  hit@20   ← fraction of queries with the correct molecule in top-k
```
Current best (step 44k): hit@1=4.67%, hit@5=12.59%, hit@20=26.54%.

---

## Step 6 · Embedding quality: test-test hexbin

Validates test-split embedding quality without running retrieval.
Loads precomputed `test_embeddings.pt` and computes `pred_mces = (1 − dot(emb_a, emb_b)) × 40`
against the GT pairs from the npy file.

**Script:** `tools/test_test_mces_hexbin.py`

**Inputs:**
- `<intermediates_dir>/test_embeddings.pt`
- `<intermediates_dir>/test_smiles.json`
- `preprocessing_msg_exact_mces_1020/ed_mces_indexes_tani_incremental_test_node0_chunk0.npy`
- `preprocessing_msg_exact_mces_1020/mapping.pkl`

**Outputs:** PNG with two panels — unbalanced and GT-balanced hexbin.
Expected: ρ ≈ 0.57–0.70 (balanced panel should match the val hexbin test panel).

**Command:**
```bash
uv run python tools/test_test_mces_hexbin.py \
    --intermediates_dir /mnt/data/nkubrakov/experiments_3_dataset/retrieval/my_run \
    --npy_pairs /mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020/ed_mces_indexes_tani_incremental_test_node0_chunk0.npy \
    --prepro_pkl /mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020/mapping.pkl \
    --output results/test_test_mces_hexbin_my_run.png
```

---

## Step 7 · Embedding quality: test-train hexbin

Cross-split analysis: how well do test embeddings predict MCES against training molecules?
Uses the nearest-neighbor train molecules found during retrieval.
Produces 5 panels: random pairs, per-bin capped, GT-balanced, and two box plots.

**Script:** `tools/test_train_mces_hexbin.py`

**Inputs:**
- `<intermediates_dir>/test_embeddings.pt`, `test_smiles.json`
- `<intermediates_dir>/train_embeddings.pt`, `train_fps.npy`
- `preprocessing_msg_exact_mces_1020/mapping.pkl`
- `preprocessing_msg_exact_mces_1020/ed_mces_indexes_tani_incremental_test_node0_chunk0.npy`

**Outputs:** PNG with 5 panels.
Expected cross-split ρ ≈ 0.68 (GT-balanced panel).

**Command:**
```bash
uv run python tools/test_train_mces_hexbin.py \
    --intermediates_dir /mnt/data/nkubrakov/experiments_3_dataset/retrieval/my_run \
    --prepro_pkl /mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020/mapping.pkl \
    --test_npy /mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020/ed_mces_indexes_tani_incremental_test_node0_chunk0.npy \
    --output results/test_train_mces_hexbin_my_run.png
```

---

## Step 8 · Oracle upper bound (optional)

Computes the structural oracle retrieval: re-rank candidates by GT Tanimoto similarity
to the true query molecule. Sets the performance ceiling for any embedding-based retrieval.

**Script:** `tools/oracle_retrieval_max_lb_hdf5.py`

**Inputs:**
- `preprocessing_msg_exact_mces_1020/mapping.pkl`
- `MassSpecGym_retrieval_candidates_mass.json`

**Outputs:** TSV with oracle hit@k metrics.

**Command:**
```bash
uv run python tools/oracle_retrieval_max_lb_hdf5.py \
    --prepro_pkl /mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020/mapping.pkl \
    --candidates /mnt/data/nkubrakov/massspecgym/data/molecules/MassSpecGym_retrieval_candidates_mass.json \
    --output results/oracle_retrieval.tsv
```

---

## Running tests

```bash
uv run pytest tests/
```

Key test files added with this pipeline:
- `tests/unit/test_load_mces.py` — `find_file` prefix-matching fix
- `tests/unit/test_embedder_multitask.py` — model forward pass
- `tests/integration/test_training_pipeline.py` — end-to-end train step

---

## Summary of pipeline data flow

```
MassSpecGym.mgf
lb_matrix.npy + all_smiles_mces.hdf5
        │
        ▼  tools/prepare_msg_max_lb_hdf5.py
preprocessing_msg_max_lb_hdf5/
  mapping.pkl + {train,val,val_official,test}_node0_chunk0.npy
        │
        ▼  tools/compute_mces_exact_1020.py  (SLURM array)
mces_exact_1020/mces_exact_10_20.npy
        │
        ▼  tools/apply_exact_mces_1020.py
preprocessing_msg_exact_mces_1020/
  mapping.pkl + updated npy files
        │
        ├──▶  uv run simba train ...          → checkpoint.ckpt
        │              │
        │              ▼  tools/run_val_hexbin.py + plot_val_hexbin_balanced.py
        │         val_hexbin_stepXXk/          → hexbin plots  (ρ ~ 0.63 balanced)
        │              │
        │              ▼  tools/simba_retrieval.py
        │         retrieval/                   → test_embeddings.pt + retrieval_results.tsv
        │                        │
        │              ┌─────────┴──────────────┐
        │              ▼                        ▼
        │   test_test_mces_hexbin.py    test_train_mces_hexbin.py
        │   (test-split embedding ρ)   (cross-split embedding ρ)
        │
        └──▶  tools/oracle_retrieval_max_lb_hdf5.py → oracle hit@k (ceiling)
```
