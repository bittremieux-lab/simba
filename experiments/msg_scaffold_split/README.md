# MSG Scaffold Split Experiment

## Motivation

The standard random train/val split allows molecules with the same Murcko scaffold to appear in both sets, which inflates validation performance. This experiment uses a scaffold-based split to give a more honest OOD evaluation: molecules sharing a scaffold stay in the same partition.

Two validation sets are maintained throughout training:
- **Scaffold val** — molecules with unseen scaffolds (OOD, harder)
- **Official val** — the original MassSpecGym random-split validation set (in-distribution, easier)

## Key changes

| File | What changed |
|------|-------------|
| `simba/workflows/training.py` | Scaffold split support; dual val dataloader; MCES-based weighted sampling (bin width 4); lightweight format loading |
| `simba/core/models/similarity_models.py` | `validation_step` accepts `dataloader_idx`; `training_step` returns `mces_pred`/`mces_target` for callback use |
| `simba/core/training/callbacks.py` | Spearman + MAE curves for train/scaffold/official; per-epoch hexbin plots |
| `simba/commands/train.py`, `sweep_train.py` | Updated to 5-tuple `load_dataset` and 8-tuple `prepare_data` return signatures |

## How to run

### 1. Preprocessing

```bash
sbatch tools/slurm/preprocess_msg_scaffold_split.slurm.sh
```

Produces `mapping.pkl` with scaffold-grouped train / val / val_official / test splits at:
```
/mnt/data2/nkubrakov/massspecgym/preprocessing_msg_scaffold_split_mces40/
```

### 2. Training

```bash
sbatch tools/slurm/train_msg_scaffold_split.slurm.sh
```

Checkpoints and per-step hexbin plots saved to:
```
/mnt/data2/nkubrakov/experiments_3_dataset/training/msg_scaffold_split_mces40/
```

### 3. Inference + evaluation plots

Run inference on all three sets (scaffold val, official val, official test) and produce balanced hexbin plots:

```bash
sbatch tools/slurm/val_hexbin.slurm.sh
```

Outputs saved to `<checkpoint_dir>/val_hexbin/`:
- `val_predictions_{scaffold,official,test}.csv` — raw predictions for all pairs
- `mces_hexbin_balanced_4panel.png` — 6-panel balanced hexbin (2 scales × 3 sets)

To rebuild plots from existing CSVs without rerunning inference:

```bash
uv run python tools/plot_val_hexbin_balanced.py \
  --val_dir /mnt/data2/nkubrakov/experiments_3_dataset/training/msg_scaffold_split_mces40/val_hexbin
```

### 4. Official splits inference

```bash
sbatch tools/slurm/inference_msg_official_splits.slurm.sh
```

## Ablation: metadata (adduct / CE / ion mode)

**Question**: does giving the model adduct, collision energy and ion mode improve similarity predictions?

Same setup as above (mces40 preprocessing, MCES-only objective, scaffold split) with `use_adduct`, `use_ce`, `use_ion_mode` all set to `true`.

```bash
sbatch tools/slurm/train_msg_scaffold_split_metadata.slurm.sh
```

Checkpoints saved to:
```
/mnt/data2/nkubrakov/experiments_3_dataset/training/msg_scaffold_split_mces40_metadata/
```
