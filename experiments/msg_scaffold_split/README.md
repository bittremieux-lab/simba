# MSG Scaffold Split Experiment

## Experiments

| Name | Key change | Commit | Job | Prepro | Results |
|------|-----------|--------|-----|--------|---------|
| discard-20 | Discard pairs with MCES > 20 | `ad31243` | — | `preprocessing_msg_scaffold_split` | `msg_scaffold_split_v3` |
| clip-40 | Keep all pairs, clip at 40, n_classes=11 | `c4f2e0b` | 7614 | `preprocessing_msg_scaffold_split_mces40` | `msg_scaffold_split_mces40` |
| clip-40-metadata | Same + adduct / CE / ion mode | `14c530f` | 7636 | `preprocessing_msg_scaffold_split_mces40` | `msg_scaffold_split_mces40_metadata` |
| sv2-mces-only | scaffold_v2 data, MCES head only, ED-based sampling, lr=3.33e-5 | — | — | `preprocessing_scaffold_v2` | `scaffold_v2_mces_only` |
| sv2-both | scaffold_v2 data, ED+MCES both objectives, lr=3.33e-5 | — | — | `preprocessing_scaffold_v2` | `scaffold_v2_both` |
| gaetan-official-clip-40 | Official MSG splits, Gaetan lb_matrix distances, scaffold 10% val, clip at 40, n_classes=11 | `26ebf0f` | 7656 | `preprocessing_msg_gaetan_official` | `msg_gaetan_official_mces40` |

All paths under `/mnt/data2/nkubrakov/` (prepro) and `/mnt/data/nkubrakov/experiments_3_dataset/training/` (results).

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
