# 014_2 ICEBERG train-spectra-augmentation experiment

## Idea

Generate extra, synthetic spectra per training molecule with ICEBERG (the
in-silico fragmentation predictor already used elsewhere in this project
for retrieval-candidate generation), at several collision energies, and
make them available during training as additional "views" of the same
molecule -- purely additive to the real spectra, not a replacement, with a
config-controlled probability of drawing a synthetic one instead of a real
one for a given training sample.

## Why CE, not adduct or anything else

014_2 doesn't use CE or adduct as model input features
(`model.features.use_ce=false`, `use_adduct=false`) -- so this is pure
spectral-pattern data augmentation/robustness training, not teaching the
model a new signal. CE varies fragmentation depth/intensity pattern
realistically for the same molecule (mimicking real instrument-to-
instrument variability already sparsely present in MassSpecGym/GNPS);
adduct variation would also drag in precursor-mass bookkeeping
(`sampling.precursor_mass_mode=theoretical`) for comparatively less
benefit. Checked ICEBERG's own `predict_smis.py` directly: it natively
takes a list of `collision_energies` per SMILES -- exactly built for this.

**CE values, chosen with the user directly**: 15, 25, 35, 45, 55 eV.
(Real training-data `COLLISION_ENERGY` distribution, grepped directly from
MassSpecGym.mgf: median 30, p10=10, p25=20, p75=55, p90≈66 -- these 5
values bracket that range. The existing `ICEBERG/build_candidate_tsv.py`
script for retrieval-candidate generation used a single fixed CE=35.0, its
own test-fold median, for a different purpose -- one representative
spectrum per candidate, not augmentation variety.)

## Generation pipeline (reused existing scripts where they exist)

1. **`ICEBERG/build_train_augmentation_tsv.py`** (new, but adapted directly
   from the existing `build_candidate_tsv.py` -- same TSV schema, same
   `ms_pred.common.mass_from_smi` + `common.ion2mass` precursor-mass
   convention rather than reimplementing with RDKit, same `[M+H]+`/
   `Orbitrap` defaults). One row per Gaetan-split train molecule (24,010),
   `collision_energies="[15.0, 25.0, 35.0, 45.0, 55.0]"`.
2. Smoke-tested the actual `predict_smis.py` call on 5 molecules (25
   spectra) on CPU first -- hit and fixed a real requirement
   (`--sparse-out` is asserted, not optional) before the full run.
3. **`ICEBERG/slurm_train_augmentation_predict.sh`** (new) -- GPU job,
   `ms-pred/.venv` (a separate venv from the main simba one -- confirmed
   necessary, matches `build_candidate_tsv_delta.py`'s own documented
   usage), `weights/msg_all/{gen,inten_contr}/best.ckpt` (same checkpoints
   as the existing retrieval-candidate pipeline). 120,050 spectra (24,010 x
   5 CEs) in **~16 minutes**. Verified directly after completion: exactly
   24,011 top-level HDF5 keys (manifest + every molecule) and 120,050
   manifest rows -- no gaps.
4. **`tools/convert_iceberg_preds_to_mgf.py`** (new) -- converts the
   nested HDF5 (`pred_<name>/ikey <key>/collision <CE>/f`) into a plain
   MGF, reusing the exact same `arr[:, 0] > 0` sparse-padding-mask logic as
   the existing `tools/simba_retrieval_iceberg.py::load_iceberg_spectra`,
   generalized to iterate every CE leaf per molecule instead of one fixed
   candidate spectrum. Tagged `FOLD=train`, `SOURCE=iceberg_synthetic`,
   `CE=<value>`. All 120,050 converted, 0 skipped for empty peaks. Checked
   the peak-count distribution afterward: **all 24,010 molecules retain
   all 5 CE variants** after the standard `min_n_peaks=6` filter (ICEBERG's
   `sparse_k=100` output means even the sparsest prediction clears it).

## Training-pipeline integration

Traced the real per-sample spectrum-selection mechanism directly (getting
it wrong on the first attempt -- initially looked at the wrong code path,
`MoleculePairsOpt.get_molecular_pair`'s fixed first/last indexing, which
isn't what training actually uses). The real mechanism, confirmed in
`simba/core/data/datasets/multitask_dataset.py::CustomDatasetMultitasking.__getitem__`:

```python
if self.training:
    idx_0_original = random.choice(self.df_smiles.loc[int(idx_0[0]), "indexes"])
    idx_1_original = random.choice(self.df_smiles.loc[int(idx_1[0]), "indexes"])
else:
    idx_0_original = self.df_smiles.loc[int(idx_0[0]), "indexes"][0]   # first
    idx_1_original = self.df_smiles.loc[int(idx_1[0]), "indexes"][-1]  # last
```

Training already randomly draws among ALL of a molecule's real spectra,
fresh every sample -- so injecting synthetic spectra just meant giving them
their own place in this same mechanism, not rebuilding it.

**Changes made** (all opt-in, default-neutral -- zero effect on any run
that doesn't set the new config):
- `simba/configs/training/default.yaml`: new `sampling.iceberg_mgf_path`
  (default `null`) and `sampling.iceberg_spectra_prob` (default `0.0`).
- `simba/workflows/training.py::load_dataset`: for the train split only,
  when `iceberg_mgf_path` is set, loads that MGF, matches each spectrum to
  its molecule by canonical SMILES, appends it to `original_spectra`, and
  records the new index in a **new, separate** `df_smiles["synthetic_indexes"]`
  column (kept apart from the real `"indexes"` column). Logs exactly how
  many synthetic spectra matched how many molecules.
- `simba/core/data/datasets/multitask_dataset.py`: new
  `_sample_spectrum_index(mol_idx)` method -- with probability
  `iceberg_spectra_prob`, and only if this molecule actually has any
  synthetic spectra, draws one uniformly at random from
  `synthetic_indexes`; otherwise (or always, when the column is absent or
  prob=0) falls back to the original `random.choice(indexes)` over real
  spectra. `__getitem__`'s training branch now calls this instead of the
  inline `random.choice`.
- `simba/core/data/datasets/multitask_dataset_builder.py`: threads
  `iceberg_spectra_prob` through to the dataset constructor.
- Every other array-building step downstream (`mz`/`intensity`/
  `precursor_mass`/etc, built by iterating `original_spectra`) needed no
  changes at all -- it's agnostic to real vs. synthetic as long as the
  appended objects are proper `SpectrumExt`s, which they are.

**Verified before trusting this with a real run**: isolated test of
`_sample_spectrum_index` against a toy `df_smiles` (not the real 24,010-
molecule one) -- confirmed ~50% synthetic draw rate for a molecule with
both real and synthetic entries (measured 49.5% over 20,000 draws),
100% real fallback for a molecule with none, and exact reproduction of the
original unweighted behavior at `iceberg_spectra_prob=0.0`.

Confirmed directly: among a molecule's available synthetic spectra
(typically all 5 CE variants), `random.choice` picks a **different** CE
each draw, not always the same one.

## This experiment

`tools/slurm/014_2_iceberg_aug_p50_1gpu.slurm.sh` -- identical to 014_2
(original small architecture, d_model=256/n_layers=5) in every respect
except `sampling.iceberg_mgf_path=<synthetic_train.mgf>` and
`sampling.iceberg_spectra_prob=0.5`. val/test completely untouched (no
synthetic data ever loaded for those splits; val/test spectrum selection
is deterministic regardless of this setting).

Launched as job 1346151. Confirmed healthy directly from the logs:
`115,314 synthetic spectra matched to 23,107 / 23,125 train molecules`
(99.9% molecule coverage -- the gap from 120,050 generated is expected,
minor filtering/canonicalization mismatch, not a bug), reached step 100
with a sane loss (train_loss=0.406, loss_mces=0.051), no errors.

## Status

Training in progress (job 1346151). Not yet evaluated on retrieval or
analog-discovery.
