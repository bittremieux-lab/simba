# 014_2 CORN-corrected ICEBERG retrieval — Gaetan-split results

Extends the comparison in `NOTES_RETRIEVAL_SPLIT_COMPARISON.md` (built on
checkpoint 005) with checkpoint 014_2, which adds an auxiliary CORN-ordinal
MCES-bucket head on top of the primary cosine-similarity regression task
(see `BASELINE_AND_DASHBOARD.md` section 11 for how that head works and how
its dashboard-side correction formula was derived/validated).

## Results — Gaetan's own split (n=14,118 test spectra)

| Method | Hit@1 | Hit@5 | Hit@20 |
|---|---:|---:|---:|
| Cosine-NN (spectral, no model) | 15.17% | 25.92% | 38.55% |
| SIMBA-NN (005, raw regression) | 8.31% | 17.55% | 31.29% |
| SIMBA-NN (014_2, raw regression) | 9.99% | 19.93% | 33.56% |
| SIMBA-NN (014_2, CORN-corrected) | 9.36% | 18.90% | 32.99% |
| Oracle GT-MCES-NN | 32.09% | 53.91% | 71.51% |
| ICEBERG + Cosine (spectral, no model) | 44.86% | 68.15% | 82.12% |
| ICEBERG + SIMBA (005, raw regression) | 11.47% | 26.94% | 48.75% |
| ICEBERG + SIMBA (014_2, raw regression) | 19.54% | 40.02% | 60.26% |
| **ICEBERG + SIMBA (014_2, CORN-corrected)** | **20.98%** | **43.48%** | **63.52%** |

Restricting to test spectra with >= 6 peaks (SIMBA's own canonical minimum,
`simba/configs/data/default.yaml`'s `min_n_peaks: 6`) changes nothing here —
Gaetan's own data-prep script already enforces that floor before assigning
folds, so all 14,118 test spectra already qualify.

**Naming note**: "Cosine" rows are model-free — plain binned spectral peak
cosine similarity, no SIMBA involved. "SIMBA" rows are the model's own
*trained regression output* — `head_mode=cosine_no_head` computes
`cosine_similarity(emb0, emb1)` as the literal mechanism, but the model was
optimized via a regression loss to make that value track `1 - MCES/max`, so
it's a learned prediction, not an independent similarity metric. Calling it
"cosine" (as an earlier draft of this table did) collides with the unrelated
spectral-cosine rows above — hence "raw regression" here.

### What "CORN-corrected" changes

The auxiliary bucket head predicts a coarse MCES bucket
(`0, (0,2], (2,4], (4,6], (6,8], (8,inf)`); the corrected score clips the
primary head's raw regression value into that bucket's range, then breaks
within-bucket ties with the raw value (`corrected*1000 + raw`, ascending —
see `tools/simba_retrieval_iceberg.py`'s `_corn_corrected_ranking_score`,
identical formula to the dashboard's). This **cannot** be computed from
cached per-spectrum embeddings the way the raw-regression ranking can — the
bucket head takes `abs(emb0-emb1)` on raw, magnitude-sensitive embeddings,
a function of the *pair*. The expensive part (transformer encoding) is still
one pass per spectrum either way; only the small bucket-head MLP runs
pairwise.

It helps on the ICEBERG candidate pool (~256 formula-matched candidates per
query, +1.4pt Hit@1) but slightly hurts on the full train-transfer search
(~24,010 unique train molecules per query, -0.6pt Hit@1) — a real result,
not a bug (see verification below), plausibly because bucket-0 tie
collisions behave differently at very different candidate-pool scales.

### Verification performed (no bugs found)

Before trusting these numbers, two things were checked directly against
code and real data rather than assumed:
1. **Formula correctness**: hand-verified the corrected/score arithmetic
   against printed per-pair values (`tools/diagnose_corn_iceberg.py`'s
   output) — exact match every time.
2. **Isolating the correction from the underlying embedding signal**: ran
   the *same* embeddings through the plain (unmodified since checkpoint 005)
   cosine-ranking code path — got 19.54/40.02/60.26, nearly identical to the
   corn-corrected 20.98/43.48/63.52 — proving the new pairwise code isn't
   the source of the gap to Cosine.
3. **Preprocessing/conditioning parity** between this retrieval pipeline and
   the real training/validation dataloader — confirmed identical on every
   axis checked (fragment-tolerance precursor-peak removal, min_intensity
   filter, max_num_peaks truncation rule, intensity sqrt+L2-normalize). The
   one real gap found (`tools/simba_retrieval.py`'s CLI lacking a
   `--precursor_mass_mode` flag) doesn't affect these numbers since
   `tools/simba_retrieval_iceberg.py` already exposes and correctly uses it
   (`--precursor_mass_mode theoretical`, matching how 014_2 itself was
   trained).
4. **Dataset validity**: confirmed the Gaetan-split `mapping.pkl` is
   byte-identical (same MD5/SHA256) between the pre-bugfix (v1) and
   bugfixed (v2) versions — the v1→v2 fix touched only an internal
   MCES-ground-truth lookup table, not fold membership — so the existing
   ICEBERG candidate/prediction files and `gaetan_test.mgf` remain fully
   valid; no GPU rebuild was needed.

Net conclusion: SIMBA's embeddings are just genuinely weaker than plain
spectral cosine on ICEBERG-predicted (synthetic) candidates — the same
qualitative pattern the original checkpoint-005 comparison already showed,
not something introduced or hidden by this session's changes.

---

## Pipeline: how to reproduce this column end to end

All paths below are relative to the repo root
(`/sofia/projects/2026_053/simba_project/`). Everything reuses the existing
v1 Gaetan-split extraction/candidate files — check first whether the split
you care about has actually changed (`md5sum` the two `mapping.pkl`s) before
re-running the expensive ICEBERG candidate-generation steps.

### 0. One-time data check (skip if already done for your split)

```bash
md5sum data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5{,_v2}/mapping.pkl
```
If these differ, the ICEBERG candidate files below need rebuilding from
scratch: re-run the extraction step against the new `mapping.pkl`
(`tools/extract_spectra_by_mgf_index.py`), then
`ICEBERG/build_candidate_tsv_delta.py`, then ICEBERG's own
`ms-pred/src/ms_pred/iceberg/predict_smis.py` (weights already present under
`ICEBERG/weights/msg_all/`). If identical, skip straight to step 1.

### 1. Cosine-NN + SIMBA-NN (no ICEBERG; train-transfer)

`experiments/retrieval_split_comparison/gaetan_test.mgf` already contains
BOTH the Gaetan train fold (141,898 spectra) and test fold (14,118 spectra)
— no separate extraction needed.

```bash
sbatch tools/slurm/nn_transfer_gaetan_test_014_2_1gpu.slurm.sh
```
Runs `tools/simba_retrieval.py` twice: plain raw-regression NN-transfer,
then `--corn_corrected` NN-transfer (`nearest_neighbor_transfer_corn_corrected`
in the same file — chunked pairwise bucket-head search over ~339M
test×train pairs, ~10 min total on one H200). Cosine-NN itself is
model-free (`tools/cosine_retrieval.py`) — re-run only if the dataset
changed; otherwise reuse `experiments/retrieval_split_comparison/gaetan_test/cosine_nn/retrieval_results.tsv`.

### 2. Oracle GT-MCES-NN

Checkpoint-independent (`tools/oracle_retrieval_gt_mces.py`) — reuse
`experiments/retrieval_split_comparison/gaetan_test/oracle_gt_mces_nn/retrieval_results.tsv`
unless the split changed.

### 3. ICEBERG + Cosine

Model-free, CPU-only, runs directly on the login node (no SLURM needed):
```bash
uv run python tools/cosine_baseline_iceberg.py \
    --mgf experiments/retrieval_split_comparison/gaetan_test.mgf \
    --candidates /sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json \
    --candidate_tsv ICEBERG/data/candidates_gaetan_test_existing_overlap.tsv ICEBERG/data/candidates_gaetan_test_new.tsv \
    --iceberg_preds ICEBERG/results/candidates_test_official/preds.hdf5 ICEBERG/results/candidates_gaetan_test_new/preds.hdf5 \
    --split test --min_peaks 6 --skip_mces \
    --output_tsv <out>/retrieval_results.tsv
```
Reuse the existing result if the dataset hasn't changed (see step 0).

### 4. ICEBERG + SIMBA (raw regression AND CORN-corrected)

```bash
sbatch tools/slurm/retrieval_iceberg_014_2_corn_corrected_1gpu.slurm.sh
```
Runs `tools/simba_retrieval_iceberg.py --corn_corrected --mces_bucket_use_mlp
--precursor_mass_mode theoretical` (the CORN-corrected row). For the
raw-regression row, run the same command without `--corn_corrected`
(or use `tools/diagnose_corn_iceberg.py`'s plain-cosine-ranking output,
step 5, which computes both from the same embeddings in one pass).

### 5. (Optional) Sanity-check a new checkpoint before trusting its numbers

```bash
sbatch tools/slurm/diagnose_corn_iceberg_014_2_1gpu.slurm.sh
```
Runs `tools/diagnose_corn_iceberg.py`: computes raw-regression Hit@k and
CORN-corrected Hit@k from the *same* embeddings in one pass (isolates
whether a correction bug vs. a weak embedding signal explains any gap), and
prints per-pair arithmetic for a handful of real queries for manual
verification. Recommended whenever evaluating a checkpoint with a new
`mces_bucket` configuration (different bin edges, `use_mlp`/`use_product`)
for the first time.

### Key file paths for this checkpoint (014_2)

- Checkpoint: `experiments/training/014_2_mces_bucket_mlp_1gpu/checkpoint-epoch=22-step=229000.ckpt`
- `head_mode=cosine_no_head`, `--mces_bucket_use_mlp` (matches
  `model.tasks.mces_bucket.use_mlp=true`; `use_product` stays off)
- `--precursor_mass_mode theoretical` (matches how every checkpoint from
  experiment 010 onward was trained — `sampling.precursor_mass_mode`)
- CORN bucket edges: `[2.0, 4.0, 6.0, 8.0]` (merged 6-class scheme; verify
  against the checkpoint's own confusion-matrix PNG before reusing this for
  a *different* checkpoint — 013 itself used an older 7-class scheme before
  the merge)

---

## Diagnostic-plot pipeline (calibration, confusion matrices, error structure)

Reproduces, for 014_2 on the Gaetan split, the 6-plot diagnostic pipeline
originally built for checkpoint 008_2 on the official split
(`NOTES_GT_MCES_RETRIEVAL.md` item 3e): `test_to_candidate_binned_box.png`,
`test_to_candidate_top1_diagnostics.png`, `test_to_test_binned_box.png`,
`confusion_matrix_simba_vs_cosine_top1.png`, `confusion_matrix_examples.png`,
`simba_mces_vs_cosine_similarity_by_cell_linear.png` — all six now land in
one shared folder, `experiments/014_2_gaetan_diagnostic_plots/`, rather than
split across two directories like the original.

### Why this needed new tooling, not just a checkpoint swap

All 6 plots ultimately depend on ground-truth MCES for the relevant
(test, candidate) pairs. The existing GT-MCES data
(`data/gt_mces_retrieval_candidates/`) only covers the **official split**,
built via `prepare_gt_mces_retrieval.py`'s asimov2-only exact-MCES pipeline
(sofia doesn't mount the packages needed to compute exact MCES itself, and
asimov2 doesn't mount `/sofia/projects` — a genuine two-machine dependency,
not something reproducible from this environment alone). No Gaetan-split
equivalent exists, and building one the same way isn't possible here.

Resolved without asimov2: `tools/oracle_retrieval_gt_mces.py` (the Oracle
GT-MCES-NN row) already proves a broader, pre-existing resource covers
every split's molecules — `max(data/massspecgym/lb_matrix.npy,
data/massspecgym/data/auxiliary/all_smiles_mces.hdf5)`, two all-vs-all
condensed MCES matrices covering the full MassSpecGym molecule universe (our
splits are just different partitions of that same space). New script
`tools/build_gt_mces_from_matrices.py` reuses that exact lookup mechanism
(same condensed-index formulas as the oracle script) to build a
`smiles.txt` + `mces_exact.npy` file for Gaetan test-to-candidate pairs, in
the same schema `load_gt_mces_lookup` already expects — a drop-in GT-MCES
source for a split that never went through the asimov2 pipeline.

**Correctness trap avoided**: `oracle_retrieval_gt_mces.py`'s own grid
lookup defaults every unresolved cell to `mces_cap` (40.0) before taking an
argmin — harmless there since a capped value can never win a minimum. The
new adapter does NOT do this: a pair unresolved by either matrix is left as
NaN (dropped downstream, matching `load_gt_mces_lookup`'s own "-1/NaN = no
reliable value" convention) rather than silently becoming a fake "GT=40".

### Known caveat: test-to-candidate GT-MCES coverage is only ~0.5%, not ~100%

`tools/build_gt_mces_from_matrices.py` resolved only **2,687 of 560,132**
possible Gaetan test-to-candidate pairs (0.48%) — `lb_matrix`/`hdf5` cover
MassSpecGym's own spectral-dataset molecules, not the broader PubChem-sourced
formula-matched decoy pool the retrieval candidates are drawn from (by
design — decoys are plausible-formula molecules from a large chemical
database, not restricted to compounds that happen to have their own mass
spec data). The official split's equivalent plots didn't hit this because
`prepare_gt_mces_retrieval.py`'s asimov2 pipeline computed exact MCES for
essentially every pair regardless (584,340/~585K, ~100%) — a cost inherent
to that two-machine pipeline, not something the matrix-lookup adapter can
match without asimov2 access.

This splits cleanly by plot:
- `test_to_test_binned_box.png` — **fully valid, no caveat**: 199,303,806/
  199,303,806 pairs resolved (100%), since test-to-test pairs are inherently
  MassSpecGym-vs-MassSpecGym, which the matrices cover completely.
- `test_to_candidate_binned_box.png` and `test_to_candidate_top1_diagnostics.png`
  — built on that 0.48% subsample: candidates that happen to *also* be real
  MassSpecGym molecules, not a representative cross-section of the actual
  decoy pool SIMBA is scored against at retrieval time. Read these as a
  rough, likely-biased signal, not a real calibration measurement — nowhere
  near equivalent in statistical power or representativeness to the
  official-split originals.
- The 3 confusion-matrix/comparison-table plots and `test_to_test_binned_box.png`
  are unaffected by this gap.

### Known caveat: Gaetan lacks the [10,20]-band exact refinement

The official split's own `test_to_test_binned_box.png` used an
*exact-refined* GT source (`apply_exact_mces_1020.py`, which replaces every
lower-bound value in `[10,20]` with an asimov2-solved exact value). No
Gaetan-split analog of that refinement step exists or can be built here.
`--test_to_test_prepro_dir` for Gaetan points at
`preprocessing_gaetan_split_max_lb_hdf5_v2/` directly (identical filenames/
schema, zero code changes needed), but its `[10,20]`-band GT values are the
looser lower bound, not the exact-solved value the 008_2/official plot had.
Not a bug — a real, if narrow, quality difference to keep in mind when
comparing the two checkpoints' `test_to_test_binned_box.png` side by side.

### Known limitation: regression-only, no CORN-bucket view for test-to-candidate

The 4 test-to-**candidate**-facing scripts (`mces_top1_diagnostics.py`,
`plot_retrieval_comparison_checks.py`, `plot_confusion_matrix_examples.py`,
`build_retrieval_comparison_table.py`) still have no CORN bucket-head
awareness — confirmed by direct reading, zero `mces_bucket`/
`corn_corrected`/`bucket_pred` references in any of them. Adding it would
mean a genuine pairwise bucket-head pass over every scored pair (hundreds
of thousands), not just each query's top-1 the way retrieval ranking
needs. Still follow-up work, not attempted here.

The test-to-**test** side, by contrast, now HAS a CORN-corrected view — see
"Does 014_2 generalize worse on test than val?" below, which computes both
raw-regression and CORN-corrected predictions for a molecule-level
test-to-test comparison (only ~3.7M pairs at that granularity, not hundreds
of thousands of times more — a genuine pairwise bucket-head pass is cheap
there).

### Small reworks made (all backward-compatible)

Three of the five scripts only accepted a single `--candidate_tsv`/
`--iceberg_preds` path; Gaetan's candidates are split across two files
(`candidates_gaetan_test_existing_overlap.tsv` + `candidates_gaetan_test_new.tsv`,
matched 1:1 with two `preds.hdf5` files) the way `simba_retrieval_iceberg.py`
already handled via `nargs="+"`. Patched the same way, reusing
`load_candidate_index`/`load_iceberg_spectra`'s existing list support rather
than duplicating merge logic:
- `tools/cosine_baseline_intermediates.py`
- `tools/build_retrieval_comparison_table.py`
- `tools/plot_confusion_matrix_examples.py`

`tools/build_retrieval_comparison_table.py` also had a genuine pre-existing
bug (predates this session): it never wrote `test_precursor_mz`/
`candidate_precursor_mz` columns, which `plot_retrieval_comparison_checks.py`
has always expected to read (crashed with a `usecols` mismatch on this run).
Fixed by extending `count_test_peaks_in_order` (renamed
`scan_test_mgf_in_order`) to also capture `PRECURSOR_MZ=` per test spectrum
in the same MGF scan, and `build_candidate_npeaks_lookup` to also return the
candidate's own theoretical precursor `m/z` from its TSV's `"precursor"`
column — both already had everything needed in hand, just weren't returning it.

See also the molecule-level test-to-test scoring fix below
(`ood_generalization_check.score_test_to_test_molecule_level`), which
`mces_calibration_plots.py`'s main `test_to_test_binned_box.png` now uses
by default.

### Pipeline: how to reproduce (014_2, Gaetan split)

```bash
sbatch tools/slurm/diagnostic_plots_014_2_gaetan_1gpu.slurm.sh
```
One SLURM job, five sequential stages (each depends on the previous, so run
as one job rather than independent ones):
1. `tools/simba_retrieval_iceberg.py --intermediates_dir ...` (regression
   only, no `--corn_corrected` — embeddings are saved before the ranking
   branch runs either way, so this also produces the plain-regression
   `retrieval_results.tsv` scripts below need).
2. `tools/cosine_baseline_intermediates.py` — the cosine-side counterpart
   (binned-spectrum sparse matrices), not yet built for Gaetan before this.
3. `tools/build_gt_mces_from_matrices.py` — the new GT-MCES adapter above.
   Loads `lb_matrix.npy` fully into RAM (~116GB, not memory-mapped — mmap'ing
   it turns ~80M scattered reads into a multi-hour job on this filesystem);
   this cluster assigns memory per CPU core automatically (no `--mem` flag
   allowed), so `--cpus-per-task=24` on the `zen4_h200` partition (~7.8GB/core)
   gives ~187GB headroom.
4. `tools/build_retrieval_comparison_table.py` — the per-pair CSV all
   plotting scripts read.
5. The 4 plotting scripts, all pointed at the same
   `experiments/014_2_gaetan_diagnostic_plots/` output directory.

**Cache warning**: `mces_calibration_plots.py` caches to
`<output_dir>/scored_pairs_cache.pkl`, keyed only on the output path (no
fingerprint of its inputs). Always use a fresh `--output_dir` for a new
checkpoint/split, or pass `--force_recompute` — reusing 008_2's output path
would silently replay its cached (stale) scored pairs.

---

## Does 014_2 generalize worse on test than val?

`test_to_test_binned_box.png`'s first version (exhaustive spectrum-vs-
spectrum, ~199M pairs) showed test-to-test self-pair calibration looking
much worse than validation's own reported numbers (median predicted MCES
3.8 vs val's 1.2 for true self-pairs). Investigated directly rather than
accepted at face value — two real methodological bugs in the *comparison*
accounted for nearly all of it, not a generalization problem in the model.

### Bug 1: wrong unit of comparison (exhaustive spectrum-pairs vs validation's molecule-pairs)

Validation pairs are **not** sampled per spectrum at all. Traced end to end
(`simba/workflows/training.py`'s `prepare_data`,
`tools/prepare_msg_gaetan_split_max_lb_hdf5.py`'s `build_pairs`,
`simba/core/data/datasets/multitask_dataset.py`'s `__getitem__`): validation
pairs are a **fixed, deterministic, exhaustive enumeration over unique
molecules** (not spectra) — C(N,2) mined cross-pairs (N≈2,731 lb_matrix-
matched val molecules) + exactly one synthetic self-pair per molecule,
injected by `_add_identity_pairs`. For any pair, position-0 is always that
molecule's **first** spectrum index, position-1 always its **last** —
deterministic, not resampled per epoch (`val_sampler = None` always,
confirmed in code and in a training script's own comment: "validation
always scores the full set"). The math checks out exactly: C(2731,2) =
3,727,815 + 2,731 = 3,730,546, matching the reported val pair count to the
pair.

An exhaustive spectrum-vs-spectrum test-to-test computation (what
`score_test_to_test_no_averaging` / the original `test_to_test_binned_box.png`
did) measures a genuinely different, unrelated thing — within-molecule
spectral-condition robustness (do a molecule's different spectra, at
different collision energies/instruments, agree with each other) — at a
wildly different pair count (~199M vs ~3.7M) and statistical character.
Comparing it against validation's numbers was never a fair comparison to
begin with, independent of the model's actual behavior.

**Fix**: new `ood_generalization_check.score_test_to_test_molecule_level`
replicates validation's exact protocol against the test fold's own mined
molecule-pairs file (`ed_mces_indexes_tani_incremental_test_node0_chunk0.npy`,
already the correct molecule-level format, no rebuild needed — it was just
never used this way before). `mces_calibration_plots.py`'s main
`test_to_test_binned_box.png` uses this now, plotted with validation's own
binning convention (`binned_box_on_ax_val_style`, dedicated self-pair box +
5-unit bins, ported from `callbacks.py::_plot_binned_box`) instead of the
unrelated bin_width=2 scheme used elsewhere in that script. The exhaustive
spectrum-level computation is kept for the mass-cutoff breakdown
(`binned_box_by_mass_cutoff.png`/`mae_spearman_by_mass_cutoff.png`), which
asks a legitimately different, exhaustive-appropriate question (does
restricting to a mass range improve calibration).

### Bug 2: val's own self-pair statistic mixes in trivial same-spectrum comparisons

Because `multitask_dataset.py` resolves a self-pair's two spectra via
`indexes[0]`/`indexes[-1]`, a validation molecule with only **one** spectrum
in that fold gets `indexes[0] == indexes[-1]` — i.e. its "self-pair" is
literally the same spectrum embedding dotted with itself (trivially
cosine_sim=1, pred_mces=0 exactly), not a genuine same-molecule-different-
spectrum comparison. `tools/benchmark_self_retrieval.py` independently
documents this: of val's 2,731 self-bucket molecules, only 2,010 have >=2
val spectra — the other 721 (26.4%) are trivial. Filtering
`val_pairs_val_consolidated.parquet`'s own `same_spectrum` column confirms
it numerically: trivial self-pairs (n=721) have median=mean=0.000 exactly;
genuine ones (n=2,010) have median=2.484, mean=3.925 — much closer to
test-to-test's numbers.

This isn't a bug to "fix" by excluding trivial pairs (the user's own
framing: some pairs being trivial is expected and fine, it's an honest
consequence of single-spectrum molecules existing in both folds) — it's
resolved automatically by fix 1 above, since the corrected test-side
computation uses the *same* `indexes[0]`/`indexes[-1]` rule and therefore
gets its *own* natural mix of trivial/genuine self-pairs (480/2,734 = 17.6%
trivial for test, vs val's 26.4%) rather than test having *zero* trivial
pairs (as the old exhaustive computation did, by explicitly excluding the
literal self-spectrum) while val had 26.4% — an apples-to-oranges mismatch
in the opposite direction from bug 1.

### Result after both fixes

With test scored using validation's exact protocol, matched almost exactly
in scale (val: 2,731 self-pairs / 3,730,546 total; test: 2,734 self-pairs /
3,738,745 total):

| | n | trivial% | median | mean |
|---|---:|---:|---:|---:|
| VAL self-pairs (all) | 2,731 | 26.4% | 1.20 | 2.89 |
| TEST-TO-TEST self-pairs (all) | 2,734 | 17.6% | 2.87 | 4.51 |

A real, modest residual gap remains (~2.4x on the self-pair statistic), and
the full calibration curve shows a similarly modest compression pattern
(test over-predicts slightly more at low GT, under-predicts slightly more
in the middle, converges again at the high end) — a believable, small
generalization gap, not the dramatic ~3x+ discrepancy the uncorrected
comparison suggested.

**CORN-correction was checked too, on both sides** (val's own
`pred_mces_bucket_step{N}` column; test via a fresh pairwise bucket-head
pass over the same ~3.7M molecule-level pairs — cheap at this granularity).
It collapses both self-pair *medians* to exactly 0 (the bucket head
correctly identifies bucket-0 for the majority of self-pairs on both
sides), but a residual *mean* gap persists (val corn mean=0.94, test corn
mean=1.49) — the classification head itself is somewhat less accurate at
identifying test self-pairs as bucket-0 than val self-pairs, a smaller
version of the same generalization gap, not something the correction erases.

**Verified consistent**: both sides use theoretical (not measured)
precursor mass — test explicitly via `load_spectra(mgf, "test", "theoretical")`,
val implicitly since 014_2's own training config sets
`sampling.precursor_mass_mode=theoretical` for both its train and val
dataloaders (`014_2_mces_bucket_mlp_1gpu.slurm.sh:81`). Not a source of the
discrepancy.

### Reproduce

```bash
sbatch tools/slurm/val_vs_test_to_test_corn_014_2_1gpu.slurm.sh
```
Runs `tools/plot_val_vs_test_to_test_binned_box.py`: loads val's raw
per-pair columns straight from `val_pairs_val_consolidated.parquet`
(`gt_mces`, `pred_mces_step{N}`, `pred_mces_bucket_step{N}`, `is_self_pair`,
`same_spectrum`), scores test-to-test via
`ood_generalization_check.score_test_to_test_molecule_level` (regression)
plus a local CORN-correction wrapper (`score_test_to_test_molecule_level_with_corn`,
same file), and saves a 2x2 grid (`val_vs_test_to_test_binned_box_with_corn.png`
in `experiments/014_2_gaetan_diagnostic_plots/`) — rows raw regression /
CORN-corrected, columns val / test-to-test, all four panels using identical
binning/styling for direct visual comparison.
