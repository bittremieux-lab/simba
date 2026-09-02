# Branch overview: `exp/mces-pipeline`

What this branch adds on top of `dev`, and how to use it. Written as a single
entry point, self-contained in this repo: the day-to-day working notes it's
distilled from live under [`notes/`](notes/) (see "Further reading" at the
bottom) rather than being merged into this file, since they're detailed logs
of *how* each result was reached, with dead-ends and intermediate numbers a
summary would just discard.

`dev` at the point this branch forked (`ed28c05`) already has SIMBA's core:
a transformer encoder over MS/MS spectra, trained to predict edit distance
and MCES between molecule pairs. Everything below was built on top of that
across ~35 commits, essentially the project's entire practical
experimentation program: new training data/splits, new model heads, a whole
retrieval-evaluation layer that didn't exist at all on `dev`, a reproduction
of the paper's analog-discovery evaluation, and several side experiments
branching off the current best checkpoint.

## 1. What's new here that isn't on `dev`

### Data / splits
- **Gaetan's own MassSpecGym split** (as opposed to `dev`'s official split
  only), with a real bug fixed along the way: the original HDF5 MCES lookup
  used the wrong triangular-index convention, silently returning wrong
  values (e.g. true MCES=1 read back as 45) — affected the older official-
  split data too, both were rebuilt (`tools/prepare_msg_gaetan_split_max_lb_hdf5.py`,
  `tools/prepare_msg_official_split_max_lb_hdf5.py`).
- A v2 of the Gaetan split (`preprocessing_gaetan_split_max_lb_hdf5_v2`),
  the one every experiment from `009` onward trains on.
- Optional **theoretical precursor mass** (recomputed from SMILES+adduct
  instead of read straight off the MGF) with MIST-CF/BUDDY-style training
  noise on top (`sampling.precursor_mass_mode`, `sampling.precursor_noise_mode`
  in `simba/configs/training/default.yaml`) — every checkpoint from `010`
  onward trains with `precursor_mass_mode=theoretical`; `009` and earlier
  stayed on the old default (`measured`). This distinction matters for
  scoring: retrieval/inference must be run with whichever mode a given
  checkpoint was actually trained with, or its precursor-mass input is
  silently wrong.
- **Identity pairs** (`sampling.add_identity_pairs`) and **MCES-bucket
  weighted resampling** (`sampling.use_resampling`) to counter the training
  distribution's natural skew.
- **Opt-in training-molecule exclusion** (`sampling.train_exclude_smiles_file`)
  — drop specific molecules from training by SMILES list, used for the
  CASMI-distance sweep below.
- **ICEBERG-generated synthetic training spectra** (`sampling.iceberg_mgf_path`,
  `sampling.iceberg_spectra_prob`) — see §4.

### Model
- **Configurable similarity head** (`model.tasks.cosine_similarity.head_mode`):
  `cosine_relu` (the `dev` default) vs `cosine_no_head`, `cosine_linear_head`,
  `distance_linear_head`, `distance_no_head`. Systematically ablated
  (experiments `008_1`-`008_9`) — **`cosine_no_head` won outright** (best
  validation metrics *and* best retrieval, with no projection layers at
  all) and is what every experiment from `009` onward uses.
- **Optional CORN-ordinal MCES-bucket head** (`model.tasks.mces_bucket`,
  fully absent from `dev`) — a second, auxiliary head that classifies each
  pair into a coarse MCES bucket (`bin_edges`, default `[2,4,6,8]`) instead
  of only regressing the continuous value. Its prediction can *correct* the
  primary regression head's output at inference time (clip the continuous
  prediction into whatever bucket the classifier picked) — this
  "CORN-corrected" score consistently beats the raw regression score on
  every retrieval benchmark below.
- **Log-transformed MCES loss** (`model.tasks.mces.use_log_loss` +
  `log_loss_a`, the latter new) — train toward `log(raw_mces + c)` instead
  of plain MSE, where `log_loss_a` sets the implied pseudocount
  (`c = 40 / log_loss_a`; default `5`→`c=8`, `40`→`c=1`). See §4.
- **Bigger-architecture support fix**: `model.transformer.d_model`/`n_layers`
  were already config-driven for *training*, but every retrieval/inference
  tool had them hardcoded to `256`/`5` — since the model never calls
  `save_hyperparameters()`, a bigger checkpoint loaded through the old code
  path would have silently dropped every shape-mismatched weight
  (`strict=False`) instead of erroring, producing a near-random model with
  no warning. Fixed across all three retrieval tools; see §4.

### Retrieval evaluation (didn't exist on `dev` at all)
Three independent ways to score a checkpoint on retrieval, all under
`experiments/retrieval_split_comparison/gaetan_test/`:
- **NN-transfer** (`tools/simba_retrieval.py`) — embed train+test, find each
  test spectrum's nearest training spectrum by cosine similarity, transfer
  its Morgan fingerprint, rank the candidate pool by Tanimoto similarity.
  The original/simplest retrieval method.
- **ICEBERG-based** (`tools/simba_retrieval_iceberg.py`) — predict an
  in-silico spectrum for every retrieval candidate with
  [ICEBERG](https://github.com/coleygroup/ms-pred) (pretrained MassSpecGym
  weights, a separate sub-project with its own venv at `ICEBERG/`), then
  rank candidates by SIMBA similarity between the real test spectrum and
  each candidate's predicted spectrum. Substantially stronger than
  NN-transfer across every checkpoint tested (e.g. `014_2`: 10.0%→19.5%
  hit@1). Supports `--corn_corrected` to use the bucket-head-corrected
  score instead of raw regression.
- **Cosine baselines** (`tools/cosine_retrieval.py`,
  `tools/cosine_baseline_iceberg.py`) — the same two methods but with plain
  binned-spectrum cosine similarity instead of any SIMBA checkpoint, to
  quantify how much the model is actually adding.
- **Ground-truth-MCES oracle** (`tools/prepare_gt_mces_retrieval.py` +
  retrieval-time `--gt_mces_dir`) — ranks candidates by their *real* MCES to
  the query instead of any predicted score; an upper bound on the
  NN-transfer paradigm specifically, checkpoint-independent.

All of it lands in a shared `retrieval_results.tsv` schema (`split, model,
[head_mode,] [corn_corrected,] [precursor_mass_mode,] [peak_filter,] n,
hit@1, hit@5, hit@20`) so results are directly comparable across
checkpoints and methods.

### Analog-discovery (CASMI) pipeline
A simplified reproduction of the SIMBA paper's Figure 2 evaluation, inference
only, no retraining (`tools/analog_discovery_embed_rank.py` and friends;
full detail in `NOTES_014_2_ANALOG_DISCOVERY.md`): SIMBA vs plain cosine,
raw (non-size-normalized) MCES, two independent reference-library searches
(NIST20+MassSpecGym, and GNPS-no-propagated), boxplot/ROC/ranking panels.

Used this pipeline to test **Wout's hypothesis** that SIMBA's edge over
cosine on analog discovery might come from training-data proximity to the
CASMI queries rather than genuine skill — a CASMI-distance-exclusion sweep,
budget-matched (fixed training-set size of 7,564 molecules throughout, only
the minimum distance from CASMI varies) to rule out a data-quantity
confound. **Result: confirmed.** ROC AUC degrades smoothly and monotonically
as the excluded distance threshold grows (0.955 unrestricted → 0.841 at
threshold 14), even with training-set size held exactly fixed — distance
from the query distribution is a real, independent driver of SIMBA's
advantage, not an artifact of how much data it happened to have.

### Monitoring dashboard
A local Streamlit app, `tools/dashboard_app.py` (`uv run streamlit run
tools/dashboard_app.py`) — browse any experiment's training curves, val-pair
diagnostics (MAE/overlap by MCES bin, confusion matrices for the bucket
head), and a cross-experiment "Compare runs" table (loss/MAE/overlap/Hit@k,
with an optional cosine-baseline reference row and CORN-corrected variant
rows). Introduced early (`e4680e3`) and extended throughout the branch
alongside whatever it needed to visualize.

## 2. Data

None of this is git-tracked (large binary domain data) — it lives under
`../data/` (one level above this repo) unless noted otherwise. Paths below
are relative to `/sofia/projects/2026_053/simba_project/`.

### Raw spectral libraries
- **MassSpecGym** (`data/massspecgym/data/auxiliary/MassSpecGym.mgf`, ~309MB)
  — the main training corpus for every experiment on this branch. Public
  benchmark dataset of real MS/MS spectra with known structures.
- **NIST20** (`data/nist20/nist20.mgf`, ~572MB, user-provided — not
  downloadable, licensed) — 681,708 spectra / ~17,800-17,990 unique
  molecules (SMILES-dedup / InChIKey-dedup respectively). Used only as a
  reference library for the analog-discovery pipeline's Search A (see "Analog-discovery (CASMI) pipeline" in §1),
  never for training.
- **GNPS, no propagated annotations**
  (`data/gnps/ALL_GNPS_NO_PROPOGATED.mgf`, ~2.2GB, downloaded from
  `external.gnps2.org`'s bulk-library page — the `AGGREGATION`-type export
  that excludes auto-propagated/molecular-networking-inferred annotations,
  keeping only curated + imported libraries) — 956,358 spectra. Reference
  library for analog discovery's independent Search B.
- **CASMI 2022** (`simba/data/casmi2022.mgf`, in-repo, ~349KB) — 169 unique
  query molecules, the actual analog-discovery evaluation set (not used in
  training at all).

### Splits
- **Official MassSpecGym split** — the benchmark's own train/val/test
  division, scaffold-resplit 90/10 for an extra validation fold
  (`tools/prepare_msg_official_split_max_lb_hdf5.py`). Used by `dev` and by
  this branch's early experiments (`004`, `006`, `008_x`).
- **Gaetan's own split** — an alternative split from
  [`spectrawl/splits/split_massspecgym.tsv`](https://github.com/bittremieux-lab/spectrawl/blob/main/spectrawl/splits/split_massspecgym.tsv),
  used from experiment `005` onward. **v1** (`preprocessing_gaetan_split_max_lb_hdf5`)
  had the triangular-index bug described in §1; **v2**
  (`preprocessing_gaetan_split_max_lb_hdf5_v2`) is the fixed version every
  experiment from `009` onward actually trains on: 24,010 train / 2,800 val
  / 2,734 test **molecules** (each with 1 or more real spectra — training
  samples one at random per molecule per epoch, see
  `CustomDatasetMultitasking.__getitem__`). The corresponding real test-fold
  spectra, extracted once as a standalone MGF for retrieval evaluation, live
  at `experiments/retrieval_split_comparison/gaetan_test.mgf` (14,118
  spectra) — every ICEBERG/NN-transfer retrieval run from `010` onward
  scores against this same file.

### Ground-truth MCES
Three tiers, cheapest/loosest to most expensive/exact, combined as
`max(lower_bound, exact_where_available)`:
1. **Lower-bound matrix** (`lb_matrix.npy` + its own SMILES index) — fast,
   approximate, always available.
2. **Exact MCES for close pairs** (an HDF5 keyed by a *condensed
   distance-matrix* index convention — the source of the v1 split bug,
   which reused a different indexing convention by mistake) — exact
   wherever the lower bound was already < 10.
3. **Exact-refined MCES for the [10,20] band** — recomputed with the same
   `myopic_mces` ILP solver used everywhere else in this project
   (threshold=20, values above that are lower bounds, not exact).

Each `preprocessing_*` directory's `mapping.pkl` bakes the final combined
value into `pair_distances` (the `(molecule_i, molecule_j, distance)` array
consumed by `simba/core/data/molecule_pairs.py::MoleculePairsOpt`) —
training never recomputes MCES itself, it's all precomputed at preprocessing
time.

### Retrieval candidates
- **Candidate pool** (`spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json`,
  shared across `spectrawl` and this project) — for each test/query
  molecule, a formula-matched pool of decoy candidates (~250-450 per query,
  510,581 unique candidate molecules for the Gaetan test fold alone). Every
  retrieval benchmark in §1 ranks within this same pool.
- **ICEBERG-predicted candidate spectra** (`ICEBERG/results/candidates_*/preds.hdf5`,
  one directory per candidate batch — `candidates_test_official`,
  `candidates_gaetan_test_new`, `candidates_gaetan_test_existing_overlap`)
  — since real reference spectra don't exist for most of the ~500K+
  candidate molecules, ICEBERG (a separate in-silico fragmentation
  predictor, its own sub-project/venv at `ICEBERG/`) generates one
  synthetic spectrum per candidate so the ICEBERG-based retrieval methods
  in §1 have something to rank against.
- **Ground-truth MCES for candidates** (`data/gt_mces_retrieval_candidates/`
  — `smiles.txt` + `mces_exact.npy`, 584,340 (test-molecule, candidate)
  pairs, mean MCES 18.71) — real MCES between each query and its own
  candidate pool, computed the same ILP way as above. Powers the
  checkpoint-independent ground-truth-MCES oracle row in every retrieval
  table (an upper bound on the NN-transfer paradigm).

### Synthetic training data
**ICEBERG train-spectra augmentation** (§4's `iceberg-aug` side experiment)
generates its own synthetic spectra for *training* molecules, not
candidates: `data/analog_discovery/iceberg_train_augmentation/synthetic_train.mgf`,
120,050 spectra (24,010 train molecules × 5 collision energies:
15/25/35/45/55 eV), consumed only when `sampling.iceberg_mgf_path` is set —
every other experiment on this branch trains on real spectra exclusively.

## 3. Experiment lineage (chronological, key findings only)

| # | What changed | Headline result |
|---|---|---|
| 001-004 | Base MCES-only training pipeline, DDP scaling, bf16 | got training working at scale |
| 005 | First Gaetan-split run (`head_mode=cosine_relu`, buggy MCES fixed) | beat the official-split equivalent on retrieval, but with data-leakage caveat (splits don't align) |
| 006 | Official split, fixed MCES, 4x H200 DDP | direct bugfix-only comparison point for 005 |
| 008_1-9 | Head-mode ablation (`cosine_relu`/`cosine_no_head`/`cosine_linear_head`/`distance_*`/CORN) | **`cosine_no_head` wins outright** — best val metrics *and* retrieval, adopted from `009` on |
| 009 | Gaetan split v2, `cosine_no_head`, no resampling, `precursor_mass_mode=measured` | new baseline, supersedes 005 |
| 010 | + `precursor_mass_mode=theoretical` + MIST-CF/BUDDY noise | ICEBERG-retrieval hit@1 13.0%→16.0% |
| 011 | + weighted resampling | →17.3% |
| 012 | + MCES-bucket-weighted resampling refinement | →18.0% |
| 013 | + auxiliary `mces_bucket` head (CORN, `use_mlp=false`) | raw hit@1 dips slightly (17.0%) — the auxiliary head's benefit only shows up once you actually use it to correct the score (CORN-corrected), not on the plain regression output |
| **014_2** | `mces_bucket` head with `use_mlp=true` (current best/reference checkpoint) | ICEBERG+SIMBA raw 19.5% hit@1, **CORN-corrected 21.0%** |

`014_2` is the checkpoint every side experiment below branches off of.

## 4. Side experiments off `014_2`

All three follow the same shape: fork `014_2`'s exact training config,
change one thing, compare via the same ICEBERG-retrieval + dashboard-table
procedure.

- **Log-loss** (`NOTES_014_2_LOGLOSS.md`) — `use_log_loss=true` at two
  strengths (`log_loss_a=5` → pseudocount 8, `log_loss_a=40` → pseudocount
  1, Gaetan's exact proposal). Mixed result: `a40` improves bucket balanced
  accuracy and CORN-corrected Hit@5/20, but is *worse* than baseline at
  Hit@1 both raw and CORN-corrected — not a clean win.
- **Bigger architecture** (`NOTES_014_2_BIGMODEL.md`) — `d_model=384,
  n_layers=8` (12.5M params vs 4.6M). Hit the 24h SLURM wall-clock timeout
  at step 131,000 (014_2 itself reached 229,000), so this comparison isn't
  apples-to-apples on training maturity yet — treat its numbers as a lower
  bound, not a verdict on the architecture change.
- **ICEBERG train-spectra augmentation** (`NOTES_014_2_ICEBERG_AUGMENTATION.md`)
  — generate 5 synthetic spectra per training molecule at different
  collision energies (15/25/35/45/55 eV) with ICEBERG, inject them into
  training with 50% probability of being drawn instead of a real spectrum.
  Reached the same training maturity as `014_2` (step 229,000, also a 24h
  timeout) — a fair comparison. **Best result of the three side
  experiments**: ICEBERG+SIMBA raw hit@1 21.2% (vs 19.5% baseline), CORN
  63.5%→**67.8%** at hit@20.

None of the three has been run through the CASMI analog-discovery pipeline
yet — only through ICEBERG retrieval and the dashboard comparison table.

## 5. How to use this branch

### Train a checkpoint
Every training run goes through `simba train` (Hydra config + CLI
overrides), launched via a SLURM script under `tools/slurm/`. To reproduce
or fork the current reference checkpoint, copy
`tools/slurm/014_2_mces_bucket_mlp_1gpu.slurm.sh` and adjust the
`uv run simba train ...` overrides — it already sets every flag described
in §1 (`head_mode=cosine_no_head`, `precursor_mass_mode=theoretical`,
`mces_bucket.enabled=true` + `use_mlp=true`, Gaetan-split-v2 preprocessing
dir). All config defaults live in `simba/configs/model/simba_default.yaml`
(model/task flags) and `simba/configs/training/default.yaml` (sampling
flags) — every override above is just a dotted-path CLI arg on top of
those.

### Run retrieval on a checkpoint
```bash
uv run python tools/simba_retrieval_iceberg.py \
    --checkpoint <path/to/checkpoint.ckpt> \
    --head_mode cosine_no_head \
    --mces_bucket_use_mlp \                 # only if the checkpoint has one, matching its own training config
    --precursor_mass_mode theoretical \     # match whatever the checkpoint was trained with
    --min_peaks 6 \
    --mgf experiments/retrieval_split_comparison/gaetan_test.mgf \
    --candidates <MassSpecGym_retrieval_candidates_formula.json> \
    --candidate_tsv <candidates.tsv...> \
    --iceberg_preds <preds.hdf5...> \
    --split test --batch_size 512 --skip_mces \
    --output_tsv <output.tsv>
```
Add `--corn_corrected` for the bucket-head-corrected score (only meaningful
if the checkpoint actually has `mces_bucket.enabled=true`). If the checkpoint
uses a non-default architecture, also pass `--d_model`/`--n_layers` — the
checkpoint doesn't record its own size, so getting this wrong silently
produces a mostly-random model, not an error (see §1). `tools/simba_retrieval.py`
(NN-transfer) and `tools/analog_discovery_embed_rank.py` (CASMI) take the
same flags for the same reasons.

### View results
```bash
uv run streamlit run tools/dashboard_app.py
```
Browse per-experiment training curves and diagnostics, or use the "Compare
runs" tab for a cross-experiment table (loss/MAE/overlap/Hit@k, optional
cosine baseline row, optional CORN-corrected row per bucket-head run).

## 6. Known caveats / open threads

- **Checkpoint-selection isn't consistent across the whole lineage.** `005`'s
  retrieval numbers use an early checkpoint (step 7,000) deliberately picked
  for peak `val_mces_spearman` at the time (that metric declines after);
  `009` through `014_2`-family experiments all just use the last checkpoint
  each run reached. Nobody has checked whether `009`/`014_2`'s *final*
  checkpoints are actually their retrieval-best point, or whether — like
  `005` — an earlier one would score higher.
- `bigmodel` and `iceberg-aug` both hit the 24h SLURM wall-clock timeout
  rather than finishing all planned epochs; `bigmodel` in particular stopped
  well short of `014_2`'s own step count, so its comparison isn't fully
  fair yet.
- Log-loss (`a40`) and bigger-architecture results are each based on one run
  — no seeds/repeats, so the sign of small deltas (e.g. log-loss's Hit@1
  regression) isn't independently confirmed.
- Contrastive/triplet-loss objective (next up, per `NEXT_STEPS.md` item 6)
  isn't started on this branch — would need the pairwise `MoleculePairsOpt`
  / `CustomDatasetMultitasking.__getitem__` data model (currently strictly
  1-vs-1 per sample) reworked to 1-vs-N.

## Further reading

Detailed working notes this overview is distilled from, all under
[`notes/`](notes/) in this repo:

- [`notes/PROGRESS_REPORT_PREPROCESSING_AND_GAETAN_SPLIT.md`](notes/PROGRESS_REPORT_PREPROCESSING_AND_GAETAN_SPLIT.md) — the MCES-lookup bug, Gaetan split v1
- [`notes/PROGRESS_REPORT_ROADMAP.md`](notes/PROGRESS_REPORT_ROADMAP.md) — items 1-5 (head-mode ablation, ICEBERG retrieval intro)
- [`notes/NEXT_STEPS.md`](notes/NEXT_STEPS.md) — the running task list this work has been checking off
- [`notes/NOTES_GT_MCES_RETRIEVAL.md`](notes/NOTES_GT_MCES_RETRIEVAL.md) — ground-truth-MCES oracle scoring (3c/3d)
- [`notes/NOTES_RETRIEVAL_SPLIT_COMPARISON.md`](notes/NOTES_RETRIEVAL_SPLIT_COMPARISON.md) — 005-era cross-split retrieval sanity check
- [`notes/NOTES_RETRIEVAL_014_2_CORN.md`](notes/NOTES_RETRIEVAL_014_2_CORN.md) — `014_2`'s CORN-corrected retrieval results
- [`notes/NOTES_014_2_ANALOG_DISCOVERY.md`](notes/NOTES_014_2_ANALOG_DISCOVERY.md) — CASMI pipeline + Wout's confound investigation
- [`notes/NOTES_014_2_LOGLOSS.md`](notes/NOTES_014_2_LOGLOSS.md),
  [`notes/NOTES_014_2_BIGMODEL.md`](notes/NOTES_014_2_BIGMODEL.md),
  [`notes/NOTES_014_2_ICEBERG_AUGMENTATION.md`](notes/NOTES_014_2_ICEBERG_AUGMENTATION.md) — the three `014_2` side experiments

Two older in-repo docs (at repo root, not under `notes/`) cover earlier
snapshots of specific pieces in more implementation detail and are still
accurate for what they describe, just not updated past their own dates:
`PIPELINE.md` (preprocessing → training → validation → retrieval, step by
step) and `BASELINE_AND_DASHBOARD.md` (the `009` baseline and the dashboard,
as of 2026-08-22).
