# tools/ — new files log

Running log of new files added to `tools/`, one section per batch, so a commit's
intent is easy to find later without digging through the diff. See
`../../NOTES_GT_MCES_RETRIEVAL.md` for the full writeup behind any of these.

## OOD generalization check (NEXT_STEPS.md item 3e)

- `ood_generalization_check.py` — core library + entry point. Shared scoring helpers
  (embedding-vs-GT scoring for both populations, self-pair inclusion) used by all
  three scripts below, plus its own MAE/Spearman summary comparing test-to-test vs.
  test-to-candidate. No embedding averaging anywhere: test-to-test scores every
  individual test spectrum against every other individual test spectrum directly
  (one dense matrix, no per-molecule embedding at all); test-to-candidate matches
  each candidate to the query's own adduct specifically (a candidate's ICEBERG
  embeddings under different adducts are kept as separate, never blended).
- `load_test_to_test_gt_mces.py` — loads the already-existing exact-refined test-fold
  GT MCES (official-split preprocessing) into a `{(smi_a, smi_b): mces}` lookup.
- `mces_pool_distribution_plots.py` — per-spectrum min/mean/max distribution plots
  (GT / SIMBA-predicted / |diff|) for both populations.
- `mces_calibration_plots.py` — GT-binned SIMBA-predicted-MCES boxplots for both
  populations — where the saturation/miscalibration is clearest.
- `mces_top1_diagnostics.py` — true-candidate prediction and wrong-top-1 error
  diagnostics for test-to-candidate retrieval (item 3c), plus their joint distribution.
  Ties for the max similarity are broken with a fair random pick among tied
  indices, not `argmax`'s first-index default (self sits at index 0 of the
  per-query array) — this barely matters for SIMBA's continuous dense
  embeddings (float ties are vanishingly rare) but was left in for
  correctness.
- `slurm/ood_generalization_check.slurm.sh` — runs the MAE/Spearman summary
  (CPU-only; uses a GPU node purely for fast, uncontended storage I/O).
- `slurm/mces_plots_all.slurm.sh` — runs all three plotting scripts sequentially
  (same CPU-only-on-a-GPU-node rationale).

## Cosine-similarity pool distribution plots (8a)

The question: `test_to_candidate_gt.png`'s "min" (GT MCES to pool) is very close
to 0 — top-1/top-2 candidates are often nearly-indistinguishable structures even
by ground truth. Does the same hold for a plain, no-SIMBA cosine-similarity
baseline — does its "max" (over the pool) cluster close to 1 too? Pure
distribution-of-raw-similarity-values question, no MCES-unit conversion, no
ranking, no ties — see `cosine_similarity_pool_distribution_plots.py`'s
module docstring for why that conversion doesn't apply here (it's only
meaningful for SIMBA's *trained* cosine_no_head head).

- `cosine_baseline_intermediates.py` — precomputes and saves the binned-spectrum
  sparse matrices (`test_mat.npz`/`candidate_mat.npz`) + SMILES/adduct JSON, in the
  same directory layout as `simba_retrieval_iceberg.py`'s `--intermediates_dir`, so
  the plotting script below doesn't have to repeat binning ~600k candidate spectra.
- `slurm/cosine_baseline_intermediates.slurm.sh` — runs the precompute step above.
- `cosine_similarity_pool_distribution_plots.py` — the plots themselves: min/mean/max
  of raw cosine similarity over each query's pool (`test_to_candidate_cosine_similarity.png` /
  `test_to_test_cosine_similarity.png`, nothing excluded except test-to-test's literal
  self-spectrum, which is trivially similarity=1.0 by construction), plus a top-1-vs-
  top-2 view for test-to-candidate (`test_to_candidate_cosine_similarity_top1_top2.png`).
- `slurm/cosine_similarity_pool_distribution_plots.slurm.sh` — runs the plotting
  script above.

## Retrieval comparison table + checks (8b)

A single reusable table, plus every SIMBA-vs-cosine comparison built on top of it.

- `build_retrieval_comparison_table.py` — the "ultimate" per-(test spectrum,
  candidate) table: one row per full-pool pair with SIMBA/cosine rank, similarity,
  MCES (`mces_max_value * (1 - similarity)`), GT MCES, `is_correct`, and raw peak
  counts (`n_peaks_test`/`n_peaks_candidate`). Directly verifies (not assumes)
  SIMBA/cosine embedding availability agreement and GT MCES coverage; every later
  script here reads only this CSV, never re-scores anything.
- `plot_confusion_matrix_examples.py` — random examples per SIMBA-top1 x cosine-top1
  confusion-matrix cell, real test spectrum vs the true candidate's ICEBERG spectrum
  as mirror plots, each shown both raw and after SIMBA's own preprocessing
  (`remove_precursor_peak` -> `filter_intensity` -> sqrt-compress -> L2-normalize,
  matching `simba_retrieval.py`'s `spectra_to_tensors` line for line).
- `plot_retrieval_comparison_checks.py` — three checks, all from the table alone:
  (1) confusion matrix + hit@k double-check against the already-committed
  `retrieval_results.tsv` numbers; (2) hit@1 rate vs. real-spectrum peak count
  (`n_peaks_candidate` is reported but not plotted — it's degenerate, always
  exactly 100, since ICEBERG was run with `--sparse-k 100 --threshold 0.0`) plus a
  check of how often cosine's top-1 "win" is an arbitrary tie among a pool that's
  entirely flat at ~0 similarity; (3) a boxplot of precursor-mass discrepancy
  (measured vs. formula-implied calculated m/z) across the 4 confusion-matrix
  cells — can't explain SIMBA picking the wrong candidate within a pool (every
  candidate in a formula-matched pool shares the same calculated precursor,
  checked directly), but shows a mild overall-difficulty signal.

## Mass-restricted MCES calibration (8c)

Extends `mces_calibration_plots.py` (item 3e, no new file) with a mass-restricted
view of the same two calibration plots — does calibration change when restricted
to a mass range closer to what training actually saw?

- Re-filters the SAME already-scored (pred, GT) pairs (no re-scoring, no
  re-embedding) by `max(mass_query, mass_other) < cutoff` (RDKit `ExactMolWt`
  on already-canonical SMILES), for `--mass_cutoffs` (default
  300/350/400/450/500/750/1000 Da) plus "no limit". Produces
  `binned_box_by_mass_cutoff.png` (2 columns x len(cutoffs)+1 rows, same
  boxplot style as the standalone plots) and `mae_spearman_by_mass_cutoff.png`
  (MAE and Spearman vs. cutoff, one line per population).
- Caches the expensive, cutoff-independent part (candidate-SMILES
  canonicalization + the dense test-to-test matmul) to
  `scored_pairs_cache.pkl` in the output dir, so re-running with different
  cutoffs skips straight to the cheap filtering/plotting step
  (`--force_recompute` to bypass).
- `mae_spearman_by_mass_cutoff.png` also overlays a GT-MCES-balanced version
  of both metrics (dashed): every non-empty GT bin (0-5, 5-10, ..., up to
  `gt_clip_max`) is resampled to exactly `--gt_balance_target_n` pairs
  (default 10,000) — oversampled with replacement if a bin has fewer — so
  the metric isn't dominated by whichever GT range has the most pairs
  (typically near-0). Revealed that raw test-to-test's "calibration keeps
  improving with no mass cap" trend reverses once GT-balanced, and that raw
  test-to-candidate's Spearman is understated at every cutoff vs. its
  GT-balanced value. The run log reports each bin's thinnest available count
  and the oversampling factor needed, since some cutoffs (especially for
  test-to-candidate) have very few pairs in the sparsest GT bin.

## Mass1 x mass2 heatmaps (8d)

- `plot_mass_heatmaps.py` — one combined figure, 2 rows (test-to-test,
  test-to-candidate) x 4 columns (MAE, signed bias, mean GT MCES, per-cell
  Spearman), each its own heatmap with its own colorbar. Test-to-test axes
  are min/max(mass_a, mass_b) (pairs are unordered, so this avoids a
  redundant symmetric square); test-to-candidate axes are query/candidate
  mass, which coincide almost exactly since candidates are formula-matched
  — confirmed directly, the heatmap concentrates on the diagonal. Fixed
  100 Da bins, every occupied cell (n >= `--min_n`, default 1,000)
  annotated in-cell with its value and a compact sample count (`235M`,
  `79K`). Reuses `mces_calibration_plots.py`'s `scored_pairs_cache.pkl` (no
  re-scoring/re-embedding); per-cell Spearman is computed via a single
  sort-and-split rather than looping a boolean mask per cell, since that
  would be O(n_cells * n) instead of O(n log n) at test-to-test's ~280M
  pairs.
- Found that GT MCES saturates at the clip ceiling for size-mismatched
  test-to-test pairs, which makes their MAE deceptively low (both GT and
  prediction sit near the ceiling) — the genuinely hard region is
  size-matched pairs, especially matched-and-large (underestimated by
  ~13 MCES). For test-to-candidate, bias flips from negative (small mass)
  to strongly positive (+10 to +12, large mass), and per-cell Spearman goes
  to ~0/negative for the two largest mass bins (n=56K and 4K, not a
  small-sample artifact) — SIMBA's retrieval ranking specifically degrades
  for the largest-mass molecules in that population.

## Retrieval split comparison — Gaetan's sanity check

Gaetan's question (Slack): is SIMBA's retrieval weakness vs. cosine specific to the
official MassSpecGym split, or does it hold up more generally? Filled in a 4x3 table
(4 retrieval methods x 3 splits) to check. See
`../../NOTES_RETRIEVAL_SPLIT_COMPARISON.md` for the full plan/journal.

- `cosine_retrieval.py` — cosine-embedding counterpart of `simba_retrieval.py`'s
  train-NN-transfer pipeline (find each test spectrum's nearest *training* spectrum,
  transfer its fingerprint, rank candidates by Tanimoto) — same pipeline, only the
  embedding step changes from a trained SIMBA checkpoint to plain binned-spectrum
  cosine similarity. Needed because no cosine counterpart of this particular
  retrieval method existed yet.
- `extract_spectra_by_mgf_index.py` — pulls a split's spectra out of the raw MGF by
  global index (scaffold-val and Gaetan's own split aren't raw MGF folds, only
  `mapping.pkl` records which spectra belong to them) into a standalone mini-MGF,
  so the existing retrieval scripts can be pointed at it unchanged. Self-verifies
  every extracted group against `mapping.pkl` before declaring success, since a
  silent index mistake here would silently corrupt everything downstream.
- `simba_retrieval_iceberg.py` / `cosine_baseline_iceberg.py` — generalized to accept
  a list of `--candidate_tsv`/`--iceberg_preds` paths (not just one), so new splits'
  candidate predictions can be combined with the existing 600K-candidate official-split
  predictions without physically merging files or re-embedding what's already computed.

Result: SIMBA loses to cosine in every cell, on every split — official, Gaetan's own
split, and the scaffold-held-out split all show the same gap, so it isn't an
artifact of the official split specifically.

## Retrieval error deep-dive: oracle upper bound, error structure, precursor mass

Continuation of Gaetan's sanity-check table plus a deeper look at *why* SIMBA loses
to cosine specifically on the official split (Wout's questions on Slack). See
`../../NOTES_RETRIEVAL_SPLIT_COMPARISON.md` for the oracle row's journal entry.

- `oracle_retrieval_gt_mces.py` — 6th table row: oracle GT-MCES NN-transfer, an
  upper bound on the train-NN-transfer rows. Same pipeline as
  `simba_retrieval.py`/`cosine_retrieval.py`, except the nearest-training-neighbor
  is picked by TRUE structural distance (`max(lb_matrix, all_smiles_mces.hdf5)`,
  both pre-existing all-vs-all matrices, no new MCES computation) instead of
  embedding similarity. Fingerprinting is deliberately reordered to run *after* the
  oracle lookup (unlike the SIMBA/cosine rows, the oracle pick doesn't depend on
  fingerprints at all) so only train molecules actually selected as a nearest
  neighbor get fingerprinted, not the whole pool. Needs the whole `lb_matrix.npy`
  (116 GB) loaded fully into RAM, not memory-mapped — mmap turned ~35M scattered
  lookups into a 2+ hour stall; a full sequential load fixed it (~4-7 min per split).
  Result: oracle roughly doubles both real NN-transfer methods on every split, still
  short of ICEBERG+Cosine — and its own diagnostic shows the official split has no
  training analog closer than GT MCES=10 for any test molecule, vs. 1-3 for the
  other two splits, explaining a chunk of the official-vs-other-splits gap directly.
- `hit1_by_mass_cutoff.py` — hit@1 (not MAE/Spearman) vs. mass cutoff, reusing item
  8b's `retrieval_comparison_table.csv` directly. Filters by the query molecule's
  own mass (a per-spectrum metric, unlike 8c's pair-level `max(mass_query,
  mass_other)`). hit@1 drops from 17.4% (<300 Da) to 10.25% (no limit) — a real but
  modest effect, not the dominant driver of SIMBA's retrieval gap.
- `synthetic_mae_vs_hit1.py` — interpolates `c = gt_mces + (simba_mces - gt_mces) * w`
  from the perfect predictor (w=0) toward SIMBA's own real prediction (w=1) on the
  same table, tracking hit@1 as it goes. w=1 exactly reproduces SIMBA's known
  official-split hit@1 (10.25%, MAE=7.18) — a built-in correctness check. Answers
  "how much would SIMBA's MAE need to improve to close the gap with cosine": an
  MAE of ~6.5 already doubles hit@1, ~5 would beat cosine outright. Ceiling isn't
  literally 100% even at w=0 — 27/17,555 test spectra have multiple candidates tied
  at GT MCES=0 (stereoisomers MCES can't distinguish), broken fairly via a
  shuffle-then-idxmin tie-break (confirmed the candidate JSON lists the true
  candidate first in 100% of groups, so an unshuffled tie-break would hand it every
  win). An earlier version also compared against independent random noise at
  matching MAE — dropped per request, kept only the real-error interpolation.
- `plot_retrieval_comparison_checks.py` — added a 4th check, `run_score_heatmaps`:
  SIMBA-predicted-MCES x cosine-similarity 2D histograms (+ per-axis marginal
  histograms) for the true candidate, one per confusion-matrix cell, saved at both
  log and linear color scales. Also: `cosine_hit1` now excludes floor-tie wins
  (cosine_similarity exactly 0 for the whole candidate pool, i.e. arbitrary
  list-order luck — confirmed 100% of these have the true candidate listed first in
  the candidate JSON) from *every* check in the file, moving cosine's table-derived
  hit@1 from 37.59% to a corrected 31.89%; `report_zero_cosine_hits` keeps a
  separate raw flag since it's specifically diagnosing this exact phenomenon. The
  precursor-mass-discrepancy boxplot now uses the FULL 17,555-query population (not
  a 100/cell sample) and reads `test_precursor_mz`/`candidate_precursor_mz` straight
  off the table (see `add_precursor_columns.py`) instead of a fresh MGF scan +
  per-row RDKit-canonicalized candidate_tsv lookup every run — that lookup alone
  used to take ~44 minutes; now the whole 4-check script runs in ~7 seconds.
- `add_precursor_columns.py` — extends `retrieval_comparison_table.csv` with
  `test_precursor_mz` (measured, one MGF scan) and `candidate_precursor_mz`
  (calculated, already sitting in `candidate_tsv` — no new mass computation, just a
  one-time canonicalization of its 600,455 raw SMILES so it joins against the
  table's already-canonical `candidate_smiles`). Caches the canonical lookup to disk
  so no future script pays that cost again. Backs up the table before overwriting.
- `plot_low_cosine_hit_examples.py` — 10 random examples of cosine "hits" with
  genuinely low but nonzero similarity (0 < sim < 0.05, i.e. not a floor-tie by the
  strict definition) — real vs. ICEBERG-predicted-candidate mirror plots, titled
  with candidate pool size and second-best cosine similarity. Most (7/10) still have
  a second-best of ~0 too, i.e. still wins by default, just with a technically
  nonzero winning value.
- `plot_mces_vs_precursor_ppm.py` — does SIMBA's predicted MCES for the true
  candidate track precursor-mass discrepancy (measured vs. calculated m/z)? Real but
  partial effect (r=0.334 on log-ppm; binned mean rises from ~13 near-zero ppm to
  ~18-23 at higher ppm) — but predicted MCES is already ~13 even at near-perfect
  mass match, so this isn't the main driver of SIMBA's confidence. Candidates used
  the calculated precursor and test spectra the measured one when passing through
  SIMBA for these predictions; worth rechecking with measured precursor used
  uniformly for both.
- `plot_confusion_matrix_examples.py` — dropped `cosine_mces` from example titles
  (not a calibrated quantity, misleading next to SIMBA's real one); fixed a
  title/content overlap (more vertical spacing + padding); excludes cosine floor-tie
  wins (rank=1 but similarity<0.01) from the example pool, same fix as the
  confusion-matrix correction above.
