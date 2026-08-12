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
