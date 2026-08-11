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
