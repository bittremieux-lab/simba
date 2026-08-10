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
- `slurm/ood_generalization_check.slurm.sh` — runs the MAE/Spearman summary
  (CPU-only; uses a GPU node purely for fast, uncontended storage I/O).
- `slurm/mces_plots_all.slurm.sh` — runs all three plotting scripts sequentially
  (same CPU-only-on-a-GPU-node rationale).
