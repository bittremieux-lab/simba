# 014_2 log-loss experiment (Gaetan's idea, 2026-08-27)

## Origin

Gaetan (Slack, 2026-08-27), recalling ~6-month-old spectrawl experiments:
predicting MCES via two different parametrizations of a log-transform gave
similar *overall* loss but a big difference in *hit rates*. His hypothesis:
"adjusting the loss might differently impact different regions of the MCES
range, like re-weighting via gradients — and in average MCES predictions
this might cancel out, but it's visible when you evaluate how well it
predicts the close candidates, i.e. hit rates." His concrete suggestion:
train with `MSE(log(MCES_true + 1), log(MCES_pred))`, i.e. predict log(MCES)
directly (cosine scaled into `[0, log(max)]`) rather than raw MCES.

## What SIMBA already had (corrected claim)

Initially misreported this — see below. SIMBA has a `use_mces20_log_loss`
flag with a `log_conversion(x, a=5)` warp (`simba/core/models/similarity_models.py`).
**First claim was wrong**: I read the Python `__init__`'s default
(`use_mces20_log_loss=True`) and assumed that's what 014_2 used. It isn't —
`training.py` always explicitly passes `cfg.model.tasks.mces.use_log_loss`,
whose actual Hydra default is `false` (`simba/configs/model/simba_default.yaml:60`),
and neither 014_2 nor 013 override it. **014_2 was trained with plain
linear MSE, no log-warp** — confirmed by checking the actual config value
reaching the model, not the dead Python-level default.

## The math: `a` ↔ pseudocount

The warp function, substituting `x = 1 - MCES/max_value` (max_value=40):

```
log_conversion(x, a) = 1 - ln((a+1) - a·x) / ln(a+1)
                      = 1 - ln(1 + a·MCES/40) / ln(a+1)
```

Matching this to the general "log with pseudocount c" form
`1 - ln(1 + MCES/c) / ln(1 + 40/c)` gives **c = 40/a**. So:
- `a=5` (the historical hardcoded default) → pseudocount `c=8` → trains
  toward `log(MCES + 8)` — a gentle compression.
- `a=40` → pseudocount `c=1` → trains toward `log(MCES + 1)` — matches
  Gaetan's proposal exactly.

Verified numerically (not just algebraically) against both closed forms
before trusting it — see session transcript, both matched exactly.

## Implementation

`a` was hardcoded in `log_conversion`, not exposed via config. Changed:
- `SimilarityModelMultitask.__init__` gained `mces_log_loss_a=5` (backward
  compatible default), stored as `self.mces_log_loss_a`.
- `log_conversion(x, a)` now takes `a` explicitly (no default) — both call
  sites (`logits2_for_loss`, `target2_for_loss`) pass `self.mces_log_loss_a`.
- New Hydra field `model.tasks.mces.log_loss_a` (default 5), threaded
  through `training.py`'s `model_kwargs`.
- Verified `a=5` still reproduces the *exact* original warp (regression
  test against the closed-form log(MCES+8)/log(MCES+1) formulas) before
  launching anything.

**Also verified (directly from code, not inference) that this only affects
training, not evaluation**: `validation_step` computes `val_mces_mae` and
everything `ValMetricsCallback` records from the model's raw forward-pass
output (`logits2`) and the raw target — entirely separate local variables
from the log-warped ones used only inside `step()`'s loss computation.
`training_step` and `validation_step` both call the same `step()`, so the
warp is symmetric across train/val. This means every run here is directly
comparable to 014_2 on `val_mces_mae`/bucket accuracy despite the different
training signal.

## Runs

Two jobs, identical to 014_2 in every other respect (`training.epochs=24`,
same Gaetan-split data, same architecture):
- Job 1342479 — `model.tasks.mces.use_log_loss=true log_loss_a=5` (pseudocount=8)
- Job 1342480 — `model.tasks.mces.use_log_loss=true log_loss_a=40` (pseudocount=1, Gaetan's exact proposal)

Both ran ~23h and reached epoch 22 / step ~221,000 (014_2's own reference
checkpoint is step 229,000 — close but not identical maturity) before being
killed on 2026-08-28 (per an unrelated instruction to kill all running jobs
while reworking the separate CASMI-exclusion sweep's sampling strategy —
not because of any problem with these two runs). Checkpoints preserved,
not deleted, available for evaluation whenever wanted:
`experiments/training/014_2_logloss_a5_1gpu/checkpoint-epoch=22-step=222000.ckpt`,
`experiments/training/014_2_logloss_a40_1gpu/checkpoint-epoch=22-step=221000.ckpt`.

## Results (real numbers, extracted from metrics.csv just now)

| | 014_2 baseline (plain MSE) | a5 (pseudocount=8) | a40 (pseudocount=1) |
|---|---|---|---|
| Last `val_mces_mae` | 4.971 | **4.885** | 5.066 |
| Best `val_mces_mae` (any checkpoint) | 4.835 | **4.805** | 4.867 |
| Last `val_mces_bucket_balanced_acc` | 0.455 | 0.459 | **0.469** |

(`val_mces_mae`/bucket-accuracy are unaffected by the log-warp per the
verification above — genuinely comparable across all three.)

## My interpretation

A real, and non-obvious, pattern: **raw MAE and bucket-classification
accuracy move in *opposite* directions as the log-compression gets
stronger.** a5 (mild warp) improves both MAE and bucket accuracy over the
baseline — a mild win across the board. a40 (Gaetan's exact proposal,
strong warp) makes raw MAE *worse* than baseline, but gives the *best*
bucket balanced accuracy of the three. This is qualitatively consistent
with what Gaetan described: a loss re-weighting that trades some precision
in the far/bulk part of the MCES range for better *ordinal* discrimination
near the bucket boundaries — which is much closer to what actually drives
hit-rate/analog-discovery performance than raw MAE is.

Caveats, so this isn't overclaimed:
- **n=1 per condition** — no repeated seeds, so some of this could be
  run-to-run training noise rather than a real effect of `a`. The
  *direction* being monotonic across all three conditions (baseline < a=5
  < a=40 for bucket accuracy; the reverse ordering for MAE) is suggestive
  of a real trend, but a single run each can't rule out noise.
- Step counts aren't perfectly matched (221k vs 229k) — a minor confound,
  probably small given how flat these curves are near convergence, but not
  eliminated.
- **The most direct test of Gaetan's actual claim — hit rate / analog-
  discovery performance, not a proxy metric — hasn't been run yet.** Bucket
  balanced accuracy is a reasonable proxy (it's literally an ordinal
  MCES-bucket classification metric) but isn't the same thing as running
  these two checkpoints through the actual CASMI analog-discovery pipeline
  (stages 2-4, same as the exclusion sweep) and checking ROC AUC / hit@K.
  That would be the definitive next step if this is worth pursuing further.

## ICEBERG retrieval benchmark (a40 only, 2026-08-28)

Followed `NOTES_RETRIEVAL_014_2_CORN.md`'s pipeline step 4 exactly (same
Gaetan-test ICEBERG candidate files, `--precursor_mass_mode theoretical`,
`--min_peaks 6`) — new launcher `tools/slurm/retrieval_iceberg_logloss_a40_1gpu.slurm.sh`,
job 1344646, runs both raw-regression and CORN-corrected
`tools/simba_retrieval_iceberg.py` in one job against a40's checkpoint
(`014_2_logloss_a40_1gpu/checkpoint-epoch=22-step=221000.ckpt`). Completed
cleanly, no errors. ICEBERG+Cosine and Oracle GT-MCES-NN rows are model-
free/checkpoint-independent, not rerun — reused from the existing table.

| | Hit@1 | Hit@5 | Hit@20 |
|---|---:|---:|---:|
| ICEBERG+SIMBA (014_2 baseline, raw) | 19.54% | 40.02% | 60.26% |
| ICEBERG+SIMBA (**a40**, raw) | 18.39% | 40.01% | **60.39%** |
| ICEBERG+SIMBA (014_2 baseline, CORN) | 20.98% | 43.48% | 63.52% |
| ICEBERG+SIMBA (**a40**, CORN) | 20.49% | **44.00%** | **64.40%** |

**Mixed result — does not cleanly confirm the training-metrics story
above.** a40 is *worse* than baseline at Hit@1 in both raw and CORN
variants (-1.15pp, -0.49pp), but slightly *better* at Hit@5/Hit@20
(+0.88pp at Hit@20, CORN). So the earlier finding (a40 gives the best
`val_mces_bucket_balanced_acc` of the three) does **not** straightforwardly
translate into "a40 wins on the actual ICEBERG hit-rate benchmark" — it's a
small, real shift in the ranking *profile* (worse at the very top rank,
mildly better a bit further down), not an unambiguous win. This tempers the
earlier interpretation: the log-loss's effect on downstream retrieval
quality is genuinely mixed on this evidence, not a confirmed net positive.
a5 (the milder warp) has not been run through ICEBERG yet — worth doing for
a complete three-way comparison, since a5 looked like the more
unambiguously-good option on the training-metrics evidence alone.

## Dashboard-style "compare runs" table (2026-08-31)

Computed directly via `tools/dashboard_app.py`'s own comparison functions
(`_compare_runs_row`, `_find_cosine_reference`, `_overlap_for_spec_cosine`
— imported and called in a plain script, not through the Streamlit UI, but
the exact same code path/values it would show) — raw-regression rows only:

| Run | Last step | Val loss | Overall MAE | Identity MAE | Bucket bal.acc | Overlap (skip0) | Overlap (skip2) | Identity overlap (skip0) | Identity overlap (skip1) | Identity overlap (skip2) |
|---|---|---|---|---|---|---|---|---|---|---|
| **014_2** (baseline) | 228,999 | 0.0398 | 4.9713 | 2.8885 | 0.4550 | 0.6626 | 0.2709 | 0.5307 | 0.3759 | 0.2392 |
| **a5** (pseudocount=8) | 220,999 | 0.0378 | 4.8849 | 2.9822 | 0.4588 | 0.6531 | 0.2649 | 0.5204 | 0.3809 | 0.2388 |
| **a40** (pseudocount=1) | 220,999 | 0.0394 | 5.0664 | 2.9373 | 0.4694 | 0.6503 | 0.2657 | 0.5384 | 0.4133 | 0.2735 |
| **Cosine** (spectral, non-learned) | — | — | — | — | — | 0.8034 | 0.5395 | 0.3789 | 0.2872 | 0.2149 |

(Val loss/MAE/Overlap all lower-is-better; cosine's MAE/loss cells are
blank — not on a comparable scale to MCES. Cosine's own Overlap numbers are
much higher than any SIMBA row, as expected: overlap coefficient between
predicted-value distributions across different true-MCES bins, lower means
better separation, so SIMBA's much-lower overlap reflects it actually
discriminating between bins where raw cosine mostly can't.)

Same pattern holds here as in the earlier training-curve comparison: a5
improves on baseline across almost every column; a40 is mixed — worse than
baseline on Overall MAE and Overlap(skip0), better on Identity overlap
(skip1/skip2) and bucket balanced accuracy.

## Status

Training stopped (not by design — an incidental "kill everything" while
reworking a different, unrelated experiment). Checkpoints preserved. a40
now has training-metrics, ICEBERG-retrieval, and dashboard-comparison
results (all mixed, see above). a5 has training-metrics and dashboard-
comparison results, not yet run through ICEBERG. Neither has been run
through the analog-discovery (CASMI) eval pipeline.
