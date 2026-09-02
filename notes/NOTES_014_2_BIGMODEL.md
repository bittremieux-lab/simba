# 014_2 bigger-model experiment

## Motivation

014_2 (and every 013/014_x variant before it) shares the same base
transformer architecture: `d_model=256, n_layers=5, n_heads=8` (~4.6M
params, confirmed from an ICEBERG retrieval run's own model summary). Only
the auxiliary mces_bucket head's internal toggles (`use_mlp`/`use_product`/
`loss_weight`/`learnable_weight`) ever varied across that lineage. User
asked to try scaling the base architecture up.

## A real bug found and fixed first

`training.py::setup_model` correctly reads `d_model`/`n_layers` from
config (`cfg.model.transformer.d_model`/`n_layers`), so training itself
was already wired correctly for any size.

**Retrieval/inference was not.** `tools/simba_retrieval.py::load_model()`
had `d_model=256, n_layers=5` **hardcoded** directly in the
`load_from_checkpoint(...)` call — every downstream tool that scores a
checkpoint goes through this one function (`simba_retrieval_iceberg.py`,
`analog_discovery_embed_rank.py`). `SimilarityModelMultitask` never calls
`save_hyperparameters()`, so the checkpoint itself doesn't record its own
architecture size — Lightning can't infer it. Loading is done with
`strict=False`, so a real checkpoint's shape-mismatched weights would have
been **silently dropped, not errored** — a bigger checkpoint loaded through
the old hardcoded path would produce a mostly-random model with zero
warning, not a crash.

Fixed: `d_model`/`n_layers` are now explicit parameters on `load_model()`
(default 256/5 — unchanged for every existing checkpoint), threaded through
as `--d_model`/`--n_layers` CLI flags on all three affected scripts
(`tools/simba_retrieval.py`, `tools/simba_retrieval_iceberg.py`,
`tools/analog_discovery_embed_rank.py`). Regression-tested against 014_2's
real checkpoint with the default values before trusting the change.

## Wiring check across 013/014_1-4 (separately verified, not assumed)

Checked each variant's actual training SLURM script directly:

| Checkpoint | head_mode | mces_bucket.use_mlp | mces_bucket.use_product | Ever run through retrieval? |
|---|---|---|---|---|
| 013 | cosine_no_head | false | false | not checked here |
| 014_1 | cosine_no_head | false (default) | false (default) | **no** -- no launcher exists |
| 014_2 | cosine_no_head | true | false | **yes** -- `retrieval_iceberg_014_2_corn_corrected_1gpu.slurm.sh`, flags verified matching |
| 014_3 | cosine_no_head | false (default) | true | **no** -- no launcher exists |
| 014_4 (combined) | cosine_no_head | true | true | **no** -- no launcher exists |
| 014_4 (learnable_weight) | cosine_no_head | false (default) | false (default) | **no** -- no launcher exists |

None of them change `d_model`/`n_layers`, so the old hardcoded 256/5 was
never actually wrong for any of them by coincidence -- but only 014_2 has
ever actually been scored by a retrieval script; the others were trained
and never evaluated that way. Retrieval's head-config flags
(`--mces_bucket_use_mlp`/`--mces_bucket_use_product`) are matched by hand
per checkpoint, not auto-detected -- same class of risk as the
d_model/n_layers bug, just not yet hit because 014_2 is the only one ever
exercised this way.

## This experiment

`tools/slurm/014_2_bigmodel_d384_l8_1gpu.slurm.sh` -- identical to 014_2 in
every other respect, `model.transformer.d_model=384` (was 256),
`model.transformer.n_layers=8` (was 5). Roughly 2.7x the parameter count.

Launched as job 1346148. Confirmed healthy directly from the logs:
**12.5M params** (vs 4.6M for 014_2 itself, matches the expected ratio for
this size change), reached step 100 with a sane loss (train_loss=0.366,
loss_mces=0.051), no errors.

## Status

Training in progress (job 1346148), same 24-epoch / same-data config as
014_2 otherwise. Not yet evaluated on retrieval or analog-discovery --
remember to pass `--d_model 384 --n_layers 8` explicitly on every
retrieval/embedding tool when the time comes, since the checkpoint itself
can't tell those tools its own size.
