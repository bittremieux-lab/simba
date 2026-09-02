# Notes: SIMBA vs cosine retrieval across splits (Gaetan's sanity check)

Working notes, not a polished report — enough to remember what we did, why, and where
the code/results live, and to track progress against the plan below by checking items
off as we go.

## Goal

Gaetan's hypothesis (Slack, 2026-08-12): "is retrieval scores of SIMBA worse than
Cosine because of its bad performance on the official split or is there something more
going on?" He sketched a 4-row x 3-column table:

|                                          | MSGym official split | MSGym own split | MSGym scaff split |
|------------------------------------------|:---:|:---:|:---:|
| Cosine to train + FP of NN retrieval     | ~9  |  ?  |  ?  |
| SIMBA to train + FP of NN retrieval      | ~5  |  ?  |  ?  |
| ICEBERG on cands + Cosine                | ~36 |  ?  |  ?  |
| ICEBERG on cands + SIMBA                 | ~10 |  ?  |  ?  |

(values are hit@1 %). If SIMBA stays behind cosine on every column, that's not just an
official-split-performance artifact — SIMBA is legitimately weaker for retrieval even
in favorable scenarios, and needs improving directly rather than via training-schedule
tweaks.

## Investigation findings (done)

- **008_2 checkpoint**: `checkpoint-epoch=02-step=24000.ckpt`, trained on
  `preprocessing_official_split_max_lb_hdf5` (MassSpecGym official split),
  `head_mode=cosine_no_head`, no metadata. Manually picked as the best
  `val_mces_spearman/official` point from `metrics.csv` — **not** the auto-saved
  `best_model.ckpt` (that one is selected by scaffold-validation loss instead).
  Confirmed via comments in
  [`tools/slurm/retrieval_008_2_cosine_no_head_1gpu.slurm.sh`](simba/tools/slurm/retrieval_008_2_cosine_no_head_1gpu.slurm.sh)
  and [`tools/slurm/retrieval_iceberg_008_2.slurm.sh`](simba/tools/slurm/retrieval_iceberg_008_2.slurm.sh).
- **Old retrieval** (rows 1-2): [`tools/simba_retrieval.py`](simba/tools/simba_retrieval.py).
  Does NOT use ICEBERG candidate spectra at all — embeds train+test, finds each test
  spectrum's nearest **training** spectrum by embedding cosine similarity
  (`nearest_neighbor_transfer`), transfers that training molecule's Morgan fingerprint,
  ranks the candidate pool by Tanimoto similarity to it. SIMBA variant already run:
  hit@1=5.89%≈"~5" (`experiments/training/008_2_cosine_no_head_1gpu/retrieval_best_official_spearman/retrieval_results.tsv`).
  No embedding averaging (consistent with our policy) — but training-side NN pool keeps
  only the first spectrum per unique training molecule (dedup, not averaging), and
  `nearest_neighbor_transfer`'s argmax has no fair tie-break (minor, likely negligible
  for continuous embeddings).
  **Cosine variant (row 1, "~9") did not exist anywhere in the repo** — see Stage A.
- **New retrieval** (rows 3-4): [`tools/simba_retrieval_iceberg.py`](simba/tools/simba_retrieval_iceberg.py) /
  [`tools/cosine_baseline_iceberg.py`](simba/tools/cosine_baseline_iceberg.py) — what
  we built/verified all through items 8a-8d this session (no-averaging, own-adduct
  matching, etc.). hit@1=10.25%≈"~10" (SIMBA), 37.59%≈"~36" (cosine) — matches.
- **Scaffold split**: not a raw MGF fold — molecules carved out of the official MGF
  *train* fold via Murcko-scaffold resplitting (90/10, seed=42) in
  [`tools/prepare_msg_official_split_max_lb_hdf5.py`](simba/tools/prepare_msg_official_split_max_lb_hdf5.py),
  used only for monitoring during 008_2's own training. 1,621 molecules / 10,450
  spectra (`mapping.pkl`'s `df_smiles_val`/`spectrum_indexes_val` in
  `preprocessing_official_split_max_lb_hdf5`). 100% already present as
  `candidates.json` keys. Candidate pools reference 340,072 unique molecules; only
  ~3,567 (1%) already have ICEBERG predictions in the existing `preds.hdf5` →
  **~336K new (candidate, adduct) ICEBERG predictions needed**, comparable in scale to
  the original 600K-candidate job.
- **Own split (Gaetan)**: training experiment
  `005_msg_gaetan_split_max_lb_hdf5_excl_mces20_identity_bf16_4gpu`. Trained with
  **default `head_mode=cosine_relu`** (no override in its slurm script) — different
  from 008_2's `cosine_no_head`. Best checkpoint by its own `val_mces_spearman/val`:
  peaks at step≈7249 (0.834), declining after → closest saved file
  `checkpoint-epoch=05-step=7000.ckpt`. Existing `005/retrieval_iceberg/` and
  `005/retrieval_official_test/` both evaluate on the **official** test set
  (n≈17,556), not Gaetan's own — neither reusable. Gaetan's own test fold: 2,734
  molecules / 14,118 spectra (`preprocessing_gaetan_split_max_lb_hdf5`'s
  `mapping.pkl`), 0 missing from `candidates.json`, 510,581 unique candidate
  molecules, only ~67K (13%) already ICEBERG-covered → **~443K new predictions
  needed**.

## Decisions made (2026-08-12)

- Cosine-NN-transfer script: write it fresh. The whole change vs.
  `simba_retrieval.py` is swapping the embedding source for the NN-transfer step from
  a SIMBA checkpoint to binned-spectrum cosine vectors — nothing else changes.
  This also means the official-split column is missing row 1 too (never computed) —
  fix that as part of Stage A/B below.
- 005 (Gaetan split)'s head_mode mismatch: use the checkpoint as-is (no fresh
  training run) — just always pass `--head_mode cosine_relu` explicitly for every
  retrieval step against this checkpoint (never rely on the script's default, even
  though it happens to coincide here).
- Accepted: both new columns require ICEBERG generation runs comparable in scale to
  the original 600K-candidate job (~336K for scaffold, ~443K for own-split). Will ask
  before submitting each one specifically, same as every SLURM job this session.

## Plan / progress

- [x] **Stage A** — [`tools/cosine_retrieval.py`](simba/tools/cosine_retrieval.py) (new):
  mirrors `simba_retrieval.py` exactly except embeds via binned-spectrum cosine
  (reusing `cosine_baseline_iceberg.py`'s `bin_spectra`) instead of a SIMBA checkpoint,
  for both train and test/val spectra. Reuses `nearest_neighbor_transfer` (adapted to
  sparse chunked matmul), `morgan_fp`/`build_fp_lookup`, `tanimoto_scores` unchanged;
  `compute_hit_rates_np` is a numpy-native copy (the original assumes torch tensors).
  Same `--split`/`--mgf` flags so it works against the mini-MGFs from Stages C/D too.
- [x] **Stage B** — official split, row 1: ran on official train+test (job 1318999).
  **hit@1 = 8.16%** (train=194,119 spectra/25,046 unique molecules after dedup,
  test=17,556) — matches Gaetan's "~9%" estimate. Official-split column now complete:
  cosine-NN 8.16%, SIMBA-NN 5.89%, ICEBERG+cosine 37.59%, ICEBERG+SIMBA 10.25%.
- [ ] **Stage C** — scaffold column:
  - [x] C1: extracted scaffold-val's 1,621 molecules / 10,450 spectra into
    [`experiments/retrieval_split_comparison/scaffold_val_as_test.mgf`](experiments/retrieval_split_comparison/scaffold_val_as_test.mgf)
    with `FOLD=test` (job 1319000). Self-verification against `mapping.pkl` passed
    exactly. **Superseded by a 2-group re-run** (see Journal) that ALSO writes the
    scaffold split's own *train* fold into the same file as `FOLD=train` — needed for
    rows 1-2's NN-transfer (which load both "train" and the query split from one
    `--mgf`); rows 3-4 just ignore the extra train rows via `--split test`. Re-run
    queued, not yet submitted.
  - [x] C2: built [`ICEBERG/data/candidates_scaffold_val_new.tsv`](ICEBERG/data/candidates_scaffold_val_new.tsv)
    (336,990 new pairs) + [`candidates_scaffold_val_existing_overlap.tsv`](ICEBERG/data/candidates_scaffold_val_existing_overlap.tsv)
    (the small subset of the original 600,455-row file this query set actually
    overlaps with — avoids re-embedding all 600K existing candidates just to reach
    that ~1% overlap). Job 1319001 first pass (336,990/443,724 new pairs, see
    Journal), job 1319006 re-run adds the overlap files.
  - [x] C3: submitted [`ICEBERG/ms-pred/slurm/predict_scaffold_val_new.slurm.sh`](ICEBERG/ms-pred/slurm/predict_scaffold_val_new.slurm.sh)
    (job 1319002) — queued, waiting on cluster resources.
  - [x] C4/C5 **scripts prepared** (not yet run, waiting on C3):
    [`tools/slurm/retrieval_scaffold_val.slurm.sh`](simba/tools/slurm/retrieval_scaffold_val.slurm.sh)
    (rows 3-4: SIMBA+ICEBERG using 008_2's same checkpoint, and cosine+ICEBERG) and
    [`tools/slurm/nn_transfer_scaffold_val.slurm.sh`](simba/tools/slurm/nn_transfer_scaffold_val.slurm.sh)
    (rows 1-2: SIMBA-NN and cosine-NN, needs the re-extracted combined mini-MGF).
- [ ] **Stage D** — own-split (Gaetan) column: same shape as Stage C.
  - [x] D1: extracted Gaetan's own 2,734-molecule/14,118-spectrum test fold into
    [`experiments/retrieval_split_comparison/gaetan_test.mgf`](experiments/retrieval_split_comparison/gaetan_test.mgf)
    (job 1319000); **superseded by the same 2-group re-run** adding Gaetan's own
    train fold as `FOLD=train` in the same file.
  - [x] D2: built [`ICEBERG/data/candidates_gaetan_test_new.tsv`](ICEBERG/data/candidates_gaetan_test_new.tsv)
    (443,724 new pairs) + [`candidates_gaetan_test_existing_overlap.tsv`](ICEBERG/data/candidates_gaetan_test_existing_overlap.tsv).
  - [x] D3: submitted [`ICEBERG/ms-pred/slurm/predict_gaetan_test_new.slurm.sh`](ICEBERG/ms-pred/slurm/predict_gaetan_test_new.slurm.sh)
    (job 1319003) — queued, waiting on cluster resources.
  - [x] D4/D5 **scripts prepared** (not yet run, waiting on D3):
    [`tools/slurm/retrieval_gaetan_test.slurm.sh`](simba/tools/slurm/retrieval_gaetan_test.slurm.sh)
    (rows 3-4, `checkpoint-epoch=05-step=7000.ckpt` + explicit `--head_mode
    cosine_relu`) and [`tools/slurm/nn_transfer_gaetan_test.slurm.sh`](simba/tools/slurm/nn_transfer_gaetan_test.slurm.sh)
    (rows 1-2).
- [x] **Stage E** — consolidated, all 12 cells filled (hit@1, %):

  |                | official | own (Gaetan) | scaffold |
  |----------------|:---:|:---:|:---:|
  | Cosine-NN      | 8.16 | 15.17 | 17.39 |
  | SIMBA-NN       | 5.89 |  8.31 | 13.47 |
  | ICEBERG+Cosine | 37.59 | 44.86 | 48.94 |
  | ICEBERG+SIMBA  | 10.25 | 11.47 | 17.49 |

  **SIMBA loses to cosine in every single cell, on every split.** The gap doesn't
  close on the splits meant to be more favorable to SIMBA (its own training split,
  or a scaffold-held-out set) — it shrinks a bit in relative terms for
  ICEBERG+SIMBA-vs-Cosine (3.67x -> 3.91x -> 2.80x official/own/scaffold) but stays
  large in absolute terms (27.3 -> 33.4 -> 31.4 points) and the NN-transfer rows show
  the same pattern throughout. Per Gaetan's original framing: this is the "still a
  gap" branch — SIMBA is legitimately weaker at retrieval, not merely because the
  official split is hard for it (OOD ICEBERG spectra / chem-space mismatch isn't the
  root cause) — improving retrieval needs work on SIMBA itself, not just training-
  schedule/data-weighting tweaks.
- [x] **Stage F** — 6th row added: oracle GT-MCES NN-transfer (upper bound on the
  "old approach"). Instead of SIMBA/cosine embedding similarity picking the nearest
  *training* spectrum, uses the TRUE structural distance (`max(lb_matrix, hdf5)` GT
  MCES, no new MCES computation — both are pre-existing all-vs-all matrices) to pick
  the actual closest training molecule, then the same Tanimoto-candidate-ranking as
  rows 1-2. See `tools/oracle_retrieval_gt_mces.py`.

  |                   | official | own (Gaetan) | scaffold |
  |-------------------|:---:|:---:|:---:|
  | Cosine-NN         | 8.16 | 15.17 | 17.39 |
  | SIMBA-NN          | 5.89 |  8.31 | 13.47 |
  | Oracle GT-MCES-NN | 14.31 | 32.09 | 35.18 |
  | ICEBERG+Cosine    | 37.59 | 44.86 | 48.94 |
  | ICEBERG+SIMBA     | 10.25 | 11.47 | 17.49 |

  Oracle roughly doubles both real NN-transfer methods on every split — confirms
  there's real headroom in the NN-transfer approach above what embedding-based
  selection achieves, though it still falls well short of ICEBERG+Cosine everywhere.

  **Bonus finding, explains a lot of the official-vs-other-splits gap**: the oracle's
  own diagnostic (mean/median/min GT MCES from each test molecule to its *actual*
  closest training molecule) is dramatically different across splits — official:
  mean=10.6, median=10.0, **min=10.0**; own (Gaetan): mean=8.4, median=10.0, min=3.0;
  scaffold: mean=6.0, median=6.0, min=1.0. **Not one single official-split test
  molecule has a training analog closer than GT MCES=10** — vs. training analogs as
  close as 1-3 edits away for the other two splits. The official split enforces much
  stronger train/test structural separation than either of the other two — so a good
  chunk of why hit@1 is systematically higher everywhere else (not just this row) is
  that the underlying task is genuinely easier there, not just retrieval-method
  strength.

### Remaining execution order (once cluster resources free up)
1. Re-submit the 2-group mini-MGF extraction (regenerates `scaffold_val_as_test.mgf` /
   `gaetan_test.mgf` as combined train+test files; safe to do in parallel with C3/D3,
   which never touch the MGF directly — only the already-built delta candidate_tsvs).
2. Wait for C3/D3 (ICEBERG generation) to complete.
3. Run `retrieval_scaffold_val.slurm.sh` / `retrieval_gaetan_test.slurm.sh` (rows 3-4).
4. Run `nn_transfer_scaffold_val.slurm.sh` / `nn_transfer_gaetan_test.slurm.sh`
   (rows 1-2) — needs step 1's regenerated file, independent of steps 2-3.
5. Stage E: consolidate.

## Journal (chronological findings, as we go)

- **2026-08-12, Stage B result**: [`tools/cosine_retrieval.py`](simba/tools/cosine_retrieval.py)
  ran clean on its first try (job 1318999, 5m51s). Official split: train=194,119 raw
  spectra → 25,046 unique molecules after dedup, test=17,556. hit@1=8.16%, hit@5=19.35%,
  hit@20=38.36%. Binned matrices: train (25046, 110001) nnz=1,250,823; test (17556,
  110001) nnz=552,532.
- **2026-08-12, Stage C1+D1 result**: [`tools/extract_spectra_by_mgf_index.py`](simba/tools/extract_spectra_by_mgf_index.py)
  (job 1319000, 11s — trivial cost, single MGF text-scan pass) confirmed the
  `spectrum_indexes_*` index semantics by direct code reading of
  `prepare_msg_official_split_max_lb_hdf5.py` (not just inference): these are
  positions in `matchms.importing.load_from_mgf(MGF_PATH)`'s own enumeration order,
  captured *before* any validity filtering (`subset_fold()`'s `spec_full[p]` lookup,
  where `spec_full = spec_idxs["train"]` was itself built from the raw
  `enumerate(matchms.importing.load_from_mgf(...))` call). Self-verification (compare
  extracted canonical-SMILES/spectrum-count against the mapping.pkl dataframe) passed
  exactly for both splits on the first attempt — no off-by-N surprises.
- **2026-08-12, ICEBERG generation timing benchmark found**: the *original* 600,455-
  candidate run (job 1306763, logged in
  [`ICEBERG/results/candidates_test_official/iceberg_predict_test_official_1306763.out`](ICEBERG/results/candidates_test_official/iceberg_predict_test_official_1306763.out))
  took exactly 3,269.97s (~54.5 min) on 1x H200. That's ~5.45s per 100 candidates,
  which extrapolates to roughly **~30 min for scaffold-val's ~336K new candidates**
  and **~40 min for Gaetan-test's ~443K new candidates** — each comfortably inside a
  single ~1-2h SLURM job, not the multi-hour ordeal the raw "comparable to the
  original 600K job" framing might have suggested. Exact launcher preserved at
  [`ICEBERG/ms-pred/slurm/predict_test_official.slurm.sh`](ICEBERG/ms-pred/slurm/predict_test_official.slurm.sh)
  — reused verbatim (same checkpoints, same flags: `--sparse-out --sparse-k 100
  --max-nodes 100 --threshold 0.0 --adduct-shift --batch-size 128 --num-gpu-workers 4
  --num-cpu-workers 16`) for the delta runs in C3/D3, just pointed at the new delta
  candidate_tsv + a fresh `--save-dir`.
- **2026-08-12, C2/D2 script**: [`ICEBERG/build_candidate_tsv_delta.py`](ICEBERG/build_candidate_tsv_delta.py)
  mirrors `build_candidate_tsv.py`'s exact pair-collection logic (same per-query-
  molecule/its-own-adducts expansion) line for line, generalized to any MGF/fold via
  CLI args, diffed against the existing `candidates_test_official.tsv` to emit only
  the missing pairs, with new `spec` IDs continuing that file's own numbering
  (`cand_600455...`) so the two files' predictions never collide once used together.
  Runs in ICEBERG's own venv (`ms-pred/.venv`, needs `ms_pred.common`), not simba's
  `uv run`. Job 1319001 (2m46s): **336,990** new pairs for scaffold-val,
  **443,724** for Gaetan-test, 0 precursor-mass failures on either — both very close
  to the earlier rough estimates (340,072 and 510,581 unique candidate molecules
  referenced minus existing overlap), confirming the earlier candidates.json-overlap
  math was sound.
- **2026-08-12, C3/D3 prepared**: two separate SLURM jobs (parallel, not sequential —
  independent save-dirs, no reason to serialize), reusing
  `predict_test_official.slurm.sh`'s exact checkpoints/flags verbatim:
  [`ICEBERG/ms-pred/slurm/predict_scaffold_val_new.slurm.sh`](ICEBERG/ms-pred/slurm/predict_scaffold_val_new.slurm.sh)
  and [`ICEBERG/ms-pred/slurm/predict_gaetan_test_new.slurm.sh`](ICEBERG/ms-pred/slurm/predict_gaetan_test_new.slurm.sh).
  Submitted as jobs 1319002/1319003 — currently queued waiting on cluster resources
  (not running yet), so moved on to preparing the next stages rather than waiting idle.
  Update: cluster was broadly resource-constrained for ~40min (all 4 pending jobs,
  including the lightweight CPU-only 1319006/1319007, stuck) before jobs started
  running. Both completed cleanly: 1319002 (scaffold-val) in 1,734.04s (~29 min,
  matches the ~30 min extrapolation closely); 1319003 (Gaetan-test) in 2,335.5s
  (~39 min, matches the ~40 min extrapolation closely). All 4 of C1-C3/D1-D3 are now
  done and verified.
- **2026-08-12, final retrieval jobs submitted**: all 4 remaining scripts (rows 1-2 +
  rows 3-4, both columns) submitted as jobs 1319141 (`nn_transfer_scaffold_val`),
  1319142 (`nn_transfer_gaetan_test`), 1319143 (`retrieval_scaffold_val`), 1319144
  (`retrieval_gaetan_test`). Once these finish, Stage E is just reading 4
  `retrieval_results.tsv` files and assembling the final table.
- **2026-08-12, 1319006/1319007 completed cleanly.** Overlap counts: 3,082 pairs for
  scaffold-val (336,990 + 3,082 = 340,072 ✓), 66,857 for Gaetan-test (443,724 + 66,857
  = 510,581 ✓) — both exactly reconcile against the earlier candidates.json-derived
  totals. Mini-MGF re-extraction: scaffold split now 147,018 spectra total (136,568
  train / 21,835 molecules + 10,450 test / 1,621 molecules, both self-verified OK);
  Gaetan split 156,016 total (141,898 train / 24,010 molecules + 14,118 test / 2,734
  molecules, both OK). This unblocks `nn_transfer_scaffold_val.slurm.sh`,
  `nn_transfer_gaetan_test.slurm.sh`, and `retrieval_scaffold_val.slurm.sh` (C3 is
  done) right now; `retrieval_gaetan_test.slurm.sh` still waits on 1319003.
- **2026-08-12, realized rows 1-2 need a train pool too**: `simba_retrieval.py` /
  `cosine_retrieval.py` both call `load_spectra(mgf, "train")` from the SAME `--mgf`
  path used for the query split -- the single-group mini-MGFs from C1/D1 only contain
  the query side, so `load_spectra(mgf, "train")` against them would silently return
  empty. Fixed by generalizing
  [`tools/extract_spectra_by_mgf_index.py`](simba/tools/extract_spectra_by_mgf_index.py)
  to accept repeatable `--group INDEX_KEY DF_KEY LABEL` triples, scanning the source
  MGF once and writing each spectrum with whichever group's label it belongs to,
  verifying each group independently against its own mapping.pkl dataframe. For
  scaffold, "train" = the *remaining* official-train molecules after scaffold-val's
  90/10 split removed them (NOT the raw MGF train fold, which would leak scaffold-val's
  own molecules back in as trivially-perfect neighbors) -- exactly
  `df_smiles_train`/`spectrum_indexes_train` in the same `preprocessing_official_split_max_lb_hdf5`
  mapping.pkl. For Gaetan, "train" = Gaetan's own 24,010-molecule/141,898-spectrum
  train fold (`preprocessing_gaetan_split_max_lb_hdf5`). Updated
  [`tools/slurm/extract_split_spectra.slurm.sh`](simba/tools/slurm/extract_split_spectra.slurm.sh)
  to the 2-group form, writing train+test into the SAME output file (rows 3-4's
  scripts only ever read `--split test`, so this is a safe superset, no need to keep
  separate single-group files).
- **2026-08-12, avoided re-embedding 600K existing candidates for a ~1% overlap**:
  since `simba_retrieval_iceberg.py`/`cosine_baseline_iceberg.py` now accept multiple
  `--candidate_tsv`/`--iceberg_preds` (generalized `load_candidate_index`/
  `load_iceberg_spectra` to take one path or a list, concatenating/merging --
  backward-compatible, single-path callers unaffected), passing the FULL original
  600,455-row `candidates_test_official.tsv` alongside the delta file would force
  embedding all 600K existing candidates just to reach the small fraction (~1-13%)
  scaffold-val/Gaetan-test actually overlap with. Fixed by having
  `build_candidate_tsv_delta.py` ALSO emit `--existing_filtered_output_tsv`: just the
  overlapping subset of the existing file (via a vectorized `MultiIndex.isin`, not a
  slow row-wise `.apply` over 600K rows). Re-ran C2/D2 as job 1319006 to get these
  (the original new-pairs counts are unaffected, this only adds the extra output).
- **2026-08-12, rows 1-2 scripts prepared for both new columns**:
  [`tools/slurm/nn_transfer_scaffold_val.slurm.sh`](simba/tools/slurm/nn_transfer_scaffold_val.slurm.sh) /
  [`tools/slurm/nn_transfer_gaetan_test.slurm.sh`](simba/tools/slurm/nn_transfer_gaetan_test.slurm.sh) —
  straight reuse of `simba_retrieval.py`/`cosine_retrieval.py` unchanged, pointed at
  the (about to be regenerated) combined mini-MGFs. Not yet submitted -- waiting on
  the 2-group extraction re-run first.
- **2026-08-13, Stage F (oracle GT-MCES-NN) discovered/adapted**: user recalled a
  pre-existing "huge matrix with all msg molecules vs all" instead of computing
  anything new. Investigation confirmed two already-existing all-vs-all MCES
  artifacts cover 100% of every split's train/test molecules used in this repo:
  `data/massspecgym/lb_matrix.npy` (240,637 molecules, condensed lower-bound MCES,
  116 GB) and `data/massspecgym/data/auxiliary/all_smiles_mces.hdf5` (34,731
  molecules, exact for MCES<10). A ready-made script,
  `tools/oracle_retrieval_max_lb_hdf5.py`, already did exactly this (oracle
  nearest-training-neighbor by true GT MCES instead of embedding similarity), but had
  a real bug: it reused lb_matrix's condensed-index formula for HDF5 too, when HDF5
  actually uses a different (scipy-pdist-style) convention --
  `prepare_msg_official_split_max_lb_hdf5.py`'s own comment documents this exact bug.
  Verified both conventions are correct by scattering 1,000 random common pairs
  between the two sources (Pearson r=0.886, coherent shape, not noise) before
  trusting either formula.
- **2026-08-13, `tools/oracle_retrieval_gt_mces.py` adaptation**: fixed the HDF5
  index bug, repointed hardcoded paths, switched to the formula-matched
  `candidates.json`, reused `simba_retrieval.py`/`cosine_retrieval.py`'s fingerprint
  and hit-rate functions instead of duplicating. Reordered fingerprinting to run
  AFTER the oracle NN lookup (the lookup doesn't depend on fingerprints at all,
  unlike the SIMBA/cosine case where the nearest neighbor is only known after
  embedding) — only the train molecules actually selected as someone's nearest
  neighbor get fingerprinted (e.g. 1,119/21,835 for scaffold), not the whole pool.
  Verified the vectorized gather logic against a brute-force per-pair reference on a
  synthetic partial-overlap example before running for real.
- **2026-08-13, mmap was catastrophically slow — real fix was loading into RAM, not
  code cleverness**: first real run (job 1319861, scaffold split, smallest grid at
  ~35M test-x-train pairs) ran 2+ hours with the `.out` log stuck at "Loading
  spectra..." — turned out to be output buffering hiding real progress (the `.err`
  log showed spectra actually finished loading in under a minute). The actual
  bottleneck was `lb_matrix.npy` being memory-mapped: ~35M scattered random-access
  reads from a 116 GB file on shared storage. Killed it. Checked the compute node's
  spec (1.5 TB RAM) and just loaded the whole matrix into RAM instead of mmap-ing it
  — one sequential ~116 GB read beats millions of scattered ones by orders of
  magnitude. This partition doesn't allow `--mem` directly (fixed 24-cpus-per-GPU
  ratio, memory follows cpu count, no CPU-only high-memory partition available to
  this account) so this needed no extra request at all once mmap was gone — 24
  cpus' default ~182 GB allocation comfortably fit the 116 GB matrix + overhead.
  Re-ran scaffold (job 1320075): **3m56s** end to end, hit@1=35.18%. Own-split (job
  1320077): 5m01s, hit@1=32.09%. Official (job 1320078): 6m40s, hit@1=14.31%. All
  three clean, no OOM, no resubmission needed.
