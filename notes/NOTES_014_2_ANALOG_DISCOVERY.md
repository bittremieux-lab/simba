# 014_2 analog discovery (CASMI 2022, simplified reproduction)

Simplified reproduction of the SIMBA paper's analog-discovery evaluation
(biorxiv 2026.06.17.733050, Figure 2), using checkpoint 014_2 (with its CORN
bucket head) — no retraining, inference only. Deliberately narrower than the
paper in several ways (agreed with the user before starting):

- **SIMBA vs plain cosine only** — no modified-cosine, Spec2Vec, MS2DeepScore,
  or DreaMS baselines.
- **Raw MCES, not molecular-size-normalized MCES** — the paper rescales MCES
  by molecular size for its boxplot/ROC threshold; we report raw MCES
  throughout.
- **Reference library — TWO separate searches, matching the paper exactly**:
  (A) NIST20 + full MassSpecGym (not just the Gaetan test fold — the user
  explicitly corrected this: use the whole `MassSpecGym.mgf`, since CASMI
  queries whose exact molecule is present in a given search's reference
  library are excluded from *that* search, which is what makes using the
  full (train-inclusive) MassSpecGym valid despite 014_2 having trained on
  most of it — same exclusion principle the paper itself uses), and
  (B) GNPS-no-propagated alone. Two independent pipelines, not one merged pool.

## Data acquired

- **CASMI 2022 queries**: already in-repo, `simba/data/casmi2022.mgf` — 169
  unique molecules (not the paper's 132; their "exclude compounds present in
  the reference library" filter isn't implemented in the repo — see below,
  we replicate this ourselves against our own actual reference library
  composition, not the paper's).
- **NIST20**: user-provided, `data/nist20/nist20.mgf` — 681,708 spectra /
  ~17,800-17,990 unique molecules (SMILES-dedup / InChIKey-dedup respectively).
  Format: `SMILES/INCHIKEY/PEPMASS/CHARGE/ADDUCT/IONMODE/TITLE` — clean,
  matches the format `tools/simba_retrieval.py` already expects reasonably well.
- **GNPS (no propagated)**: downloaded from the official GNPS2 bulk-library
  page (`https://external.gnps2.org/gnpslibrary/ALL_GNPS_NO_PROPOGATED.mgf`,
  confirmed via the page's own embedded library-list data, not guessed —
  `AGGREGATION`-type entry, excludes auto-propagated/molecular-networking-
  inferred annotations, keeping only the curated GNPS + imported libraries).
  `data/gnps/ALL_GNPS_NO_PROPOGATED.mgf`, 2,226,992,929 bytes (size matches
  the server's reported content-length exactly, download integrity
  confirmed), 956,358 spectra. Format: `SMILES/PEPMASS/CHARGE/IONMODE/...` —
  **no `ADDUCT=` field**, `INCHI=` frequently `"N/A"`, `CHARGE=0` seen in
  sampled entries — community-contributed metadata, meaningfully messier
  than NIST20/MassSpecGym. Needs real cleaning (see "Data cleaning" below).
- **MassSpecGym (full, not just Gaetan test)**: `data/massspecgym/data/auxiliary/MassSpecGym.mgf`,
  309,138,720 bytes. Includes molecules 014_2 was trained on -- explicitly OK
  per the user's instruction, since per-search CASMI exclusion (see below)
  removes any query whose own molecule is present in that search's library,
  which is the standard/paper-matching leakage guard, not full-library
  novelty.

## Checkpoint / model details (014_2, confirmed, not assumed)

- `experiments/training/014_2_mces_bucket_mlp_1gpu/checkpoint-epoch=22-step=229000.ckpt`
- `head_mode=cosine_no_head`, `use_mces_bucket_head=True`, `mces_bucket_use_mlp=True`,
  `mces_bucket_use_product=False` (default), bucket edges `[2,4,6,8]` (merged
  6-class scheme) — same config verified multiple times earlier this session.
- `model.tasks.edit_distance.enabled=false` — **014_2 does NOT predict edit
  distance**, confirmed directly from its training config. The paper's
  "rank by MCES, tie-break by edit distance" procedure can't use a real
  edit-distance prediction here. Substitute: the already-validated
  `corrected*1000 + raw` CORN-corrected ranking score (bucket = coarse class,
  raw regression = fine tie-break) — same formula validated for retrieval
  earlier this session (`tools/simba_retrieval_iceberg.py::_corn_corrected_ranking_score`).
- `sampling.precursor_mass_mode=theoretical` — query/library precursor mass
  must be computed as theoretical (from SMILES+adduct), not read from the
  file's own measured value, matching every other 014_2 evaluation this
  session.

## Existing codebase, what's reusable vs stale

- `simba/analog_discovery/*` (`Simba.load_model()`, `FcLayerAnalogDiscovery`) —
  **stale for 014_2**: their checkpoint loader doesn't pass `head_mode`/
  `corn_bin_edges`/`use_mces_bucket_head` to `load_from_checkpoint`, so the
  CORN bucket head would silently never be constructed (`strict=False` drops
  the unmatched weights). Same class of bug already fixed elsewhere this
  session in `tools/simba_retrieval.py::load_model`. Not used here — new
  inference code built on the validated `tools/simba_retrieval.py` pattern
  instead.
- `simba/analog_discovery/simba_analog_discovery.py::AnalogDiscovery.compute_ranking` —
  gives the right lexsort *idea* (round MCES, tie-break by a second signal)
  but assumes real edit-distance. Not reused directly; the CORN-corrected
  score formula serves the same structural role.
- `simba/core/chemistry/similarity_metrics.py::MolecularSimilarityMetrics.compute_mces` —
  genuine exact-MCES computation, `myopic_mces` package + local `PULP_CBC_CMD`
  ILP solver, **fully local, no asimov2 needed**. This is the GT-MCES engine
  used below.
- `tools/cosine_baseline_iceberg.py::bin_spectra` — genuine plain (non-modified)
  cosine, reusable building block for the cosine baseline, though wired for a
  different (bounded-candidate-pool) task — needs a new query×library scoring
  wrapper, not the existing `rank_candidates_cosine`.
- No prior analog-discovery run of any kind exists in `experiments/` — nothing
  to reuse there.

## Cost estimate for exact GT-MCES (empirically measured, not guessed)

Timed `MolecularSimilarityMetrics.compute_mces`'s underlying `myopic_mces`
call directly on 300 real CASMI×NIST20 molecule pairs (same solver config
used throughout the codebase: `threshold=20`, `PULP_CBC_CMD`, `threads=1`,
10s per-pair time limit): **14.9 pairs/sec single-threaded** (median 4.4ms,
mean 67ms — inflated by a small number of genuinely-similar, slower-to-solve
pairs; the large majority of dissimilar pairs reject fast).

Actual exhaustive pair counts, from the real `tools/prepare_analog_discovery_data.py`
run (see "Data cleaning" below for the cleaning that produced these numbers):
- Search A (136 queries × 48,125 library molecules): **6,545,000 pairs**
- Search B (138 queries × 66,819 library molecules): **9,221,022 pairs**
- Total: ~15.77M pairs ≈ 294 hours single-threaded serial

All independent per-pair, trivially parallelizable — with ~150 parallel
workers (matching this cluster's 192-CPU nodes, or whatever the target CPU
server offers), ~2 hours wall-clock. This is why the exhaustive query×
library GT-MCES matrix (not just top-10) is used as the common data source
for all three result panels below, rather than a cheaper top-10-only subset
— confirmed cheap, per the user's "estimate it" instruction, not assumed.

## Data cleaning (done — `tools/prepare_analog_discovery_data.py`)

Filters applied to every source (CASMI/NIST20/MassSpecGym/GNPS alike):
RDKit-parseable SMILES, ≥6 peaks (SIMBA's canonical `min_n_peaks`), positive
ion mode (absent `IONMODE=` treated as positive, matching MassSpecGym/CASMI's
own convention elsewhere this session), and `ADDUCT` present in
`ADDUCT_TO_MASS` (`simba/core/chemistry/chem_utils.py`) since
`theoretical_precursor_mz()` raises on anything else — GNPS has no `ADDUCT=`
field at all, defaulted to `[M+H]+` (the dominant real-world adduct) for
positive-mode entries. Deduplicated to one representative spectrum per
canonical-SMILES molecule.

Real run output:
- CASMI: 170 spectra → 169 clean unique molecules.
- NIST20: 681,708 spectra → 17,801 unique molecules kept (234,352 dropped for
  unresolvable adduct — confirms the earlier concern about NIST20's exotic
  in-source-fragment/isotope adducts; 123,746 dropped for non-positive
  ionmode; 80,883 dropped for <6 peaks).
- MassSpecGym (full): 231,104 spectra → 30,747 unique molecules kept (34,440
  dropped for <6 peaks; no adduct/ionmode issues — MassSpecGym's own mgf is
  uniformly `[M+H]+`/positive).
- GNPS (no propagated): 956,358 spectra → 66,819 unique molecules kept
  (93,492 dropped for bad SMILES, 182,665 for <6 peaks, 172,331 for
  non-positive ionmode — confirms the earlier "meaningfully messier than
  NIST20/MassSpecGym" expectation; 0 dropped for adduct, since GNPS entries
  all got the `[M+H]+` default).
- Library A (NIST20 ∪ MassSpecGym, 423 molecules overlapping, NIST20 kept):
  **48,125 unique molecules**.
- Library B (GNPS alone): **66,819 unique molecules**.
- CASMI queries after per-search exclusion: **136 for search A** (33
  excluded, present in library A), **138 for search B** (31 excluded,
  present in library B).

Output: `data/analog_discovery/search_A_nist_msg.mgf` (52.2 MB, combined
query+library with `FOLD=` field), `data/analog_discovery/search_B_gnps.mgf`
(292.4 MB, same convention) — both directly loadable by
`tools/simba_retrieval.py::load_spectra(mgf, fold, "theoretical")` unmodified
(confirmed by reading that function: it reads `FOLD=`/`SMILES=`/`ADDUCT=` off
whatever matchms parses, no format changes needed on that side).

## Pipeline stages

1. **Data cleaning** — done, see above. `tools/prepare_analog_discovery_data.py`.
2. **Embedding + ranking** (GPU, this cluster) — `tools/analog_discovery_embed_rank.py`,
   launched per search via `tools/slurm/analog_discovery_embed_rank_014_2.slurm.sh`
   (`sbatch --export=SEARCH=A|B ...`). Reuses `tools/simba_retrieval.py`'s
   validated `load_model`/`embed_spectra`/`load_spectra` (CORN-bucket-aware,
   `--precursor_mass_mode theoretical`) — NOT `simba/analog_discovery/*`
   (stale, see below). Embeds all queries + the full library, then scores
   every query×library molecule pair under three schemes, each saved as a
   dense `(n_query, n_library)` matrix (small: tens of MB, no need for a
   top-K-only format):
   - `score_simba_raw.npy` — `(1 - cosine(raw_emb_q, raw_emb_lib)) * mces_max_value`,
     via `model.compute_from_embeddings` (pairwise, since the bucket head is
     magnitude-sensitive and not decomposable into cached per-spectrum
     embeddings — same reasoning as `simba_retrieval_iceberg.py::rank_candidates_corn_corrected`).
   - `score_simba_corn.npy` — `corrected*1000 + raw` CORN-corrected ranking
     score, substituting for the paper's real-edit-distance tiebreak (014_2
     has no edit-distance head, see "Checkpoint / model details" above).
   - `score_cosine.npy` — plain binned-peak spectral cosine (`tools/cosine_baseline_iceberg.py::bin_spectra`,
     bin_width=0.01 Da, max_mz=1100 Da), reported as `1 - cosine` so all
     three scores share an ascending-is-closer convention.
   Smoke-tested end-to-end on a tiny CPU fixture (3 queries × 5 library
   entries, real 014_2 checkpoint) — model loads, embeds, and scores
   correctly; not yet run at full scale (needs a GPU node).
3. **Exact GT-MCES** (CPU, "most likely on another CPU server" per the user)
   — `tools/analog_discovery_exact_mces.py`, same restart-safe
   prepare/compute_block/status/combine block pattern as
   `tools/compute_mces_exact_1020.py` / `tools/prepare_gt_mces_retrieval.py`
   (`tools/slurm/mces_exact_retrieval_candidates.slurm.sh`'s own array-job
   template), but using `myopic_mces.myopic_mces.MCES` directly
   (`PULP_CBC_CMD`) instead of `metabo_depthcharge`'s asimov2-only worker —
   confirmed fully local to this repo's `.venv`, no asimov2 dependency.
   Output schema (`smiles.txt` + `mces_exact.npy` as `(N,3)`
   `[mol_idx_a, mol_idx_b, mces]`) matches `prepare_gt_mces_retrieval.py`
   exactly, so `simba_retrieval_iceberg.py::load_gt_mces_lookup`'s format is
   reusable unmodified if ever needed elsewhere.
   `prepare` already run for real on both searches (fast, CPU-only, done
   directly rather than deferred):
   - `data/analog_discovery/search_A_exact_mces/` — 6,545,000 pairs, 100 blocks.
   - `data/analog_discovery/search_B_exact_mces/` — 9,221,022 pairs, 140 blocks.
   `compute_block`/`status`/`combine` smoke-tested end-to-end on a tiny
   fixture (restart-safety confirmed: re-running a `.done` block is skipped).
   **Not yet submitted at full scale** — `tools/slurm/analog_discovery_exact_mces.slurm.sh`
   is ready (`sbatch --export=OUTPUT_DIR=... ...`, array bounds must match
   `--n_blocks`: `0-99` for search A, `0-139` for search B) but partition/
   nodelist are placeholders since the user said this will likely run on a
   different CPU server — copy the `search_*_exact_mces/` directory there
   (meta.json/smiles.txt/pairs.npy only, `blocks/` is empty until compute
   runs) and adjust the `#SBATCH -p`/nodelist lines for that cluster first.
   Values above `threshold=20` in the output are lower bounds, not exact
   distances (`myopic_mces`'s own documented early-rejection behavior) —
   doesn't affect any of the three panels below, which only need "is this
   pair close" at thresholds well under 20.
4. **Analysis / plots** — `tools/analog_discovery_analyze.py`, run once per
   search once stages 2 and 3 have both completed for it:
   - (a) boxplot of best-true-MCES among each query's top-10 inferred
     candidates, SIMBA-raw / SIMBA-corn / Cosine.
   - (b) ROC curve, true-analog binarized at `--roc_threshold` (default
     **4.0** raw MCES — the paper's own 0.3 is on *normalized* MCES, which we
     don't compute by design; 4.0 was picked as a raw-MCES stand-in because
     it's one of 014_2's own trained CORN bucket edges `[2,4,6,8]`, not an
     arbitrary number, and is exposed as a CLI flag so it can be swept
     post-hoc without recomputing the underlying exhaustive GT matrix).
     Pooled over every resolved `(query, library)` pair in the exhaustive GT
     matrix, not just each method's own top-10 — this is the whole reason
     stage 3 computes exhaustively rather than per-method top-10-only (three
     methods have three different top-10s; a shared exhaustive GT source is
     what makes the three ROC curves comparable).
   - (c) ranking-performance: for each query, the single library molecule
     with the lowest true GT MCES (the "gold-standard best analog") is
     located in each method's own ranking; cumulative fraction of queries
     where it falls within rank K is plotted for K=1..20. (This specific
     algorithmic definition is my own reasonable reading of "ranking-
     performance plot" — the paper doesn't spell out the exact procedure
     beyond the general idea — documented here rather than left implicit.)
   Smoke-tested end-to-end on the same tiny fixture as stages 2-3 (all three
   plots + summary.json produced correctly; AUC came back NaN only because
   the 3-query/5-library smoke sample happened to contain zero true-analog
   pairs — an sklearn artifact of the degenerate tiny fixture, not a bug).

### Pipeline run to completion

All four stages have now run for real, both searches:

- **Stage 2** (GPU, this cluster): jobs 1338969 (search A) / 1338970 (search
  B), `sbatch --export=SEARCH=A|B tools/slurm/analog_discovery_embed_rank_014_2.slurm.sh`.
  Both COMPLETED in under 2 minutes each, no errors.
- **Stage 3** (CPU, computed on a different server by another agent, per the
  original plan — not run on this cluster): both `mces_exact.npy` results
  verified byte-exact via md5sum and copied back to
  `data/analog_discovery/search_{A,B}_exact_mces/mces_exact.npy` (search A:
  78.5MB, search B: 110.7MB, all 6,545,000 / 9,221,022 pairs computed across
  100/140 blocks, `threshold=20` as prepared). Only 284 (A) / 1 (B) pairs
  unresolved — essentially complete coverage.
- **Stage 4**: `tools/analog_discovery_analyze.py`, run per search. Boxplot
  panel extended with a **4th "oracle" box** (best GT MCES anywhere in the
  *whole* library per query, ignoring ranking entirely — the ceiling no
  method can beat, since it isn't constrained to any top-k). Ranking-
  performance panel extended to `--max_rank_k 1000` (was 20). A new
  companion script, `tools/analog_discovery_roc_sweep.py`, sweeps the ROC
  "true analog" threshold across 3..13 (11 subplots in one grid figure per
  search, each titled with its `%label=1` rate) — reuses `load_scores`/
  `build_gt_matrix` from `analog_discovery_analyze.py` directly rather than
  recomputing anything.

### Results

| | Search A (NIST20+MassSpecGym) | Search B (GNPS) |
|---|---|---|
| ROC AUC — SIMBA raw | 0.967 | 0.963 |
| ROC AUC — SIMBA CORN | 0.967 | 0.963 |
| ROC AUC — cosine | 0.761 | 0.725 |
| Best-in-top10 MCES (median) — SIMBA raw | 4.0 | 4.75 |
| Best-in-top10 MCES (median) — cosine | 5.0 | 5.5 |
| Oracle best-in-library MCES (median) | 2.0 | 2.0 |
| hit@10 — SIMBA raw | 0.169 | 0.203 |
| hit@10 — cosine | 0.125 | 0.130 |

SIMBA clearly beats plain cosine on both searches by every metric. SIMBA-raw
edges out SIMBA-CORN slightly (the CORN tiebreak formula is a substitute for
a real edit-distance head 014_2 doesn't have — see "Checkpoint / model
details" above — so this isn't surprising). ROC positive-class rate is tiny
throughout the sweep (0.01% at threshold=3 up to ~5.8% at threshold=13) —
true analogs are rare even at a loose cutoff, worth keeping in mind when
reading AUC alongside that imbalance.

Outputs: `data/analog_discovery/search_{A,B}_results/` — `analog_discovery_boxplot.png`
(4 boxes), `analog_discovery_roc.png` (single-threshold=4.0), `analog_discovery_roc_sweep.png`
(11-threshold grid), `analog_discovery_ranking_performance.png` (K=1..1000),
`summary.json`.

### Follow-up: does SIMBA's edge come from CASMI being close to its training data?

Raised by Wout (Slack, 2026-08-27) after seeing the above: SIMBA's large
margin over cosine here might not reflect general analog-discovery skill —
it could be an artifact of (a) 014_2's own train/test splits being
deliberately built at *large* MCES distances (to force generalization to
unfamiliar structures), unlike this benchmark, and (b) heavy overlap between
014_2's training data (MassSpecGym) and the search libraries. His suggested
test: sweep MassSpecGym train/test split distance and see where SIMBA's
advantage over cosine starts to erode.

First step (this session): characterize how far CASMI queries actually are
from 014_2's real training set. **Reused already-computed data, no new
exact-MCES compute** — `search_A_exact_mces/mces_exact.npy` already has
exact GT MCES from every CASMI query to every molecule in library A
(NIST20 ∪ full MassSpecGym), which by construction includes every molecule
in the Gaetan-split train set (`data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2/mapping.pkl`,
`df_smiles_train`, 24,010 molecules, the split 014_2 was ACTUALLY trained
on — confirmed by reading `tools/slurm/014_2_mces_bucket_mlp_1gpu.slurm.sh`,
NOT the raw MassSpecGym.mgf's own default `FOLD=train` field, which is a
different, larger split of 25,046 molecules and was my first, wrong,
instinct). Filtered `mces_exact.npy` down to (CASMI query, Gaetan-train
molecule) pairs (all 24,010 train molecules present in library A, 3.27M
resolved pairs), took the min GT MCES to any CASMI query per train molecule,
and plotted the survival curve (`data/analog_discovery/casmi_vs_msg_train_distance_survival.png`):
x = MCES threshold, y = count of train molecules with min-distance-to-any-
CASMI-query ≥ x. Median distance from a train molecule to its nearest CASMI
query: **11.0**. At x=10, 58.2% of train molecules (13,983/24,010) survive;
at x=20, only 7.5% (1,806/24,010) — most training molecules are structurally
far from CASMI, but a large minority (up to ~42% at threshold 10) are close.

This is the number behind the reply to Wout: enforcing "every training
molecule ≥ MCES 10 from every CASMI molecule" would shrink the Gaetan train
set from 24,010 to 13,983 molecules — with the caveat that redoing the
analog-discovery eval at increasing thresholds (4/6/8/10/12 proposed) would
confound two effects at once: a fairer/harder distance-controlled eval AND
less training data, both of which push SIMBA's measured performance down.

**Note for reruns of `tools/analog_discovery_exact_mces.py`'s SLURM
launcher on THIS cluster** (sofia, not "another server"): fixed three
cluster-specific bugs in `tools/slurm/analog_discovery_exact_mces.slurm.sh`
while debugging a since-abandoned attempt to recompute CASMI-vs-train
distances from scratch (before realizing the data already existed, see
above) — `--mem` is rejected outright here (memory is per-CPU-core
automatic), `module load uv` doesn't reliably load on the `zen5_vis`
partition (switched to calling `.venv/bin/python` directly), and
`${BASH_SOURCE[0]}`-derived paths break because SLURM copies the script to
a `/var/spool/...` staging path before execution (switched to a hardcoded
`SIMBA_DIR`, matching `analog_discovery_embed_rank_014_2.slurm.sh`'s own
convention). Also: this cluster's CPU-capable partition for this user's
account is `zen5_vis` with `--account=vsc` (not `zen4_h200`, which is
GPU-only and only usable with the `zen4-h200-...` account).

### Retraining pilot: CASMI-distance-excluded 014_2 variant (threshold=12)

Directly operationalizes the reply to Wout: retrain 014_2, identical config
in every respect, except Gaetan-split TRAIN molecules within a chosen MCES
distance of any CASMI query are dropped before training. **val/test are
never touched** — only train, so the analog-discovery eval (this same
pipeline, stages 2-4) stays comparable across thresholds. 014_2 itself is
the threshold=0 (no exclusion) baseline — no separate baseline run needed.

Explicitly NOT re-running `prepare_msg_gaetan_split_max_lb_hdf5.py` (the
heavy GPU+116GB-RAM preprocessing step that builds
`preprocessing_gaetan_split_max_lb_hdf5_v2/`) — that data is reused
completely unmodified. The exclusion happens purely at training-DATA-LOAD
time:

- **`simba/configs/training/default.yaml`**: new opt-in field
  `sampling.train_exclude_smiles_file` (default `null` — zero behavior
  change for every other experiment; every existing training run reads this
  as `None` and the new code path is a no-op).
- **`simba/workflows/training.py::load_dataset`**: for the `train` split
  only, drops any molecule whose `canon_smiles` is in that file, reusing the
  *exact same* `mol_idx_remap` mechanism the code already used for dropping
  molecules with missing spectra (`_apply_remap` in `prepare_data()` then
  filters/reindexes the precomputed `ed_mces_indexes_tani_incremental_train_*.npy`
  pair array automatically, logging `kept X / Y pairs after filtering` —
  this is where "pairs dropped / pairs left" gets logged, no new code
  needed for that). Verified the drop/remap logic in isolation against a
  toy 5-molecule example before trusting it with a 24h GPU job (see session
  transcript) — both molecule-index remapping and pair filtering matched
  hand-computed expected output exactly.
- **Exclusion file**: `data/analog_discovery/train_exclude_smiles_threshold12.txt`
  — built from the SAME already-computed CASMI-vs-train distances used for
  the survival-curve plot above (zero new MCES compute). **13,277 / 24,010**
  Gaetan-train molecules excluded at threshold=12, leaving **10,733**.
- **SLURM launcher**: `tools/slurm/014_2_casmi_excl12_1gpu.slurm.sh` — exact
  copy of `014_2_mces_bucket_mlp_1gpu.slurm.sh`'s config, only two
  differences: adds `sampling.train_exclude_smiles_file=...`, and
  `training.epochs=240` (10x 014_2's 24 — fewer train pairs means fewer
  steps/epoch at the same `limit_train_batches=10000`, so more epochs are
  needed for a comparable total step count). SLURM `--time=24:00:00`
  unchanged — the run isn't expected to reach 240 epochs, it's meant to be
  monitored and killed manually once validation loss plateaus (checkpoints
  save periodically throughout, same as every other experiment).
  Output: `experiments/training/014_2_casmi_excl12_1gpu/`.

### Full sweep launched (all 6 thresholds)

Threshold=12 was launched first as a pilot (job 1341673) and verified
healthy (exclusion filter fired correctly, `mol_idx_remap` filtered pairs
as expected, training loop running with sane losses) before committing to
the rest. Once confirmed, thresholds 4/6/8/10/14 were generated from the
same template (`sed`-parameterized copies of `014_2_casmi_excl12_1gpu.slurm.sh`)
and launched too — all 6 running concurrently on `zen4_h200`:

| Threshold | Job | Excluded | Train molecules remaining | Pairs remaining |
|---|---|---|---|---|
| 4  | 1341860 | 982    | 22,172 / 24,010 | 245,787,706 / 288,228,045 |
| 6  | 1341861 | 3,507  | 19,747 / 24,010 | 194,962,131 / 288,228,045 |
| 8  | 1341862 | 6,795  | 16,580 / 24,010 | 137,439,910 / 288,228,045 |
| 10 | 1341863 | 10,027 | 13,466 / 24,010 | 90,659,845 / 288,228,045  |
| 12 | 1341673 | 13,277 | 10,315 / 24,010 | 53,194,455 / 288,228,045  |
| 14 | 1341864 | 16,446 | 7,234 / 24,010  | 26,161,761 / 288,228,045  |

("Excluded" counts are CASMI-distance exclusions only; "remaining" also
nets out the normal missing-spectra drop every run has, hence not exactly
`24,010 - excluded`.) All 6 confirmed healthy directly from their SLURM
`.err` logs (Python's `logger.info`/`logger.warning` go to stderr, not
stdout — check `.err`, not `.out`, for the `[TRAIN]`/`[VAL]` progress lines
and the exclusion-filter confirmation) — no errors, losses stable, well
past step 100 on first check.

**Live monitoring dashboard**: `tools/dashboard_app.py`, `list_experiments()`
filtered via `_is_shown_experiment()` to only: the 014_2 baseline, `excl14`
(already budget-matched by construction), every `014_2_casmi_budget7564_T*`
run, and the two log-loss runs (`014_2_logloss_a5_1gpu`/`_a40_1gpu`) —
excludes the superseded, non-budget-matched excl4/6/8/10/12 runs. Launched
with `uv run --extra dashboard streamlit run tools/dashboard_app.py
--server.port 8579 --server.address 0.0.0.0 --server.headless true` (kill
and relaunch manually as needed — not left running by default). Only lists
experiments that already have `mces_binned_box_*.png` diagnostic plots
(generated post-training) — a run in progress won't appear until its first
post-training diagnostic step lands.

Plan: once each run's validation loss plateaus (manually monitored, killed
by hand per the user's own call — no automatic early stopping), rerun this
same analog-discovery pipeline (stages 2-4) against each resulting
checkpoint to see where, if anywhere, cosine starts to catch up to SIMBA as
training data is pushed farther from CASMI — the actual answer to Wout's
question.

### Wout's reply, cancelling the sweep, and the grid comparison (2026-08-27/28)

Posted the boxplot/ROC/ranking plots to Wout. His reply:

- On "retrieving the best candidate when all candidates share mass+formula
  feels like a different task from open-database analog retrieval": agreed
  — "for the former a contrastive loss should be more relevant than the
  regression task SIMBA is primarily trained on. Hence the idea to combine
  those losses" — not a near-term priority, just a note to keep the right
  evaluation setting in mind for SIMBA generally.
- On sweeping the exclusion threshold: yes, but **"keep the training budget
  comparable across those runs (i.e. comparable number of molecules in each
  training task). This won't lead to the best models possible, but that
  doesn't matter that much. We're more interested in relative differences
  than absolute performance."** — flags exactly the confound already noted
  above (smaller thresholds keep more data, larger thresholds have both
  "farther from CASMI" AND "less data" baked in together) — addressed below.
- Requested output shape: **2 final plots (one per reference library)**,
  each a grid with **one row per threshold** (starting at no-exclusion/014_2
  baseline, increasing thereafter) and **3 columns**: boxplot, ROC at a
  single fixed threshold (10, per his screenshot), ranking-performance.
- Told me to cancel the running threshold sweep jobs, but **not** the two
  log-loss jobs.

Actions taken:
1. Cancelled all 6 CASMI-exclusion training jobs (1341673, 1341860-1341864)
   — kept the 2 log-loss jobs (1342479 a5, 1342480 a40) running.
2. Took each cancelled run's **last saved checkpoint** (all had reached
   epoch 19-22 / steps ~200k-227k by cancellation time, close to their
   24h SLURM wall-clock cap anyway):
   - excl4: `checkpoint-epoch=19-step=200000.ckpt`
   - excl6: `checkpoint-epoch=20-step=207000.ckpt`
   - excl8: `checkpoint-epoch=21-step=214000.ckpt`
   - excl10: `checkpoint-epoch=21-step=219000.ckpt`
   - excl12: `checkpoint-epoch=22-step=227000.ckpt`
   - excl14: `checkpoint-epoch=22-step=224000.ckpt`
3. New generic launcher `tools/slurm/analog_discovery_embed_rank_checkpoint.slurm.sh`
   (`sbatch --export=CHECKPOINT=...,LABEL=...`) — re-runs stage 2 (embed+rank)
   for an arbitrary checkpoint against **both** searches in one job, output
   to `data/analog_discovery/search_{A,B}_scores_<LABEL>/`. Submitted 6 jobs
   (1343451-1343456), all COMPLETED cleanly, verified output files present
   for all 12 (6 checkpoints × 2 searches) before proceeding. **No new
   exact-MCES computation needed** — the exact-GT-MCES data
   (`search_{A,B}_exact_mces/`) doesn't depend on which checkpoint scored
   the queries, only on the CASMI queries and reference library molecules,
   both fixed — reused as-is for every row.
4. New `tools/analog_discovery_grid.py` — reuses `load_scores`/
   `build_gt_matrix` from `analog_discovery_analyze.py` (same underlying
   data/logic as the single-model plots), draws N rows × 3 columns onto one
   figure instead of separate per-model files. Ran once per search:
   `data/analog_discovery/search_A_grid.png` and `search_B_grid.png`, 7 rows
   each (014_2 baseline + excl4/6/8/10/12/14), `--roc_threshold 10
   --max_rank_k 1000`.

**Result — directly answers Wout's original question**: SIMBA's ROC AUC
(at threshold=10) degrades steadily and monotonically as the exclusion
threshold increases, in both reference libraries, while cosine's AUC stays
flat (it has no training data to be close/far from):

| Threshold | Search A AUC (SIMBA raw) | Search B AUC (SIMBA raw) |
|---|---|---|
| 0 (014_2 baseline) | 0.955 | 0.949 |
| 4  | 0.947 | 0.943 |
| 6  | 0.937 | 0.932 |
| 8  | 0.916 | 0.913 |
| 10 | 0.903 | 0.898 |
| 12 | 0.854 | 0.847 |
| 14 | 0.841 | 0.835 |
| Cosine (flat, any threshold) | 0.646 | 0.650 |

SIMBA's advantage narrows a lot but never fully vanishes within this range
on AUC. The ranking-performance panel is more nuanced: for search B at
excl14 specifically, the SIMBA and cosine curves nearly overlap by K=1000 —
SIMBA's *ranking* edge (not just AUC) has largely eroded there even though
AUC hasn't. This confirms Wout's hypothesis directionally, but the
threshold/data-size confound he flagged is still live in these numbers (see
next section).

### Budget-matched exclusion sets — controlling for training-set size (2026-08-28)

Wout's confound, addressed: each threshold above changes BOTH "how far
training data must be from CASMI" AND "how much training data there is" at
once (24,010 → 7,564 eligible train molecules from threshold 0 to 14) — so
the AUC degradation above could be partly/wholly a data-quantity effect,
not a distance effect. This stage fixes the training-set SIZE across every
condition and varies ONLY the minimum-distance constraint.

**Design**: recomputed each Gaetan-train molecule's min GT-MCES distance to
any CASMI query (same already-computed `search_A_exact_mces` data as
before, zero new MCES compute — 24,010/24,010 resolved). Confirmed
eligible-count-per-threshold directly (not from memory):

| Threshold T | Eligible (min_dist ≥ T) |
|---|---|
| 0  | 24,010 |
| 4  | 23,028 |
| 6  | 20,503 |
| 8  | 17,215 |
| 10 | 13,983 |
| 12 | 10,733 |
| 14 | **7,564** |

**BUDGET = 7,564** (threshold=14's own eligible count — the smallest, and
already an existing run, so it needs no rebuild). For each of the other 6
conditions (T = 0, 4, 6, 8, 10, 12), instead of excluding *only* molecules
closer than T (which would leave a different-sized pool each time), select
exactly **7,564 molecules** as the new train set: among molecules with
min_dist ≥ T, sort by min_dist ascending and take the 7,564 **closest to
CASMI** (chosen over random subsampling from the eligible pool — sorting
and taking the closest gives a tight, well-defined distance *band* per
condition rather than a wide/noisy random sample, so the "typical distance
from CASMI" dose-response across conditions is monotonic and interpretable
by construction, not just in expectation). The complement (24,010 - 7,564 =
16,446 molecules, same count every time) becomes that condition's
`train_exclude_smiles_file`.

Resulting selected-molecule distance bands (min → max among the 7,564
chosen), confirming the sliding-window design works as intended:

| T | Selected molecules' min_dist range |
|---|---|
| 0  | [0.0, 8.0]   |
| 4  | [4.0, 9.0]   |
| 6  | [6.0, 10.0]  |
| 8  | [8.0, 12.5]  |
| 10 | [10.0, 14.5] |
| 12 | [12.0, 17.5] |
| 14 (existing run, unchanged) | [14.0, max] |

Files written (all 16,446 lines, verified):
`data/analog_discovery/train_exclude_smiles_budget7564_T{0,4,6,8,10,12}.txt`.

Note the T=0 row here is NOT the same thing as the existing "014_2 baseline"
row in the grid above — the existing baseline trains on all 24,010
molecules unrestricted, while this new T=0 condition trains on only the
7,564 molecules *closest* to CASMI (budget-matched like every other row
here). Both are worth keeping: the old baseline shows "full data, no
distance control"; this new sweep isolates the distance effect at fixed
data size, with the T=0 row as its own "closest possible, same budget"
anchor point.

**Stage 1 (data prep) landed, then stage 2 (training) launched**: 6 new
launchers (`tools/slurm/014_2_casmi_budget7564_T{0,4,6,8,10,12}_1gpu.slurm.sh`,
copied from the `excl12` pattern) — `training.epochs` reverted from the
earlier sweep's 240 back to 014_2's own 24 (no longer needed now that every
condition has identical molecule count), `training.val_check_interval`
changed 1000 → 3000. Launched jobs 1343709-1343714, all confirmed healthy
(remaining-molecule counts tightly clustered 7,273-7,330, as expected from
the shared 7,564 budget). Also launched the two log-loss jobs a bit
earlier: 1342479 (a5) and 1342480 (a40).

### Sampling refinement: closest-1000 + random fill, not closest-N (2026-08-28)

After ~5h of training on the pure "N=7,564 closest to CASMI" design, revised
the selection strategy: pure closest-N packs every condition's training set
into an artificially NARROW distance band (e.g. T=12's selected molecules
were confined to [12.0, 17.5] — nothing farther than 17.5 ever seen), which
distorts the training distribution away from what 014_2 itself trains on
and isn't really "the same task, closer or farther data" so much as "a
different, narrow-band task at every T." Revised to a hybrid: **the 1,000
closest-to-CASMI molecules (among those with min_dist ≥ T) are always kept
in** (guarantees near-CASMI structural examples exist at every threshold,
even T=12), **and the remaining 7,564 − 1,000 = 6,564 slots are filled by
random sampling** (seed 42) from the rest of the eligible pool — giving each
condition a training-distance distribution that's a genuine, representative
sample of "everything at least T away" rather than an artificially
compressed sliver near the boundary.

Resulting distance distributions (min is always exactly T by construction,
median increases meaningfully and monotonically with T — the real signal —
max reflects the true long tail of the pool, not an artificial cutoff):

| T | min | median | max |
|---|---|---|---|
| 0  | 0.0  | 10.0 | 44.0 |
| 4  | 4.0  | 10.5 | 53.0 |
| 6  | 6.0  | 11.5 | 42.5 |
| 8  | 8.0  | 12.5 | 44.0 |
| 10 | 10.0 | 14.0 | 53.0 |
| 12 | 12.0 | 15.0 | 50.0 |

**Actions**: killed all 8 running jobs (the 6 budget-matched sweep +
both log-loss runs — log-loss jobs stopped per the user's "kill all
currently running jobs," not because anything was wrong with them; not
relaunched yet, pending a separate decision). Cleared all 6
`experiments/training/014_2_casmi_budget7564_T{T}_1gpu/` directories
(deleted ~69 files each — checkpoints, metrics.csv, diagnostic PNGs from
the now-superseded closest-N run) so the fresh runs start clean, not mixed
with stale artifacts from the old sampling strategy. Rebuilt all 6
`train_exclude_smiles_budget7564_T{T}.txt` files with the new hybrid
selection (same 16,446-exclusion count each, verified). SLURM launchers
themselves are unchanged (same file paths) — only the exclusion files'
*content* differs, so the existing 6 scripts are being resubmitted as-is.

### Budget-matched sweep: final results (2026-08-28)

All 6 jobs (1344564-1344569) completed cleanly, full 24 epochs / step
240,000 each. Re-ran stage 2 (embed+rank, jobs 1345846-1345851) against
both searches using each final checkpoint, then rebuilt the grid plots:
`data/analog_discovery/search_{A,B}_grid_budget7564.png`, 8 rows (014_2
full baseline + T0/4/6/8/10/12 budget-matched + T14, which needs no rerun
since its eligible pool exactly equals the budget, forcing 100% inclusion
regardless of sampling strategy).

**ROC AUC (threshold=10), SIMBA raw regression:**

| | Search A | Search B |
|---|---|---|
| 014_2 (full, unrestricted) | 0.955 | 0.949 |
| T0 (budget 7,564) | 0.941 | 0.940 |
| T4 (budget 7,564) | 0.936 | 0.933 |
| T6 (budget 7,564) | 0.930 | 0.928 |
| T8 (budget 7,564) | 0.918 | 0.915 |
| T10 (budget 7,564) | 0.886 | 0.877 |
| T12 (budget 7,564) | 0.875 | 0.868 |
| T14 (budget 7,564, forced full) | 0.841 | 0.835 |
| Cosine (flat, any condition) | 0.646 | 0.650 |

**This directly answers Wout's confound concern**: even with training-set
size held exactly fixed across every condition (7,564 molecules each), AUC
still degrades smoothly and monotonically as the minimum distance from
CASMI increases. The earlier (non-budget-matched) sweep's degradation was
NOT primarily a data-quantity artifact — distance from the query
distribution is a real, independent driver of SIMBA's advantage over
cosine, confirming Wout's original hypothesis on properly controlled data.
