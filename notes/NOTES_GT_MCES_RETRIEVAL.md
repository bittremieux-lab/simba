# Notes: GT MCES for ICEBERG retrieval (NEXT_STEPS 3c / 3d)

Working notes, not a polished report — just enough to remember what we did, why, and
where the code/results live, for whoever (including future us) picks this back up.

## Goal

NEXT_STEPS.md items 3c and 3d, both extending the ICEBERG-based retrieval work from
item 3 (see PROGRESS_REPORT_ROADMAP.md):

- **3c**: extend hit@1/5/20 with a "how close did we get" metric — for the ranked
  top-1/5/20 candidates, the MIN ground-truth MCES to the true molecule (not max —
  min = the best/closest wrong guess).
- **3d**: a baseline that ranks the same ICEBERG-predicted candidate spectra by plain
  spectral cosine similarity to the real test spectrum — no SIMBA at all — to see how
  much SIMBA is actually adding (Gaetan's suggestion).

Both need ground-truth (exact) MCES between test molecules and their ~256-per-query
PubChem candidates — we didn't have that before, only SIMBA's own *predicted* MCES.

## What we built

- [`simba/tools/prepare_gt_mces_retrieval.py`](simba/tools/prepare_gt_mces_retrieval.py)
  — builds the full (test molecule, candidate molecule) pair set (584,340 pairs, self/
  true-match pairs excluded since GT MCES=0 trivially) and reuses `compute_block` /
  `status` / `combine` **unmodified** from the existing
  [`compute_mces_exact_1020.py`](simba/tools/compute_mces_exact_1020.py) tool (same
  threshold=20 exact-MCES ILP solver + watchdog already used elsewhere in SIMBA).
  Two-machine split: `prepare` runs here on sofia (has the source MGF + candidates
  json), `compute_block`/`combine` run on asimov2 (has `metabo_depthcharge` + spare
  CPUs, which this repo's sofia env doesn't have). Paths in `meta.json` get re-derived
  from `--output_dir` on every call so the directory survives being copied between
  machines.
- [`simba/tools/slurm/mces_exact_retrieval_candidates.slurm.sh`](simba/tools/slurm/mces_exact_retrieval_candidates.slurm.sh)
  — the asimov2 array job (60 blocks × ~9.7k pairs). Ran to completion, all 60/60
  blocks done, only 13/584,340 pairs unresolved (solver failed).
- Combined output: `data/gt_mces_retrieval_candidates/` (`smiles.txt`, `mces_exact.npy`
  — renamed from the reused `combine()`'s default `mces_exact_10_20.npy`, which was a
  misnomer here since there's no [10,20] filter, just the same threshold=20 cap).
  Stats: mean MCES 18.71, range 0–69, 91.5% ≤ 20. Candidates are formula-matched so
  they're not *wildly* different from the truth, but clearly not close either.
- [`simba/tools/simba_retrieval_iceberg.py`](simba/tools/simba_retrieval_iceberg.py)
  — added `load_gt_mces_lookup()` (reads the combined npy + smiles.txt into a
  `{(smi_a, smi_b): mces}` dict) and rewrote `compute_mces_stats()` to use it instead
  of the old on-the-fly `myopic_mces` subprocess calls (removed entirely — this is 3c).
- [`simba/tools/cosine_baseline_iceberg.py`](simba/tools/cosine_baseline_iceberg.py)
  (new) — 3d. Bins each spectrum onto a fixed m/z grid (0.01 Da, up to 1100 Da),
  sqrt-compresses + L2-normalizes, so cosine similarity is one dot product and ranking
  is one small sparse matmul per query against its own ≤256 candidates. No SIMBA, no
  GPU — ran directly on the login node in a couple of minutes.
- Updated [`retrieval_iceberg_005.slurm.sh`](simba/tools/slurm/retrieval_iceberg_005.slurm.sh)
  and [`retrieval_iceberg_008_2.slurm.sh`](simba/tools/slurm/retrieval_iceberg_008_2.slurm.sh)
  to pass `--gt_mces_dir`.

## Bug found + fixed (worth remembering)

First pass gave nonsensical numbers: cosine baseline had hit@20=79.2% but
`mces_min_top20_median=4.0` — should be 0 if the true molecule is in the top-20 for
>50% of queries. Root cause: `prepare_gt_mces_retrieval.py` deliberately doesn't store
true-match (self) pairs in the lookup ("trivially 0, don't bother computing"), but
`compute_mces_stats` treated a lookup miss as "no data, drop this pair" — silently
discarding every hit instead of crediting it as MCES=0. Fixed by special-casing
`c_canon == q_canon -> 0.0` before the lookup, and switched from a compacted list to
None-padding so a genuine miss doesn't shift which rank ends up in the `v[:k]` window.
Confirmed with a unit test where the true match sits at rank 3 in a top-20 list.
Re-ran everything after the fix (see results below — this is the corrected version).

## Results (test split, n=17,555)

| Model | hit@1 | hit@5 | hit@20 | mces_top1 (mean/median) | mces_min_top5 | mces_min_top20 |
|---|---|---|---|---|---|---|
| **Cosine baseline (no SIMBA)** | 37.6% | 61.4% | 79.2% | 7.38 / 2.0 | 3.86 / 0.0 | 1.73 / 0.0 |
| 008_2 (cosine_no_head) | 10.2% | 25.9% | 48.0% | 14.43 / 20.0 | 9.80 / 8.0 | 5.63 / 2.0 |
| 005 (Gaetan split, cosine_relu) | 10.1% | 25.9% | 49.3% | 13.91 / 19.0 | 9.34 / 6.0 | 5.36 / 2.0 |

Result files:
- Cosine baseline: `experiments/cosine_baseline_iceberg/retrieval_results.tsv` (run log: `experiments/cosine_baseline_iceberg/run.log`)
- 008_2: `experiments/training/008_2_cosine_no_head_1gpu/retrieval_iceberg/retrieval_results.tsv`
- 005: `experiments/training/005_msg_gaetan_split_max_lb_hdf5_excl_mces20_identity_bf16_4gpu/retrieval_iceberg/retrieval_results.tsv`

## The surprising bit

The no-SIMBA cosine baseline beats both SIMBA models by ~3.7x on hit@1. Checked for an
obvious bug (hit-rate code is identical between scripts, candidate spectra only ever
come from ICEBERG predictions, MCES stats are internally consistent) and didn't find
one. Current best guess: **domain mismatch** — SIMBA was trained exclusively on real
measured MS/MS spectra, and has never seen ICEBERG's theoretical/predicted spectra
during training, so it may be poorly suited to scoring them specifically, while plain
binned cosine similarity doesn't care what distribution the spectrum came from. Not
confirmed — worth raising with Gaetan/Wout given the size of the gap. 3e (the
OOD-generalization check) is a natural next step to help disentangle this further,
picking up right where we paused it earlier.

## 3e: does SIMBA generalize, or is it an ICEBERG-domain problem?

**Goal**: 3d found SIMBA loses badly to plain cosine on ICEBERG-candidate retrieval.
Two competing explanations: (a) SIMBA doesn't generalize past its training molecules
(Gaetan's OOD-chemical-space hypothesis), or (b) SIMBA is fine on new molecules but
specifically struggles with ICEBERG's *predicted* spectra (a different modality than
the real spectra it was trained on). 3e compares SIMBA's own predicted-MCES quality on
test-to-test pairs (in-distribution molecules, real spectra both sides) against
test-to-candidate pairs (same test molecules, but ICEBERG-predicted spectra on the
other side) — same solver, same model, same metric, two populations.

**Good news, no new asimov2 job needed**: exact GT MCES for test-to-test already
existed — `data/massspecgym/preprocessing_msg_exact_mces_1020/ed_mces_indexes_tani_incremental_test_node0_chunk0.npy`,
from the official-split preprocessing's own [10,20]-lb exact-refinement. Covers
2,959/3,170 test molecules (near-full cross product, 4.38M pairs). New loader:
[`simba/tools/load_test_to_test_gt_mces.py`](simba/tools/load_test_to_test_gt_mces.py).

**Two real bugs caught along the way (both worth remembering):**
1. First version averaged each test molecule's several real spectra into one
   embedding before scoring. Denoises the query side in a way the real evaluation
   never benefits from — caught via a hit@1 discrepancy (16.3% averaged vs. 10.2%
   real, from the actual retrieval run). Fixed everywhere: the query side now always
   uses each individual test *spectrum's* own embedding; the "other side" (candidates,
   or the other test molecule for test-to-test) stays molecule-level since GT doesn't
   depend on which spectrum represents a molecule.
2. The test-to-test dense cross-molecule scorer computed raw cosine similarity but
   never converted it to predicted MCES (`mces_max_value * (1 - sim)`) — caught via a
   suspicious `max-of-max=1.0` in one of the plots (should be ≤40). Fixed in
   `score_test_to_test_by_spectrum`.

**Self-inclusion, added per your request** (previously self was excluded everywhere,
for both populations):
- test-to-candidate: the true candidate (GT=0) is now included, scored against its
  own ICEBERG-predicted spectrum specifically — needed fixing `combined_smi_to_emb`'s
  merge priority (candidate embedding must win over the real-spectrum one on overlap,
  via `build_combined_other_side_embeddings`). One exception: the GT plot's "min"
  series is still computed self-*excluded*, since GT=0 for self trivially makes "min"
  a meaningless 0-spike otherwise; mean/max use the self-included pool everywhere.
- test-to-test: same-molecule-*different*-spectrum pairs (GT=0, genuine spectrum-vs-
  spectrum, not an averaged embedding) are now included alongside the cross-molecule
  pairs — previously the whole molecule was excluded, not just the literal self
  spectrum, which was a real gap rather than a deliberate choice. New:
  `build_same_molecule_spectrum_pairs` + `combine_row_stats` (sum/count-based
  combining across the dense cross-molecule part and this ragged extra part — can't
  just average two sub-means when their counts differ per query).

**Numbers** (final, all fixes applied):

| Population | MAE | Spearman | n |
|---|---|---|---|
| test-to-test (self spectrum excluded, same-molecule spectra included) | 5.774 | 0.718 | 50,299,912 |
| test-to-candidate (self included) | 6.735 | 0.220 | 2,908,770 |

MAE is close for both; Spearman is ~3.3x worse for test-to-candidate. **This is a
ranking/ordering failure, not a magnitude/calibration failure** — SIMBA isn't wildly
biased in scale on candidates, it just can't order them correctly by true similarity.

**Plots** (all in `experiments/mces_pool_distribution_plots/`):
- `test_to_{candidate,test}_{gt,simba,abs_dif_gt_simba}.png` — per-spectrum min/mean/max
  distributions ([`mces_pool_distribution_plots.py`](simba/tools/mces_pool_distribution_plots.py)).
- `test_to_{candidate,test}_binned_box.png` — GT-binned calibration boxplots, whis=(5,95),
  n-per-box labeled, GT clipped to 40 for a shared scale
  ([`mces_calibration_plots.py`](simba/tools/mces_calibration_plots.py)). This is where the
  mechanism is clearest: test-to-candidate's predicted-MCES medians *saturate* around
  20-25 once GT exceeds ~30 (can't tell "quite far" from "very far" apart); test-to-test
  tracks the diagonal much more evenly across the full 0-40 range.
- `test_to_candidate_top1_diagnostics.png` — 4-panel: (a) SIMBA's predicted MCES for the
  true candidate (should cluster at 0, actually means 14.96), (b) GT-minus-predicted for
  wrong top-1 picks (skews positive, mean 7.35 — SIMBA systematically *underestimates*
  distance for its own wrong top choice, i.e. overconfident false positives, not just
  noisy in both directions), (c)/(d) their joint distribution, log- and linear-colored
  ([`mces_top1_diagnostics.py`](simba/tools/mces_top1_diagnostics.py)).

**My read on this**: every independent angle — MAE/Spearman, the binned-box saturation,
the true-candidate/wrong-top1 diagnostics — converges on the same story, and it's a
*modality* story more than a *molecule-distribution* story. The clearest evidence: the
same-molecule-different-spectrum comparisons (real spectrum vs. a different real
spectrum of the same molecule — no ICEBERG involved on either side) show SIMBA
recognizing them as near-identical almost every time (the |diff| plot's "min" is a huge
spike at exactly 0). So SIMBA isn't generically bad at recognizing "this is basically the
same molecule" — it's specifically bad when one side is an ICEBERG-*predicted* spectrum
instead of a real one. That leans the original OOD question (Gaetan's chemical-space
hypothesis) toward "probably not the dominant factor" and item 3's domain-mismatch guess
toward "probably right" — though this still isn't a clean, isolated test of molecule
OOD-ness alone (test-to-test's molecules are the same ones used in test-to-candidate;
we haven't tested SIMBA on real spectra of molecules it's never seen in *any* form). If
that's the real mechanism, the fix isn't "train on more diverse molecules" (item 2) so
much as some form of domain adaptation — training SIMBA on at least some ICEBERG-style
predicted spectra, or otherwise closing the gap between what ICEBERG outputs and what
SIMBA was trained to expect. Worth raising with Gaetan/Wout alongside the 3d finding.


Now I will write presise message about updatese on 3c,d,e

New updates
So first one was 3c (Add MCES to the selected candidate to the retrieval scoring: extend hit@1/5/20 by, for the retrieved candidates, taking the MIN GT MCES-based similarity to the true one, to see how close we really got even when we missed the exact match.)
See the first 2 rows in the attached table, this for the aformentioned 005 and 008_2 experiments now apart from only showing hit rates gives also information about how well was the retrieval and unfortunately it looks like for at least 50% of the cases the top1 retrieved candidate with SIMBA has GT MCES of 20+. It only kinda gets more or less reasonable with 20 candidates, than indeed for the 50% of the cases we have amoung them candidate with GT MCES <=2, but at average best candidate is more like 5-6 GT MCES. So my hypotesis that we maybe retrieve bad, bad retrive a good analog seems to be disapproved.
Next up is what Geatan suggested above, sanity-check: for each test spectrum, rank the ICEBERG-predicted candidate spectra by just cosine similarity to the real spectrum directly (no SIMBA). I call it 3d. And this is the third row in the table. So cosine baseline beats SIMBA models by ~3.7x on hit@1. Well maybe there is a domain mismatch, because SIMBA was trained on real
spectra, and has never seen ICEBERG's spectra
during training, while cosine doesn't care what distribution the spectrum came from. Another might be a leakage of test to the ICEBERG. But the sertqin conlusion here for now is that cosine works on this benchmark way better than SIMBA does.
And one more point, it was also suggested by Gaetan, so the idea is MSG is narrow/metabolomics-focused, while PubChem candidates take much more of chemical space, so test-to-candidate pairs are typically far apart in a way test-to-test pairs rarely are. So goal was to check test-to-candidate pairs, compute true MCES and compare with SIMBA's predictions by MAE/spearman on those against test-to-test, and if test-to-candidate is much worse, SIMBA isn't generalizing beyong its training distribution. I called this point 3e
So see the second attached image. I has test to test pairs on the right and test to candidates on the left. Both show how GT MCES is distributed. And we actually have lower MCES in general for test to candidates pairs. And it seems like for this retrieval task it would be more relevant to be able to distinguish between MCES <20 ratrher than be able to predict accurately for higher MCES. And we try to do that by having weighted sampling based on GT MCES, but perhaps not enought. Next attached image is again 2 plots now it is GT vs predicted MCES again on both sets. Boxes instead of the hex beacuse distribution is super non uniform and would be hard to see anything. The numbers are
test-to-test MAE:  6.840, Spearman: 0.725
test-to-candidate MAE:  6.465, Spearman: 0.287
And performance on test-to-test is not ideal, especily unfortunately in the low MCES range, and the first box is actually pairs where we have same molecule different spectra and you see how median is 10, which is bad. And for test to candidqtes it is almost like 2 flat levels, one before 20 GT MCES and one after. So we can confirm distribution mismatch I guess, but test to test also not doing great.
Last attached plot as a first subplot has a distribution of MCES distance predicted for the test vs GT candidate (should be 0 idealy, but around 15 on average) and next subplot is for the SIMBA retrieved wrongly as the top1 pick, how big was the error on it, and it turns out to be underpredicted MCES at average by 7.35. Interesting how this distribution is bimodal, probably small peak at 0 is indeed hard cases, where gt candidate and next ones were really close and simba just mixed them but the high peak is SIMBA true errors
The second row of plots is just heatmap where now x axis is x-axis of first plot, y axis is axis of second plot and color is number of samples. So like the brightest point here shows that more or less the most common situation is for the actual candidate SIMBA predicts 10 instad of 0 and there is just some other candidate which simba underpredicts MCES by 14 and it gots the best score.

We had a discussion of all this with Gaetan (thanks a lot!) and seems like before we move on to the next points in our plan we want to make some more analysis. The main directions are checking on cosine similarity a bit more and get more understanding of OOD problem between what SIMBA is trained on and what it gets on test to test and test to candidate benchmarks. Beacuse it seems that we might have 3 contributing factors:
1 official splits, leading to very different mass distribution between train and test spectra
2 MSG is narrow, while PubChem candidates come from larger chemical space
3 SIMBA never seen ICEBERG generated spectra
So the list of checks for now goes like this (most-likely will be extended in the process)
8a same plot as second attached but with cosine similarity, so we can have idea how hard the task is for cosine, so for MCES on the test to candidate set  the distribution of min is very low and close to 0, meaning it is very easy actualy to confuse top1 and top2 candiadtes if your model is not perfect, but interesting to check will it be the case for max on cosine similarity plot, wil it be very close to 1, or not
8b just plot few ICEBERG spectra and real MSG spectra side by side to see how different they are
8c Rebuild 3rd atached plot separetely for like mass bellow 500 or vary it, which is much closer to the training mass distribution
8d heatmap where 1 axis is mass1 another mass2 and with color we might have average GT MCES or MAE between GT and simba or something like this, on test to test and test to candidate, to see how different performance depending on mass

And later it might bring us to the following additional popint to experements:
using not only MCES but also mass for the weighted sampling
use ICEBRG generated spectra during training
