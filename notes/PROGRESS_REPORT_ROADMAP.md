7a: Preprocessing audit
Checked spectrum preprocessing pipeline and confirmed train and validation use the identical steps (precursor-peak removal → intensity floor/top-N cap → sqrt+L2 normalize). Found that my retrieval never removed the precursor peak, fixed. The fix gave a very small improvement on the same checkpoint hit@1 5.58% -> 5.83% (+4.5%) and less than 1% for hit@5 and hit@20.
Main result: preprocessing is not the problem.

1: Train SIMBA on Gaetan's split
Built a Gaetan-split based train and val data and trained the model (experiment 005, using here my internal experiment numbers) and a matching official-split counterpart (experiment 006). 006 has both a scaffold and an official validation set, 005 only its own. On validation 005 sits between 006's validations, as I would expect since scaffold split is easier than Gaetan's splits but the official split is harder:  4.28 (006 scaffold) vs MCES MAE 4.85 (005) vs 7.06 (006 official), spearman  0.869 (006 scaffold) vs 0.645 (006 official) vs 0.834 (005). On retrieval it flips: 005 beats 006, hit@1 6.5% vs 5.7%, hit@5 16.7% vs 16.1%, hit@20 35.1% vs 34.1%.
Main result: 005 looks better on retrieval (still kinda low), but there's a possibility of leakage (Gaetan's split doesn't align with the official one, so some official-test or candidates molecules may sit in 005's training set) But I did not check that yet.

3: Retrieval without using train (ICEBERG)
Set up ICEBERG (coleygroup/ms-pred, pretrained MassSpecGym weights) and generated a spectrum for every one of our ~600k official-test retrieval candidates, all at one fixed collision energy of 35 (median across MSG test-fold spectra). Built a new scoring script that ranks candidates by SIMBA similarity between the real test spectrum and each candidate's predicted spectrum. For 005, for example, the jump is large: hit@1 6.5% -> 10.1%, hit@5 16.7% -> 25.9%, hit@20 35.1% -> 49.3%.
Main result: this is a real, large improvement over nearest-train-neighbor retrieval, not just for 005, again leakage is possible now with ICEBERG if it was trained on candidates or official test, but again I did not check that.

5: Explore alternative heads/objectives
The transformer encoder produces embedding per spectrum, call them emb0/emb1. What we're varying is only the steps that turn (emb0, emb1) into the MCES score. Trained 5 such variants on the same official-split data (experiments 008_1..5):
- cosine_relu (current SIMBA): emb0 and emb1 each go through Linear(256,256) -> Dropout -> ReLU -> Linear(256,256) -> ReLU, then cosine similarity between the two results.
- cosine_no_head: cosine similarity directly on emb0/emb1, no projection at all.
- cosine_linear_head: same 2-layer projection as cosine_relu (Linear -> Dropout -> ReLU -> Linear) but drop the final ReLU, then cosine similarity.
- distance_linear_head: same projection as cosine_linear_head (no final ReLU), then L2-normalize both projected vectors, take their Euclidean distance, map through exp(-dist) to get a similarity.
- distance_no_head: L2-normalize emb0/emb1 directly (no projection), Euclidean distance, exp(-dist).
cosine_no_head wins on every metric (see table). On retrieval it also wins 005 on hit@1 (10.3% vs 10.1%) and ties it on hit@5 (25.9% vs 25.9%) and unlike 005 has no leakage risk from splits.
Main result: dropping the extra projection layers helps and shows good results on validation and retrieval.

Updated plan
Where things stand against the original list from @wout / @Gaetan De Waele, done items struck through, new items in bold:

1. ~~Train SIMBA on Gaetan's MSG split (not the official one)~~ https://github.com/bittremieux-lab/spectrawl/blob/main/spectrawl/splits/split_massspecgym.tsv
2. Add more training data: NIST20 and Spectraverse, find molecules with > 10 MCES from every official val and test molecules to enrich training molecules.
3. ~~Retrieval evaluation, without using the train.~~ Currently, retrieval works by finding the closest train spectra to a test spectrum via SIMBA. We can instead:
   - ~~Predict each candidate's molecule spectrum with ICEBERG~~
   - ~~Rank those predicted spectra against the test spectrum using SIMBA's MCES prediction~~
   - **Add MCES to the selected candidate to the retrieval scoring**
4. The weighted sampler currently targets a uniform MCES distribution in training (with a bit more weight for MCES <10). Can check whether specific MCES ranges should have more weight than others.
5. Explore alternative heads and/or objectives.
   - ~~5a: Regression~~
   - 5b: Conditional-ordinal-classification head on binned MCES instead of a regression
6. Investigate a contrastive loss paradigm.

Smaller things, still might be worth doing:
- ~~Preprocessing is currently spread across several modules, check it end-to-end generally and to confirm it is identical for train and validation.~~
- Encode precursor mass properly: switch to MIST-CF-style resampling by using the spectrawl implementation (lift it into a public method in metabo-depthcharge, then wire it into SIMBA's precursor-mass processing).
- Sinusoidal m/z embeddings -> RoPE.
- CLS-token pooling -> attention pooling.

Next up: 5b (already running) -> 7b, precursor mass encoding (well-specified now, and addresses a shortcut-learning issue that likely affects every experiment so far, not just one head or split) -> 4, weighted sampler tuning (cheap to test, can slot in alongside 7b) -> 2, more training data (biggest potential lift, but also the biggest lift in effort). 6 (contrastive loss) intentionally parked, not near-term.
