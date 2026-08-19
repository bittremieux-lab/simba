"""Dry-run diagnostic for the MCES-bucket + mass-tier weighted training
sampler (simba/workflows/training.py's prepare_data, use_mces_sampling
branch).

Builds the REAL train_sampler exactly as `simba train` would (same
load_dataset + prepare_data call, same weights), then simulates drawing from
it (same weighted-with-replacement logic CustomWeightedRandomSampler.__iter__
uses, just a smaller sample size so it's fast) to answer a concrete question:
even though a bucket's aggregate *draw share* can look reasonable (e.g. ~33%
of draws landing in MCES>10 buckets combined), how many *distinct* pairs is
that share spread over, and how many times does each pair in a small,
heavily-upweighted group (self-pairs; a fine MCES bucket's bottom mass-diff
decile) get repeated within that many draws, compared to a big MCES>10
bucket? A group that's a large fraction of *draws* but a tiny, heavily
repeated set of *distinct examples* behaves very differently in training
than the same fraction spread over millions of distinct examples, even
though both show the same aggregate bucket share.

Usage:
    uv run python tools/dry_test_resampling_weights.py --n_draws 2000000
"""

import argparse

import numpy as np
from hydra import compose, initialize_config_dir

from simba.core.chemistry.chem_utils import mass_lookup_from_df_smiles
from simba.utils.config_utils import get_config_path
from simba.workflows.training import (
    MASS_TIER_LOW_SHARE,
    MASS_TIER_QUANTILE,
    MCES_SAMPLING_BIN_LABELS,
    MCES_SAMPLING_BUCKET_MULTIPLIERS,
    MCES_SAMPLING_EDGES,
    load_dataset,
    prepare_data,
)


DEFAULT_PREPRO_DIR = (
    "/sofia/projects/2026_053/simba_project/data/massspecgym/"
    "preprocessing_gaetan_split_max_lb_hdf5_v2"
)
DEFAULT_MGF = (
    "/sofia/projects/2026_053/simba_project/data/massspecgym/"
    "data/auxiliary/MassSpecGym.mgf"
)


def report_group(label, bucket_of_draw, draws, n_draws, in_group_mask, n_unique_total):
    mask = in_group_mask[draws]
    n_here = int(mask.sum())
    vals = draws[mask]
    n_unique_hit = len(np.unique(vals)) if n_here else 0
    avg_repeats = n_here / n_unique_hit if n_unique_hit else 0.0
    print(
        f"  {label:>28s}: {n_unique_total:>10,} unique pairs total | "
        f"{n_here:>9,} draws ({100 * n_here / n_draws:5.2f}% of {n_draws:,}) | "
        f"{n_unique_hit:>9,} distinct pairs hit | "
        f"avg {avg_repeats:6.1f} repeats/pair"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preprocessing_dir", default=DEFAULT_PREPRO_DIR)
    parser.add_argument("--mgf_path", default=DEFAULT_MGF)
    parser.add_argument("--n_draws", type=int, default=2_000_000)
    args = parser.parse_args()

    overrides = [
        f"paths.preprocessing_dir={args.preprocessing_dir}",
        f"paths.preprocessing_dir_train={args.preprocessing_dir}",
        "paths.preprocessing_pickle_file=mapping.pkl",
        f"paths.mgf_path={args.mgf_path}",
        "sampling.add_identity_pairs=true",
        "sampling.use_resampling=true",
        "model.tasks.edit_distance.enabled=false",
    ]
    config_path = get_config_path()
    with initialize_config_dir(
        config_dir=str(config_path.absolute()), version_base=None
    ):
        cfg = compose(config_name="config", overrides=overrides)

    print("Loading dataset ...")
    (
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_val_official,
        molecule_pairs_test,
        uniformed_molecule_pairs_test,
    ) = load_dataset(cfg)

    print("Running prepare_data (builds the REAL training sampler) ...")
    (
        dataset_train,
        train_sampler,
        dataset_val,
        val_sampler,
        dataset_val_official,
        val_official_sampler,
        weights_ed,
        bins_ed,
    ) = prepare_data(
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_test,
        uniformed_molecule_pairs_test,
        cfg,
        molecule_pairs_val_official=molecule_pairs_val_official,
    )
    assert train_sampler is not None, "sampling.use_resampling produced no sampler"

    weights_tr = train_sampler.weights.numpy().astype(np.float64)
    p = weights_tr / weights_tr.sum()
    n_pairs = len(p)

    mces_raw = (1.0 - molecule_pairs_train.extra_distances) * 40.0
    n_buckets = len(MCES_SAMPLING_BIN_LABELS)
    bin_idx = np.clip(
        np.searchsorted(MCES_SAMPLING_EDGES, mces_raw).astype(int), 0, n_buckets - 1
    )

    print(
        f"\nBucket multipliers in effect: {MCES_SAMPLING_BUCKET_MULTIPLIERS.tolist()}"
    )
    print("\n=== Per-bucket weight share (theory) vs pair count ===")
    bucket_weight_share = np.bincount(bin_idx, weights=p, minlength=n_buckets)
    bucket_counts = np.bincount(bin_idx, minlength=n_buckets)
    for i, label in enumerate(MCES_SAMPLING_BIN_LABELS):
        print(
            f"  {label:>16s}: weight_share={bucket_weight_share[i] * 100:6.2f}%  "
            f"n_pairs={bucket_counts[i]:>10,}"
        )
    print(
        f"\n  Combined MCES>10 (buckets 5-{n_buckets - 1}) weight share: "
        f"{bucket_weight_share[5:].sum() * 100:.2f}%"
    )

    # Identify bucket 1's ((0,2.5]) low mass-difference tier, same rule as
    # training.py's mass-tier reweighting, for a direct apples-to-apples group.
    mol_mass = mass_lookup_from_df_smiles(molecule_pairs_train.df_smiles)
    mol_idx_0 = molecule_pairs_train.pair_distances[:, 0].astype(int)
    mol_idx_1 = molecule_pairs_train.pair_distances[:, 1].astype(int)
    mass_diff = np.abs(mol_mass[mol_idx_0] - mol_mass[mol_idx_1])

    bucket1_mask = bin_idx == 1
    bucket1_diffs = mass_diff[bucket1_mask]
    q_low = np.nanquantile(bucket1_diffs, MASS_TIER_QUANTILE)
    bucket1_low_global_mask = bucket1_mask & (mass_diff <= q_low)
    print(
        f"\nBucket (0,2.5] mass-tier: q{MASS_TIER_QUANTILE} = {q_low:.2f} Da, "
        f"{int(bucket1_low_global_mask.sum()):,} pairs at/below it "
        f"(share of bucket 1's own weight going to this group: "
        f"{MASS_TIER_LOW_SHARE * 100:.0f}%)"
    )

    print(
        f"\nSimulating {args.n_draws:,} draws from the REAL sampler weights (seed=0) ..."
    )
    rng = np.random.default_rng(0)
    draws = rng.choice(n_pairs, size=args.n_draws, p=p, replace=True)

    print("\n=== Per-bucket empirical draw share (should match theory above) ===")
    draw_bucket = bin_idx[draws]
    empirical_share = np.bincount(draw_bucket, minlength=n_buckets) / args.n_draws
    for i, label in enumerate(MCES_SAMPLING_BIN_LABELS):
        print(f"  {label:>16s}: empirical={empirical_share[i] * 100:6.2f}%")

    print("\n=== Distinct-pair / repetition analysis ===")
    unique_draws, counts = np.unique(draws, return_counts=True)
    print(f"Total draws: {args.n_draws:,}")
    print(
        f"Distinct pair indices touched overall: {len(unique_draws):,} "
        f"({100 * len(unique_draws) / args.n_draws:.2f}% of draws are 'first sightings')"
    )
    print(f"Max repeats for any single pair: {counts.max()}")

    print()
    self_mask = bin_idx == 0
    report_group(
        "self (MCES=0)",
        bin_idx,
        draws,
        args.n_draws,
        self_mask,
        int(self_mask.sum()),
    )
    report_group(
        "(0,2.5] bucket, all",
        bin_idx,
        draws,
        args.n_draws,
        bucket1_mask,
        int(bucket1_mask.sum()),
    )
    report_group(
        "(0,2.5] low-mass tier",
        bin_idx,
        draws,
        args.n_draws,
        bucket1_low_global_mask,
        int(bucket1_low_global_mask.sum()),
    )
    big_bucket_idx = n_buckets - 1
    big_mask = bin_idx == big_bucket_idx
    report_group(
        MCES_SAMPLING_BIN_LABELS[big_bucket_idx] + " (big MCES>10 bucket)",
        bin_idx,
        draws,
        args.n_draws,
        big_mask,
        int(big_mask.sum()),
    )
    mces_gt10_mask = bin_idx >= 5
    report_group(
        "all MCES>10 combined",
        bin_idx,
        draws,
        args.n_draws,
        mces_gt10_mask,
        int(mces_gt10_mask.sum()),
    )


if __name__ == "__main__":
    main()
