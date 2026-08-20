"""Raw spectral cosine similarity for every validation pair (classical,
non-learned baseline to compare against SIMBA's own predicted MCES).

Computed once per val set, not once per experiment or per training step:
every experiment sharing this preprocessing dir builds its validation pairs
the same deterministic way (fixed pair list, `shuffle=False` DataLoader, and
`CustomDatasetMultitasking.__getitem__`'s val branch always resolves the
same molecule to the same spectrum index), so a pair's (spec_idx_0,
spec_idx_1) identity -- and therefore its cosine similarity -- is identical
across every run built from this val set (confirmed directly: it comes from
`molecule_pairs_val.original_spectra`, an order-preserving copy of the same
list every experiment's dataset builder iterates).

Critically, this uses each spectrum's RAW peaks (as loaded from the MGF,
before SIMBA's own Preprocessor runs: precursor-peak removal, intensity
filtering, max-peak truncation) -- not SIMBA's preprocessed model input --
matching how the existing tools/cosine_baseline_*.py scripts already treat
"cosine baseline" throughout this project (see bin_spectra's docstring in
tools/cosine_baseline_iceberg.py). Reuses that exact function rather than
reimplementing binning, so this is consistent with prior baseline work.

`spec_idx_0`/`spec_idx_1` (which specific spectrum was used for each side of
a pair -- not just which molecule, since a molecule can have several
spectra) are read from an existing experiment's consolidated parquet purely
as a source of pair identity, not its predictions -- any experiment sharing
this val set works interchangeably as that "reference".

Usage:
    uv run python tools/compute_val_cosine.py \\
        --preprocessing_dir /path/to/preprocessing_gaetan_split_max_lb_hdf5_v2 \\
        --mgf_path /path/to/MassSpecGym.mgf \\
        --reference_exp_dir /path/to/experiments/training/011_..._1gpu \\
        --val_name val
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from cosine_baseline_iceberg import bin_spectra
from hydra import compose, initialize_config_dir

from simba.utils.config_utils import get_config_path
from simba.workflows.training import load_dataset


DEFAULT_PREPRO_DIR = (
    "/sofia/projects/2026_053/simba_project/data/massspecgym/"
    "preprocessing_gaetan_split_max_lb_hdf5_v2"
)
DEFAULT_MGF = (
    "/sofia/projects/2026_053/simba_project/data/massspecgym/"
    "data/auxiliary/MassSpecGym.mgf"
)
DEFAULT_REFERENCE_EXP_DIR = (
    "/sofia/projects/2026_053/simba_project/experiments/training/"
    "011_msg_gaetan_split_v2_theoretical_precursor_mist_cf_resampling_1gpu"
)
# Same convention as tools/cosine_baseline_iceberg.py / cosine_retrieval.py /
# cosine_baseline_intermediates.py -- kept identical so "cosine baseline"
# means the same thing everywhere in this project.
BIN_WIDTH = 0.01
MAX_MZ = 1100.0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preprocessing_dir", default=DEFAULT_PREPRO_DIR)
    parser.add_argument("--mgf_path", default=DEFAULT_MGF)
    parser.add_argument("--val_name", default="val", choices=["val", "official"])
    parser.add_argument(
        "--reference_exp_dir",
        default=DEFAULT_REFERENCE_EXP_DIR,
        help="Any experiment dir sharing this val set -- only its "
        "spec_idx_0/1 + mol_idx_0/1 pair identity is used, not its "
        "predictions.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to {preprocessing_dir}/val_cosine_{val_name}.parquet "
        "-- lives with the val set itself since it's experiment-independent.",
    )
    args = parser.parse_args()

    reference_exp_dir = Path(args.reference_exp_dir)
    consolidated_path = (
        reference_exp_dir / f"val_pairs_{args.val_name}_consolidated.parquet"
    )
    print(f"Reading pair identity from {consolidated_path} ...")
    pair_ids = pd.read_parquet(
        consolidated_path,
        columns=["mol_idx_0", "mol_idx_1", "spec_idx_0", "spec_idx_1"],
    )
    print(f"{len(pair_ids):,} validation pairs")

    overrides = [
        f"paths.preprocessing_dir={args.preprocessing_dir}",
        f"paths.preprocessing_dir_train={args.preprocessing_dir}",
        "paths.preprocessing_pickle_file=mapping.pkl",
        f"paths.mgf_path={args.mgf_path}",
    ]
    config_path = get_config_path()
    with initialize_config_dir(
        config_dir=str(config_path.absolute()), version_base=None
    ):
        cfg = compose(config_name="config", overrides=overrides)

    print("Loading dataset (raw validation spectra) ...")
    _, molecule_pairs_val, molecule_pairs_val_official, _, _ = load_dataset(cfg)
    val_by_name = {"val": molecule_pairs_val, "official": molecule_pairs_val_official}
    molecule_pairs = val_by_name.get(args.val_name)
    if molecule_pairs is None:
        raise ValueError(
            f"No val split named {args.val_name!r} in this preprocessing dir "
            f"(val_official is {'present' if molecule_pairs_val_official else 'absent'})"
        )

    spectra = molecule_pairs.original_spectra
    print(
        f"Binning {len(spectra):,} raw validation spectra "
        f"(bin_width={BIN_WIDTH} Da, max_mz={MAX_MZ} Da) ..."
    )
    mat = bin_spectra(spectra, BIN_WIDTH, MAX_MZ)

    spec_idx_0 = pair_ids["spec_idx_0"].to_numpy()
    spec_idx_1 = pair_ids["spec_idx_1"].to_numpy()
    n_spectra = mat.shape[0]
    if spec_idx_0.max() >= n_spectra or spec_idx_1.max() >= n_spectra:
        raise ValueError(
            f"spec_idx out of range for {n_spectra:,} loaded spectra -- "
            f"reference_exp_dir's val set doesn't match preprocessing_dir/mgf_path"
        )

    print(f"Computing cosine similarity for {len(pair_ids):,} pairs ...")
    mat_0 = mat[spec_idx_0]
    mat_1 = mat[spec_idx_1]
    cosine = np.asarray(mat_0.multiply(mat_1).sum(axis=1)).flatten()

    out = pd.DataFrame(
        {
            "mol_idx_0": pair_ids["mol_idx_0"].to_numpy(),
            "mol_idx_1": pair_ids["mol_idx_1"].to_numpy(),
            "spec_idx_0": spec_idx_0,
            "spec_idx_1": spec_idx_1,
            "cosine": cosine,
        }
    )
    output = Path(
        args.output or f"{args.preprocessing_dir}/val_cosine_{args.val_name}.parquet"
    )
    out.to_parquet(output, index=False)
    print(f"Saved {len(out):,} rows to {output}")


if __name__ == "__main__":
    main()
