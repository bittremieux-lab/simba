"""Precompute sparse binned-spectrum ("cosine baseline") intermediates for
item 8a — the no-SIMBA counterpart of simba_retrieval_iceberg.py's
--intermediates_dir output, consumed by ood_generalization_check.py and the
mces_*_plots.py scripts' --embedding_source cosine mode.

Bins every real test spectrum and every ICEBERG-predicted candidate
spectrum (see cosine_baseline_iceberg.bin_spectra: fixed m/z grid, sum
intensity per bin, sqrt-compress, L2-normalize — so a plain dot product
between two rows is their cosine similarity) and saves the results as
scipy-sparse .npz files + the parallel smiles/adduct JSON lists, in the same
directory layout as simba_retrieval_iceberg.py's --intermediates_dir
(test_mat.npz/candidate_mat.npz instead of test_embeddings.pt/
candidate_embeddings.pt; same test_smiles.json/test_adducts.json/
candidate_smiles.json/candidate_adducts.json).

A separate precompute step (rather than binning inline inside every
plotting script) because binning ~600k candidate spectra is the expensive
part and none of the 4 downstream scripts should have to repeat it.

Usage:
    uv run python tools/cosine_baseline_intermediates.py \\
        --mgf /path/to/MassSpecGym.mgf \\
        --candidate_tsv /path/to/candidates_test_official.tsv \\
        --iceberg_preds /path/to/preds.hdf5 \\
        --intermediates_dir /path/to/cosine_baseline_intermediates
"""

import argparse
import json
from pathlib import Path

import scipy.sparse as sp
from cosine_baseline_iceberg import bin_spectra
from simba_retrieval import load_spectra
from simba_retrieval_iceberg import (
    build_candidate_spectra,
    load_candidate_index,
    load_iceberg_spectra,
)


def run(
    mgf: str,
    candidate_tsv: str | list[str],
    iceberg_preds: str | list[str],
    intermediates_dir: str,
    split: str = "test",
    bin_width: float = 0.01,
    max_mz: float = 1100.0,
) -> None:
    out_dir = Path(intermediates_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {split}-fold real spectra from {mgf} ...")
    test_smiles, test_spectra = load_spectra(mgf, split)
    test_adducts = [s.adduct for s in test_spectra]
    print(f"  {len(test_smiles)} real test spectra")

    print(f"\nLoading candidate index from {candidate_tsv} ...")
    cand_index = load_candidate_index(candidate_tsv)
    all_cand_ids = cand_index["spec"].tolist()

    print(f"\nLoading ICEBERG-predicted spectra from {iceberg_preds} ...")
    iceberg_specs = load_iceberg_spectra(iceberg_preds, all_cand_ids)
    print(
        f"  {len(iceberg_specs)} / {len(all_cand_ids)} candidates have a predicted spectrum"
    )

    print("\nBuilding candidate SpectrumExt objects ...")
    cand_smiles, cand_spectra = build_candidate_spectra(cand_index, iceberg_specs)
    cand_adducts = [s.adduct for s in cand_spectra]
    print(f"  {len(cand_spectra)} candidate spectra ready")

    print(f"\nBinning spectra (bin_width={bin_width} Da, max_mz={max_mz} Da) ...")
    test_mat = bin_spectra(test_spectra, bin_width, max_mz)
    cand_mat = bin_spectra(cand_spectra, bin_width, max_mz)
    print(f"  test matrix: {test_mat.shape}, nnz={test_mat.nnz:,}")
    print(f"  candidate matrix: {cand_mat.shape}, nnz={cand_mat.nnz:,}")

    sp.save_npz(out_dir / "test_mat.npz", test_mat)
    sp.save_npz(out_dir / "candidate_mat.npz", cand_mat)
    (out_dir / "test_smiles.json").write_text(json.dumps(test_smiles))
    (out_dir / "test_adducts.json").write_text(json.dumps(test_adducts))
    (out_dir / "candidate_smiles.json").write_text(json.dumps(cand_smiles))
    (out_dir / "candidate_adducts.json").write_text(json.dumps(cand_adducts))
    print(f"\nIntermediates saved to {out_dir}/")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mgf", required=True, help="MassSpecGym MGF file")
    p.add_argument(
        "--candidate_tsv",
        required=True,
        nargs="+",
        help="ICEBERG candidate TSV(s) (smiles/ionization/precursor) -- one path, "
        "or several (e.g. the original plus a delta file), matched 1:1 by "
        "position with --iceberg_preds",
    )
    p.add_argument(
        "--iceberg_preds",
        required=True,
        nargs="+",
        help="ICEBERG predictions HDF5(s) -- one path, or several to merge",
    )
    p.add_argument(
        "--intermediates_dir",
        required=True,
        help="Directory to save the binned-spectrum sparse matrices + SMILES/adducts",
    )
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--bin_width", type=float, default=0.01, help="m/z bin width (Da)")
    p.add_argument(
        "--max_mz",
        type=float,
        default=1100.0,
        help="Max m/z covered by the binning grid (max precursor mass in this "
        "candidate set is ~997 Da)",
    )
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
