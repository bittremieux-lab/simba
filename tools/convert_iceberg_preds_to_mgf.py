"""Convert ICEBERG's preds.hdf5 (from build_train_augmentation_tsv.py +
predict_smis.py, see NOTES_014_2_ICEBERG_AUGMENTATION.md) into an MGF of
synthetic train spectra, one per (molecule, CE) pair, consumable by
simba/workflows/training.py::load_dataset's iceberg_mgf_path option.

Reuses the same (mz, intensity) extraction logic as
tools/simba_retrieval_iceberg.py::load_iceberg_spectra (arr[:, 0] > 0 masks
out sparse_k's zero-padding) -- generalized here to iterate over ALL CE
leaves per molecule instead of just one fixed candidate spectrum.

Usage:
    uv run python tools/convert_iceberg_preds_to_mgf.py \\
        --preds_hdf5 ../ICEBERG/results/train_augmentation/preds.hdf5 \\
        --labels_tsv ../ICEBERG/data/train_augmentation_labels.tsv \\
        --output_mgf ../data/analog_discovery/iceberg_train_augmentation/synthetic_train.mgf
"""

import argparse

import h5py
import pandas as pd


def run(preds_hdf5: str, labels_tsv: str, output_mgf: str):
    print(f"Loading labels from {labels_tsv} ...")
    labels = pd.read_csv(labels_tsv, sep="\t").set_index("spec")

    print(f"Loading predictions from {preds_hdf5} ...")
    n_written = 0
    n_skipped_empty = 0
    with h5py.File(preds_hdf5, "r") as f, open(output_mgf, "w") as out:
        manifest = f["__predspec_manifest__"]
        for name, leaf, ce_key in zip(
            manifest["name"][:], manifest["leaf_path"][:], manifest["ce_key"][:]
        ):
            name = name.decode()
            leaf = leaf.decode()
            ce_key = ce_key.decode()
            spec_id = name.removeprefix("pred_")
            if spec_id not in labels.index:
                continue
            row = labels.loc[spec_id]

            arr = f[leaf]["f"][:]
            mask = arr[:, 0] > 0
            mz = arr[mask, 0]
            intensity = arr[mask, 1]
            if len(mz) == 0:
                n_skipped_empty += 1
                continue

            ce_val = ce_key.removeprefix("collision ")

            out.write("BEGIN IONS\n")
            out.write(f"SMILES={row['smiles']}\n")
            out.write(f"ADDUCT={row['ionization']}\n")
            out.write(f"PEPMASS={row['precursor']}\n")
            out.write("CHARGE=1\n")
            out.write("IONMODE=positive\n")
            out.write("FOLD=train\n")
            out.write(f"CE={ce_val}\n")
            out.write("SOURCE=iceberg_synthetic\n")
            for m, i in zip(mz, intensity):
                out.write(f"{m} {i}\n")
            out.write("END IONS\n")
            n_written += 1

    if n_skipped_empty:
        print(f"  WARNING: {n_skipped_empty} predictions had zero peaks, skipped")
    print(f"Wrote {n_written} synthetic spectra to {output_mgf}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preds_hdf5", required=True)
    p.add_argument("--labels_tsv", required=True)
    p.add_argument("--output_mgf", required=True)
    args = p.parse_args()
    run(args.preds_hdf5, args.labels_tsv, args.output_mgf)


if __name__ == "__main__":
    main()
