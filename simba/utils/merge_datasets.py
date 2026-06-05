import dill
import numpy as np
import pandas as pd
from pathlib import Path
import glob


SPLITS = ["train", "val", "test"]
PAIR_PREFIX = "ed_mces_indexes_tani_incremental"


def count_mgf_spectra(mgf_path):
    n = 0
    with open(mgf_path, "r", errors="ignore") as f:
        for line in f:
            if line.strip().upper() == "BEGIN IONS":
                n += 1
    return n


def concat_mgf(mgf_a, mgf_b, output_mgf):
    with open(output_mgf, "w") as out:
        for mgf in [mgf_a, mgf_b]:
            with open(mgf, "r", errors="ignore") as f:
                out.write(f.read().strip())
                out.write("\n\n")


def find_pair_files(folder, split):
    pattern = str(Path(folder) / f"{PAIR_PREFIX}_{split}*.npy")
    return sorted(glob.glob(pattern))


def merge_pair_files(folder_a, folder_b, split, unique_offset_b, output_folder):
    files_a = find_pair_files(folder_a, split)
    files_b = find_pair_files(folder_b, split)

    if len(files_a) == 0:
        raise FileNotFoundError(f"No pair files found for A split={split}")
    if len(files_b) == 0:
        raise FileNotFoundError(f"No pair files found for B split={split}")

    arrs = []

    for f in files_a:
        arrs.append(np.load(f))

    for f in files_b:
        arr = np.load(f).copy()
        arr[:, 0] += unique_offset_b
        arr[:, 1] += unique_offset_b
        arrs.append(arr)

    merged = np.vstack(arrs)

    out_path = Path(output_folder) / f"{PAIR_PREFIX}_{split}.npy"
    np.save(out_path, merged)

    return merged


def merge_split(ds_a, ds_b, split, mgf_offset_b):
    df_key = f"df_smiles_{split}"
    idx_key = f"spectrum_indexes_{split}"

    if df_key not in ds_a or df_key not in ds_b:
        return None, None, None

    df_a = ds_a[df_key].copy()
    df_b = ds_b[df_key].copy()

    spectrum_list_offset_b = len(ds_a[idx_key])
    unique_offset_b = len(df_a)

    # df_smiles indexes point into spectrum_indexes_split, not raw MGF
    df_b["indexes"] = df_b["indexes"].apply(
        lambda xs: [int(x) + spectrum_list_offset_b for x in xs]
    )

    df_merged = pd.concat([df_a, df_b], ignore_index=True)

    spectrum_indexes_a = [int(x) for x in ds_a[idx_key]]
    spectrum_indexes_b = [int(x) + mgf_offset_b for x in ds_b[idx_key]]

    spectrum_indexes_merged = spectrum_indexes_a + spectrum_indexes_b

    return df_merged, spectrum_indexes_merged, unique_offset_b


def validate_split(dataset, pair_distances, split):
    df = dataset[f"df_smiles_{split}"]
    spectrum_indexes = dataset[f"spectrum_indexes_{split}"]

    n_spectra = len(spectrum_indexes)
    n_unique = len(df)

    for i, xs in enumerate(df["indexes"]):
        for x in xs:
            if x < 0 or x >= n_spectra:
                raise ValueError(
                    f"[{split}] invalid df_smiles index {x} at row {i}; "
                    f"valid range: 0–{n_spectra - 1}"
                )

    pair_idx = pair_distances[:, :2].astype(int)

    if pair_idx.min() < 0 or pair_idx.max() >= n_unique:
        raise ValueError(
            f"[{split}] invalid pair index range "
            f"{pair_idx.min()}–{pair_idx.max()}; valid: 0–{n_unique - 1}"
        )

    print(
        f"[{split}] OK | molecules={n_unique}, spectra={n_spectra}, pairs={len(pair_distances)}"
    )


def merge_lightweight_datasets_all_splits(
    dataset_a_pkl,
    dataset_b_pkl,
    preprocessing_dir_a,
    preprocessing_dir_b,
    output_dir,
    output_dataset_name="preprocessing_merged.pkl",
    output_mgf_name="merged_spectra.mgf",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_a_pkl, "rb") as f:
        ds_a = dill.load(f)

    with open(dataset_b_pkl, "rb") as f:
        ds_b = dill.load(f)

    assert ds_a.get("format_version") == "lightweight"
    assert ds_b.get("format_version") == "lightweight"

    mgf_a = Path(ds_a["mgf_path"])
    mgf_b = Path(ds_b["mgf_path"])
    output_mgf = output_dir / output_mgf_name

    mgf_offset_b = count_mgf_spectra(mgf_a)
    concat_mgf(mgf_a, mgf_b, output_mgf)

    merged = dict(ds_a)
    merged["mgf_path"] = str(output_mgf)
    merged["merged_from"] = [str(dataset_a_pkl), str(dataset_b_pkl)]

    offsets = {}

    for split in SPLITS:
        df_merged, spectrum_indexes_merged, unique_offset_b = merge_split(
            ds_a, ds_b, split, mgf_offset_b
        )

        if df_merged is None:
            print(f"[{split}] skipped; split not found in both datasets")
            continue

        merged[f"df_smiles_{split}"] = df_merged
        merged[f"spectrum_indexes_{split}"] = spectrum_indexes_merged

        pair_distances = merge_pair_files(
            preprocessing_dir_a,
            preprocessing_dir_b,
            split,
            unique_offset_b,
            output_dir,
        )

        validate_split(merged, pair_distances, split)

        offsets[split] = {
            "mgf_offset_b": mgf_offset_b,
            "spectrum_list_offset_b": len(ds_a[f"spectrum_indexes_{split}"]),
            "unique_offset_b": unique_offset_b,
        }

    merged["merge_offsets"] = offsets

    output_dataset_pkl = output_dir / output_dataset_name
    with open(output_dataset_pkl, "wb") as f:
        dill.dump(merged, f)

    return merged


merge_lightweight_datasets_all_splits(
    dataset_a_pkl="/data/simba_files/distance_files/tfs_ms2_ref/mapping_unique_smiles.pkl",
    dataset_b_pkl="/data/simba_files/distance_files/tfs_ms2_auto/mapping_unique_smiles.pkl",
    preprocessing_dir_a="/data/simba_files/distance_files/tfs_ms2_ref/",
    preprocessing_dir_b="/data/simba_files/distance_files/tfs_ms2_auto/",
    output_dir="/data/simba_files/tfs_ms2_merged_ref_auto/",
)