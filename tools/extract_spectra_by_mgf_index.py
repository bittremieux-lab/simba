"""Extract specific subsets of spectra from the raw MassSpecGym MGF, by
global spectrum index (as saved in a prepare_msg_*_max_lb_hdf5.py-style
mapping.pkl's spectrum_indexes_* list -- confirmed by reading
prepare_msg_official_split_max_lb_hdf5.py directly: these are positions in
matchms.importing.load_from_mgf(MGF_PATH)'s own enumeration order, captured
before any validity filtering), into one standalone mini-MGF with the FOLD=
field rewritten per group -- so existing scripts (simba_retrieval.py,
simba_retrieval_iceberg.py, cosine_baseline_iceberg.py, cosine_retrieval.py)
can be pointed at it via --split <label> --mgf <output_mgf> with zero
changes to any of them.

Supports multiple groups (e.g. a split's own train fold labeled "train" AND
its own val/test query fold labeled "test", written into the SAME output
file) in one pass over the source MGF -- needed for
tools/simba_retrieval.py / tools/cosine_retrieval.py's train-NN-transfer
rows, which load both "train" and the query split from the same --mgf path
in one invocation.

Needed because some splits (scaffold-val, carved out of the official MGF's
train fold; Gaetan's own train/val/test) don't correspond to a raw MGF
FOLD= value at all -- only mapping.pkl records which raw spectra belong to
them.

SELF-VERIFICATION (important: a silent index misalignment here would
silently corrupt everything downstream, including expensive ICEBERG
generation): counts, while writing, the canonical SMILES and per-molecule
spectrum count actually extracted per group, and compares that against the
mapping.pkl dataframe's own df_smiles_<fold>['canon_smiles' /
'number_indexes'] -- refuses to report success unless every group matches
exactly. This is a manual text scan (fast -- avoids matchms's slow
per-spectrum object construction for a ~230k-spectrum file), matching the
technique already used in plot_confusion_matrix_examples.py.

Usage:
    uv run python tools/extract_spectra_by_mgf_index.py \\
        --mgf /path/to/MassSpecGym.mgf \\
        --mapping_pkl /path/to/preprocessing_official_split_max_lb_hdf5/mapping.pkl \\
        --group spectrum_indexes_train df_smiles_train train \\
        --group spectrum_indexes_val df_smiles_val test \\
        --output_mgf /path/to/scaffold_val_combined.mgf
"""

import argparse
import pickle
from collections import Counter

from rdkit import Chem


def canonicalize(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def run(
    mgf: str,
    mapping_pkl: str,
    groups: list[tuple[str, str, str]],
    output_mgf: str,
) -> None:
    print(f"Loading {mapping_pkl} ...")
    with open(mapping_pkl, "rb") as fh:
        mapping = pickle.load(fh)

    index_to_label: dict[int, str] = {}
    group_defs = []
    for index_key, df_key, label in groups:
        target_indices = set(mapping[index_key])
        df = mapping[df_key]
        overlap = set(index_to_label) & target_indices
        if overlap:
            raise SystemExit(
                f"Index overlap between groups at label={label!r} "
                f"({len(overlap)} indices) -- ambiguous, refusing to proceed"
            )
        for idx in target_indices:
            index_to_label[idx] = label
        group_defs.append((label, index_key, df_key, target_indices, df))
        print(
            f"  [{label}] {len(target_indices)} target spectra ({index_key}), "
            f"{len(df)} molecules ({df_key})"
        )

    print(f"\nScanning {mgf} ...")
    n_written_by_label: Counter[str] = Counter()
    extracted_counts_by_label: dict[str, Counter[str]] = {
        label: Counter() for label, *_ in group_defs
    }
    global_idx = -1
    in_block = False
    current_label: str | None = None
    block_lines: list[str] = []
    current_smiles: str | None = None

    with open(mgf) as src, open(output_mgf, "w") as dst:
        for line in src:
            stripped = line.rstrip("\n")
            if stripped == "BEGIN IONS":
                global_idx += 1
                current_label = index_to_label.get(global_idx)
                in_block = current_label is not None
                if in_block:
                    block_lines = [line]
                    current_smiles = None
                continue
            if stripped == "END IONS":
                if in_block:
                    block_lines.append(line)
                    dst.writelines(block_lines)
                    n_written_by_label[current_label] += 1
                    if current_smiles is not None:
                        extracted_counts_by_label[current_label][
                            canonicalize(current_smiles)
                        ] += 1
                in_block = False
                block_lines, current_smiles, current_label = [], None, None
                continue
            if in_block:
                if stripped.startswith("FOLD="):
                    block_lines.append(f"FOLD={current_label}\n")
                elif stripped.startswith("SMILES="):
                    current_smiles = stripped[len("SMILES=") :]
                    block_lines.append(line)
                else:
                    block_lines.append(line)

    print(f"Wrote {sum(n_written_by_label.values())} spectra total to {output_mgf}")

    print("\nSelf-verification against mapping.pkl (per group) ...")
    all_ok = True
    for label, _index_key, _df_key, target_indices, df in group_defs:
        n_written = n_written_by_label[label]
        extracted_counts = extracted_counts_by_label[label]
        expected_counts = dict(
            zip(df["canon_smiles"].tolist(), df["number_indexes"].tolist())
        )
        ok = True
        if n_written != len(target_indices):
            ok = False
            print(
                f"  [{label}] FAIL: wrote {n_written} spectra, expected "
                f"{len(target_indices)} (index misalignment or "
                "duplicate/out-of-range indices)"
            )
        if set(extracted_counts) != set(expected_counts):
            ok = False
            missing = set(expected_counts) - set(extracted_counts)
            extra = set(extracted_counts) - set(expected_counts)
            print(
                f"  [{label}] FAIL: molecule set mismatch -- {len(missing)} expected "
                f"molecules missing, {len(extra)} unexpected molecules extracted"
            )
        else:
            mismatched = [
                s for s in expected_counts if expected_counts[s] != extracted_counts[s]
            ]
            if mismatched:
                ok = False
                print(
                    f"  [{label}] FAIL: {len(mismatched)}/{len(expected_counts)} "
                    "molecules have the wrong per-molecule spectrum count"
                )
        if ok:
            print(
                f"  [{label}] OK: {len(expected_counts)} molecules, {n_written} "
                "spectra, all counts match mapping.pkl exactly"
            )
        all_ok = all_ok and ok

    if not all_ok:
        raise SystemExit(
            "Extraction verification FAILED for at least one group -- do not use "
            "this mini-MGF; the index semantics assumption needs re-checking."
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mgf", required=True)
    p.add_argument("--mapping_pkl", required=True)
    p.add_argument(
        "--group",
        dest="groups",
        nargs=3,
        action="append",
        required=True,
        metavar=("INDEX_KEY", "DF_KEY", "LABEL"),
        help="e.g. --group spectrum_indexes_val df_smiles_val test "
        "(repeatable; each spectrum index must belong to exactly one group)",
    )
    p.add_argument("--output_mgf", required=True)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
