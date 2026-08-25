"""8b: for each cell of the SIMBA-top1 x cosine-top1 confusion matrix, pick
a few random examples and plot the real test spectrum against the ICEBERG-
predicted spectrum of the TRUE candidate (never the wrongly-ranked one) as a
mirror plot — to see, visually, whether SIMBA/cosine's success or failure on
a given query tracks how similar the real and ICEBERG spectra actually look.

Each example gets 2 stacked rows: the raw spectra as-is, and directly below,
the exact same pair after SIMBA's own preprocessing (remove_precursor_peak +
filter_intensity/max_num_peaks, then sqrt-compress + L2-normalize — see
simba_preprocess() below, which mirrors simba_retrieval.py's
spectra_to_tensors() call to Preprocessor.preprocess_spectrum() line for
line) — i.e. what the model actually sees, not what's in the raw file.

Reads build_retrieval_comparison_table.py's CSV to pick examples (no
re-scoring), then pulls the actual peak data directly:
  - Real spectrum: a manual MGF text scan (like prepare_gt_mces_retrieval.py's
    _load_test_fold_smiles) rather than matchms's per-spectrum object
    construction, which is very slow for this file.
  - ICEBERG spectrum: candidate_tsv (smiles, adduct) -> cand_id, then
    preds.hdf5's manifest -> leaf -> (mass, intensity). candidate_tsv's own
    "smiles" column is RAW (not canonicalized — build_candidate_tsv.py only
    canonicalizes the query-molecule keys it groups by, not the candidate-
    list values), so matching it against this table's canonical
    candidate_smiles column naively would mean canonicalizing all 600,455
    rows with RDKit — the same cost every other script touching this
    candidate list pays via a 15-25 min SLURM job. For just a dozen lookups
    that's wasteful: candidate_tsv's "precursor" column plus the query's own
    measured PRECURSOR_MZ (same molecule + adduct, so the same mass) narrows
    each lookup to that one query's own candidate pool (~185 rows, a plain
    numeric filter, no RDKit) before canonicalizing only that tiny subset.

Usage:
    uv run python tools/plot_confusion_matrix_examples.py \\
        --table_csv /path/to/retrieval_comparison_table.csv \\
        --mgf /path/to/MassSpecGym.mgf \\
        --candidate_tsv /path/to/candidates_test_official.tsv \\
        --iceberg_preds /path/to/preds.hdf5 \\
        --output_dir /path/to/output \\
        --n_per_cell 3 --seed 42
"""

import argparse
import contextlib
import copy
from pathlib import Path

import h5py
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simba_retrieval import canonicalize

from simba.core.data.preprocessor import Preprocessor
from simba.core.data.spectrum import SpectrumExt


_PP = Preprocessor()


def to_spectrum_ext(
    precursor_mz: float, mz: np.ndarray, intensity: np.ndarray
) -> SpectrumExt:
    """Minimal SpectrumExt wrapper around raw (mz, intensity) arrays — only
    the fields preprocess_spectrum actually touches (precursor_mz, charge,
    mz, intensity) carry real values; everything else is an unused
    placeholder, same convention as build_candidate_spectra in
    simba_retrieval_iceberg.py (precursor_charge=1)."""
    return SpectrumExt(
        identifier="",
        precursor_mz=precursor_mz,
        precursor_charge=1,
        mz=mz,
        intensity=intensity,
        retention_time=np.nan,
        params={},
        library=None,
        inchi=None,
        smiles="",
        ionmode="positive",
        adduct="",
        ce=35.0,
        ion_activation=None,
        ionization_method=None,
        bms=None,
        superclass=None,
        classe=None,
        subclass=None,
    )


def simba_preprocess(
    precursor_mz: float, mz: np.ndarray, intensity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly SIMBA's own preprocessing (simba_retrieval.py's
    spectra_to_tensors -> Preprocessor.preprocess_spectrum with
    fragment_tol_mass=10, fragment_tol_mode="ppm", min_intensity=0.01,
    max_num_peaks=100, scale_intensity=None -- which removes the precursor
    peak then keeps only the top max_num_peaks peaks above 1% of the max),
    followed by spectra_to_tensors' own sqrt-compress + L2-normalize
    (torch.sqrt then divide by the L2 norm clamped to a 1e-8 floor) --
    identical code path used for both real and ICEBERG-origin spectra."""
    if len(mz) == 0:
        return mz, intensity
    spec = to_spectrum_ext(
        precursor_mz,
        np.array(mz, dtype=np.float64),
        np.array(intensity, dtype=np.float64),
    )
    processed = _PP.preprocess_spectrum(
        copy.copy(spec),
        fragment_tol_mass=10,
        fragment_tol_mode="ppm",
        min_intensity=0.01,
        max_num_peaks=100,
        scale_intensity=None,
    )
    p_mz = np.asarray(processed.mz, dtype=np.float64)
    p_int = np.asarray(processed.intensity, dtype=np.float64)
    p_int = np.sqrt(np.clip(p_int, 0.0, None))
    norm = max(float(np.sqrt((p_int**2).sum())), 1e-8)
    p_int = p_int / norm
    return p_mz, p_int


CELLS = [
    ("both_correct", "SIMBA correct, cosine correct"),
    ("simba_only", "SIMBA correct, cosine wrong"),
    ("cosine_only", "SIMBA wrong, cosine correct"),
    ("neither", "SIMBA wrong, cosine wrong"),
]


def pick_examples(
    table_csv: str, n_per_cell: int, seed: int
) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(
        table_csv,
        usecols=[
            "test_spec_idx",
            "test_smiles",
            "test_adduct",
            "candidate_smiles",
            "candidate_adduct",
            "simba_rank",
            "cosine_rank",
            "simba_similarity",
            "simba_mces",
            "cosine_similarity",
            "cosine_mces",
        ],
    )
    correct = df[df["test_smiles"] == df["candidate_smiles"]].copy()
    correct["simba_hit1"] = correct["simba_rank"] == 1
    correct["cosine_hit1"] = correct["cosine_rank"] == 1

    # Cells where cosine "wins" (both_correct, cosine_only) exclude floor-tie
    # wins -- cosine_similarity < 0.01 means the whole candidate pool was
    # ~flat at 0 (see plot_retrieval_comparison_checks.py's
    # report_zero_cosine_hits), so cosine's rank-1 there is arbitrary sort
    # order, not a real example of cosine "working." Doesn't apply to
    # simba_only/neither -- cosine didn't win there regardless of its score.
    cosine_real_win = correct["cosine_hit1"] & (correct["cosine_similarity"] >= 0.01)
    masks = {
        "both_correct": correct["simba_hit1"] & cosine_real_win,
        "simba_only": correct["simba_hit1"] & ~correct["cosine_hit1"],
        "cosine_only": ~correct["simba_hit1"] & cosine_real_win,
        "neither": ~correct["simba_hit1"] & ~correct["cosine_hit1"],
    }
    rng = np.random.default_rng(seed)
    picked = {}
    for key, mask in masks.items():
        pool = correct[mask]
        n = min(n_per_cell, len(pool))
        picked[key] = pool.sample(n=n, random_state=rng.integers(0, 2**31 - 1))
    return picked


def extract_test_spectra_by_index(
    mgf_path: str, target_indices: set[int]
) -> dict[int, tuple[str, str, float, np.ndarray, np.ndarray]]:
    """Manual MGF scan (fast — avoids matchms's slow per-spectrum object
    construction). Returns {test_idx: (smiles, adduct, precursor_mz, mz,
    intensity)} for the given 0-indexed positions within the fold==test
    sequence, in the SAME order load_spectra(mgf, 'test') would enumerate
    them. precursor_mz is carried through so the true candidate's
    ICEBERG-side row can be found by mass instead of by canonicalizing the
    whole candidate_tsv (see load_iceberg_peaks)."""
    result = {}
    test_idx = -1
    current = None
    n_targets = len(target_indices)
    with open(mgf_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "BEGIN IONS":
                current = {
                    "smiles": None,
                    "adduct": None,
                    "fold": None,
                    "precursor_mz": None,
                    "mz": [],
                    "intensity": [],
                }
                continue
            if line == "END IONS":
                if current["fold"] == "test":
                    test_idx += 1
                    if test_idx in target_indices:
                        result[test_idx] = (
                            current["smiles"],
                            current["adduct"],
                            current["precursor_mz"],
                            np.array(current["mz"], dtype=np.float64),
                            np.array(current["intensity"], dtype=np.float64),
                        )
                        if len(result) == n_targets:
                            return result
                current = None
                continue
            if current is None:
                continue
            if line.startswith("SMILES="):
                current["smiles"] = line[len("SMILES=") :]
            elif line.startswith("ADDUCT="):
                current["adduct"] = line[len("ADDUCT=") :]
            elif line.startswith("FOLD="):
                current["fold"] = line[len("FOLD=") :]
            elif line.startswith("PRECURSOR_MZ="):
                with contextlib.suppress(ValueError):
                    current["precursor_mz"] = float(line[len("PRECURSOR_MZ=") :])
            elif current["fold"] == "test" and "=" not in line and line.strip():
                # FOLD= always appears before the peak lines in this file's
                # block layout, so by here we already know whether this
                # spectrum is train/val (skip parsing its peaks entirely --
                # the vast majority of the file) or test (worth parsing).
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz, inten = float(parts[0]), float(parts[1])
                        current["mz"].append(mz)
                        current["intensity"].append(inten)
                    except ValueError:
                        pass
    return result


def load_iceberg_peaks(
    candidate_tsv: str | list[str],
    iceberg_preds: str | list[str],
    wanted: list[tuple[str, str, float]],
    mass_tol: float = 0.02,
) -> dict[tuple[str, str], tuple[float, np.ndarray, np.ndarray]]:
    """wanted: list of (canonical smiles, adduct, query precursor_mz)
    triples — precursor_mz narrows candidate_tsv down to that one query's
    own candidate pool (~185 rows, a plain numeric filter) before
    canonicalizing only that tiny subset to find the exact match, instead of
    canonicalizing all 600,455 rows up front (see module docstring). Returns
    {(canonical smiles, adduct): (candidate's own precursor_mz, mass,
    intensity)} -- the candidate's own precursor_mz (not necessarily bit-
    identical to the query's) is carried through so SIMBA-preprocessing this
    spectrum removes the right peak.

    candidate_tsv/iceberg_preds each accept one path or several (matched 1:1
    by position, same delta-file convention as simba_retrieval_iceberg.py)
    -- concatenated/merged before the mass-filtered lookup."""
    tsv_paths = (
        [candidate_tsv] if isinstance(candidate_tsv, str) else list(candidate_tsv)
    )
    preds_paths = (
        [iceberg_preds] if isinstance(iceberg_preds, str) else list(iceberg_preds)
    )
    print(
        f"  Loading {tsv_paths} (mass-filtered lookup, no full-table canonicalization) ..."
    )
    cand_index = pd.concat(
        [pd.read_csv(p, sep="\t") for p in tsv_paths], ignore_index=True
    )

    cand_ids = {}
    cand_precursor_mz = {}
    for c_smi, adduct, prec_mz in wanted:
        subset = cand_index[
            (cand_index["ionization"] == adduct)
            & (cand_index["precursor"].sub(prec_mz).abs() < mass_tol)
        ]
        if subset.empty:
            print(
                f"    no mass-matched rows for {c_smi[:50]}... / {adduct} (prec_mz={prec_mz})"
            )
            continue
        canon_smi = subset["smiles"].map(canonicalize)
        match = subset[canon_smi == c_smi]
        if match.empty:
            print(
                f"    {len(subset)} mass-matched rows for {c_smi[:50]}... / {adduct}, "
                "but none canonicalize to the target — unexpected"
            )
            continue
        cand_ids[(c_smi, adduct)] = match.iloc[0]["spec"]
        cand_precursor_mz[(c_smi, adduct)] = float(match.iloc[0]["precursor"])

    wanted_ids = set(cand_ids.values())
    print(f"  Reading {len(wanted_ids)} predicted spectra from {preds_paths} ...")
    result_by_id = {}
    for preds_path in preds_paths:
        with h5py.File(preds_path, "r") as f:
            manifest = f["__predspec_manifest__"]
            for name, leaf in zip(manifest["name"][:], manifest["leaf_path"][:]):
                name_str = name.decode().removeprefix("pred_")
                if name_str not in wanted_ids or name_str in result_by_id:
                    continue
                arr = f[leaf.decode()]["f"][:]
                mask = arr[:, 0] > 0
                result_by_id[name_str] = (arr[mask, 0], arr[mask, 1])

    return {
        key: (cand_precursor_mz[key], *result_by_id[cid])
        for key, cid in cand_ids.items()
        if cid in result_by_id
    }


def plot_mirror(
    ax,
    real_mz: np.ndarray,
    real_intensity: np.ndarray,
    iceberg_mz: np.ndarray,
    iceberg_intensity: np.ndarray,
    title: str,
) -> None:
    real_norm = (
        real_intensity / real_intensity.max() if len(real_intensity) else real_intensity
    )
    iceberg_norm = (
        iceberg_intensity / iceberg_intensity.max()
        if len(iceberg_intensity)
        else iceberg_intensity
    )
    ax.vlines(real_mz, 0, real_norm, color="tab:blue", linewidth=1)
    ax.vlines(iceberg_mz, 0, -iceberg_norm, color="tab:red", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=7, pad=8)
    ax.set_ylim(-1.15, 1.15)
    ax.tick_params(labelsize=7)


def run(
    table_csv: str,
    mgf: str,
    candidate_tsv: str | list[str],
    iceberg_preds: str | list[str],
    output_dir: str,
    n_per_cell: int = 3,
    seed: int = 42,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Picking random examples per confusion-matrix cell ...")
    picked = pick_examples(table_csv, n_per_cell, seed)
    for key, _ in CELLS:
        print(f"  {key}: {len(picked[key])} examples")

    all_rows = pd.concat(picked.values())
    target_indices = set(all_rows["test_spec_idx"].astype(int).tolist())
    print(f"\nExtracting {len(target_indices)} real test spectra from {mgf} ...")
    real_spectra = extract_test_spectra_by_index(mgf, target_indices)

    wanted_triples = []
    for _, row in all_rows.iterrows():
        spec_idx = int(row["test_spec_idx"])
        if spec_idx not in real_spectra:
            continue
        _, _, prec_mz, _, _ = real_spectra[spec_idx]
        wanted_triples.append(
            (row["candidate_smiles"], row["candidate_adduct"], prec_mz)
        )
    print(f"\nLooking up {len(wanted_triples)} true-candidate ICEBERG spectra ...")
    iceberg_peaks = load_iceberg_peaks(candidate_tsv, iceberg_preds, wanted_triples)
    print(f"  {len(iceberg_peaks)} / {len(wanted_triples)} resolved")

    n_cat = len(CELLS)
    n_cols = n_per_cell
    n_rows = n_cat * 2
    # 3.6 (up from 3.2) + an explicit hspace below: the 3-line titles were
    # colliding with the row of subplots above them -- tight_layout alone
    # wasn't leaving enough room for them.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for cat_i, (key, label) in enumerate(CELLS):
        raw_row, proc_row = cat_i * 2, cat_i * 2 + 1
        cell_df = picked[key]
        for col_i in range(n_cols):
            raw_ax, proc_ax = axes[raw_row, col_i], axes[proc_row, col_i]
            if col_i >= len(cell_df):
                raw_ax.axis("off")
                proc_ax.axis("off")
                continue
            row = cell_df.iloc[col_i]
            spec_idx = int(row["test_spec_idx"])
            cand_key = (row["candidate_smiles"], row["candidate_adduct"])

            if spec_idx not in real_spectra or cand_key not in iceberg_peaks:
                raw_ax.set_title("missing data", fontsize=8)
                raw_ax.axis("off")
                proc_ax.axis("off")
                continue

            _, _, real_prec_mz, real_mz, real_intensity = real_spectra[spec_idx]
            cand_prec_mz, iceberg_mz, iceberg_intensity = iceberg_peaks[cand_key]

            title = (
                f"idx={spec_idx} [raw]\n"
                f"SIMBA: r={row['simba_rank']:.0f} sim={row['simba_similarity']:.2f} "
                f"mces={row['simba_mces']:.1f}\n"
                f"cosine: r={row['cosine_rank']:.0f} sim={row['cosine_similarity']:.2f}"
                # cosine_mces dropped -- not a calibrated quantity (raw cosine
                # similarity was never trained to predict MCES, see
                # cosine_similarity_pool_distribution_plots.py's module
                # docstring), just a same-formula convenience transform that
                # doesn't mean anything to show alongside SIMBA's real one.
            )
            plot_mirror(
                raw_ax, real_mz, real_intensity, iceberg_mz, iceberg_intensity, title
            )

            p_real_mz, p_real_intensity = simba_preprocess(
                real_prec_mz, real_mz, real_intensity
            )
            p_iceberg_mz, p_iceberg_intensity = simba_preprocess(
                cand_prec_mz, iceberg_mz, iceberg_intensity
            )
            plot_mirror(
                proc_ax,
                p_real_mz,
                p_real_intensity,
                p_iceberg_mz,
                p_iceberg_intensity,
                "SIMBA-preprocessed\n(precursor removed, top-100, sqrt+L2-norm)",
            )

            if col_i == 0:
                raw_ax.set_ylabel(label + "\n(raw)", fontsize=8)
                proc_ax.set_ylabel(label + "\n(preprocessed)", fontsize=8)

    fig.suptitle(
        "Real test spectrum (blue, up) vs ICEBERG-predicted true-candidate spectrum (red, down) — "
        "raw above, SIMBA-preprocessed below",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.9, top=0.94)
    out_path = out_dir / "confusion_matrix_examples.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--table_csv", required=True)
    p.add_argument("--mgf", required=True)
    p.add_argument("--candidate_tsv", required=True, nargs="+")
    p.add_argument("--iceberg_preds", required=True, nargs="+")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_per_cell", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
