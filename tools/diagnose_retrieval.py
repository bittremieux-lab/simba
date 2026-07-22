"""
Diagnostic analysis of SIMBA retrieval quality vs oracle.

For each test spectrum, compares:
  1. Rank of the oracle-best training molecule in SIMBA's cosine-sim ordering.
  2. Calibration error for the oracle pair: SIMBA_pred_MCES - GT_MCES.
  3. Calibration error for SIMBA's own retrieved pair: GT_MCES - SIMBA_pred_MCES.
  4. Oracle GT MCES distribution (with coverage breakdown).

Second figure — molecular property analysis:
  5. Tanimoto similarity: oracle pair vs SIMBA pair.
  6. Oracle rank vs Oracle GT MCES (scatter — are hard cases due to chemistry?).
  7. SIMBA GT vs Oracle GT MCES (how much structure is lost by SIMBA's pick).

GT MCES: max(Gaetan lb_matrix, MSG HDF5) — same source as training.

Requires:
  - Retrieval intermediates from simba_retrieval.py --intermediates_dir
  - /mnt/data2 access (lb_matrix + HDF5) — do NOT run as asimov SLURM job

Usage:
    uv run python tools/diagnose_retrieval.py \\
        --intermediates_dir /mnt/data/nkubrakov/experiments_3_dataset/retrieval/bs2048_v2_step44k \\
        --output results/retrieval_diagnostics_bs2048_v2_step44k.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matchms
import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from tqdm.auto import tqdm


MCES_CAP = 40.0
LB_MATRIX = Path("/mnt/data2/gdewaele/lb_matrix.npy")
LB_SMILES = Path("/mnt/data2/gdewaele/lb_matrix.smiles.txt")
HDF5_PATH = Path("/mnt/data2/nkubrakov/massspecgym/data/auxiliary/all_smiles_mces.hdf5")
MGF_DEFAULT = Path("/mnt/data2/nkubrakov/massspecgym/data/auxiliary/MassSpecGym.mgf")
N_SANITY = 5  # printed examples for mapping verification
SPEC_BIN_SIZE = 1.0  # Da per bin for spectral cosine
SPEC_N_BINS = 2000  # 0–2000 Da


def canonicalize(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def condensed_idx(i, j):
    hi = np.maximum(i, j).astype(np.int64)
    lo = np.minimum(i, j).astype(np.int64)
    return hi * (hi - 1) // 2 + lo


def gt_mces_to_all_train(
    test_lb_idx: int,
    test_hdf5_idx: int,
    train_lb_idxs: np.ndarray,
    train_hdf5_idxs: np.ndarray,
    lb: np.ndarray,
    hdf5_mces: np.ndarray,
) -> np.ndarray:
    n = len(train_lb_idxs)
    vals = np.full(n, MCES_CAP, dtype=np.float32)
    if test_lb_idx >= 0:
        mask = train_lb_idxs >= 0
        if mask.any():
            vals[mask] = lb[
                condensed_idx(np.int64(test_lb_idx), train_lb_idxs[mask])
            ].astype(np.float32)
    if test_hdf5_idx >= 0:
        mask = train_hdf5_idxs >= 0
        if mask.any():
            vals[mask] = np.maximum(
                vals[mask],
                hdf5_mces[
                    condensed_idx(np.int64(test_hdf5_idx), train_hdf5_idxs[mask])
                ].astype(np.float32),
            )
    return np.clip(vals, 0.0, MCES_CAP)


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    inter = int((fp_a & fp_b).sum())
    union = int((fp_a | fp_b).sum())
    return inter / union if union > 0 else 0.0


def heavy_atom_count(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumHeavyAtoms() if mol else 0


def exact_mass(smi: str) -> float:
    mol = Chem.MolFromSmiles(smi)
    return Descriptors.ExactMolWt(mol) if mol else float("nan")


def tanimoto_matrix(query_fp: np.ndarray, cand_fps: np.ndarray) -> np.ndarray:
    inter = (query_fp & cand_fps).sum(axis=1)
    union = (query_fp | cand_fps).sum(axis=1)
    return np.where(union > 0, inter / union, 0.0)


def spectrum_to_vector(spec) -> np.ndarray | None:
    """Bin spectrum into fixed-length L2-normalized vector (sqrt intensities, 1 Da bins)."""
    mzs = spec.peaks.mz
    ints = spec.peaks.intensities
    if mzs is None or len(mzs) == 0:
        return None
    ints = np.sqrt(ints.astype(np.float32))
    v = np.zeros(SPEC_N_BINS, dtype=np.float32)
    idx = np.floor(mzs / SPEC_BIN_SIZE).astype(int)
    mask = (idx >= 0) & (idx < SPEC_N_BINS)
    np.add.at(v, idx[mask], ints[mask])
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else None


def load_spectral_vectors(mgf_path: Path, fold: str, n_expected: int) -> np.ndarray:
    """Load spectra for a fold from MGF, apply same filter as simba_retrieval.py, return (N, SPEC_N_BINS)."""
    vecs = []
    for spec in tqdm(
        matchms.importing.load_from_mgf(str(mgf_path)),
        desc=f"  spectral vecs ({fold})",
        unit="spec",
    ):
        if spec.get("fold") != fold:
            continue
        if not spec.get("smiles"):
            continue
        if spec.peaks is None or len(spec.peaks.mz) < 1:
            continue
        v = spectrum_to_vector(spec)
        vecs.append(v if v is not None else np.zeros(SPEC_N_BINS, dtype=np.float32))
    arr = np.stack(vecs, axis=0)
    if len(arr) != n_expected:
        print(
            f"  WARNING: loaded {len(arr)} spectral vectors but expected {n_expected}"
        )
    return arr


def compute_oracle_and_simba_hits(
    rows: list,
    cand_lookup: dict,
    fp_cache: dict,
    ks: tuple = (1, 5, 20),
) -> tuple[dict, dict]:
    oracle_hits = dict.fromkeys(ks, 0)
    simba_hits = dict.fromkeys(ks, 0)
    n = 0
    for r in tqdm(rows, desc="Hit rates"):
        q_canon = canonicalize(r["test_smi"])
        cands = cand_lookup.get(q_canon, [])
        if not cands:
            continue
        cand_fps = np.stack(
            [
                fp_cache.get(c)
                if fp_cache.get(c) is not None
                else np.zeros(2048, dtype=np.uint8)
                for c in cands
            ]
        )
        for tag, smi_key in (("oracle", "oracle_smi"), ("simba", "simba_smi")):
            qfp = fp_cache.get(r[smi_key])
            if qfp is None:
                continue
            scores = tanimoto_matrix(qfp, cand_fps)
            ranked = [cands[i] for i in np.argsort(-scores)]
            for k in ks:
                if any(canonicalize(c) == q_canon for c in ranked[:k]):
                    (oracle_hits if tag == "oracle" else simba_hits)[k] += 1
        n += 1
    total = n or 1
    oracle_rates = {f"hit@{k}": oracle_hits[k] / total for k in ks}
    simba_rates = {f"hit@{k}": simba_hits[k] / total for k in ks}
    return oracle_rates, simba_rates


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--intermediates_dir", required=True)
    p.add_argument("--output", default="results/retrieval_diagnostics.png")
    p.add_argument(
        "--candidates",
        default=None,
        help="MSG candidates JSON for oracle/SIMBA hit rates",
    )
    p.add_argument(
        "--mgf",
        default=str(MGF_DEFAULT),
        help="Path to MassSpecGym.mgf for spectral cosine",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    intermediates_dir = Path(args.intermediates_dir)
    device = torch.device(args.device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    fig2_path = out_path.with_name(out_path.stem + "_molprops.png")

    # ── Load intermediates ────────────────────────────────────────────────────
    print("Loading intermediates ...")
    train_embs = torch.load(intermediates_dir / "train_embeddings.pt").to(device)
    test_embs = torch.load(intermediates_dir / "test_embeddings.pt").to(device)
    nn_indices = torch.load(intermediates_dir / "test_nn_indices.pt").cpu().numpy()
    train_smiles = json.loads((intermediates_dir / "train_smiles.json").read_text())
    test_smiles = json.loads((intermediates_dir / "test_smiles.json").read_text())
    N_train, N_test = len(train_smiles), len(test_smiles)
    print(f"  train spectra: {N_train:,}   test spectra: {N_test:,}")

    print("Canonicalizing SMILES ...")
    train_can = [canonicalize(s) for s in tqdm(train_smiles, desc="train")]
    test_can = [canonicalize(s) for s in tqdm(test_smiles, desc="test")]

    # ── Load spectral vectors for peak-based cosine ───────────────────────────
    mgf_path = Path(args.mgf)
    print(f"\nLoading spectral vectors from {mgf_path} ...")
    train_spec_vecs = load_spectral_vectors(mgf_path, "train", N_train)
    test_spec_vecs = load_spectral_vectors(mgf_path, "test", N_test)
    print(
        f"  train_spec_vecs: {train_spec_vecs.shape}, test_spec_vecs: {test_spec_vecs.shape}"
    )

    # ── Load GT MCES sources ──────────────────────────────────────────────────
    print("Loading lb SMILES index ...")
    lb_smi2idx: dict[str, int] = {}
    with open(LB_SMILES) as fh:
        for i, line in enumerate(fh):
            lb_smi2idx[line.strip()] = i
    print(f"  {len(lb_smi2idx):,} mols in lb_matrix")

    print("Opening lb_matrix.npy (mmap) ...")
    lb = np.load(LB_MATRIX, mmap_mode="r")

    print("Loading HDF5 MCES into RAM ...")
    with h5py.File(HDF5_PATH, "r") as hf:
        hdf5_smi_list = [
            s.decode() if isinstance(s, bytes) else s
            for s in hf["mces_smiles_order"][:]
        ]
        # HDF5 stores non-canonical SMILES; canonicalize so we can match train_can/test_can correctly.
        hdf5_smi2idx = {}
        for i, s in enumerate(hdf5_smi_list):
            mol = Chem.MolFromSmiles(s)
            if mol:
                hdf5_smi2idx[Chem.MolToSmiles(mol)] = i
        hdf5_mces = hf["mces"][:].astype(np.float32)
    print(f"  {len(hdf5_smi2idx):,} mols in HDF5 (canonical keys)")

    # ── Map to source indices ─────────────────────────────────────────────────
    train_lb_idxs = np.array([lb_smi2idx.get(s, -1) for s in train_can], dtype=np.int64)
    train_hdf5_idxs = np.array(
        [hdf5_smi2idx.get(s, -1) for s in train_can], dtype=np.int64
    )
    print(
        f"  train: lb_missing={int((train_lb_idxs < 0).sum())}, "
        f"hdf5_missing={int((train_hdf5_idxs < 0).sum())}"
    )

    unique_test = list(dict.fromkeys(test_can))
    test_lb_map = {s: lb_smi2idx.get(s, -1) for s in unique_test}
    test_hdf5_map = {s: hdf5_smi2idx.get(s, -1) for s in unique_test}

    n_lb_covered = sum(1 for v in test_lb_map.values() if v >= 0)
    n_hdf5_covered = sum(1 for v in test_hdf5_map.values() if v >= 0)
    n_uncovered = sum(
        1 for s in unique_test if test_lb_map[s] < 0 and test_hdf5_map[s] < 0
    )
    print(
        f"  test ({len(unique_test)} unique mols): lb_covered={n_lb_covered}, "
        f"hdf5_covered={n_hdf5_covered}, UNCOVERED={n_uncovered}"
    )

    # Map test mol SMILES → spectrum indices (for batching same-mol test spectra)
    spec_by_mol: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(test_can):
        spec_by_mol[s].append(i)

    # Map train mol SMILES → spectrum indices (for oracle rank: best rank across all spectra of oracle mol)
    train_mol_to_specs: dict[str, np.ndarray] = defaultdict(list)
    for i, s in enumerate(train_can):
        train_mol_to_specs[s].append(i)
    train_mol_to_specs = {
        k: np.array(v, dtype=np.int64) for k, v in train_mol_to_specs.items()
    }

    # ── Main diagnostic loop ──────────────────────────────────────────────────
    print("\nComputing diagnostics ...")

    rows = []  # per-spectrum records for CSV
    sanity_rows = []  # first N_SANITY covered mols for mapping check

    for test_smi in tqdm(unique_test, desc="unique test mols"):
        lb_idx = test_lb_map[test_smi]
        hdf5_idx = test_hdf5_map[test_smi]
        covered = lb_idx >= 0 or hdf5_idx >= 0

        gt = gt_mces_to_all_train(
            lb_idx, hdf5_idx, train_lb_idxs, train_hdf5_idxs, lb, hdf5_mces
        )

        oracle_nn = int(np.argmin(gt))
        oracle_gt = float(gt[oracle_nn])
        oracle_smi = train_can[oracle_nn]

        # Cosine sims for all spectra of this mol at once
        spec_idxs = spec_by_mol[test_smi]
        sims = (
            (test_embs[spec_idxs] @ train_embs.T).cpu().float().numpy()
        )  # (B, N_train)

        # Sanity examples: first N_SANITY covered mols
        if covered and len(sanity_rows) < N_SANITY:
            sanity_rows.append(
                {
                    "test_smi": test_smi,
                    "oracle_smi": oracle_smi,
                    "oracle_gt_mces": oracle_gt,
                    "lb_idx": lb_idx,
                    "hdf5_idx": hdf5_idx,
                }
            )

        # Spectral cosine sims for all test spectra of this mol vs all train spectra
        spec_cos_block = test_spec_vecs[spec_idxs] @ train_spec_vecs.T  # (B, N_train)

        for b, spec_i in enumerate(spec_idxs):
            sim_row = sims[b]
            simba_nn = int(nn_indices[spec_i])
            simba_smi = train_can[simba_nn]

            # Oracle rank: use the best-scoring spectrum of the oracle molecule,
            # not just the one arbitrarily selected by argmin(gt).
            oracle_specs = train_mol_to_specs[oracle_smi]
            oracle_sim = float(sim_row[oracle_specs].max())
            rank = 1 + int((sim_row > oracle_sim).sum())
            oracle_pred = (1.0 - oracle_sim) * 40.0
            oracle_err = oracle_pred - oracle_gt

            simba_pred = (1.0 - float(sim_row[simba_nn])) * 40.0
            simba_gt = float(gt[simba_nn])
            simba_err = simba_gt - simba_pred

            # Spectral cosine: test spectrum vs SIMBA-picked train spectrum (embedding argmax)
            spec_cos_row = spec_cos_block[b]
            simba_spectral_cos = float(spec_cos_row[simba_nn])
            # Oracle spectral cosine: max over all spectra of oracle mol
            oracle_spectral_cos = float(spec_cos_row[oracle_specs].max())

            rows.append(
                {
                    "spec_idx": spec_i,
                    "test_smi": test_smi,
                    "oracle_smi": oracle_smi,
                    "simba_smi": simba_smi,
                    "oracle_gt_mces": oracle_gt,
                    "simba_gt_mces": simba_gt,
                    "oracle_rank": rank,
                    "oracle_pred_mces": oracle_pred,
                    "oracle_err": oracle_err,
                    "simba_pred_mces": simba_pred,
                    "simba_err": simba_err,
                    "simba_spectral_cos": simba_spectral_cos,
                    "oracle_spectral_cos": oracle_spectral_cos,
                    "lb_covered": lb_idx >= 0,
                    "hdf5_covered": hdf5_idx >= 0,
                    "covered": covered,
                }
            )

    # ── Sanity check: print mapping examples ──────────────────────────────────
    print("\n=== Mapping sanity check (first covered test mols) ===")
    for r in sanity_rows:
        print(
            f"  test: {r['test_smi'][:60]}\n"
            f"  oracle train: {r['oracle_smi'][:60]}\n"
            f"  GT MCES={r['oracle_gt_mces']:.1f}  lb_idx={r['lb_idx']}  hdf5_idx={r['hdf5_idx']}\n"
        )

    # ── Arrays ───────────────────────────────────────────────────────────────
    covered_mask = np.array([r["covered"] for r in rows])
    oracle_ranks = np.array([r["oracle_rank"] for r in rows])
    oracle_errors = np.array([r["oracle_err"] for r in rows])
    simba_errors = np.array([r["simba_err"] for r in rows])
    oracle_gt_arr = np.array([r["oracle_gt_mces"] for r in rows])
    simba_gt_arr = np.array([r["simba_gt_mces"] for r in rows])

    # Covered-only versions for cleaner stats
    om = covered_mask
    print(f"\n=== Diagnostics — ALL {N_test:,} test spectra ===")
    _print_stats(oracle_ranks, oracle_errors, simba_errors, oracle_gt_arr, simba_gt_arr)
    if om.sum() < N_test:
        print(
            f"\n=== Diagnostics — COVERED only ({om.sum():,} spectra, {100 * om.mean():.1f}%) ==="
        )
        _print_stats(
            oracle_ranks[om],
            oracle_errors[om],
            simba_errors[om],
            oracle_gt_arr[om],
            simba_gt_arr[om],
        )

    # ── Tanimoto ─────────────────────────────────────────────────────────────
    print("\nComputing Tanimoto similarities ...")
    oracle_tanis, simba_tanis = [], []
    fp_cache: dict[str, np.ndarray | None] = {}

    def get_fp(smi):
        if smi not in fp_cache:
            fp_cache[smi] = morgan_fp(smi)
        return fp_cache[smi]

    for r in tqdm(rows, desc="Tanimoto"):
        fp_test = get_fp(r["test_smi"])
        fp_oracle = get_fp(r["oracle_smi"])
        fp_simba = get_fp(r["simba_smi"])
        oracle_tanis.append(
            tanimoto(fp_test, fp_oracle)
            if fp_test is not None and fp_oracle is not None
            else float("nan")
        )
        simba_tanis.append(
            tanimoto(fp_test, fp_simba)
            if fp_test is not None and fp_simba is not None
            else float("nan")
        )

    oracle_tanis = np.array(oracle_tanis)
    simba_tanis = np.array(simba_tanis)

    # ── Oracle + SIMBA hit rates (optional) ──────────────────────────────────
    oracle_rates, simba_rates = {}, {}
    if args.candidates:
        print("\nLoading candidates for hit rate computation ...")
        with open(args.candidates) as fh:
            cand_json = json.load(fh)
        cand_lookup = {canonicalize(k): v for k, v in cand_json.items()}

        # Pre-populate fp_cache with all candidate SMILES
        all_cand_smi = {c for cands in cand_lookup.values() for c in cands}
        print(f"  Computing FPs for {len(all_cand_smi):,} candidate SMILES ...")
        for c in tqdm(all_cand_smi, desc="Cand FPs"):
            get_fp(c)

        oracle_rates, simba_rates = compute_oracle_and_simba_hits(
            rows, cand_lookup, fp_cache
        )

        hits_path = out_path.with_name(out_path.stem + "_hits.json")
        import json as _json

        hits_path.write_text(
            _json.dumps(
                {
                    "oracle": oracle_rates,
                    "simba_cosine_nn": simba_rates,
                },
                indent=2,
            )
        )
        print("\n=== Hit rates (n with candidates) ===")
        print(
            "  Oracle NN transfer:  "
            + "  ".join(f"hit@{k}={v * 100:.2f}%" for k, v in oracle_rates.items())
        )
        print(
            "  SIMBA cosine NN:     "
            + "  ".join(f"hit@{k}={v * 100:.2f}%" for k, v in simba_rates.items())
        )
        print(f"  Saved → {hits_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()) + ["oracle_tanimoto", "simba_tanimoto"]
        )
        writer.writeheader()
        for r, ot, st in zip(rows, oracle_tanis, simba_tanis):
            writer.writerow({**r, "oracle_tanimoto": ot, "simba_tanimoto": st})
    print(
        f"  simba_spectral_cos: mean={np.mean([r['simba_spectral_cos'] for r in rows]):.4f}"
    )
    print(
        f"  oracle_spectral_cos: mean={np.mean([r['oracle_spectral_cos'] for r in rows]):.4f}"
    )
    print(f"CSV saved → {csv_path}")

    # ── Figure 1: 4 subplots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Rank histogram
    ax = axes[0, 0]
    max_rank = int(oracle_ranks.max())
    bins = np.unique(
        np.concatenate(
            [
                np.arange(1, min(21, max_rank + 1)),
                np.logspace(np.log10(20), np.log10(max_rank + 1), 40).astype(int),
            ]
        )
    )
    ax.hist(
        oracle_ranks[om],
        bins=bins,
        color="#4E9A7A",
        edgecolor="none",
        label="covered",
        alpha=0.9,
    )
    if (~om).any():
        ax.hist(
            oracle_ranks[~om],
            bins=bins,
            color="#aaa",
            edgecolor="none",
            label="uncovered",
            alpha=0.7,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Rank of oracle-best training mol in SIMBA ordering")
    ax.set_ylabel("# test spectra")
    ax.set_title("1 · Oracle mol rank in SIMBA", fontweight="bold")
    pct1 = 100 * (oracle_ranks[om] == 1).mean() if om.any() else 0
    pct10 = 100 * (oracle_ranks[om] <= 10).mean() if om.any() else 0
    pct100 = 100 * (oracle_ranks[om] <= 100).mean() if om.any() else 0
    ax.text(
        0.97,
        0.97,
        f"rank=1: {pct1:.1f}%\nrank≤10: {pct10:.1f}%\nrank≤100: {pct100:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 2. Oracle pair calibration error
    ax = axes[0, 1]
    _hist_with_stats(
        ax,
        oracle_errors[om],
        "#5B8DB8",
        "SIMBA pred MCES − GT MCES  (oracle pair)",
        "2 · Calibration error — oracle pair",
    )

    # 3. SIMBA pair calibration error
    ax = axes[1, 0]
    _hist_with_stats(
        ax,
        simba_errors[om],
        "#E07B54",
        "GT MCES − SIMBA pred MCES  (SIMBA's pick)",
        "3 · Calibration error — SIMBA's pick",
    )

    # 4. Oracle GT MCES distribution
    ax = axes[1, 1]
    bins4 = np.arange(0, 42.5, 2.5)
    ax.hist(
        oracle_gt_arr[om],
        bins=bins4,
        color="#7B5EA7",
        edgecolor="none",
        label="covered",
    )
    if (~om).any():
        ax.hist(
            oracle_gt_arr[~om],
            bins=bins4,
            color="#aaa",
            edgecolor="none",
            label="uncovered (capped at 40)",
            alpha=0.7,
        )
    ax.set_xlabel("Oracle GT MCES (best possible training match)")
    ax.set_ylabel("# test spectra")
    ax.set_title("4 · Oracle GT MCES distribution", fontweight="bold")
    ax.text(
        0.97,
        0.97,
        f"mean={oracle_gt_arr[om].mean():.1f}\nmedian={np.median(oracle_gt_arr[om]):.1f}\n"
        f"≤10: {100 * (oracle_gt_arr[om] <= 10).mean():.1f}%\n≤20: {100 * (oracle_gt_arr[om] <= 20).mean():.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"SIMBA Retrieval Diagnostics — {N_test:,} test spectra  "
        f"(GT = max(Gaetan lb, HDF5))\n"
        f"covered: {om.sum():,}/{N_test:,} ({100 * om.mean():.1f}%)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Figure 1 saved → {out_path}")

    # ── Figure 2: Tanimoto distributions (panel 5) ───────────────────────────
    fig2, ax = plt.subplots(1, 1, figsize=(6, 5))
    bins5 = np.linspace(0, 1, 41)
    ax.hist(
        oracle_tanis[om],
        bins=bins5,
        color="#4E9A7A",
        edgecolor="none",
        alpha=0.75,
        label=f"oracle pair (μ={np.nanmean(oracle_tanis[om]):.2f})",
    )
    ax.hist(
        simba_tanis[om],
        bins=bins5,
        color="#E07B54",
        edgecolor="none",
        alpha=0.75,
        label=f"SIMBA pair (μ={np.nanmean(simba_tanis[om]):.2f})",
    )
    ax.set_xlabel("Tanimoto similarity (Morgan FP, r=2)")
    ax.set_ylabel("# test spectra")
    ax.set_title("Tanimoto: oracle vs SIMBA pair", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig2.suptitle("SIMBA Retrieval — Tanimoto Analysis", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=140, bbox_inches="tight")
    print(f"Figure 2 saved → {fig2_path}")


def _print_stats(ranks, oracle_errors, simba_errors, oracle_gt, simba_gt):
    n = len(ranks)
    pct1 = 100 * (ranks == 1).mean()
    pct10 = 100 * (ranks <= 10).mean()
    pct100 = 100 * (ranks <= 100).mean()
    print(
        f"  n={n:,}  Oracle rank: median={np.median(ranks):.0f}  rank=1:{pct1:.1f}%  ≤10:{pct10:.1f}%  ≤100:{pct100:.1f}%"
    )
    print(
        f"  Oracle GT MCES: mean={oracle_gt.mean():.1f}  median={np.median(oracle_gt):.1f}  ≤10:{100 * (oracle_gt <= 10).mean():.1f}%"
    )
    print(
        f"  Calibration (oracle pair): mean err={oracle_errors.mean():.1f}  std={oracle_errors.std():.1f}"
    )
    print(
        f"  Calibration (SIMBA pair):  mean err={simba_errors.mean():.1f}  std={simba_errors.std():.1f}"
    )
    print(
        f"  SIMBA GT MCES: mean={simba_gt.mean():.1f}  median={np.median(simba_gt):.1f}"
    )


def _hist_with_stats(ax, data, color, xlabel, title):
    err_lo, err_hi = -40, 40
    ax.hist(
        np.clip(data, err_lo, err_hi),
        bins=80,
        range=(err_lo, err_hi),
        color=color,
        edgecolor="none",
    )
    ax.axvline(0, color="red", lw=1.2, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# test spectra")
    ax.set_title(title, fontweight="bold")
    ax.text(
        0.97,
        0.97,
        f"mean={data.mean():.1f}\nstd={data.std():.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
    )
    ax.grid(True, alpha=0.2)


if __name__ == "__main__":
    main()
