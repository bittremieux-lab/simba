"""
GT MCES vs predicted MCES hexbin for cross-split (test spec, train spec) pairs.

GT MCES: max(Gaetan lb_matrix, MSG HDF5) — same source as training.
Pred MCES: (1 - cosine_sim(test_emb, train_emb)) * 40 from precomputed embeddings.

Pairs are GT-MCES-balanced (same cap logic as the balanced val hexbins).

Runs on login node (needs /mnt/data2 for lb_matrix + HDF5, and --intermediates_dir
for the precomputed train/test embeddings and SMILES produced by simba_retrieval.py).

Usage:
    uv run python tools/test_train_mces_hexbin.py \
        --intermediates_dir /mnt/data/nkubrakov/experiments_3_dataset/retrieval/bs2048_v2_step44k \
        --output results/test_train_mces_hexbin_bs2048_v2_step44k.png
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib


matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from scipy.stats import spearmanr
from tqdm.auto import tqdm


MCES_CAP = 40.0
LB_MATRIX = Path("/mnt/data2/gdewaele/lb_matrix.npy")
LB_SMILES = Path("/mnt/data2/gdewaele/lb_matrix.smiles.txt")
HDF5_PATH = Path("/mnt/data2/nkubrakov/massspecgym/data/auxiliary/all_smiles_mces.hdf5")

BIN_STEP = 2.5
BINS_MCES = np.arange(0, 40 + BIN_STEP, BIN_STEP)
COLOR = "#4E9A7A"


def canonicalize(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def condensed_idx(i, j):
    hi = np.maximum(i, j).astype(np.int64)
    lo = np.minimum(i, j).astype(np.int64)
    return hi * (hi - 1) // 2 + lo


def gt_mces_to_all_train(
    test_lb_idx, test_hdf5_idx, train_lb_idxs, train_hdf5_idxs, lb, hdf5_mces
):
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


def balance_by_gt(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bin_ids = np.clip(np.digitize(gt, BINS_MCES) - 1, 0, len(BINS_MCES) - 2)
    counts = np.bincount(bin_ids, minlength=len(BINS_MCES) - 1)
    cap = int(1.5 * counts[counts > 0].min())
    idx = []
    for b in range(len(BINS_MCES) - 1):
        where = np.where(bin_ids == b)[0]
        if len(where):
            idx.append(rng.choice(where, size=min(len(where), cap), replace=False))
    return np.concatenate(idx)


def _hexbin_panel(fig, subplot_spec, gt, pred, title, cmap="Greens"):
    r, _ = spearmanr(gt, pred)
    mse = float(np.mean((pred - gt) ** 2))
    bins = np.arange(0, 42.5, 2.5)

    inner = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=subplot_spec,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.04,
        wspace=0.04,
    )
    ax_top = fig.add_subplot(inner[0, 0])
    ax_main = fig.add_subplot(inner[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)
    fig.add_subplot(inner[0, 1]).set_visible(False)

    ax_main.hexbin(gt, pred, gridsize=16, cmap=cmap, mincnt=1, extent=[0, 40, 0, 40])
    ax_main.plot([0, 40], [0, 40], "r--", lw=1)
    ax_main.set_xlabel("GT MCES (test mol vs train mol)", fontsize=9)
    ax_main.set_ylabel("SIMBA predicted MCES", fontsize=9)
    ax_main.set_xlim(0, 40)
    ax_main.set_ylim(0, 40)
    ax_main.grid(True, alpha=0.2)

    ax_top.hist(gt, bins=bins, color=COLOR, edgecolor="none")
    ax_top.set_xlim(0, 40)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_ylabel("count", fontsize=7)
    ax_top.tick_params(labelsize=6)
    ax_top.set_title(f"{title}\nn={len(gt):,}  ρ={r:.3f}  MSE={mse:.2f}", fontsize=9)
    ax_top.grid(True, alpha=0.2)

    ax_right.hist(
        pred, bins=bins, orientation="horizontal", color=COLOR, edgecolor="none"
    )
    ax_right.set_ylim(0, 40)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.set_xlabel("count", fontsize=7)
    ax_right.tick_params(labelsize=6)
    ax_right.grid(True, alpha=0.2)


def _boxplot_panel(ax, gt_all, pred_all, title):
    bin_ids = np.clip(np.digitize(gt_all, BINS_MCES) - 1, 0, len(BINS_MCES) - 2)
    positions, data, labels = [], [], []
    for b in range(len(BINS_MCES) - 1):
        where = np.where(bin_ids == b)[0]
        if len(where) < 5:
            continue
        positions.append(BINS_MCES[b] + BIN_STEP / 2)
        data.append(pred_all[where])
        labels.append(f"{int(BINS_MCES[b])}")

    ax.boxplot(
        data,
        positions=positions,
        widths=BIN_STEP * 0.7,
        whis=[5, 95],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#c6e8d6", "color": "#2e7d55"},
        medianprops={"color": "#c0392b", "lw": 1.5},
        whiskerprops={"color": "#2e7d55"},
        capprops={"color": "#2e7d55"},
    )
    for pos, d in zip(positions, data):
        p95 = float(np.percentile(d, 95))
        ax.text(
            pos,
            min(p95 + 0.8, 39),
            f"n={len(d):,}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#333",
        )

    ax.plot([0, 40], [0, 40], "r--", lw=1, alpha=0.7)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_xlabel("GT MCES bin start", fontsize=9)
    ax.set_ylabel("SIMBA predicted MCES", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")


def plot_figure(
    gt_rand, pred_rand, gt_raw, pred_raw, gt_bal, pred_bal, step, output_path
):
    fig = plt.figure(figsize=(24, 14))
    outer = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.40)

    # ── Row 0: hexbins ────────────────────────────────────────────────────────
    _hexbin_panel(
        fig, outer[0, 0], gt_rand, pred_rand, "1 · Truly random pairs", cmap="Oranges"
    )
    _hexbin_panel(
        fig,
        outer[0, 1],
        gt_raw,
        pred_raw,
        "2 · Per-bin capped (≤3k/bin/test mol)",
        cmap="Blues",
    )
    _hexbin_panel(
        fig, outer[0, 2], gt_bal, pred_bal, "3 · GT-MCES balanced", cmap="Greens"
    )

    # ── Row 1: box plots + legend ─────────────────────────────────────────────
    ax_box_rand = fig.add_subplot(outer[1, 0])
    _boxplot_panel(
        ax_box_rand,
        gt_rand,
        pred_rand,
        f"4 · Random pairs — box plot  (n={len(gt_rand):,})\nwhiskers 5–95%, no outliers",
    )

    ax_box_raw = fig.add_subplot(outer[1, 1])
    _boxplot_panel(
        ax_box_raw,
        gt_raw,
        pred_raw,
        f"5 · Per-bin capped — box plot  (n={len(gt_raw):,})\nwhiskers 5–95%, no outliers",
    )

    ax_leg = fig.add_subplot(outer[1, 2])
    ax_leg.axis("off")
    legend_text = (
        "Plot descriptions\n"
        "─────────────────────────────────────\n\n"
        "1 · Truly random pairs\n"
        "   For each covered test mol, sample ~200 random\n"
        "   train spectra with no MCES-bin constraint.\n"
        "   Shows the real distribution: most test-train\n"
        "   pairs are structurally dissimilar (GT ≈ 40),\n"
        "   so low-MCES region is nearly empty.\n\n"
        "2 · Per-bin capped\n"
        "   Same pass, but sample up to 3k train spectra\n"
        "   per GT MCES bin per test mol. High-MCES bins\n"
        "   are heavily subsampled. Shows SIMBA behaviour\n"
        "   across the full MCES range, but bin counts are\n"
        "   not proportional to reality.\n\n"
        "3 · GT-MCES balanced\n"
        "   Apply 1.5× cap across bins on the per-bin\n"
        "   collection so all occupied bins have equal\n"
        "   representation. Directly comparable to the\n"
        "   val/test Inference hexbins.\n\n"
        "4 · Random box plot\n"
        "   pred MCES distribution per GT MCES bin from\n"
        "   the random sample. n= shows how many pairs\n"
        "   fall in each bin naturally.\n\n"
        "5 · Per-bin capped box plot\n"
        "   Same but from the per-bin collection. n= is\n"
        "   capped at 3k × #test mols, so bins look more\n"
        "   equal but that is an artefact of sampling.\n\n"
        "Red dashed diagonal = perfect prediction.\n"
        "GT = max(Gaetan lb_matrix, MSG HDF5)."
    )
    ax_leg.text(
        0.03,
        0.97,
        legend_text,
        transform=ax_leg.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "fc": "#f7f7f7", "ec": "#ccc", "lw": 0.8},
    )

    fig.suptitle(
        f"Test vs Train cross-split pairs — bs2048_v2 · step {step:,}   "
        f"(GT = max(Gaetan lb_matrix, MSG HDF5))",
        fontsize=12,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intermediates_dir",
        required=True,
        help="Dir with train/test embeddings + smiles from simba_retrieval.py",
    )
    parser.add_argument("--output", default="results/test_train_mces_hexbin.png")
    parser.add_argument(
        "--pairs_per_bin",
        type=int,
        default=3000,
        help="Train spec candidates to collect per MCES bin per test mol (default 3000)",
    )
    parser.add_argument(
        "--random_per_mol",
        type=int,
        default=200,
        help="Truly random train spec samples per test mol for raw panel (default 200)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    intermediates_dir = Path(args.intermediates_dir)

    # ── Load embeddings and SMILES ─────────────────────────────────────────────
    print("Loading precomputed embeddings and SMILES ...")
    train_embs = torch.load(intermediates_dir / "train_embeddings.pt").cpu().float()
    test_embs = torch.load(intermediates_dir / "test_embeddings.pt").cpu().float()
    train_smiles = json.loads((intermediates_dir / "train_smiles.json").read_text())
    test_smiles = json.loads((intermediates_dir / "test_smiles.json").read_text())
    N_train, N_test = len(train_smiles), len(test_smiles)
    print(f"  train spectra: {N_train:,}   test spectra: {N_test:,}")

    # ── Canonicalize ───────────────────────────────────────────────────────────
    print("Canonicalizing SMILES ...")
    train_can = [canonicalize(s) for s in tqdm(train_smiles, desc="train")]
    test_can = [canonicalize(s) for s in tqdm(test_smiles, desc="test")]

    # ── Load GT MCES sources ──────────────────────────────────────────────────
    print("Loading lb_matrix SMILES index ...")
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
        hdf5_smi2idx = {}
        for i, s in enumerate(hdf5_smi_list):
            mol = Chem.MolFromSmiles(s)
            if mol:
                hdf5_smi2idx[Chem.MolToSmiles(mol)] = i
        hdf5_mces = hf["mces"][:].astype(np.float32)
    print(f"  {len(hdf5_smi2idx):,} mols in HDF5")

    # ── Build index arrays ────────────────────────────────────────────────────
    train_lb_idxs = np.array([lb_smi2idx.get(s, -1) for s in train_can], dtype=np.int64)
    train_hdf5_idxs = np.array(
        [hdf5_smi2idx.get(s, -1) for s in train_can], dtype=np.int64
    )

    unique_test = list(dict.fromkeys(test_can))
    test_lb_map = {s: lb_smi2idx.get(s, -1) for s in unique_test}
    test_hdf5_map = {s: hdf5_smi2idx.get(s, -1) for s in unique_test}

    # One representative test spectrum per unique test mol (first occurrence)
    test_mol_to_spec: dict[str, int] = {}
    for i, s in enumerate(test_can):
        if s not in test_mol_to_spec:
            test_mol_to_spec[s] = i

    covered = [s for s in unique_test if test_lb_map[s] >= 0 or test_hdf5_map[s] >= 0]
    print(f"  covered test mols: {len(covered):,} / {len(unique_test):,}")

    # ── Collect cross-split pairs per MCES bin ────────────────────────────────
    print(
        f"\nCollecting pairs (up to {args.pairs_per_bin} train specs per MCES bin per test mol) ..."
    )
    collected_test_spec = []  # test spectrum index  (per-bin collection)
    collected_train_spec = []  # train spectrum index
    collected_gt = []  # GT MCES

    random_test_spec = []  # truly random collection
    random_train_spec = []
    random_gt = []

    n_bins = len(BINS_MCES) - 1

    for test_smi in tqdm(covered, desc="test mols"):
        lb_idx = test_lb_map[test_smi]
        hdf5_idx = test_hdf5_map[test_smi]
        t_spec = test_mol_to_spec[test_smi]

        gt_vec = gt_mces_to_all_train(
            lb_idx, hdf5_idx, train_lb_idxs, train_hdf5_idxs, lb, hdf5_mces
        )

        # ── Per-bin collection ─────────────────────────────────────────────
        bin_ids = np.clip(np.digitize(gt_vec, BINS_MCES) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            where = np.where(bin_ids == b)[0]
            if len(where) == 0:
                continue
            sample = rng.choice(
                where, size=min(len(where), args.pairs_per_bin), replace=False
            )
            collected_test_spec.extend([t_spec] * len(sample))
            collected_train_spec.extend(sample.tolist())
            collected_gt.extend(gt_vec[sample].tolist())

        # ── Truly random collection ────────────────────────────────────────
        rand_sample = rng.choice(
            N_train, size=min(N_train, args.random_per_mol), replace=False
        )
        random_test_spec.extend([t_spec] * len(rand_sample))
        random_train_spec.extend(rand_sample.tolist())
        random_gt.extend(gt_vec[rand_sample].tolist())

    collected_gt = np.array(collected_gt, dtype=np.float32)
    collected_test_spec = np.array(collected_test_spec, dtype=np.int64)
    collected_train_spec = np.array(collected_train_spec, dtype=np.int64)
    random_gt = np.array(random_gt, dtype=np.float32)
    random_test_spec = np.array(random_test_spec, dtype=np.int64)
    random_train_spec = np.array(random_train_spec, dtype=np.int64)
    print(f"  Per-bin collected: {len(collected_gt):,} pairs before balancing")
    print(f"  Truly random:      {len(random_gt):,} pairs")

    # ── Compute pred MCES from embeddings ────────────────────────────────────
    print("Computing pred MCES from embeddings ...")
    t_emb = train_embs[collected_train_spec]
    te_emb = test_embs[collected_test_spec]
    pred_raw = (1.0 - (te_emb * t_emb).sum(dim=1).numpy()) * 40.0
    gt_raw = collected_gt

    tr_emb = train_embs[random_train_spec]
    te_emb2 = test_embs[random_test_spec]
    pred_rand = (1.0 - (te_emb2 * tr_emb).sum(dim=1).numpy()) * 40.0
    gt_rand = random_gt

    # ── GT-balance ────────────────────────────────────────────────────────────
    bal_idx = balance_by_gt(gt_raw, rng)
    gt_bal = gt_raw[bal_idx]
    pred_bal = pred_raw[bal_idx]
    print(f"  Per-bin: {len(gt_raw):,} → balanced {len(gt_bal):,}")
    print(f"  Random:  {len(gt_rand):,} pairs")

    r, _ = spearmanr(gt_bal, pred_bal)
    mse = float(np.mean((pred_bal - gt_bal) ** 2))
    print(
        f"  Balanced — rho={r:.4f}  MSE={mse:.4f}  GT mean={gt_bal.mean():.2f}  pred mean={pred_bal.mean():.2f}"
    )

    # ── Determine step from intermediates_dir name ────────────────────────────
    step = 0
    for part in intermediates_dir.name.replace("-", "_").split("_"):
        if part.startswith("step") and part[4:].isdigit():
            step = int(part[4:])

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_figure(
        gt_rand, pred_rand, gt_raw, pred_raw, gt_bal, pred_bal, step, args.output
    )
    print("Done.")


if __name__ == "__main__":
    main()
