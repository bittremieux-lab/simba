"""
Validation script: reproduce the Inference hexbin [test] plot using
precomputed embeddings instead of running the model.

Loads the same npy pairs as run_val_hexbin.py, maps molecule indices to
test_embeddings.pt via df_smiles_test from mapping.pkl, and computes
  pred_mces = (1 - dot(emb_a, emb_b)) * 40
Expected: rho / MSE close to "Inference hexbin · step 44k official test"
(rho=0.707, MSE=45.73).

Also shows a second panel with GT from max(lb_matrix, HDF5) instead of
the npy column, to validate GT source consistency.

Usage:
    uv run python tools/test_test_mces_hexbin.py \
        --intermediates_dir /mnt/data/nkubrakov/experiments_3_dataset/retrieval/bs2048_v2_step44k \
        --npy_pairs /mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020/ed_mces_indexes_tani_incremental_test_node0_chunk0.npy \
        --prepro_pkl /mnt/data/nkubrakov/massspecgym/preprocessing_msg_exact_mces_1020/mapping.pkl \
        --output results/test_test_mces_hexbin_bs2048_v2_step44k.png
"""

import argparse
import json
import pickle
from pathlib import Path

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
BIN_STEP = 2.5
BINS_MCES = np.arange(0, 40 + BIN_STEP, BIN_STEP)
COLOR = "#4E9A7A"


def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def balance_by_gt(gt, rng):
    bin_ids = np.clip(np.digitize(gt, BINS_MCES) - 1, 0, len(BINS_MCES) - 2)
    counts = np.bincount(bin_ids, minlength=len(BINS_MCES) - 1)
    cap = int(1.5 * counts[counts > 0].min())
    idx = []
    for b in range(len(BINS_MCES) - 1):
        where = np.where(bin_ids == b)[0]
        if len(where):
            idx.append(rng.choice(where, size=min(len(where), cap), replace=False))
    return np.concatenate(idx)


def hexbin_panel(fig, subplot_spec, gt, pred, title, n):
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

    ax_main.hexbin(
        gt, pred, gridsize=16, cmap="Greens", mincnt=1, extent=[0, 40, 0, 40]
    )
    ax_main.plot([0, 40], [0, 40], "r--", lw=1)
    ax_main.set_xlabel("GT MCES", fontsize=10)
    ax_main.set_ylabel("Pred MCES (embeddings)", fontsize=10)
    ax_main.set_xlim(0, 40)
    ax_main.set_ylim(0, 40)
    ax_main.grid(True, alpha=0.2)

    ax_top.hist(gt, bins=bins, color=COLOR, edgecolor="none")
    ax_top.set_xlim(0, 40)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_ylabel("count", fontsize=7)
    ax_top.grid(True, alpha=0.2)
    ax_top.tick_params(labelsize=6)
    ax_top.set_title(f"{title}\nρ={r:.3f}  MSE={mse:.2f}  n={n:,}", fontsize=8)

    ax_right.hist(
        pred, bins=bins, orientation="horizontal", color=COLOR, edgecolor="none"
    )
    ax_right.set_ylim(0, 40)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.set_xlabel("count", fontsize=7)
    ax_right.grid(True, alpha=0.2)
    ax_right.tick_params(labelsize=6)

    return r, mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--intermediates_dir", required=True)
    p.add_argument("--npy_pairs", required=True)
    p.add_argument("--prepro_pkl", required=True)
    p.add_argument("--output", default="results/test_test_mces_hexbin.png")
    args = p.parse_args()

    idir = Path(args.intermediates_dir)

    # ── Load test embeddings ──────────────────────────────────────────────────
    print("Loading test embeddings and SMILES ...")
    test_embs = torch.load(idir / "test_embeddings.pt").cpu().float()
    test_smiles = json.loads((idir / "test_smiles.json").read_text())
    test_can = [canonicalize(s) for s in test_smiles]
    mol_to_spec = {}
    for i, s in enumerate(test_can):
        if s not in mol_to_spec:
            mol_to_spec[s] = i
    print(f"  test spectra: {len(test_smiles):,}  unique mols: {len(mol_to_spec):,}")

    # ── Load npy pairs: (mol_a, mol_b, _, gt_mces) ───────────────────────────
    print("Loading npy pairs ...")
    npy = np.load(args.npy_pairs)
    mol_a_idx = npy[:, 0].astype(np.int64)
    mol_b_idx = npy[:, 1].astype(np.int64)
    gt_npy = npy[:, 3].astype(np.float32)
    print(f"  {len(gt_npy):,} pairs  GT mean={gt_npy.mean():.2f}")

    # ── Map npy mol_idx → embedding idx via df_smiles_test ───────────────────
    print("Loading prepro pkl for mol-idx→SMILES mapping ...")
    with open(args.prepro_pkl, "rb") as f:
        prepro = pickle.load(f)
    df_test = prepro["df_smiles_test"]  # row i = mol index i in npy
    mol_idx_to_emb = {}
    skipped = 0
    for mol_idx, row in enumerate(df_test.itertuples()):
        smi = canonicalize(row.canon_smiles)
        emb_idx = mol_to_spec.get(smi, -1)
        if emb_idx >= 0:
            mol_idx_to_emb[mol_idx] = emb_idx
        else:
            skipped += 1
    print(f"  mapped {len(mol_idx_to_emb):,}/{len(df_test):,} mols  skipped {skipped}")

    # Filter pairs where both mols have an embedding
    mask = np.array(
        [
            a in mol_idx_to_emb and b in mol_idx_to_emb
            for a, b in zip(mol_a_idx, mol_b_idx)
        ]
    )
    mol_a_idx = mol_a_idx[mask]
    mol_b_idx = mol_b_idx[mask]
    gt_npy = gt_npy[mask]
    print(f"  after filter: {len(gt_npy):,} pairs")

    emb_a_idx = np.array([mol_idx_to_emb[i] for i in mol_a_idx])
    emb_b_idx = np.array([mol_idx_to_emb[i] for i in mol_b_idx])

    # ── Compute pred in batches ───────────────────────────────────────────────
    print("Computing pred MCES from embeddings ...")
    BATCH = 100_000
    pred_chunks = []
    for start in tqdm(range(0, len(emb_a_idx), BATCH)):
        ea = test_embs[emb_a_idx[start : start + BATCH]]
        eb = test_embs[emb_b_idx[start : start + BATCH]]
        pred_chunks.append((1.0 - (ea * eb).sum(dim=1).numpy()) * 40.0)
    pred = np.concatenate(pred_chunks).astype(np.float32)

    # ── GT-balance (same 1.5-rule as mces_hexbin_balanced.png) ───────────────
    rng = np.random.default_rng(42)
    bal = balance_by_gt(gt_npy, rng)
    gt_b = gt_npy[bal]
    pr_b = pred[bal]
    print(f"  Balanced: {len(bal):,} pairs (from {len(gt_npy):,})")

    r_raw, _ = spearmanr(gt_npy, pred)
    mse_raw = float(np.mean((pred - gt_npy) ** 2))
    r_bal, _ = spearmanr(gt_b, pr_b)
    mae_bal = float(np.mean(np.abs(pr_b - gt_b)))
    print(f"  Unbalanced: rho={r_raw:.4f}  MSE={mse_raw:.3f}")
    print(
        f"  Balanced:   rho={r_bal:.4f}  MAE={mae_bal:.3f}  "
        f"(reference: rho=0.628  MAE=6.91)"
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7))
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    step = 0
    for part in idir.name.replace("-", "_").split("_"):
        if part.startswith("step") and part[4:].isdigit():
            step = int(part[4:])

    hexbin_panel(
        fig,
        outer[0],
        gt_npy,
        pred,
        f"Unbalanced (all {len(gt_npy):,} npy pairs) — step {step:,}",
        len(gt_npy),
    )
    hexbin_panel(
        fig, outer[1], gt_b, pr_b, f"GT-balanced (1.5-rule) — step {step:,}", len(gt_b)
    )

    fig.suptitle(
        "Validation: test-test hexbin via precomputed embeddings  "
        "[ GT = npy col3,  Pred = (1−dot(emb_a, emb_b))×40 ]\n"
        "Right panel should match mces_hexbin_balanced.png test panel "
        "(rho=0.628  MAE=6.91)",
        fontsize=9,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
