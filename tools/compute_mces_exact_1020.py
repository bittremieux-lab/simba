"""Compute exact MCES(threshold=20) for training pairs with lb in [10, 20].

Stages — run in this order:
  prepare        extract pairs from preprocessing, write meta + smiles.txt + pairs.npy
  compute_block  one SLURM array task: compute exact MCES for its pair block
  status         show progress (run any time, no side effects)
  combine        concatenate block results → mces_exact_10_20.npy

Directory layout:
  OUTPUT_DIR/
    meta.json              block plan and path pointers
    smiles.txt             one canonical SMILES per line (line N = mol_idx N)
    pairs.npy              (N_PAIRS, 2) int32 — [mol_idx_a, mol_idx_b], a < b
    blocks/
      block_00000.npy      (n_pairs,) float32 — exact MCES capped at 20
      block_00000.done     sentinel written after successful block completion
      ...
    mces_exact_10_20.npy   (N_PAIRS, 3) float32 — [mol_idx_a, mol_idx_b, mces]
                           written by combine

Typical workflow:
  # 1. Prepare locally (reads /mnt/data, writes ~500 MB to OUTPUT_DIR — ~60 s)
  uv run python tools/compute_mces_exact_1020.py prepare

  # 2. Submit SLURM array (asimov2, 16 CPUs/task, up to 5 concurrent)
  sbatch tools/slurm/mces_exact_1020.slurm.sh

  # 3. Monitor
  python tools/compute_mces_exact_1020.py status
  # or: watch -n 60 'ls /mnt/data2/nkubrakov/mces_exact_1020/blocks/*.done | wc -l'

  # 4. Combine when all done
  uv run python tools/compute_mces_exact_1020.py combine

  # Restart failed blocks: just resubmit the same sbatch — done blocks are skipped.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


# ── Default paths ──────────────────────────────────────────────────────────
PREPRO_DIR = Path("/mnt/data/nkubrakov/massspecgym/preprocessing_msg_max_lb_hdf5")
TRAIN_PAIRS_NPY = PREPRO_DIR / "ed_mces_indexes_tani_incremental_train_node0_chunk0.npy"
MAPPING_PKL = PREPRO_DIR / "mapping.pkl"
DEFAULT_OUTPUT_DIR = Path("/mnt/data2/nkubrakov/mces_exact_1020")

LB_MIN, LB_MAX = 10.0, 20.0
THRESHOLD = 20


# ── prepare ────────────────────────────────────────────────────────────────


def prepare(output_dir: Path, n_blocks: int) -> None:
    """Extract pairs in [LB_MIN, LB_MAX], write smiles + pairs + meta."""
    import pickle

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blocks").mkdir(exist_ok=True)

    print("Loading mapping.pkl ...")
    with open(MAPPING_PKL, "rb") as f:
        mapping = pickle.load(f)
    smiles_list: list[str] = mapping["df_smiles_train"]["canon_smiles"].tolist()
    n_mols = len(smiles_list)
    print(f"  {n_mols:,} unique train molecules")

    smiles_path = output_dir / "smiles.txt"
    smiles_path.write_text("\n".join(smiles_list) + "\n")
    print(f"Wrote {smiles_path}")

    print("Loading training pairs ...")
    d = np.load(TRAIN_PAIRS_NPY)
    mask = (d[:, 3] >= LB_MIN) & (d[:, 3] <= LB_MAX)
    pairs = d[mask, :2].astype(np.int32)
    n_pairs = len(pairs)
    print(f"  {n_pairs:,} pairs with lb in [{LB_MIN}, {LB_MAX}]")

    # Shuffle so each block is a representative sample of the full distribution,
    # not just consecutive molecule indices. Fixed seed for reproducibility.
    rng = np.random.default_rng(seed=42)
    rng.shuffle(pairs)
    print("  Shuffled pairs (seed=42).")

    pairs_path = output_dir / "pairs.npy"
    np.save(pairs_path, pairs)
    size_mb = pairs_path.stat().st_size / 1e6
    print(f"Wrote {pairs_path} ({size_mb:.0f} MB)")

    # Equal-size blocks by pair count
    bounds = [int(round(i * n_pairs / n_blocks)) for i in range(n_blocks + 1)]
    bounds[-1] = n_pairs  # exact end
    block_sizes = [bounds[i + 1] - bounds[i] for i in range(n_blocks)]

    meta = {
        "n_pairs": n_pairs,
        "n_mols": n_mols,
        "lb_min": LB_MIN,
        "lb_max": LB_MAX,
        "threshold": THRESHOLD,
        "n_blocks": n_blocks,
        "bounds": bounds,
        "pairs_path": str(pairs_path.resolve()),
        "smiles_path": str(smiles_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}")

    min_sz, max_sz = min(block_sizes), max(block_sizes)
    print(f"\nReady. {n_blocks} blocks of {min_sz:,}–{max_sz:,} pairs each.")
    print("Submit: sbatch tools/slurm/mces_exact_1020.slurm.sh")


# ── compute_block ──────────────────────────────────────────────────────────


def compute_block(
    output_dir: Path,
    task_id: int,
    n_jobs: int,
    timeout: float | None,
) -> None:
    """Compute exact MCES for one block of pairs and write the result."""
    from metabo_depthcharge.chem.similarities import MCESDistance

    meta = json.loads((output_dir / "meta.json").read_text())
    n_blocks = meta["n_blocks"]

    if task_id >= n_blocks:
        print(f"task_id {task_id} >= n_blocks {n_blocks}; nothing to do.")
        return

    done_file = output_dir / "blocks" / f"block_{task_id:05d}.done"
    out_npy = output_dir / "blocks" / f"block_{task_id:05d}.npy"

    if done_file.exists():
        print(f"Block {task_id}: already done. Skipping.")
        return

    bounds = meta["bounds"]
    i0, i1 = bounds[task_id], bounds[task_id + 1]
    n_pair = i1 - i0
    print(f"Block {task_id}: pairs [{i0:,}, {i1:,})  —  {n_pair:,} pairs")

    smiles = Path(meta["smiles_path"]).read_text().splitlines()
    pairs = np.load(meta["pairs_path"], mmap_mode="r")[i0:i1]
    smiles_a = np.array([smiles[i] for i in pairs[:, 0]])
    smiles_b = np.array([smiles[j] for j in pairs[:, 1]])

    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    print(
        f"  MCESDistance(threshold={THRESHOLD}, always_stronger_bound=True,"
        f" n_jobs={n_jobs}, timeout={timeout})"
    )

    mces = MCESDistance(
        threshold=THRESHOLD,
        always_stronger_bound=True,
        n_jobs=n_jobs,
        solver_options={"msg": 0},
        timeout=timeout,
        progress=True,
    )
    result = np.asarray(mces(smiles_a, smiles_b), dtype=np.float32)

    np.save(out_npy, result)
    done_file.write_text(f"block={task_id} n_pairs={n_pair}\n")
    print(f"Block {task_id}: wrote {out_npy.name}. Done.")


# ── status ─────────────────────────────────────────────────────────────────


def status(output_dir: Path) -> None:
    """Print completion progress across all blocks."""
    meta_path = output_dir / "meta.json"
    if not meta_path.exists():
        print(f"No meta.json at {output_dir}. Run prepare first.")
        return

    meta = json.loads(meta_path.read_text())
    n_blocks = meta["n_blocks"]
    bounds = meta["bounds"]

    done_ids = [
        i
        for i in range(n_blocks)
        if (output_dir / "blocks" / f"block_{i:05d}.done").exists()
    ]
    pending_ids = [i for i in range(n_blocks) if i not in set(done_ids)]

    pairs_done = sum(bounds[i + 1] - bounds[i] for i in done_ids)
    n_pairs = meta["n_pairs"]

    print(
        f"Blocks : {len(done_ids):4d} / {n_blocks} done ({100 * len(done_ids) / n_blocks:.1f}%)"
    )
    print(f"Pairs  : {pairs_done:,} / {n_pairs:,} ({100 * pairs_done / n_pairs:.1f}%)")

    if pending_ids:
        sample = pending_ids[:15]
        suffix = (
            f" ... (+{len(pending_ids) - 15} more)" if len(pending_ids) > 15 else ""
        )
        print(f"Pending: {sample}{suffix}")
    else:
        print(
            "All blocks complete. Run: uv run python tools/compute_mces_exact_1020.py combine"
        )


# ── combine ────────────────────────────────────────────────────────────────


def combine(output_dir: Path) -> None:
    """Merge block result files into a single (N, 3) array."""
    meta = json.loads((output_dir / "meta.json").read_text())
    n_blocks = meta["n_blocks"]

    missing = [
        i
        for i in range(n_blocks)
        if not (output_dir / "blocks" / f"block_{i:05d}.done").exists()
    ]
    if missing:
        print(
            f"WARNING: {len(missing)} blocks not done: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    print(f"Loading {n_blocks} block files ...")
    chunks = []
    for i in range(n_blocks):
        p = output_dir / "blocks" / f"block_{i:05d}.npy"
        if p.exists():
            chunks.append(np.load(p))
        else:
            bounds = meta["bounds"]
            n_pair = bounds[i + 1] - bounds[i]
            print(f"  Block {i} missing — filling with NaN ({n_pair:,} pairs)")
            chunks.append(np.full(n_pair, np.nan, dtype=np.float32))

    mces_values = np.concatenate(chunks).astype(np.float32)
    pairs = np.load(meta["pairs_path"])

    result = np.column_stack([pairs.astype(np.float32), mces_values])
    out_path = output_dir / "mces_exact_10_20.npy"
    np.save(out_path, result)
    size_gb = out_path.stat().st_size / 1e9
    print(f"Wrote {out_path}  shape={result.shape}  size={size_gb:.2f} GB")

    valid = ~np.isnan(mces_values)
    print(f"Valid  : {valid.sum():,} / {len(mces_values):,}")
    if valid.any():
        v = mces_values[valid]
        print(f"Mean   : {v.mean():.2f}")
        print(f"Range  : {v.min():.1f} – {v.max():.1f}")
        for thresh in [10, 15, 20]:
            pct = (v <= thresh).mean() * 100
            print(f"  ≤ {thresh} : {pct:.1f}%")


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute exact MCES(threshold=20) for training pairs with lb in [10, 20].",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser(
        "prepare", help="Extract pairs and write meta/smiles/pairs.npy."
    )
    pp.add_argument(
        "--n_blocks", type=int, default=200, help="Number of SLURM array tasks."
    )

    pb = sub.add_parser(
        "compute_block", help="Compute one block (one SLURM array task)."
    )
    pb.add_argument(
        "--task_id",
        type=int,
        required=True,
        help="Block index (= $SLURM_ARRAY_TASK_ID).",
    )
    pb.add_argument(
        "--n_jobs", type=int, default=-1, help="Parallel workers (-1 = all CPUs)."
    )
    pb.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-pair solver timeout in seconds.",
    )

    sub.add_parser("status", help="Print completion progress.")
    sub.add_parser("combine", help="Merge blocks into mces_exact_10_20.npy.")

    a = p.parse_args()
    if a.cmd == "prepare":
        prepare(a.output_dir, a.n_blocks)
    elif a.cmd == "compute_block":
        compute_block(a.output_dir, a.task_id, a.n_jobs, a.timeout)
    elif a.cmd == "status":
        status(a.output_dir)
    elif a.cmd == "combine":
        combine(a.output_dir)


if __name__ == "__main__":
    main()
