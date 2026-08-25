"""014_2 analog discovery (see NOTES_014_2_ANALOG_DISCOVERY.md), stage 3:
exhaustive exact-MCES(threshold=20) for one search's full query x library
molecule-pair space.

Structurally this is the SAME prepare/compute_block/status/combine,
restart-safe, watchdog-protected block pipeline as
tools/compute_mces_exact_1020.py / tools/prepare_gt_mces_retrieval.py (see
tools/slurm/mces_exact_retrieval_candidates.slurm.sh for the SLURM array
pattern) -- _dispatch_with_watchdog below is carried over essentially
unmodified, since it's solver-agnostic (it just runs picklable worker_args
through a Pool with a per-pair wall-clock deadline and kills+restarts the
pool on a stall).

The ONE thing that's different, and the reason this is a new script rather
than a new --split option on the existing one: the existing tool's
compute_block lazily imports metabo_depthcharge.chem.similarities._mces_worker,
which requires an asimov2-only install this project's own sofia checkout
does not have. This script uses myopic_mces.myopic_mces.MCES directly
instead (PULP_CBC_CMD solver) -- confirmed fully local to this repo's own
.venv, no asimov2 dependency, and the exact same call used by
simba/core/chemistry/similarity_metrics.py::MolecularSimilarityMetrics.compute_mces
elsewhere in this codebase. Empirically timed at ~15 pairs/sec single-
threaded on real CASMI x NIST20 pairs (see NOTES_014_2_ANALOG_DISCOVERY.md,
"Cost estimate") -- this is why exhaustive, not top-K-only, is affordable
here with enough parallel workers.

Output schema matches tools/prepare_gt_mces_retrieval.py exactly (smiles.txt
+ mces_exact.npy as (N,3) float32 [mol_idx_a, mol_idx_b, mces]), so
tools/simba_retrieval_iceberg.py::load_gt_mces_lookup reads this file
unmodified -- no new lookup code needed downstream.

Directory layout (OUTPUT_DIR):
  meta.json              block plan and path pointers
  smiles.txt             one canonical SMILES per line (line N = mol_idx N;
                          query molecules first, then library molecules not
                          already covered by query overlap)
  pairs.npy              (N_PAIRS, 2) int32 -- [query_mol_idx, library_mol_idx]
  blocks/
    block_00000.npy      (n_pairs,) float32 -- exact MCES, -1 = solver failed/timed out
    block_00000.done     sentinel written after successful block completion
  mces_exact.npy          (N_PAIRS, 3) float32 -- [mol_idx_a, mol_idx_b, mces], written by combine

Dependencies (document clearly -- this is meant to run on "another CPU
server", possibly not this exact environment): rdkit, numpy, tqdm, and the
myopic_mces PyPI package with a working PULP_CBC_CMD binary (comes bundled
with the `pulp` package's own CBC binary on most platforms -- no separate
install needed, confirmed working on this repo's own .venv).

Typical workflow (mirrors the existing exact-MCES tools in this repo):
  # 1. Prepare (fast, seconds -- just parses the two SMILES lists + builds
  #    the exhaustive pair list). Run wherever the search_*.mgf file is.
  uv run python tools/analog_discovery_exact_mces.py \\
      --output_dir data/analog_discovery/search_A_exact_mces \\
      prepare --mgf data/analog_discovery/search_A_nist_msg.mgf --n_blocks 100

  # 2. Copy OUTPUT_DIR (meta.json, smiles.txt, pairs.npy -- blocks/ doesn't
  #    exist yet) to the target CPU server if different from where you ran
  #    prepare, then submit the SLURM array there:
  #    sbatch tools/slurm/analog_discovery_exact_mces.slurm.sh
  uv run python tools/analog_discovery_exact_mces.py \\
      --output_dir data/analog_discovery/search_A_exact_mces \\
      compute_block --task_id "$SLURM_ARRAY_TASK_ID" --n_jobs "$SLURM_CPUS_PER_TASK"

  # 3. Monitor / combine when done
  uv run python tools/analog_discovery_exact_mces.py --output_dir ... status
  uv run python tools/analog_discovery_exact_mces.py --output_dir ... combine
"""

import argparse
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
from rdkit import Chem


THRESHOLD = 20


def canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def _load_fold_smiles(mgf_path: str, fold: str) -> set[str]:
    """Fast manual MGF scan for one FOLD= value's SMILES -- same approach as
    prepare_gt_mces_retrieval.py::_load_test_fold_smiles (matchms's own
    loader is far slower on large files here, see that script's docstring).
    """
    smis = set()
    cur_fold = None
    cur_smi = None
    with open(mgf_path) as fh:
        for line in fh:
            line = line.strip()
            if line == "BEGIN IONS":
                cur_fold, cur_smi = None, None
            elif line.upper().startswith("FOLD="):
                cur_fold = line.split("=", 1)[1].strip()
            elif line.upper().startswith("SMILES="):
                cur_smi = line.split("=", 1)[1].strip()
            elif line == "END IONS" and cur_fold == fold and cur_smi:
                smis.add(cur_smi)
    return smis


# ── prepare ──────────────────────────────────────────────────────────────────


def prepare(output_dir: Path, mgf: str, n_blocks: int) -> None:
    """Build the exhaustive query x library molecule-pair list for one search."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blocks").mkdir(exist_ok=True)

    print(f"Scanning {mgf} for query/library SMILES ...")
    query_raw = _load_fold_smiles(mgf, "query")
    library_raw = _load_fold_smiles(mgf, "library")

    query_canon = sorted({c for s in query_raw if (c := canon(s))})
    library_canon = sorted({c for s in library_raw if (c := canon(s))})
    print(
        f"  {len(query_canon)} query molecules, {len(library_canon)} library molecules"
    )

    overlap = set(query_canon) & set(library_canon)
    if overlap:
        print(
            f"  WARNING: {len(overlap)} molecules present in BOTH query and library "
            "-- expected to be zero after prepare_analog_discovery_data.py's "
            "per-search CASMI exclusion; proceeding, but check the source data."
        )

    smiles_list = list(query_canon)
    mol_to_idx = {s: i for i, s in enumerate(smiles_list)}
    for s in library_canon:
        if s not in mol_to_idx:
            mol_to_idx[s] = len(smiles_list)
            smiles_list.append(s)

    query_idx = np.asarray([mol_to_idx[s] for s in query_canon], dtype=np.int32)
    library_idx = np.asarray([mol_to_idx[s] for s in library_canon], dtype=np.int32)

    print("Building exhaustive pair list (query x library) ...")
    pairs = np.empty((len(query_idx) * len(library_idx), 2), dtype=np.int32)
    pairs[:, 0] = np.repeat(query_idx, len(library_idx))
    pairs[:, 1] = np.tile(library_idx, len(query_idx))
    n_pairs = len(pairs)
    print(f"  {n_pairs:,} pairs")

    # Shuffle so blocks have similar difficulty distribution, not e.g. all of
    # one query's pairs (which share a molecule and so correlate in solve
    # time) landing in the same block. Fixed seed for reproducibility.
    rng = np.random.default_rng(seed=42)
    rng.shuffle(pairs)

    smiles_path = output_dir / "smiles.txt"
    smiles_path.write_text("\n".join(smiles_list) + "\n")
    pairs_path = output_dir / "pairs.npy"
    np.save(pairs_path, pairs)

    bounds = [int(round(i * n_pairs / n_blocks)) for i in range(n_blocks + 1)]
    bounds[-1] = n_pairs
    block_sizes = [bounds[i + 1] - bounds[i] for i in range(n_blocks)]

    meta = {
        "n_pairs": n_pairs,
        "n_mols": len(smiles_list),
        "n_query": len(query_idx),
        "n_library": len(library_idx),
        "threshold": THRESHOLD,
        "n_blocks": n_blocks,
        "bounds": bounds,
        "pairs_path": str(pairs_path.resolve()),
        "smiles_path": str(smiles_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    size_mb = pairs_path.stat().st_size / 1e6
    print(f"Wrote {smiles_path}, {pairs_path} ({size_mb:.1f} MB), meta.json")
    print(
        f"\nReady. {n_blocks} blocks of {min(block_sizes):,}-{max(block_sizes):,} pairs each."
    )
    print(
        f"At the empirically measured ~15 pairs/sec/worker (see NOTES doc), "
        f"{n_pairs:,} pairs / (15 * n_jobs * n_concurrent_tasks) seconds serial-per-slot."
    )


# ── compute_block ────────────────────────────────────────────────────────────


def _mces_worker(args):
    """Picklable (module-scope) worker: one exact-MCES pair via myopic_mces.
    Mirrors MolecularSimilarityMetrics.compute_mces's own call exactly (same
    solver + options), catch_errors=True so a single bad pair returns a
    sentinel instead of raising into the pool."""
    smi_a, smi_b, threshold = args
    from myopic_mces.myopic_mces import MCES as MCES2

    try:
        result = MCES2(
            smi_a,
            smi_b,
            threshold=threshold,
            i=0,
            solver="PULP_CBC_CMD",
            solver_options={"threads": 1, "msg": False, "timeLimit": 10},
            no_ilp_threshold=False,
            always_stronger_bound=True,
            catch_errors=True,
        )
        return float(result[1])
    except Exception:
        return -1.0


def _dispatch_with_watchdog(worker_args, n_jobs, per_pair_timeout, show_progress):
    """Same pool-restart watchdog as tools/compute_mces_exact_1020.py's own
    _dispatch_with_watchdog -- a pair stuck inside the ILP solver's native
    code never returns control to Python, so the only reliable way to bound
    per-pair wall time is to kill the whole worker process and resume the
    remaining pairs in a fresh pool (see that script's docstring for why a
    signal-based timeout does not work here)."""
    from tqdm.auto import tqdm

    n = len(worker_args)
    results = [None] * n
    todo = list(range(n))
    bar = tqdm(total=n, disable=not show_progress, unit="pair")

    while todo:
        pool = multiprocessing.Pool(n_jobs)
        it = pool.imap(_mces_worker, (worker_args[i] for i in todo))
        stuck_at = None
        try:
            for pos, idx in enumerate(todo):
                try:
                    results[idx] = it.next(timeout=per_pair_timeout)
                except multiprocessing.TimeoutError:
                    stuck_at = pos
                    break
                bar.update(1)
        finally:
            if stuck_at is not None:
                pool.terminate()
                pool.join()
                results[todo[stuck_at]] = -1.0
                bar.update(1)
                todo = todo[stuck_at + 1 :]
            else:
                pool.close()
                pool.join()
                todo = []

    bar.close()
    return results


def compute_block(
    output_dir: Path, task_id: int, n_jobs: int, timeout: float | None
) -> None:
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
    print(f"Block {task_id}: pairs [{i0:,}, {i1:,})  --  {n_pair:,} pairs")

    smiles = Path(meta["smiles_path"]).read_text().splitlines()
    pairs = np.load(meta["pairs_path"], mmap_mode="r")[i0:i1]
    smiles_a = [smiles[i] for i in pairs[:, 0]]
    smiles_b = [smiles[j] for j in pairs[:, 1]]

    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    per_pair_timeout = timeout if timeout else 300.0
    print(
        f"  MCES(threshold={meta['threshold']}, PULP_CBC_CMD, always_stronger_bound=True,"
        f" n_jobs={n_jobs}, per-pair timeout={per_pair_timeout}s, pool-restart watchdog)"
    )

    worker_args = [(a, b, meta["threshold"]) for a, b in zip(smiles_a, smiles_b)]
    result = np.asarray(
        _dispatch_with_watchdog(
            worker_args, n_jobs, per_pair_timeout, show_progress=True
        ),
        dtype=np.float32,
    )

    n_failed = int((result == -1.0).sum() + np.isnan(result).sum())
    if n_failed:
        print(
            f"  {n_failed:,} / {n_pair:,} pairs failed to solve or timed out -- recorded as -1."
        )
    result = np.where(np.isnan(result), -1.0, result).astype(np.float32)

    np.save(out_npy, result)
    done_file.write_text(f"block={task_id} n_pairs={n_pair} n_failed={n_failed}\n")
    print(f"Block {task_id}: wrote {out_npy.name}. Done.")


# ── status ───────────────────────────────────────────────────────────────────


def status(output_dir: Path) -> None:
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
        print("All blocks complete. Run: ... combine")


# ── combine ──────────────────────────────────────────────────────────────────


def combine(output_dir: Path) -> None:
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
            print(f"  Block {i} missing -- filling with NaN ({n_pair:,} pairs)")
            chunks.append(np.full(n_pair, np.nan, dtype=np.float32))

    mces_values = np.concatenate(chunks).astype(np.float32)
    pairs = np.load(meta["pairs_path"])

    result = np.column_stack([pairs.astype(np.float32), mces_values])
    out_path = output_dir / "mces_exact.npy"
    np.save(out_path, result)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path}  shape={result.shape}  size={size_mb:.1f} MB")

    n_solver_failed = int((mces_values == -1.0).sum())
    valid = ~np.isnan(mces_values) & (mces_values != -1.0)
    print(f"Valid  : {valid.sum():,} / {len(mces_values):,}")
    if n_solver_failed:
        print(
            f"Failed : {n_solver_failed:,} pairs recorded as -1 (solver could not solve)"
        )
    if valid.any():
        v = mces_values[valid]
        print(f"Mean   : {v.mean():.2f}")
        print(f"Range  : {v.min():.1f} - {v.max():.1f}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output_dir", type=Path, required=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="Build smiles.txt/pairs.npy/meta.json.")
    pp.add_argument(
        "--mgf", required=True, help="Combined search_*.mgf with FOLD=query/library"
    )
    pp.add_argument(
        "--n_blocks", type=int, default=100, help="Number of SLURM array tasks."
    )

    pb = sub.add_parser(
        "compute_block", help="Compute one block, one SLURM array task."
    )
    pb.add_argument("--task_id", type=int, required=True)
    pb.add_argument("--n_jobs", type=int, default=-1, help="-1 = all CPUs.")
    pb.add_argument(
        "--timeout", type=float, default=None, help="Per-pair wall-clock deadline (s)."
    )

    sub.add_parser("status", help="Print completion progress.")
    sub.add_parser("combine", help="Merge blocks into mces_exact.npy.")

    a = p.parse_args()
    if a.cmd == "prepare":
        prepare(a.output_dir, a.mgf, a.n_blocks)
    elif a.cmd == "compute_block":
        compute_block(a.output_dir, a.task_id, a.n_jobs, a.timeout)
    elif a.cmd == "status":
        status(a.output_dir)
    elif a.cmd == "combine":
        combine(a.output_dir)


if __name__ == "__main__":
    main()
