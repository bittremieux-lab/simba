"""
Fast pair computation: generate all pairs → cache hit → compute misses.

Workers receive explicit (idx0, idx1) arrays; no RNG, no per-chunk pools.
One multiprocessing.Pool per split, created only when there are pairs to compute.
"""

import multiprocessing
import os
import time

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from simba.core.chemistry.edit_distance.edit_distance import (
    simba_solve_pair_edit_distance,
    simba_solve_pair_mces,
)
from simba.core.data.molecule_pairs import MoleculePairsOpt
from simba.core.training.train_utils import TrainUtils
from simba.utils.logger_setup import logger


VERY_HIGH_DISTANCE = 666.0


def _compute_worker(
    idx0,
    idx1,
    all_smiles,
    fps,
    mols,
    threshold_mces,
    output_path,
    mces_precomputed=None,
):
    """Compute ED + MCES for each pair.

    mces_precomputed: float array of length n, NaN means compute normally.
    Only called for non-trivial pairs (Tanimoto >= 0.2, both mols <= 40 atoms).
    """
    n = len(idx0)
    out = np.empty((n, 4))
    out[:, 0] = idx0
    out[:, 1] = idx1
    for k in range(n):
        i, j = int(idx0[k]), int(idx1[k])
        out[k, 2], _ = simba_solve_pair_edit_distance(
            all_smiles[i], all_smiles[j], fps[i], fps[j], mols[i], mols[j]
        )
        if mces_precomputed is not None and not np.isnan(mces_precomputed[k]):
            out[k, 3] = mces_precomputed[k]
        else:
            out[k, 3], _ = simba_solve_pair_mces(
                all_smiles[i],
                all_smiles[j],
                fps[i],
                fps[j],
                mols[i],
                mols[j],
                threshold_mces,
            )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, out)


def _chunk_path(
    preprocessing_dir, prefix, identifier, current_node, num_nodes, chunk_idx
):
    tag = f"node{current_node}_chunk{chunk_idx}" if num_nodes > 1 else str(chunk_idx)
    return f"{preprocessing_dir}{prefix}indexes_tani_incremental{identifier}_{tag}.npy"


def _save_chunk(arr, preprocessing_dir, identifier, current_node, num_nodes, chunk_idx):
    ed_mces = arr
    ed = arr[:, :3]
    mces = arr[:, [0, 1, 3]]
    for prefix, data in [
        ("ed_mces_", ed_mces),
        ("edit_distance_", ed),
        ("mces_", mces),
    ]:
        path = _chunk_path(
            preprocessing_dir, prefix, identifier, current_node, num_nodes, chunk_idx
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, data)


def compute_pairs_for_split(
    all_spectra,
    preprocessing_dir,
    identifier,
    num_workers,
    current_node=0,
    num_nodes=1,
    precomputed_cache=None,
    threshold_mces=20,
    hdf5_mces_cache=None,
    hdf5_mces_threshold=10.0,
):
    if current_node is None:
        current_node, num_nodes = 0, 1

    spectra_unique, df_smiles = TrainUtils.get_unique_spectra(all_spectra)
    N = len(spectra_unique)
    all_smiles = [s.params["smiles"] for s in spectra_unique]

    mols = [Chem.MolFromSmiles(s) for s in all_smiles]
    fpgen = AllChem.GetRDKitFPGenerator(maxPath=3, fpSize=512)
    fps = [fpgen.GetFingerprint(m) for m in mols]

    node_rows = np.array_split(np.arange(N), num_nodes)[current_node]
    idx0 = np.repeat(node_rows, N).astype(np.int32)
    idx1 = np.tile(np.arange(N), len(node_rows)).astype(np.int32)
    logger.info(
        f"{identifier}: {len(idx0):,} total pairs for node {current_node}/{num_nodes}"
    )

    chunk_idx = 0

    # ── precomputed cache (full ed+mces) ──────────────────────────────────────
    if precomputed_cache is not None:
        ed_vals = np.full(len(idx0), np.nan)
        mces_vals = np.full(len(idx0), np.nan)
        hit = np.zeros(len(idx0), dtype=bool)
        for k in range(len(idx0)):
            entry = precomputed_cache.get(
                tuple(sorted([all_smiles[idx0[k]], all_smiles[idx1[k]]]))
            )
            if entry is not None:
                hit[k] = True
                ed_vals[k] = entry[0]
                mces_vals[k] = entry[1]
        logger.info(f"{identifier}: {hit.sum():,} / {len(idx0):,} pairs from cache")
        if hit.any():
            cached_arr = np.column_stack(
                [idx0[hit], idx1[hit], ed_vals[hit], mces_vals[hit]]
            )
            _save_chunk(
                cached_arr,
                preprocessing_dir,
                identifier,
                current_node,
                num_nodes,
                chunk_idx,
            )
            chunk_idx += 1
        idx0 = idx0[~hit]
        idx1 = idx1[~hit]

    if len(idx0) == 0:
        logger.info(f"{identifier}: all pairs served from cache, no workers needed")
        return MoleculePairsOpt(
            original_spectra=all_spectra,
            unique_spectra=spectra_unique,
            df_smiles=df_smiles,
            pair_distances=np.empty((0, 3)),
        )

    # ── trivial pre-filter (Tanimoto < 0.2, self-pairs, >40 atoms) ───────────
    # Use BulkTanimotoSimilarity row-by-row (C-speed) to classify pairs without
    # spawning workers. Workers only see pairs that actually need ILP computation.
    atom_counts = np.array([m.GetNumAtoms() if m is not None else 9999 for m in mols])
    right_large = atom_counts > 40  # reusable mask over all N molecules

    # Rebuild idx0/idx1 as a structured view: k_row * N + j for row k_row, col j
    # We iterate row by row to reuse BulkTanimotoSimilarity results
    n_rows = len(node_rows)
    trivial_mask = np.zeros(len(idx0), dtype=bool)
    triv_ed = np.full(len(idx0), np.nan)
    triv_mces = np.full(len(idx0), np.nan)

    for k_row in tqdm(
        range(n_rows), desc=f"{identifier} pre-filter", unit="rows", disable=False
    ):
        row_i = int(node_rows[k_row])
        start, end = k_row * N, (k_row + 1) * N

        bulk_tani = np.array(DataStructs.BulkTanimotoSimilarity(fps[row_i], fps))
        low_tani = bulk_tani < 0.2
        large = (atom_counts[row_i] > 40) | right_large

        triv_slice = low_tani | large
        triv_slice[row_i] = True  # self-pair

        ed_slice = np.full(N, np.nan)
        mces_slice = np.full(N, np.nan)
        ed_slice[low_tani] = VERY_HIGH_DISTANCE
        mces_slice[low_tani] = VERY_HIGH_DISTANCE
        ed_slice[row_i] = 0.0  # self-pair overrides low_tani (Tanimoto=1.0, never low)
        mces_slice[row_i] = 0.0
        # large & ~low_tani stays NaN

        trivial_mask[start:end] = triv_slice
        triv_ed[start:end] = ed_slice
        triv_mces[start:end] = mces_slice

    n_trivial = int(trivial_mask.sum())
    logger.info(
        f"{identifier}: {n_trivial:,} / {len(idx0):,} pairs are trivial"
        f" (Tanimoto<0.2, self, or >40 atoms) — saved without workers"
    )
    if n_trivial > 0:
        triv_arr = np.column_stack(
            [
                idx0[trivial_mask],
                idx1[trivial_mask],
                triv_ed[trivial_mask],
                triv_mces[trivial_mask],
            ]
        )
        _save_chunk(
            triv_arr, preprocessing_dir, identifier, current_node, num_nodes, chunk_idx
        )
        chunk_idx += 1

    idx0 = idx0[~trivial_mask]
    idx1 = idx1[~trivial_mask]

    if len(idx0) == 0:
        logger.info(f"{identifier}: all pairs trivial, no workers needed")
        return MoleculePairsOpt(
            original_spectra=all_spectra,
            unique_spectra=spectra_unique,
            df_smiles=df_smiles,
            pair_distances=np.empty((0, 3)),
        )

    # ── HDF5 MCES cache (mces-only, applied to non-trivial pairs) ────────────
    # Only values ≤ hdf5_mces_threshold are used; these skip the MCES ILP in workers.
    mces_from_hdf5 = None
    if hdf5_mces_cache is not None:
        mces_from_hdf5 = np.full(len(idx0), np.nan)
        hdf5_hits = 0
        for k in range(len(idx0)):
            val = hdf5_mces_cache.lookup(all_smiles[idx0[k]], all_smiles[idx1[k]])
            if val is not None and val <= hdf5_mces_threshold:
                mces_from_hdf5[k] = val
                hdf5_hits += 1
        logger.info(
            f"{identifier}: {hdf5_hits:,} / {len(idx0):,} non-trivial pairs"
            f" have HDF5 MCES ≤ {hdf5_mces_threshold}"
        )
        if hdf5_hits == 0:
            mces_from_hdf5 = None

    logger.info(
        f"{identifier}: computing {len(idx0):,} non-trivial pairs"
        f" with {num_workers} workers"
    )

    # ── worker pool ───────────────────────────────────────────────────────────
    # Pairs are split into fixed-size batches (PAIRS_PER_BATCH).  Each batch
    # saves immediately on completion so progress survives a job restart:
    # existing batch files are skipped, only missing ones are recomputed.
    PAIRS_PER_BATCH = 1000
    batch_starts = list(range(0, len(idx0), PAIRS_PER_BATCH))
    n_batches = len(batch_starts)

    def _batch_path(b):
        return f"{preprocessing_dir}worker_{identifier}_node{current_node}_b{b}.npy"

    pending_batches = [
        b for b in range(n_batches) if not os.path.exists(_batch_path(b))
    ]
    n_skip = n_batches - len(pending_batches)
    if n_skip:
        logger.info(
            f"{identifier}: resuming — {n_skip}/{n_batches} batches already done,"
            f" {len(pending_batches)} remaining"
        )
    logger.info(
        f"{identifier}: {n_batches} batches of ≤{PAIRS_PER_BATCH} pairs,"
        f" pool size {num_workers}"
    )

    if pending_batches:
        with multiprocessing.Pool(num_workers) as pool:
            handles = []
            for b in pending_batches:
                s = slice(batch_starts[b], batch_starts[b] + PAIRS_PER_BATCH)
                handles.append(
                    pool.apply_async(
                        _compute_worker,
                        (
                            idx0[s],
                            idx1[s],
                            all_smiles,
                            fps,
                            mols,
                            threshold_mces,
                            _batch_path(b),
                            mces_from_hdf5[s] if mces_from_hdf5 is not None else None,
                        ),
                    )
                )
            pool.close()
            n_handles = len(handles)
            log_interval = 60
            last_log = time.monotonic()
            while True:
                done = sum(1 for h in handles if h.ready())
                if done == n_handles:
                    logger.info(f"{identifier}: all {n_handles} pending batches done")
                    break
                if time.monotonic() - last_log >= log_interval:
                    total_done = n_skip + done
                    logger.info(
                        f"{identifier}: {total_done}/{n_batches} batches done"
                        f" ({100 * total_done // n_batches}%)"
                    )
                    last_log = time.monotonic()
                time.sleep(5)
            pool.join()
            for h in handles:
                h.get()

    batch_files = [_batch_path(b) for b in range(n_batches)]
    computed_arr = np.concatenate([np.load(f) for f in batch_files], axis=0)
    for f in batch_files:
        os.remove(f)

    _save_chunk(
        computed_arr, preprocessing_dir, identifier, current_node, num_nodes, chunk_idx
    )

    return MoleculePairsOpt(
        original_spectra=all_spectra,
        unique_spectra=spectra_unique,
        df_smiles=df_smiles,
        pair_distances=computed_arr[:, :3],
    )
