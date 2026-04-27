"""
Fast pair computation: generate all pairs → cache hit → compute misses.

Workers receive explicit (idx0, idx1) arrays; no RNG, no per-chunk pools.
One multiprocessing.Pool per split, created only when there are pairs to compute.
"""

import multiprocessing
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from simba.core.chemistry.edit_distance.edit_distance import (
    simba_solve_pair_edit_distance,
    simba_solve_pair_mces,
)
from simba.core.data.molecule_pairs import MoleculePairsOpt
from simba.core.training.train_utils import TrainUtils
from simba.utils.logger_setup import logger


def _compute_worker(idx0, idx1, all_smiles, fps, mols, threshold_mces, output_path):
    n = len(idx0)
    out = np.empty((n, 4))
    out[:, 0] = idx0
    out[:, 1] = idx1
    for k in range(n):
        i, j = int(idx0[k]), int(idx1[k])
        out[k, 2], _ = simba_solve_pair_edit_distance(
            all_smiles[i], all_smiles[j], fps[i], fps[j], mols[i], mols[j]
        )
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

    logger.info(
        f"{identifier}: computing {len(idx0):,} pairs with {num_workers} workers"
    )

    splits = np.array_split(np.arange(len(idx0)), num_workers)
    worker_files = []
    with multiprocessing.Pool(num_workers) as pool:
        handles = []
        for w, s in enumerate(splits):
            if len(s) == 0:
                continue
            fpath = f"{preprocessing_dir}tmp_worker_{identifier}_node{current_node}_w{w}.npy"
            worker_files.append(fpath)
            handles.append(
                pool.apply_async(
                    _compute_worker,
                    (idx0[s], idx1[s], all_smiles, fps, mols, threshold_mces, fpath),
                )
            )
        pool.close()
        pool.join()
        for h in handles:
            h.get()

    computed_arr = np.concatenate([np.load(f) for f in worker_files], axis=0)
    for f in worker_files:
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
