import os
import sys
from functools import lru_cache

import dill
import numpy as np
import pandas as pd
from myopic_mces.myopic_mces import MCES as MCES2
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from tqdm import tqdm

import simba.core.chemistry.edit_distance.mol_utils as mu
from simba.utils.logger_setup import logger


VERY_HIGH_DISTANCE = 666

_GLOBAL_CACHE = None


def set_global_cache(cache: dict) -> None:
    """Set global cache before forking workers."""
    global _GLOBAL_CACHE
    _GLOBAL_CACHE = cache


def get_global_cache() -> dict:
    """Get global cache in worker process."""
    return _GLOBAL_CACHE


class PrecomputedDistancesCache:
    """
    Memory-efficient precomputed distances cache using sorted numpy arrays + binary search.

    Replaces a Python dict (which needs ~100M+ entries) with per-split sorted
    numpy arrays. Loading is fast (numpy mmap I/O); lookup is O(log N) binary search.
    Interface mirrors dict: supports .get(key), len(), bool().
    """

    def __init__(self) -> None:
        # Each entry: (smiles_to_idx, i_col, j_col, ed_col, mces_col)
        self._splits: list[tuple] = []

    def add_split(
        self,
        smiles_list: list[str],
        ed_mces_files: list[str],
    ) -> None:
        """Load one split's ed_mces_*.npy files into a sorted numpy array."""
        if not smiles_list or not ed_mces_files:
            return

        smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}

        arrays = []
        for f in ed_mces_files:
            try:
                arrays.append(np.load(f))
            except Exception as e:
                logger.warning(f"Error loading {f}: {e}")
        if not arrays:
            return

        arr = np.concatenate(arrays, axis=0)  # (N, 4): [i, j, ed, mces]

        # Normalize so i <= j (handles symmetric / full-matrix storage)
        i_col = arr[:, 0].astype(np.int32)
        j_col = arr[:, 1].astype(np.int32)
        swap = i_col > j_col
        i_col[swap], j_col[swap] = j_col[swap].copy(), i_col[swap].copy()

        # Sort lexicographically by (i, j)
        sort_idx = np.lexsort((j_col, i_col))
        i_col = i_col[sort_idx]
        j_col = j_col[sort_idx]
        ed_col = arr[sort_idx, 2].astype(np.float32)
        mces_col = arr[sort_idx, 3].astype(np.float32)

        # Remove duplicate (i, j) pairs introduced by normalization
        n = len(smiles_list)
        packed = i_col.astype(np.int64) * (n + 1) + j_col
        _, uniq = np.unique(packed, return_index=True)
        i_col = i_col[uniq]
        j_col = j_col[uniq]
        ed_col = ed_col[uniq]
        mces_col = mces_col[uniq]

        self._splits.append((smiles_to_idx, i_col, j_col, ed_col, mces_col))
        logger.info(
            f"  split cached: {len(i_col):,} unique pairs ({len(smiles_list):,} molecules)"
        )

    def get(self, key: tuple) -> list | None:
        """Return [ed, mces] for (smiles_a, smiles_b), or None if not found."""
        smiles_a, smiles_b = key
        for smiles_to_idx, i_col, j_col, ed_col, mces_col in self._splits:
            ia = smiles_to_idx.get(smiles_a)
            ib = smiles_to_idx.get(smiles_b)
            if ia is None or ib is None:
                continue
            i, j = (ia, ib) if ia <= ib else (ib, ia)
            lo = int(np.searchsorted(i_col, i))
            hi = int(np.searchsorted(i_col, i, side="right"))
            if lo >= hi:
                continue
            k = int(np.searchsorted(j_col[lo:hi], j))
            if k >= (hi - lo) or j_col[lo + k] != j:
                continue
            row = lo + k
            return [float(ed_col[row]), float(mces_col[row])]
        return None

    def bulk_lookup(
        self,
        smiles_list: list[str],
        idx0: np.ndarray,
        idx1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized bulk lookup for arrays of pair indices.

        Parameters
        ----------
        smiles_list : list[str]
            The molecule SMILES list for the current run (idx0/idx1 index into this).
        idx0, idx1 : np.ndarray of int
            Pair indices into smiles_list.

        Returns
        -------
        hit : bool array, shape (len(idx0),)
        ed  : float32 array — ED values for hits (NaN for misses)
        mces: float32 array — MCES values for hits (NaN for misses)
        """
        n = len(idx0)
        ed_out = np.full(n, np.nan, dtype=np.float32)
        mces_out = np.full(n, np.nan, dtype=np.float32)
        hit = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)  # pairs not yet resolved

        for smiles_to_idx, i_col, j_col, ed_col, mces_col in self._splits:
            if not remaining.any():
                break

            ridx = np.where(remaining)[0]

            # Map current-run SMILES → old-split integer index (-1 if absent)
            # Build new→old index array
            new_to_old = np.full(len(smiles_list), -1, dtype=np.int32)
            for new_i, s in enumerate(smiles_list):
                old_i = smiles_to_idx.get(s)
                if old_i is not None:
                    new_to_old[new_i] = old_i

            old_i = new_to_old[idx0[ridx]]
            old_j = new_to_old[idx1[ridx]]

            # Only pairs where both SMILES exist in this split
            valid = (old_i >= 0) & (old_j >= 0)
            if not valid.any():
                continue

            vidx = ridx[valid]
            oi = old_i[valid].astype(np.int64)
            oj = old_j[valid].astype(np.int64)

            # Normalize so oi <= oj
            swap = oi > oj
            oi[swap], oj[swap] = oj[swap].copy(), oi[swap].copy()

            # Pack into single int64 for binary search
            n_split = len(smiles_to_idx)
            packed_cache = i_col.astype(np.int64) * (n_split + 1) + j_col
            packed_query = oi * (n_split + 1) + oj

            pos = np.searchsorted(packed_cache, packed_query)
            found = (pos < len(packed_cache)) & (packed_cache[pos] == packed_query)

            found_vidx = vidx[found]
            found_pos = pos[found]

            ed_out[found_vidx] = ed_col[found_pos]
            mces_out[found_vidx] = mces_col[found_pos]
            hit[found_vidx] = True
            remaining[found_vidx] = False

        return hit, ed_out, mces_out

    def __len__(self) -> int:
        return sum(len(ic) for _, ic, *_ in self._splits)

    def __bool__(self) -> bool:
        return bool(self._splits)


def filter_cache_by_smiles(cache, smiles_list: list[str]):
    """Filter cache to only keep entries for given SMILES.

    For PrecomputedDistancesCache the numpy lookup naturally handles absent SMILES
    (returns None), so filtering is a no-op — just return the same object.
    For legacy dict caches, filter as before.
    """
    if not cache:
        return {}

    if isinstance(cache, PrecomputedDistancesCache):
        # No filtering needed; binary search handles missing SMILES transparently.
        logger.info(f"Precomputed cache: {len(cache):,} pairs (no filtering needed)")
        return cache

    smiles_set = set(smiles_list)
    filtered = {
        key: val
        for key, val in cache.items()
        if key[0] in smiles_set and key[1] in smiles_set
    }

    logger.info(
        f"Cache: {len(filtered):,} entries kept out of {len(cache):,} ({len(filtered) / len(cache) * 100:.1f}%)"
    )
    return filtered


def load_precomputed_distances_cache(
    preprocessing_dirs: list[str],
) -> "PrecomputedDistancesCache":
    """
    Load precomputed distances from preprocessing directories.

    Auto-discovers files in each directory:
    - mapping_unique_smiles.pkl / mapping.pkl (for per-split SMILES lists)
    - ed_mces_*_{split}_*.npy files (4-column arrays: [i, j, ed, mces])

    Returns a PrecomputedDistancesCache backed by sorted numpy arrays.
    Lookups are O(log N) binary search — avoids building a 100M+ entry Python dict.

    Parameters
    ----------
    preprocessing_dirs : list[str]
        List of preprocessing directory paths.

    Returns
    -------
    PrecomputedDistancesCache
    """
    cache = PrecomputedDistancesCache()

    for prep_dir in preprocessing_dirs:
        if not os.path.exists(prep_dir):
            logger.warning(f"Preprocessing directory not found: {prep_dir}")
            continue

        # Find the pickle file
        pickle_path = os.path.join(prep_dir, "mapping_unique_smiles.pkl")
        if not os.path.exists(pickle_path):
            pickle_path = os.path.join(prep_dir, "mapping.pkl")
        if not os.path.exists(pickle_path):
            logger.warning(f"No mapping pickle found in {prep_dir}")
            continue

        # Discover ed_mces_*.npy files per split
        ed_mces_files: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for filename in sorted(os.listdir(prep_dir)):
            if not filename.startswith("ed_mces_") or not filename.endswith(".npy"):
                continue
            for split in ("train", "val", "test"):
                if f"_{split}_" in filename or f"_{split}." in filename:
                    ed_mces_files[split].append(os.path.join(prep_dir, filename))
                    break

        total = sum(len(v) for v in ed_mces_files.values())
        if total == 0:
            logger.warning(f"No ed_mces_*.npy files found in {prep_dir} — skipping")
            continue

        logger.info(
            f"Auto-discovered in {prep_dir}: "
            + ", ".join(
                f"{split}={len(ed_mces_files[split])}"
                for split in ("train", "val", "test")
            )
            + " ed_mces files"
        )

        # Load per-split SMILES lists from pickle
        try:
            with open(pickle_path, "rb") as f:
                data = dill.load(f)
        except Exception as e:
            logger.error(f"Error loading pickle {pickle_path}: {e}")
            continue

        for split in ("train", "val", "test"):
            if not ed_mces_files[split]:
                continue

            split_smiles = _extract_split_smiles(data, split)
            if not split_smiles:
                logger.warning(f"No SMILES found for split '{split}' in {pickle_path}")
                continue

            logger.info(
                f"Loading {len(ed_mces_files[split])} ed_mces files for split '{split}' "
                f"({len(split_smiles):,} molecules) ..."
            )
            cache.add_split(split_smiles, ed_mces_files[split])

    logger.info(f"Precomputed distances cache ready: {len(cache):,} unique pairs total")
    return cache


def _extract_split_smiles(data: dict, split: str) -> list[str]:
    """Extract ordered unique SMILES for one split from a loaded mapping pickle."""
    for key_prefix in ("molecule_pairs_", "df_smiles_"):
        key = key_prefix + split
        if key not in data or data[key] is None:
            continue
        if key_prefix == "molecule_pairs_":
            if not hasattr(data[key], "df_smiles"):
                continue
            df = data[key].df_smiles
        else:
            df = data[key]
        if df is None or df.empty:
            continue
        if "canon_smiles" in df.columns:
            raw = df["canon_smiles"].tolist()
        elif "smiles" in df.columns:
            raw = df["smiles"].tolist()
        else:
            raw = df.index.tolist()
        seen: set[str] = set()
        result = []
        for s in raw:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result
    return []


def _load_smiles_from_pickle(pickle_path: str) -> list[str]:
    """Extract unique SMILES list from preprocessing pickle."""
    try:
        with open(pickle_path, "rb") as f:
            data = dill.load(f)

        unique_smiles = []

        # Handle both old format (molecule_pairs_X) and new lightweight format (df_smiles_X)
        for key_prefix in ["molecule_pairs_", "df_smiles_"]:
            for split in ["train", "val", "test"]:
                key = key_prefix + split
                if key not in data or data[key] is None:
                    continue

                # Extract DataFrame
                if key_prefix == "molecule_pairs_":
                    if not hasattr(data[key], "df_smiles"):
                        continue
                    df = data[key].df_smiles
                else:
                    df = data[key]

                if df is None or df.empty:
                    continue

                # Extract SMILES from DataFrame
                if "canon_smiles" in df.columns:
                    smiles_list = df["canon_smiles"].tolist()
                elif "smiles" in df.columns:
                    smiles_list = df["smiles"].tolist()
                elif hasattr(df.index, "tolist"):
                    smiles_list = df.index.tolist()
                else:
                    continue

                unique_smiles.extend(smiles_list)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for s in unique_smiles:
            if s not in seen:
                seen.add(s)
                result.append(s)

        logger.info(f"Extracted {len(result)} unique SMILES from {pickle_path}")
        return result

    except Exception as e:
        logger.error(f"Error loading SMILES from {pickle_path}: {e}")
        return []


def _load_distances_from_files(
    distance_files: list[str], smiles_list: list[str]
) -> dict:
    """Load distances from multiple .npy files."""
    distance_cache = {}

    for dist_file in sorted(distance_files):
        try:
            distances = np.load(dist_file)
            for row in distances:
                idx1, idx2 = int(row[0]), int(row[1])

                # Skip out of bounds
                if idx1 >= len(smiles_list) or idx2 >= len(smiles_list):
                    continue

                # Map to SMILES
                smiles1, smiles2 = smiles_list[idx1], smiles_list[idx2]
                key = tuple(sorted([smiles1, smiles2]))

                # Store distance (3rd column)
                distance_cache[key] = float(row[2])

        except Exception as e:
            logger.warning(f"Error loading {dist_file}: {e}")

    return distance_cache


def create_input_df(smiles, indexes_0, indexes_1):
    df = pd.DataFrame()
    logger.info(f"Number of spectra: {len(smiles)}")

    df["smiles_0"] = [smiles[int(r)] for r in indexes_0]
    df["smiles_1"] = [smiles[int(r)] for r in indexes_1]

    return df


def compute_ed_and_mces_both(
    smiles: list[str],
    sampled_index: np.int64,
    batch_size: int,
    identifier: int,
    random_sampling: bool,
    preprocessing_dir: str,
    compute_specific_pairs: bool,
    threshold_mces: int,
    fps: list[ExplicitBitVect],
    mols: list[Mol],
    output_file: str = None,
    progress_position: int = 0,
    progress_desc: str = "Computing",
) -> np.ndarray | dict:
    """
    Compute BOTH edit distance AND MCES for a batch of molecule pairs in a single pass.

    This is the optimized version that eliminates redundant computations:
    - Load molecules once (not twice)
    - Generate fingerprints once (not twice)
    - Calculate Tanimoto similarity once (not twice)
    - Compute both ED and MCES for each pair

    Parameters
    ----------
    smiles : List[str]
        List of SMILES strings.
    sampled_index : np.int64
        Index to sample from the smiles list.
    batch_size : int
        The size of the batch to process.
    identifier : int
        An identifier for the batch (used for random seed).
    random_sampling : bool
        Whether to use random sampling of pairs.
    preprocessing_dir : str
        Directory for preprocessing files.
    compute_specific_pairs : bool
        Whether to compute specific pairs.
    threshold_mces : int
        MCES threshold value.
    fps : List[ExplicitBitVect]
        List of fingerprints corresponding to the smiles.
    mols : List[Mol]
        List of RDKit Mol objects corresponding to the smiles.
    output_file : str, optional
        If provided, saves results directly to this file instead of returning data.
    progress_position : int, optional
        Position for the tqdm progress bar (default: 0). Used for stacking multiple progress bars.
    progress_desc : str, optional
        Description text for the tqdm progress bar (default: "Computing").

    Returns
    -------
    np.ndarray | dict
        If output_file is None: A 2D numpy array with each row containing (index1, index2, ed_distance, mces_distance).
        If output_file is provided: A dict with metadata {"rows": int, "output_file": str}.
    """
    # 2D array to store the indexes and BOTH distances
    pair_distances = np.zeros((int(batch_size), 4))  # 4 columns: idx1, idx2, ED, MCES

    # Initialize randomness
    if random_sampling:
        np.random.seed(identifier)
        pair_distances[:, 0] = np.random.randint(0, len(smiles), int(batch_size))
        pair_distances[:, 1] = np.random.randint(0, len(smiles), int(batch_size))
    else:
        if batch_size > len(smiles):
            raise ValueError(
                f"batch_size ({batch_size}) cannot exceed the number of molecules ({len(smiles)})"
            )
        pair_distances[:, 0] = sampled_index
        pair_distances[:, 1] = np.arange(0, batch_size)

    ed_distances = []
    mces_distances = []

    # Track cache hits/misses
    cache_hits = 0
    cache_misses = 0

    # Get global cache (set before pool creation)
    precomputed_cache = get_global_cache()

    for index in tqdm(
        range(pair_distances.shape[0]),
        desc=progress_desc,
        position=progress_position,
        leave=False,
        unit="pair",
        bar_format="{desc}: {n_fmt}/{total_fmt} pairs [{elapsed}<{remaining}, {rate_fmt}]",
        disable=not sys.stdout.isatty(),
    ):
        pair = pair_distances[index]

        s0 = smiles[int(pair[0])]
        s1 = smiles[int(pair[1])]

        # Check precomputed cache first
        if precomputed_cache is not None:
            cache_key = tuple(sorted([s0, s1]))
            cached_values = precomputed_cache.get(cache_key)
            if cached_values is not None:
                ed_dist = cached_values[0]
                mces_dist = cached_values[1]
                cache_hits += 1
                ed_distances.append(ed_dist)
                mces_distances.append(mces_dist)
                continue

        # Not in cache, compute
        cache_misses += 1
        fp0 = fps[int(pair[0])]
        fp1 = fps[int(pair[1])]
        mol0 = mols[int(pair[0])]
        mol1 = mols[int(pair[1])]

        # Compute BOTH metrics in single pass
        ed_dist, _ = simba_solve_pair_edit_distance(s0, s1, fp0, fp1, mol0, mol1)
        mces_dist, _ = simba_solve_pair_mces(
            s0, s1, fp0, fp1, mol0, mol1, threshold_mces
        )

        ed_distances.append(ed_dist)
        mces_distances.append(mces_dist)

    pair_distances[:, 2] = ed_distances
    pair_distances[:, 3] = mces_distances

    # Log cache statistics
    if precomputed_cache is not None:
        total = cache_hits + cache_misses
        hit_rate = (cache_hits / total * 100) if total > 0 else 0
        logger.info(
            f"Cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate)"
        )

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, pair_distances)
        return {"rows": pair_distances.shape[0], "output_file": output_file}

    return pair_distances


def compute_ed_or_mces(
    smiles: list[str],
    sampled_index: np.int64,
    batch_size: int,
    identifier: int,
    random_sampling: bool,
    preprocessing_dir: str,
    compute_specific_pairs: bool,
    threshold_mces: int,
    fps: list[ExplicitBitVect],
    mols: list[Mol],
    use_edit_distance: bool,
    output_file: str = None,
    progress_position: int = 0,
    progress_desc: str = "Computing",
) -> np.ndarray | dict:
    """
    Compute the edit distance or MCES for a batch of molecule pairs.

    Parameters
    ----------
    smiles : List[str]
        List of SMILES strings.
    sampled_index : np.int64
        Index to sample from the smiles list.
    batch_size : int
        The size of the batch to process.
    identifier : int
        An identifier for the batch (used for random seed).
    random_sampling : bool
        Whether to use random sampling of pairs.
    preprocessing_dir : str
        Directory for preprocessing files.
    compute_specific_pairs : bool
        Whether to compute specific pairs.
    threshold_mces : int
        MCES threshold value.
    fps : List[ExplicitBitVect]
        List of fingerprints corresponding to the smiles.
    mols : List[Mol]
        List of RDKit Mol objects corresponding to the smiles.
    use_edit_distance : bool
        Whether to compute edit distance (True) or MCES (False).
    output_file : str, optional
        If provided, saves results to this file and returns metadata dict instead of array.
    progress_position : int, optional
        Position for the tqdm progress bar (default: 0). Used for stacking multiple progress bars.
    progress_desc : str, optional
        Description text for the tqdm progress bar (default: "Computing").

    Returns
    -------
    np.ndarray or dict
        If output_file is None: A 2D numpy array with each row containing (index1, index2, distance).
        If output_file is provided: A dict with keys 'rows' (int) and 'output_file' (str).
    """
    # 2D array to store the indexes and distances
    pair_distances = np.zeros(
        (int(batch_size), 3),
    )
    # initialize randomness
    if random_sampling:
        np.random.seed(identifier)
        pair_distances[:, 0] = np.random.randint(0, len(smiles), int(batch_size))
        pair_distances[:, 1] = np.random.randint(0, len(smiles), int(batch_size))
    else:
        if batch_size > len(smiles):
            raise ValueError(
                f"batch_size ({batch_size}) cannot exceed the number of molecules ({len(smiles)})"
            )
        pair_distances[:, 0] = sampled_index
        pair_distances[:, 1] = np.arange(0, batch_size)

    distances = []

    for index in tqdm(
        range(0, pair_distances.shape[0]),
        desc=progress_desc,
        position=progress_position,
        leave=False,
        unit="pair",
        bar_format="{desc}: {n_fmt}/{total_fmt} pairs [{elapsed}<{remaining}, {rate_fmt}]",
        disable=not sys.stdout.isatty(),
    ):
        pair = pair_distances[index]

        s0 = smiles[int(pair[0])]
        s1 = smiles[int(pair[1])]
        fp0 = fps[int(pair[0])]
        fp1 = fps[int(pair[1])]
        mol0 = mols[int(pair[0])]
        mol1 = mols[int(pair[1])]
        if use_edit_distance:
            dist, _ = simba_solve_pair_edit_distance(s0, s1, fp0, fp1, mol0, mol1)
        else:
            dist, _ = simba_solve_pair_mces(
                s0, s1, fp0, fp1, mol0, mol1, threshold_mces
            )
        distances.append(dist)

    pair_distances[:, 2] = distances

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, pair_distances)
        return {"rows": pair_distances.shape[0], "output_file": output_file}

    return pair_distances


def get_number_of_modification_edges(mol, substructure):
    if not mol.HasSubstructMatch(substructure):
        #    raise ValueError("The substructure is not a substructure of the molecule.")
        return None
    matches = mol.GetSubstructMatch(substructure)
    intersect = set(matches)
    modification_edges = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in intersect and bond.GetEndAtomIdx() in intersect:
            continue
        if bond.GetBeginAtomIdx() in intersect or bond.GetEndAtomIdx() in intersect:
            modification_edges.append(bond.GetIdx())

    return modification_edges


def get_edit_distance_from_smiles(smiles1, smiles2, return_nans=True):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    return simba_get_edit_distance(mol1, mol2, return_nans=return_nans)


def simba_get_edit_distance(mol1, mol2, return_nans=True):
    """
    Calculate the edit distance between two molecules.

    Parameters
    ----------
    mol1 : Mol
        First molecule.
    mol2 : Mol
        Second molecule.
    return_nans : bool, optional
        Whether to return NaN for dissimilar molecules (default: True).

    Returns
    -------
    float or np.nan
        Edit distance between mol1 and mol2.
    """

    mcs1 = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs1.smartsString)
    if return_nans:
        if (
            mcs_mol.GetNumAtoms() < mol1.GetNumAtoms() // 2
            and mcs_mol.GetNumAtoms() < mol2.GetNumAtoms() // 2
        ):
            return np.nan
        if mcs_mol.GetNumAtoms() < 2:
            return np.nan

    dist1, dist2 = mu.get_edit_distance_detailed(mol1, mol2, mcs_mol)
    distance = dist1 + dist2

    return distance


@lru_cache(
    maxsize=1000
)  # Set maxsize to None for an unbounded cache or a specific integer for a bounded cache
def return_mol(smiles):
    # Simulate some processing
    return Chem.MolFromSmiles(smiles)


def simba_solve_pair_edit_distance(s0, s1, fp0, fp1, mol0, mol1):
    tanimoto = DataStructs.TanimotoSimilarity(fp0, fp1)

    if tanimoto < 0.2:
        distance = VERY_HIGH_DISTANCE
    else:
        if (mol0.GetNumAtoms() > 40) or (mol1.GetNumAtoms() > 40):
            return np.nan, tanimoto
        else:
            distance = simba_get_edit_distance(mol0, mol1)

    return distance, tanimoto


def simba_solve_pair_mces(
    s0,
    s1,
    fp0,
    fp1,
    mol0,
    mol1,
    threshold,
    TIME_LIMIT=2,  # 2 seconds
):
    tanimoto = DataStructs.TanimotoSimilarity(fp0, fp1)

    if tanimoto < 0.2:
        distance = VERY_HIGH_DISTANCE
    else:
        if (mol0.GetNumAtoms() > 40) or (mol1.GetNumAtoms() > 40):
            return np.nan, tanimoto
        else:
            result = MCES2(
                s0,
                s1,
                threshold=threshold,
                i=0,
                # solver='CPLEX_CMD',       # or another fast solver you have installed
                solver="HiGHS",
                solver_options={
                    "threads": 1,
                    "msg": False,
                    "timeLimit": TIME_LIMIT,  # Stop CBC after 1 seconds
                },
                # solver_options={'threads': 1, 'msg': False},  # use single thread + no console messages
                no_ilp_threshold=False,  # allow the ILP to stop early once the threshold is exceeded
                always_stronger_bound=True,
                catch_errors=True,  # return distance=-1 on solver failure instead of raising
            )
            distance = result[1]
            time_taken = result[2]
            exact_answer = result[3]

            if distance == -1 or (
                time_taken >= (0.9 * TIME_LIMIT) and (exact_answer != 1)
            ):
                distance = np.nan

    return distance, tanimoto


def get_data(data, index, batch_count):
    batch_size = len(data) // batch_count
    if index < len(data) % batch_count:
        batch_size += 1
        start = index * batch_size
        end = start + batch_size
    else:
        start = index * batch_size + len(data) % batch_count
        end = start + batch_size

    if end > len(data):
        end = len(data)

    res = data.iloc[start:end]
    # reset index
    res = res.reset_index(drop=True)
    return res
