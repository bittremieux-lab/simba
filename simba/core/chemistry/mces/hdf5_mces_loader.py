"""
Loader for pre-computed MCES distances stored in HDF5 format (MassSpecGym).

The HDF5 file contains:
  - "mces_smiles_order": array of N SMILES strings (not necessarily RDKit-canonical)
  - "mces": condensed upper-triangle distance matrix, length N*(N-1)//2, float32
"""

from functools import lru_cache

import h5py
import numpy as np
from rdkit import Chem

from simba.utils.logger_setup import logger


@lru_cache(maxsize=65536)
def _canonical(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else Chem.MolToSmiles(mol)


def _condensed_index(i: int, j: int, n: int) -> int:
    """Index in the condensed upper-triangle array for row i, col j (requires i < j)."""
    return n * i - i * (i + 1) // 2 + j - i - 1


class HDF5MCESCache:
    """Index over a MassSpecGym MCES HDF5 file for O(1) distance lookups.

    The condensed array (~2.4 GB) is kept as a numpy array loaded into RAM on
    first use. The SMILES-to-index dict is built at load time by canonicalizing
    all N SMILES once.
    """

    def __init__(self, smiles_to_idx: dict[str, int], mces_array: np.ndarray):
        self._smiles_to_idx = smiles_to_idx
        self._mces = mces_array
        self._n = len(smiles_to_idx)

    @classmethod
    def load(cls, hdf5_path: str) -> "HDF5MCESCache":
        with h5py.File(hdf5_path, "r") as f:
            raw_smiles = [
                s.decode() if isinstance(s, bytes) else s
                for s in f["mces_smiles_order"][:]
            ]
            mces_array = f["mces"][:]

        smiles_to_idx = {
            canon: idx
            for idx, smi in enumerate(raw_smiles)
            if (canon := _canonical(smi)) is not None
        }
        skipped = len(raw_smiles) - len(smiles_to_idx)
        if skipped:
            logger.warning(
                f"HDF5MCESCache: {skipped} SMILES could not be parsed and were skipped"
            )
        logger.info(
            f"HDF5MCESCache: loaded {len(smiles_to_idx)} SMILES from {hdf5_path}"
        )
        return cls(smiles_to_idx, mces_array)

    def lookup(self, smi_a: str, smi_b: str) -> float | None:
        """Return MCES distance for two SMILES, or None if either is not in the cache."""
        ca = _canonical(smi_a)
        cb = _canonical(smi_b)
        if ca is None or cb is None:
            return None
        ia = self._smiles_to_idx.get(ca)
        ib = self._smiles_to_idx.get(cb)
        if ia is None or ib is None:
            return None
        if ia == ib:
            return 0.0
        i, j = (ia, ib) if ia < ib else (ib, ia)
        return float(self._mces[_condensed_index(i, j, self._n)])
