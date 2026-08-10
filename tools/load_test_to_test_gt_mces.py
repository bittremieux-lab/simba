"""Load ground-truth test-to-test MCES for the official split (3e).

Unlike test-to-candidate (tools/prepare_gt_mces_retrieval.py, computed fresh
on asimov2), this data already exists: the official-split preprocessing
pipeline's own pair-mining, with the [10,20] lower-bound band exact-refined
by tools/apply_exact_mces_1020.py — see
data/massspecgym/preprocessing_msg_exact_mces_1020/. It covers 2,959 of the
3,170 unique official-test-fold molecules (the rest were dropped somewhere
in SIMBA's own preprocessing filters) — essentially the full pairwise
cross-product, 4,376,361 pairs.

Usage:
    from load_test_to_test_gt_mces import load_test_to_test_gt_lookup
    lookup = load_test_to_test_gt_lookup(
        "/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_msg_exact_mces_1020"
    )
"""

import pickle
from pathlib import Path

import numpy as np


def load_test_to_test_gt_lookup(prepro_dir: str) -> dict[tuple[str, str], float]:
    """Return a symmetric {(canon_smi_a, canon_smi_b): mces} dict covering
    every mined official-test-fold molecule pair.

    mol_idx -> canonical SMILES comes from mapping.pkl's df_smiles_test
    (row order = mol_idx, confirmed: max index in the pairs file == len(df)-1).
    The distance column (col 3) is exact wherever the pair's original lower
    bound fell in [10,20]; outside that band it's still the lower-bound
    approximation, not the exact solver's value.
    """
    prepro = Path(prepro_dir)
    with open(prepro / "mapping.pkl", "rb") as fh:
        mapping = pickle.load(fh)
    smiles = mapping["df_smiles_test"]["canon_smiles"].tolist()

    arr = np.load(prepro / "ed_mces_indexes_tani_incremental_test_node0_chunk0.npy")

    lookup: dict[tuple[str, str], float] = {}
    for a, b, _ed, mces in arr:
        smi_a, smi_b = smiles[int(a)], smiles[int(b)]
        lookup[(smi_a, smi_b)] = float(mces)
        lookup[(smi_b, smi_a)] = float(mces)
    return lookup
