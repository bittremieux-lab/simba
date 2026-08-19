import re

import numpy as np


ADDUCT_TO_MASS = {
    "[M+3H]3+": 1.007276 * 3,
    "[M+2H+Na]3+": 8.334590 * 3,
    "[M+H+2Na]3+": 15.7661904 * 3,
    "[M+3Na]3+": 22.989218 * 3,
    "[M+2H]2+": 1.007276 * 2,
    "[M+H+NH4]2+": 9.520550 * 2,
    "[M+H+Na]2+": 11.998247 * 2,
    "[M+H+K]2+": 19.985217 * 2,
    "[M+ACN+2H]2+": 21.520550 * 2,
    "[M+2Na]2+": 22.989218 * 2,
    "[M+2ACN+2H]2+": 42.033823 * 2,
    "[M+3ACN+2H]2+": 62.547097 * 2,
    "[M+H]+": 1.007276,
    "[M+NH4]+": 18.033823,
    "[M+Na]+": 22.989218,
    "[M+CH3OH+H]+": 33.033489,
    "[M+K]+": 38.963158,
    "[M+ACN+H]+": 42.033823,
    "[M+2Na-H]+": 44.971160,
    "[M+IsoProp+H]+": 61.06534,
    "[M+ACN+Na]+": 64.015765,
    "[M+2K-H]+": 76.919040,
    "[M+DMSO+H]+": 79.02122,
    "[M+2ACN+H]+": 83.060370,
    "[M+IsoProp+Na+H]+": 84.05511,
    "[M-H2O+H]+": -18.010565 + 1.007276,
    "[M-H4O2+H]+": -18.010565 * 2 + 1.007276,
    "[2M+H]+": 1.007276,
    "[2M+NH4]+": 18.033823,
    "[2M+Na]+": 22.989218,
    "[2M+K]+": 38.963158,
    "[2M+ACN+H]+": 42.033823,
    "[2M+ACN+Na]+": 64.015765,
    "[M]+": 0.0,
    "[M-3H]3-": -1.007276 * 3,
    "[M-2H]2-": -1.007276 * 2,
    "[M-H2O-H]-": -19.01839,
    "[M-H]-": -1.007276,
    "[M+Na-2H]-": 20.974666,
    "[M+Cl]-": 34.969402,
    "[M+K-2H]-": 36.948606,
    "[M+FA-H]-": 44.998201,
    "[M+Hac-H]-": 59.013851,
    "[M+Br]-": 78.918885,
    "[M+TFA-H]-": 112.985586,
    "[2M-H]-": -1.007276,
    "[2M+FA-H]-": 44.998201,
    "[2M+Hac-H]-": 59.013851,
    "[3M-H]-": -1.007276,  # was +1.007276 -- sign-flipped vs [2M-H]-/[M-H]- (same single deprotonation)
}


def ion_to_mass(adduct: str) -> float:
    """
    Convert an adduct string to its corresponding mass.

    Parameters
    ----------
    adduct : str
        The adduct string to convert.

    Returns
    -------
    float
        The mass corresponding to the adduct. Returns None if the adduct is not found.
    """
    return ADDUCT_TO_MASS.get(adduct)


_ADDUCT_NMER_RE = re.compile(r"^\[(\d*)M")
_ADDUCT_CHARGE_RE = re.compile(r"(\d*)[+-]$")


def _adduct_nmer(adduct: str) -> int:
    """'[2M+H]+' -> 2, '[M+H]+' -> 1 (no leading digit means a single monomer)."""
    m = _ADDUCT_NMER_RE.match(adduct)
    if not m:
        raise ValueError(f"Cannot parse monomer count from adduct {adduct!r}")
    digits = m.group(1)
    return int(digits) if digits else 1


def _adduct_charge(adduct: str) -> int:
    """'[M+2H]2+' -> 2, '[M+H]+' -> 1 (magnitude only; sign doesn't matter -- m/z
    divides by |charge| either way)."""
    m = _ADDUCT_CHARGE_RE.search(adduct)
    if not m:
        raise ValueError(f"Cannot parse charge from adduct {adduct!r}")
    digits = m.group(1)
    return int(digits) if digits else 1


def theoretical_precursor_mz(neutral_mass: float, adduct: str) -> float:
    """Theoretical m/z for a neutral monoisotopic mass under a given adduct.

    ADDUCT_TO_MASS's values are the adduct's own mass contribution (already
    x charge magnitude, per how the table above was built -- e.g.
    "[M+3H]3+": 1.007276 * 3) but NOT yet divided by charge, and NOT scaled
    by monomer count for "nM" adducts (e.g. [2M+H]+) -- both depend on the
    adduct string itself, parsed here:
        precursor_mz = (nmer * neutral_mass + ADDUCT_TO_MASS[adduct]) / charge
    """
    offset = ADDUCT_TO_MASS.get(adduct)
    if offset is None:
        raise KeyError(f"Unknown adduct: {adduct!r}")
    nmer = _adduct_nmer(adduct)
    charge = _adduct_charge(adduct)
    return (nmer * neutral_mass + offset) / charge


# Per-instrument-type ppm tolerance (BUDDY's table, as used by MIST-CF's
# precursor resampling -- see metabo_depthcharge.mist_cf.common.chem_utils
# in the sibling spectrawl_project, and Wout/Gaetan's Slack discussion).
INSTRUMENT_PPM_TOLERANCE = {
    "orbitrap": 5,
    "qtof": 10,
    "iontrap": 15,
    "fticr": 5,
    "unknown": 15,
}

# Raw MGF INSTRUMENT_TYPE strings (case-insensitive) -> canonical
# INSTRUMENT_PPM_TOLERANCE key. MassSpecGym's own MGF only ever has
# "Orbitrap"/"QTOF", but a few common variants are included for robustness.
_RAW_INSTRUMENT_TO_TYPE = {
    "orbitrap": "orbitrap",
    "qtof": "qtof",
    "q-tof": "qtof",
    "q tof": "qtof",
    "iontrap": "iontrap",
    "ion trap": "iontrap",
    "fticr": "fticr",
    "ft-icr": "fticr",
}


def normalize_instrument_type(raw: str | None) -> str:
    """Raw MGF INSTRUMENT_TYPE value -> one of INSTRUMENT_PPM_TOLERANCE's
    keys. Falls back to "unknown" (BUDDY's own least-precise tier) for
    anything missing or unrecognized, rather than raising -- this only
    affects how much noise gets added to a theoretical precursor mass, not
    correctness."""
    if not raw:
        return "unknown"
    return _RAW_INSTRUMENT_TO_TYPE.get(raw.strip().lower(), "unknown")


def resample_precursor_mz(theoretical_mz: float, instrument: str | None, rng) -> float:
    """MIST-CF/BUDDY-style precursor m/z resampling: Gaussian noise with std
    = instrument-specific ppm tolerance / 5. The original MIST-CF formulation
    redraws if the sample falls outside the instrument tolerance, but at this
    std that's a >=5-sigma event (p ~ 6e-7), so it's a no-op and dropped here
    -- a plain Gaussian is equivalent in practice.

    `instrument` should already be normalized to one of
    INSTRUMENT_PPM_TOLERANCE's keys (e.g. via a raw-string -> canonical-type
    lookup at load time); unrecognized/missing falls back to "unknown".
    `rng` is a numpy Generator (e.g. np.random.default_rng(seed)) -- passed
    in rather than created here so callers control reproducibility.
    """
    tol_ppm = INSTRUMENT_PPM_TOLERANCE.get(
        (instrument or "unknown").lower(), INSTRUMENT_PPM_TOLERANCE["unknown"]
    )
    std_ppm = tol_ppm / 5
    return theoretical_mz + rng.normal(0, theoretical_mz * std_ppm / 1e6)


def mass_lookup_from_df_smiles(df_smiles) -> np.ndarray:
    """mol_idx -> RDKit ExactMolWt, indexed by df_smiles' own 0-based index
    (the same molecule-index space pair_distances' columns 0/1 use).
    Cached by SMILES string so repeated molecules are only computed once.
    `df_smiles` needs an index plus a "canon_smiles" column -- the same
    shape as MoleculePairsOpt.df_smiles. RDKit is imported lazily since not
    every chem_utils consumer needs it.
    """
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt

    masses = np.full(int(df_smiles.index.max()) + 1, np.nan, dtype=np.float64)
    cache: dict[str, float] = {}
    for idx, smi in zip(df_smiles.index, df_smiles["canon_smiles"]):
        if smi not in cache:
            mol = Chem.MolFromSmiles(smi)
            cache[smi] = ExactMolWt(mol) if mol is not None else float("nan")
        masses[idx] = cache[smi]
    return masses
