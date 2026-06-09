"""
Convert NIST20 MSP file to MGF format for SIMBA preprocessing.

Strategy:
- Parse MSP using NistLoader (already handles all field extraction)
- Build InChIKey → canonical SMILES lookup from Sebastian's df_smiles CSVs using RDKit
- Write one MGF entry per spectrum, with SMILES injected from the lookup

Usage:
    cd /home/nkubrakov/simba
    uv run python tools/nist20_msp_to_mgf.py

Output:
    /mnt/data2/nkubrakov/nist20/nist20.mgf
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))
from simba.core.data.loaders.nist_loader import NistLoader


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

MSP_PATH = Path(
    "/mnt/data2/nkubrakov/preprocessing_ed_mces_20250123/hr_msms_nist_all.MSP"
)
SMILES_DIR = Path("/mnt/data2/nkubrakov/preprocessing_ed_mces_20250123")
OUT_DIR = Path("/mnt/data2/nkubrakov/nist20")
OUT_MGF = OUT_DIR / "nist20.mgf"


def build_inchikey_to_smiles(smiles_dir: Path) -> dict[str, str]:
    """Build InChIKey → canonical SMILES from Sebastian's CSV files (NIST rows only)."""
    log.info("Building InChIKey → SMILES lookup from Sebastian's CSVs...")
    inchikey_to_smiles = {}
    failed = 0

    for split in ["train", "val", "test"]:
        df = pd.read_csv(smiles_dir / f"df_smiles_{split}.csv", index_col=0)
        nist_df = df[df["library"] == "nist"]
        log.info(f"  {split}: {len(nist_df)} NIST molecules")

        for smiles in tqdm(
            nist_df["canon_smiles"], desc=f"  RDKit {split}", leave=False
        ):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed += 1
                    continue
                inchi = MolToInchi(mol)
                if inchi is None:
                    failed += 1
                    continue
                inchikey = InchiToInchiKey(inchi)
                if inchikey is None:
                    failed += 1
                    continue
                # InChIKey in MSP is just the first 14-char block (connectivity)
                # but the $:28 field contains the full InChIKey — store both
                inchikey_to_smiles[inchikey] = smiles
                # Also store with just the first block as fallback
                inchikey_to_smiles[inchikey.split("-")[0]] = smiles
            except Exception:
                failed += 1

    log.info(
        f"  Built lookup: {len(inchikey_to_smiles)} entries ({failed} RDKit failures)"
    )
    return inchikey_to_smiles


def write_mgf(spectra: list[dict], out_path: Path, inchikey_to_smiles: dict[str, str]):
    """Write spectra to MGF format compatible with SIMBA."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_smiles = 0
    skipped_no_mz = 0
    skipped_no_peaks = 0

    with open(out_path, "w") as f:
        for i, spec in enumerate(tqdm(spectra, desc="Writing MGF")):
            inchikey = spec.get("inchi_key", "").strip()
            precursor_mz = spec.get("precursor_mz", 0)
            mz_arr = spec.get("mz", [])
            intensity_arr = spec.get("intensity", [])

            # Require peaks
            if len(mz_arr) == 0 or (hasattr(mz_arr, "__len__") and len(mz_arr) == 0):
                skipped_no_peaks += 1
                continue

            if precursor_mz == 0 or precursor_mz is None:
                skipped_no_mz += 1
                continue

            # Look up SMILES: try full InChIKey, then just first block
            smiles = inchikey_to_smiles.get(inchikey)
            if smiles is None and "-" in inchikey:
                smiles = inchikey_to_smiles.get(inchikey.split("-")[0])
            if smiles is None:
                skipped_no_smiles += 1
                continue

            charge = spec.get("precursor_charge", 1)
            adduct_raw = spec.get("adduct", "")
            # Reconstruct full adduct string from the raw parsed value (e.g. "M+H" -> "[M+H]+")
            # NistLoader strips brackets; try to re-read from params if available
            params = spec.get("params", {})
            adduct = params.get("adduct", adduct_raw)
            if adduct and not adduct.startswith("["):
                adduct = f"[{adduct}]+"  # best-effort reconstruction
            ionmode = spec.get("ionmode", "")

            f.write("BEGIN IONS\n")
            f.write(f"SMILES={smiles}\n")
            f.write(f"INCHIKEY={inchikey}\n")
            # pyteomics parses PEPMASS into params["pepmass"] as a tuple;
            # simba's is_valid_spectrum_janssen reads spectrum["params"]["pepmass"][0]
            f.write(f"PEPMASS={precursor_mz}\n")
            # pyteomics parses CHARGE=1+ into params["charge"] = [1]
            f.write(f"CHARGE={charge}+\n")
            if adduct:
                f.write(f"ADDUCT={adduct}\n")
            if ionmode:
                f.write(f"IONMODE={ionmode}\n")
            f.write(f"TITLE=nist20_spectrum_{i}\n")

            for mz_val, int_val in zip(mz_arr, intensity_arr):
                f.write(f"{mz_val:.5f} {int_val:.5f}\n")

            f.write("END IONS\n\n")
            written += 1

    log.info(f"Written: {written} spectra")
    log.info(f"Skipped — no SMILES match: {skipped_no_smiles}")
    log.info(f"Skipped — no precursor m/z:  {skipped_no_mz}")
    log.info(f"Skipped — no peaks:          {skipped_no_peaks}")
    return written, skipped_no_smiles


def main():
    log.info(f"MSP: {MSP_PATH}")
    log.info(f"Output: {OUT_MGF}")

    # Step 1: build InChIKey → SMILES lookup
    inchikey_to_smiles = build_inchikey_to_smiles(SMILES_DIR)

    # Step 2: parse MSP
    log.info("Parsing MSP file (this may take a few minutes)...")
    spectra, n_lines = NistLoader.parse_file(MSP_PATH)
    log.info(f"Parsed {len(spectra)} spectra from {n_lines} lines")

    # Step 3: quick stats on InChIKey coverage before writing
    found = sum(
        1
        for s in spectra
        if inchikey_to_smiles.get(s.get("inchi_key", "")) is not None
        or inchikey_to_smiles.get((s.get("inchi_key", "") or "").split("-")[0])
        is not None
    )
    log.info(
        f"InChIKey lookup hits: {found}/{len(spectra)} ({100 * found / max(1, len(spectra)):.1f}%)"
    )

    # Step 4: write MGF
    written, skipped = write_mgf(spectra, OUT_MGF, inchikey_to_smiles)

    log.info("Done.")
    log.info(
        f"Output MGF: {OUT_MGF} ({OUT_MGF.stat().st_size / 1e6:.1f} MB)"
        if written > 0
        else ""
    )


if __name__ == "__main__":
    main()
