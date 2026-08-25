"""Build the two analog-discovery search inputs (014_2, simplified
reproduction of the SIMBA paper's Figure 2 -- see NOTES_014_2_ANALOG_DISCOVERY.md):

  Search A: CASMI 2022 queries vs (NIST20 + full MassSpecGym) reference library
  Search B: CASMI 2022 queries vs GNPS (no propagated) reference library

For each search, produces ONE combined MGF with a `FOLD=` field
(`FOLD=query` / `FOLD=library`) so the existing tools/simba_retrieval.py
load_spectra(mgf, fold) can read both sides straight off it -- same
convention gaetan_test.mgf already uses for train+test.

Cleaning applied to every source (library and query alike), matching this
project's own established conventions:
  - SMILES must RDKit-parse (Chem.MolFromSmiles succeeds).
  - >= 6 peaks (SIMBA's own canonical min_n_peaks, simba/configs/data/default.yaml).
  - Positive ion mode only (absent IONMODE, e.g. MassSpecGym/CASMI's own mgf
    convention, is treated as positive -- matches how every other 014_2
    evaluation this session was built; explicit "Negative" is dropped).
  - ADDUCT must be a key in ADDUCT_TO_MASS (simba/core/chemistry/chem_utils.py)
    -- theoretical_precursor_mz() raises on anything else, and real-world
    libraries (NIST20 especially) have many exotic in-source-fragment/isotope
    adducts this project's table was never built to cover. Dropped, not
    guessed at.
  - GNPS has no ADDUCT field at all -- defaulted to "[M+H]+" for positive-mode
    entries (the dominant real-world adduct by a wide margin, a standard
    simplifying assumption when adduct is unrecorded), dropped otherwise.
  - Deduplicated to ONE representative spectrum per canonical-SMILES molecule
    within each source, then again across sources when merging (first spectrum
    encountered wins) -- keeps embedding/GT-MCES cost to a molecule-level
    count, matching the molecule-level ranking convention already established
    for 014_2 elsewhere this session, not spectrum-level.

Per-search CASMI exclusion: a query molecule present (by canonical SMILES) in
THAT search's own reference library is dropped from that search's query set
-- same leakage guard the paper itself uses, applied per-search (a molecule
excluded from search A because it's in NIST20+MassSpecGym is NOT necessarily
excluded from search B, and vice versa).

Usage:
    uv run python tools/prepare_analog_discovery_data.py \\
        --casmi_mgf simba/data/casmi2022.mgf \\
        --nist20_mgf data/nist20/nist20.mgf \\
        --massspecgym_mgf data/massspecgym/data/auxiliary/MassSpecGym.mgf \\
        --gnps_mgf data/gnps/ALL_GNPS_NO_PROPOGATED.mgf \\
        --output_dir data/analog_discovery
"""

import argparse
from pathlib import Path

from rdkit import Chem


MIN_N_PEAKS = 6


def _parse_mgf_blocks(path: str):
    """Minimal manual MGF block parser -- returns a list of dicts (field ->
    raw string value, uppercased keys) plus 'MZ'/'INTENSITY' peak arrays.
    Deliberately not matchms (this project's own established reason: slow
    per-spectrum object construction on large files -- see
    prepare_gt_mces_retrieval.py's module docstring for the same call)."""
    blocks = []
    cur = None
    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                cur = {"_mz": [], "_intensity": []}
                continue
            if line == "END IONS":
                if cur is not None:
                    blocks.append(cur)
                cur = None
                continue
            if cur is None:
                continue
            if "=" in line and not line[0].isdigit():
                key, _, val = line.partition("=")
                cur[key.strip().upper()] = val.strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        inten = float(parts[1])
                    except ValueError:
                        continue
                    cur["_mz"].append(mz)
                    cur["_intensity"].append(inten)
    return blocks


def _clean_source(
    blocks: list[dict], adduct_to_mass: dict, default_adduct_if_missing: str | None
) -> dict[str, dict]:
    """Filter + dedup a parsed source to {canonical_smiles: block}. First
    spectrum seen per molecule wins; everything else about the block is kept
    as-is (mz/intensity/adduct/precursor) for later MGF writing."""
    out: dict[str, dict] = {}
    n_seen = 0
    n_bad_smiles = 0
    n_too_few_peaks = 0
    n_bad_ionmode = 0
    n_bad_adduct = 0
    for b in blocks:
        n_seen += 1
        smi = b.get("SMILES")
        if not smi:
            n_bad_smiles += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_bad_smiles += 1
            continue
        canon = Chem.MolToSmiles(mol)

        if len(b["_mz"]) < MIN_N_PEAKS:
            n_too_few_peaks += 1
            continue

        ionmode = (b.get("IONMODE") or "").strip().lower()
        if ionmode not in ("", "positive"):
            n_bad_ionmode += 1
            continue

        adduct = b.get("ADDUCT")
        if not adduct and default_adduct_if_missing:
            adduct = default_adduct_if_missing
        if not adduct or adduct not in adduct_to_mass:
            n_bad_adduct += 1
            continue
        b["ADDUCT"] = adduct

        if canon in out:
            continue  # first spectrum per molecule wins
        b["SMILES"] = canon
        out[canon] = b

    print(
        f"    {n_seen} spectra -> {len(out)} unique molecules kept "
        f"(dropped: {n_bad_smiles} bad SMILES, {n_too_few_peaks} <{MIN_N_PEAKS} peaks, "
        f"{n_bad_ionmode} non-positive ionmode, {n_bad_adduct} unresolvable adduct)"
    )
    return out


def _write_mgf(
    path: Path, entries: dict[str, dict], fold_label: str, mode: str
) -> None:
    with open(path, mode) as fh:
        for smi, b in entries.items():
            fh.write("BEGIN IONS\n")
            fh.write(f"SMILES={smi}\n")
            fh.write(f"ADDUCT={b['ADDUCT']}\n")
            fh.write(f"PEPMASS={b.get('PEPMASS', '0')}\n")
            fh.write(f"CHARGE={b.get('CHARGE', '1')}\n")
            fh.write("IONMODE=positive\n")
            fh.write(f"FOLD={fold_label}\n")
            for mz, inten in zip(b["_mz"], b["_intensity"]):
                fh.write(f"{mz} {inten}\n")
            fh.write("END IONS\n")


def run(
    casmi_mgf: str,
    nist20_mgf: str,
    massspecgym_mgf: str,
    gnps_mgf: str,
    output_dir: str,
):
    from simba.core.chemistry.chem_utils import ADDUCT_TO_MASS

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nParsing CASMI queries from {casmi_mgf} ...")
    casmi_blocks = _parse_mgf_blocks(casmi_mgf)
    casmi_clean = _clean_source(
        casmi_blocks, ADDUCT_TO_MASS, default_adduct_if_missing=None
    )

    print(f"\nParsing NIST20 from {nist20_mgf} ...")
    nist_blocks = _parse_mgf_blocks(nist20_mgf)
    nist_clean = _clean_source(
        nist_blocks, ADDUCT_TO_MASS, default_adduct_if_missing=None
    )

    print(f"\nParsing MassSpecGym from {massspecgym_mgf} ...")
    msg_blocks = _parse_mgf_blocks(massspecgym_mgf)
    msg_clean = _clean_source(
        msg_blocks, ADDUCT_TO_MASS, default_adduct_if_missing=None
    )

    print(f"\nParsing GNPS from {gnps_mgf} ...")
    gnps_blocks = _parse_mgf_blocks(gnps_mgf)
    gnps_clean = _clean_source(
        gnps_blocks, ADDUCT_TO_MASS, default_adduct_if_missing="[M+H]+"
    )

    # Library A: NIST20 union MassSpecGym, first-seen wins on overlap.
    library_a = dict(nist_clean)
    n_overlap_a = 0
    for smi, b in msg_clean.items():
        if smi in library_a:
            n_overlap_a += 1
            continue
        library_a[smi] = b
    print(
        f"\nLibrary A (NIST20 + MassSpecGym): {len(library_a)} unique molecules "
        f"({n_overlap_a} overlap between the two sources, NIST20 kept)"
    )

    library_b = gnps_clean
    print(f"Library B (GNPS, no propagated): {len(library_b)} unique molecules")

    queries_a = {smi: b for smi, b in casmi_clean.items() if smi not in library_a}
    queries_b = {smi: b for smi, b in casmi_clean.items() if smi not in library_b}
    print(
        f"\nCASMI queries: {len(casmi_clean)} clean -> "
        f"{len(queries_a)} for search A (excluded {len(casmi_clean) - len(queries_a)} "
        f"present in library A), {len(queries_b)} for search B (excluded "
        f"{len(casmi_clean) - len(queries_b)} present in library B)"
    )

    search_a_path = out_dir / "search_A_nist_msg.mgf"
    search_b_path = out_dir / "search_B_gnps.mgf"

    print(f"\nWriting {search_a_path} ...")
    _write_mgf(search_a_path, queries_a, "query", mode="w")
    _write_mgf(search_a_path, library_a, "library", mode="a")

    print(f"Writing {search_b_path} ...")
    _write_mgf(search_b_path, queries_b, "query", mode="w")
    _write_mgf(search_b_path, library_b, "library", mode="a")

    print("\nDone.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--casmi_mgf", required=True)
    p.add_argument("--nist20_mgf", required=True)
    p.add_argument("--massspecgym_mgf", required=True)
    p.add_argument("--gnps_mgf", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
