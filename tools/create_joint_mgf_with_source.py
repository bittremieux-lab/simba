"""
Add DATASET= field to each spectrum in the joint MGF based on global index ranges.

Source boundaries (0-indexed spectrum count):
  0 – 231,103        → MassSpecGym   (231,104 spectra)
  231,104 – 912,811  → NIST20        (681,708 spectra)
  912,812 – 1,401,608 → Spectraverse (488,797 spectra)

Usage:
  cd /home/nkubrakov/simba
  uv run python tools/create_joint_mgf_with_source.py
"""

from pathlib import Path

from tqdm import tqdm


INPUT = Path("/mnt/data2/nkubrakov/joint/joint_msg_nist20_sv.mgf")
OUTPUT = Path("/mnt/data2/nkubrakov/joint/joint_msg_nist20_sv_with_source.mgf")

BOUNDARIES = [
    (0, 231_103, "MassSpecGym"),
    (231_104, 912_811, "NIST20"),
    (912_812, 1_401_608, "Spectraverse"),
]


def source_for(idx: int) -> str:
    for lo, hi, name in BOUNDARIES:
        if lo <= idx <= hi:
            return name
    return "Unknown"


spectrum_idx = -1
total_spectra = sum(hi - lo + 1 for lo, hi, _ in BOUNDARIES)

with INPUT.open() as fin, OUTPUT.open("w") as fout:
    for line in tqdm(fin, desc="rewriting", unit="lines", total=95_000_000):
        fout.write(line)
        if line.rstrip() == "BEGIN IONS":
            spectrum_idx += 1
            fout.write(f"DATASET={source_for(spectrum_idx)}\n")

print(f"Done. Wrote {spectrum_idx + 1:,} spectra to {OUTPUT}")
