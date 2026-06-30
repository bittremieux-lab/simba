from copy import deepcopy
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


# ----------------------------
# RDKit: compute MCS
# ----------------------------
def compute_mcs_mol(mol1, mol2, timeout=10):
    if mol1 is None or mol2 is None:
        return None
    mcs_result = rdFMCS.FindMCS(
        [mol1, mol2],
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        matchValences=False,
        timeout=timeout,
    )
    smarts = mcs_result.smartsString
    return Chem.MolFromSmarts(smarts) if smarts else None


# ----------------------------
# RDKit PNG drawing
# ----------------------------
def draw_mcs_diff_png(
    mol,
    mcs_mol,
    size=(300, 300),
    col_common=(0.55, 0.85, 0.55),
    col_diff=(1.0, 0.55, 0.55),
    line_width=2.5,
):
    if mol is None:
        return Image.new("RGBA", size, (255, 255, 255, 0))

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.setBackgroundColour((1, 1, 1))
    opts.bondLineWidth = line_width
    opts.useBWAtomPalette()

    if mcs_mol is None:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        return Image.open(BytesIO(png))

    match = mol.GetSubstructMatch(mcs_mol)

    if not match:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        return Image.open(BytesIO(drawer.GetDrawingText()))

    mcs_to_mol = {i: match[i] for i in range(len(match))}
    common_atoms = set(match)

    common_bonds = set()
    for b in mcs_mol.GetBonds():
        a1 = mcs_to_mol[b.GetBeginAtomIdx()]
        a2 = mcs_to_mol[b.GetEndAtomIdx()]
        mb = mol.GetBondBetweenAtoms(a1, a2)
        if mb is not None:
            common_bonds.add(mb.GetIdx())

    diff_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in common_atoms]
    diff_bonds = [b.GetIdx() for b in mol.GetBonds() if b.GetIdx() not in common_bonds]

    highlight_atoms = list(common_atoms) + diff_atoms
    highlight_bonds = list(common_bonds) + diff_bonds

    atom_colors = dict.fromkeys(common_atoms, col_common)
    atom_colors.update(dict.fromkeys(diff_atoms, col_diff))

    bond_colors = dict.fromkeys(common_bonds, col_common)
    bond_colors.update(dict.fromkeys(diff_bonds, col_diff))

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        highlightAtomRadii=dict.fromkeys(common_atoms, 0.35),
    )

    drawer.FinishDrawing()

    return Image.open(BytesIO(drawer.GetDrawingText()))


# ----------------------------
# MAIN (PNG pipeline)
# ----------------------------
def plot_pair_mols_plus_spectrum_png(
    pair_index,
    all_spectrums_query,
    all_spectrums_reference,
    pairs_interesting,
    *,
    fragment_tol_mass=10,
    fragment_tol_mode="ppm",
    mz_min=0,
    mz_max=500,
    mol_size=(300, 300),
    spec_fig_size=(7.5, 5.0),
    spec_dpi=300,
    out_dir=".",
    out_name_tpl="pair_{pair_index}.png",
    metrics=None,
    mcs_timeout=10,
):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_idx = pairs_interesting[pair_index]["indexes"][0]
    r_idx = pairs_interesting[pair_index]["indexes"][1]

    s1 = deepcopy(all_spectrums_query[q_idx])
    s2 = deepcopy(all_spectrums_reference[r_idx])

    mol1 = Chem.MolFromSmiles(s1.params.get("smiles"))
    mol2 = Chem.MolFromSmiles(s2.params.get("smiles"))

    Chem.RemoveStereochemistry(mol1)
    Chem.RemoveStereochemistry(mol2)

    mcs_mol = compute_mcs_mol(mol1, mol2, timeout=mcs_timeout)

    img1 = draw_mcs_diff_png(mol1, mcs_mol, size=mol_size)
    img2 = draw_mcs_diff_png(mol2, mcs_mol, size=mol_size)

    # ----------------------------
    # Spectrum (PNG)
    # ----------------------------
    fig = plt.figure(figsize=spec_fig_size)
    ax = fig.add_subplot(111)

    sup.mirror(
        s1.remove_precursor_peak(fragment_tol_mass, fragment_tol_mode).filter_intensity(
            0.01
        ),
        s2.remove_precursor_peak(fragment_tol_mass, fragment_tol_mode).filter_intensity(
            0.01
        ),
        ax=ax,
    )
    ax.set_title(
        f"MCES ground truth: {metrics['mces_gt']:.2f} MCES pred: {metrics['mces_pred']:.2f}"
    )
    ax.set_xlim(mz_min, mz_max)
    ax.grid(False)
    ax.minorticks_off()

    spec_path = out_dir / f"_tmp_spec_{pair_index}.png"
    fig.savefig(spec_path, dpi=spec_dpi, bbox_inches="tight")
    plt.close(fig)

    spec_img = Image.open(spec_path)

    # ----------------------------
    # Compose final image
    # ----------------------------
    left_width = mol_size[0]
    total_height = mol_size[1] * 2

    # resize spectrum to match height
    spec_img = spec_img.resize(
        (int(spec_img.width * total_height / spec_img.height), total_height)
    )

    total_width = left_width + spec_img.width

    final_img = Image.new("RGB", (total_width, total_height), "white")

    final_img.paste(img1, (0, 0))
    final_img.paste(img2, (0, mol_size[1]))
    final_img.paste(spec_img, (left_width, 0))

    out_path = out_dir / out_name_tpl.format(pair_index=pair_index)
    final_img.save(out_path, dpi=(300, 300))

    spec_path.unlink(missing_ok=True)

    return str(out_path)
