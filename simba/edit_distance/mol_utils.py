"""
GNPS Utils - Molecule Utils
---------------------------
This file contains utility functions around molecules and molecules modification based on RDKit library.
"""

from rdkit import Chem
from rdkit.Chem import rdFMCS
from copy import deepcopy

import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')

def get_modification_nodes(mol1, mol2, in_mol1 = True):
    """
        Calculates the modification sites between two molecules when one molecule is a substructure of the other molecule

        Input:
            :mol1: first molecule 
            :mol2: second molecule
            :in_mol1: bool, if True, the modification sites are given in the mol1, if False, the modification sites are given in the mol2

        Output:
            list of modification sites
    """

    copy_mol1, copy_mol2 = _get_molecules(mol1, mol2)
    
    if copy_mol1.HasSubstructMatch(copy_mol2):
        return _calculateModificationSites(copy_mol1, copy_mol2, in_mol1)
    elif copy_mol2.HasSubstructMatch(copy_mol1):
        return _calculateModificationSites(copy_mol2, copy_mol1, not in_mol1)
    else:
        raise ValueError("One molecule should be a substructure of the other molecule")


def get_modification_edges(mol1, mol2, only_outward_edges = False):
    """
        Calculates the modification edges between two molecules when one molecule is a substructure of the other molecule

        Input:
            :mol1: first molecule
            :mol2: second molecule
            :only_outward_edges: bool, if True, only the modification edges that go from atoms in the substructure to atoms outside the substructure are returned

        Output:
            list of the the modification edges in the parent molecule as tuples of atom indices
    """

    copy_mol1, copy_mol2 = _get_molecules(mol1, mol2)
    
    if not mol1.HasSubstructMatch(mol2):
        if mol2.HasSubstructMatch(mol1):
            copy_mol1, copy_mol2 = copy_mol2, copy_mol1
        else:
            raise ValueError("One molecule should be a substructure of the other molecule")
    
    matches = _find_minimal_modification_edges_match(copy_mol1, copy_mol2)
    modificationEdgesOutward, modificationEdgesInside, _ = _get_edge_modifications(copy_mol1, copy_mol2, matches)
    if only_outward_edges:
        return modificationEdgesOutward
    else:
        return modificationEdgesOutward + modificationEdgesInside


def get_edit_distance(mol1, mol2):
    """
        Calculates the edit distance between mol1 and mol2.

        Input:
            :mol1: first molecule
            :mol2: second molecule

        Output:
            :edit_distance: edit distance between mol1 and mol2
    """
    copy_mol1, copy_mol2 = _get_molecules(mol1, mol2)
    mcs1 = rdFMCS.FindMCS([copy_mol1, copy_mol2])
    mcs_mol = Chem.MolFromSmarts(mcs1.smartsString)
    dist1 = get_modification_edges(copy_mol1, mcs_mol)
    dist2 = get_modification_edges(copy_mol2, mcs_mol)
    return len(dist1) + len(dist2)


def get_edit_distance_detailed(mol1, mol2, mcs = None):
    """
        Calculates the edit distance between mol1 and mol2.

        Input:
            :mol1: first molecule
            :mol2: second molecule
            :mcs: the maximum common substructure between mol1 and mol2

        Output:
            :removed edges: the removed modification edges
            :added edges: the added modification edges
    """
    copy_mol1, copy_mol2 = _get_molecules(mol1, mol2)
    if mcs is None:
        mcs1 = rdFMCS.FindMCS([copy_mol1, copy_mol2])
        mcs_mol = Chem.MolFromSmarts(mcs1.smartsString)
    else:
        mcs_mol = Chem.Mol(mcs)
    dist1 = get_modification_edges(copy_mol1, mcs_mol)
    dist2 = get_modification_edges(copy_mol2, mcs_mol)
    return len(dist1), len(dist2)


def get_transition(input1, input2):
    """
        Calculates the transition between mol1 and mol2.

        Input:
            :input1: first molecule
            :input2: second molecule

        Output:
            :result: a dictionary with the following keys:
                'merged_mol': the merged molecule
                'common_bonds': the common bonds between mol1 and mol2
                'common_atoms': the common atoms between mol1 and mol2
                'removed_atoms': the removed atoms from mol1
                'added_atoms': the added atoms from mol2
                'modified_added_edges_inside': the added edges inside the common substructure
                'modified_added_edges_bridge': the added edges between the common substructure and the added atoms
                'modified_removed_edges_inside': the removed edges inside the common substructure
                'modified_removed_edges_bridge': the removed edges between the common substructure and the removed
                'added_edges': the added edges that are not modification edges
                'removed_edges': the removed edges that are not modification edges
    """
    mol1, mol2 = _get_molecules(input1, input2)
    copy_mol1, copy_mol2 = Chem.Mol(mol1), Chem.Mol(mol2)
    # finding the maximum common substructure
    mcs1 = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs1.smartsString)

    # finding the mapping between the atoms of the common substructure
    mol1_indices = _find_minimal_modification_edges_match(mol1, mcs_mol)
    mol2_indices = _find_minimal_modification_edges_match(mol2, mcs_mol)

    map_mcs_from_mol1_to_mol2 = dict()
    map_mcs_from_mol2_to_mol1 = dict()
    for i in range(mcs_mol.GetNumAtoms()):
        map_mcs_from_mol1_to_mol2[mol1_indices[i]] = mol2_indices[i]
        map_mcs_from_mol2_to_mol1[mol2_indices[i]] = mol1_indices[i]

    # calculating the removed atoms from mol1
    removed_atoms = list(set(range(mol1.GetNumAtoms())) - set(mol1_indices))
    modified_removed_edges_bridge, modified_removed_edges_inside, removed_edges = _get_edge_modifications(mol1, mcs_mol, mol1_indices)

    # calculating the added atoms from mol2
    added_atoms = set(range(mol2.GetNumAtoms())) - set(mol2_indices)
    modified_added_edges_bridge, modified_added_edges_inside, added_edges = _get_edge_modifications(mol2, mcs_mol, mol2_indices)

    # creating the merged molecule
    editable_mol = Chem.EditableMol(copy_mol1)
    ## adding the atoms from mol2
    n = mol1.GetNumAtoms()
    map_old_mol2_to_added = dict() # mapping from the old mol2 indices to the new indices in the merged molecule (for the added atoms)
    for atom in added_atoms:
        editable_mol.AddAtom(mol2.GetAtomWithIdx(atom))
        map_old_mol2_to_added[atom] = n
        n += 1
    for atom in mol2_indices:
        map_old_mol2_to_added[atom] = map_mcs_from_mol2_to_mol1[atom]
    ## adding the bonds from mol2
    updated_modified_added_edges_inside = []
    for bond in modified_added_edges_inside:
        editable_mol.AddBond(map_old_mol2_to_added[bond[0]], map_old_mol2_to_added[bond[1]], order=copy_mol2.GetBondBetweenAtoms(bond[0], bond[1]).GetBondType())
        updated_modified_added_edges_inside.append((map_old_mol2_to_added[bond[0]], map_old_mol2_to_added[bond[1]]))
    updated_modified_added_edges_bridge = []
    for bond in modified_added_edges_bridge:
        if bond[0] in added_atoms:
            index1 = map_old_mol2_to_added[bond[0]]
            index2 = map_mcs_from_mol2_to_mol1[bond[1]]
        else:
            index1 = map_mcs_from_mol2_to_mol1[bond[0]]
            index2 = map_old_mol2_to_added[bond[1]]
        updated_modified_added_edges_bridge.append((index1, index2))
        editable_mol.AddBond(index1, index2, order=copy_mol2.GetBondBetweenAtoms(bond[0], bond[1]).GetBondType())
    updated_added_edges = []
    for bond in added_edges:
        index1 = map_old_mol2_to_added[bond[0]]
        index2 = map_old_mol2_to_added[bond[1]]
        updated_added_edges.append((index1, index2))
        editable_mol.AddBond(index1, index2, order=copy_mol2.GetBondBetweenAtoms(bond[0], bond[1]).GetBondType())
    ## adding the common bonds
    common_bonds = []
    for bond in mol1.GetBonds():
        pair = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if pair in modified_removed_edges_inside or pair in modified_removed_edges_bridge or pair in removed_edges:
            continue
        pair2 = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
        if pair2 in modified_removed_edges_inside or pair2 in modified_removed_edges_bridge or pair2 in removed_edges:
            continue
        common_bonds.append(pair)

    
    common_atoms = mol1_indices
    mol1_atoms = removed_atoms
    mol2_atoms = [map_old_mol2_to_added[i] for i in added_atoms]

    # create a structure for result
    result = {
        'merged_mol': editable_mol.GetMol(),
        'common_bonds': common_bonds,
        'common_atoms': common_atoms,
        'removed_atoms': mol1_atoms,
        'added_atoms': mol2_atoms,
        'added_edges': updated_added_edges,
        'removed_edges': removed_edges,
        'modified_added_edges_inside': updated_modified_added_edges_inside,
        'modified_added_edges_bridge': updated_modified_added_edges_bridge,
        'modified_removed_edges_inside': modified_removed_edges_inside,
        'modified_removed_edges_bridge': modified_removed_edges_bridge
    }

    return result


def get_modification_graph(main_struct, sub_struct):
    """
        Calculates the substructure difference between main_struct and sub_struct when there is exactly one modification edge,
          if there are multiple modification edges, one of the modifications is returned randomly.

        Input:
            :main_struct: main molecule
            :sub_struct: substructure molecule
        Output:
            :all_modifications: a list of the modifications structures, each modification is a tuple of:
                1. the modified subgraph molecule (as an rdkit editable molecule)
                2. a dictionary that maps the wildcard atom indices in subgraph to its true index in the main molecule
                3. the SMARTS representation of the modification
    """
    main_struct, sub_struct = _get_molecules(main_struct, sub_struct)
    if not main_struct.HasSubstructMatch(sub_struct):
        if sub_struct.HasSubstructMatch(main_struct):
            return get_modification_graph(sub_struct, main_struct)
        else:
            raise ValueError("One molecule should be a substructure of the other molecule")

    atoms_of_substructure = _find_minimal_modification_edges_match(main_struct, sub_struct)
    modificationEdgesOutward, modificationEdgesInside, noneModificationDifferentEdges = _get_edge_modifications(main_struct, sub_struct, atoms_of_substructure)
    all_modification_edges = modificationEdgesOutward + modificationEdgesInside + noneModificationDifferentEdges

    def dfs(mol, index, visited, color, all_modification_edges):
        if index in visited:
            return
        visited[index] = color
        
        for bond in all_modification_edges:
            if bond[0] == index:
                target = bond[1]
            elif bond[1] == index:
                target = bond[0]
            else:
                continue
            
            if target not in visited:
                dfs(mol, target, visited, color, all_modification_edges)
        return

    visited = {}
    color = 0
    for atom in range(main_struct.GetNumAtoms()):
        if atom not in visited and atom not in atoms_of_substructure:
            dfs(main_struct, atom, visited, color, all_modification_edges)
            color += 1


    all_modifications = []
    for modification in range(color):
        true_map = dict()
        edit_mol = Chem.RWMol(main_struct)
        # delete any atom that is not in the modification
        for atom in range(main_struct.GetNumAtoms()-1, -1, -1):
            if atom not in visited or visited[atom] != modification:
                edit_mol.RemoveAtom(atom)
            elif atom in atoms_of_substructure:
                edit_mol.GetAtomWithIdx(atom).SetProp('atomNote', f'substructure_{atom}')
        
        new_edit_mol = Chem.RWMol()
        for atom in edit_mol.GetAtoms():
            try:
                note = atom.GetProp('atomNote')
            except:
                note = None
            if note is not None and note.startswith('substructure'):
                new_edit_mol.AddAtom(Chem.Atom(0))
                true_map[atom.GetIdx()] = note.split('_')[1]
            else:
                new_edit_mol.AddAtom(atom)
        
        for bond in edit_mol.GetBonds():
            new_edit_mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
            
        # new_edit_mol = new_edit_mol.GetMol()
        all_modifications.append((new_edit_mol, true_map, Chem.MolToSmarts(new_edit_mol)))
        
    return all_modifications


def attach_mols(main_mol, attachment_mol, attach_location_main, attach_location_attachment, bond_type):
    """
        Attaches the attachment structure to main molecule at the attach_location_main and attach_location_attachment with bond_type.

        Input:
            :main_mol: rdkit molecule of the main molecule
            :attachment_mol: rdkit molecule of the attachment molecule
            :attach_location_main: the index of the atom in the main molecule where the attachment should be done
            :attach_location_attachment: the index of the atom in the attachment molecule where the attachment should be done
            :bond_type: the type of the bond between the main molecule and the attachment molecule

        Output:
            :new_mol: the new molecule after attachment
    """

    new_mol = Chem.CombineMols(main_mol, attachment_mol)
    new_mol = Chem.EditableMol(new_mol)
    new_mol.AddBond(attach_location_main, attach_location_attachment + main_mol.GetNumAtoms(), bond_type)
    new_mol = new_mol.GetMol()
    return new_mol


def generate_possible_stuctures(main_struct, sub_struct):
    """
        Generates all possible structures after attaching sub_struct to main_struct.

        Input:
            :main_struct: main molecule
            :sub_struct: substructure molecule
        Output:
            :list of possible_structures: all possible structures after attachment with the index of the atom
    """
    main_struct, sub_struct = _get_molecules(main_struct, sub_struct)
    if not main_struct.HasSubstructMatch(sub_struct):
        if sub_struct.HasSubstructMatch(main_struct):
            return generate_possible_stuctures(sub_struct, main_struct)
        else:
            raise ValueError("One molecule should be a substructure of the other molecule")

    all_modifications = get_modification_graph(main_struct, sub_struct)
    if len(all_modifications) == 0:
        raise ValueError("No modification is found")
    if len(all_modifications) > 1:
        raise ValueError("Multiple modifications are found")
    if len(all_modifications[0][1]) > 1:
        raise ValueError("Multiple modifications are found")
    if len(all_modifications[0][1]) == 0:
        raise ValueError("ISSUE WITH CODE, PLEASE REPORT")
    
    wild_atom = list(all_modifications[0][1].keys())[0]
    neighbor = all_modifications[0][0].GetAtomWithIdx(wild_atom).GetNeighbors()[0]
    bondType = all_modifications[0][0].GetBondBetweenAtoms(wild_atom, neighbor.GetIdx()).GetBondType()
    all_modifications[0][0].RemoveAtom(wild_atom)

    frag = all_modifications[1][0]
    index_in_frag = neighbor.GetIdx()

    structs = []
    for atom in sub_struct.GetAtoms():
        temp_struct = _attach_struct_try(sub_struct, frag, atom.GetIdx(), index_in_frag, bondType)
        if temp_struct is None:
            continue
        structs.append((atom.GetIdx(), temp_struct))
    
    return structs
    

def _attach_struct_try(main_struct, frag, main_location, frag_location, bond_type):
    mol = attach_mols(main_struct, frag, main_location, frag_location, bond_type)
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        for bond in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
            if bond != bond_type:
                try:
                    mol = attach_mols(main_struct, frag, main_location, frag_location, bond)
                    Chem.SanitizeMol(mol)
                    return mol
                except Exception:
                    continue
    return None


def _find_minimal_modification_sites_match(mol, substructure):
    """
        Finds the matching that results in the minimum number of modification sites, if multiple matches are found.
        If multiple results are found, one random result is returned.
        Input:
            mol: first molecule
            substructure: substructure molecule
        Output:
            match array
    """
    matches = mol.GetSubstructMatches(substructure)
    if len(matches) == 1:
        return matches[0]
    else:
        min_modifications = float('inf')
        min_match = None
        for match in matches:
            modif_sites = set()
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in match and ~(bond.GetEndAtomIdx() in match):
                    modif_sites.add(bond.GetBeginAtomIdx())
                if bond.GetEndAtomIdx() in match and ~(bond.GetBeginAtomIdx() in match):
                    modif_sites.add(bond.GetEndAtomIdx())
            if len(modif_sites) < min_modifications:
                min_modifications = len(modif_sites)
                min_match = match
        return min_match


def _find_minimal_modification_edges_match(mol, substructure):
    """
        Finds the matching that results in the minimum number of modification edges, if multiple matches are found.
        If multiple matches have the minimum number, one random result is returned.
        Input:
            mol: first molecule
            substructure: substructure molecule
        Output:
            match array
    """
    if not mol.HasSubstructMatch(substructure):
        raise ValueError("The substructure is not a substructure of the molecule")
    matches = mol.GetSubstructMatches(substructure)
    if len(matches) == 1:
        return matches[0]
    else:
        min_modifications = float('inf')
        min_match = None
        for match in matches:
            modificationEdgesOutward, modificationEdgesInside, _ = _get_edge_modifications(mol, substructure, match)
            num_modifications = len(modificationEdgesOutward) + len(modificationEdgesInside)
            if num_modifications < min_modifications:
                min_modifications = num_modifications
                min_match = match
        return min_match


def _get_edge_modifications(mol, substructure, match):
    """
        Calculates the modification edges to get mol from substructure given the mapping of the atoms
        Input:
            mol: first molecule
            substructure: substructure molecule
            match: the match array, mapping atoms from substructure to mol
        Output:
            modificationEdgesOutward: the modification edges that go from atoms in the substructure to atoms outside the substructure
            modificationEdgesInside: the modification edges that happen within the atoms in the substructure
            noneModificationDifferentEdges: the edges that are not modification edges but are different between the two molecules (edges that are completely outside the substructure)
    """
    reverse_match = {v: k for k, v in enumerate(match)}
    intersect = set(match)
    modificationEdgesOutward = []
    modificationEdgesInside = []
    noneModificationDifferentEdges = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in intersect or bond.GetEndAtomIdx() in intersect:
            if bond.GetBeginAtomIdx() in intersect and bond.GetEndAtomIdx() in intersect:
                a1 = reverse_match[bond.GetBeginAtomIdx()]
                a2 = reverse_match[bond.GetEndAtomIdx()]
                if substructure.GetBondBetweenAtoms(a1, a2) is None:
                    modificationEdgesInside.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            else:
                modificationEdgesOutward.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        else:
            noneModificationDifferentEdges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    
    return modificationEdgesOutward, modificationEdgesInside, noneModificationDifferentEdges


def _calculateModificationSites(mol, substructure, inParent = True):
    """
        Calculates the number of modification sites to get mol from substructure (works when one moleculte is a substructure of the other molecule)
        if multiple matches are found, the modification sites that result in the minimum number of modifications are returned
        Input:
            mol1: first molecule
            substructure: substructure molecule
            inParent: bool, if True, the modification sites are given in the parent molecule, if False, the modification sites are given in the substructure
        Output:
            count: modification sites
    """

    matches = _find_minimal_modification_edges_match(mol, substructure)
    intersect = set(matches)

    
    modificationSites = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in intersect:
            for tempAtom in intersect:
                if mol.GetBondBetweenAtoms(atom.GetIdx(), tempAtom) is not None:
                    modificationSites.append(tempAtom)

    if inParent:
        return modificationSites
    else:
        res = []
        if type(matches[0]) is tuple:
            # multiple matches (substructure is happening multiple times in the parent molecule)
            for match in matches:
                subMatches = list(match)
                temp_res = []
                for atom in modificationSites:
                    idx = subMatches.index(atom) # based on rdkit library, the value in matches array are sorted by index
                    temp_res.append(idx)
                res.append(temp_res)
        else:
            for atom in modificationSites:
                subMatches = list(matches)
                idx = subMatches.index(atom) # based on rdkit library, the value in matches array are sorted by index
                res.append(idx)
        return res


def _get_molecules(mol1, mol2):
    copy_mol1 = deepcopy(mol1)
    copy_mol2 = deepcopy(mol2)
    copy_mol1 = _get_molecule(copy_mol1)
    copy_mol2 = _get_molecule(copy_mol2)
    if copy_mol1 is None or copy_mol2 is None:
        raise ValueError("The molecules are not valid")
    return copy_mol1, copy_mol2


def _get_molecule(identifier = None, smiles=None, inchi=None, molblock=None, smarts=None, **kwargs):
    """
    Get the rdkit molecule from smiles, inchi or GNPS identifier
    
    Parameters:
    ----------
    identifier: str
        GNPS identifier, smiles, inchi, molblock or smart
    smiles: str
        smiles string
    inchi: str
        inchi string
    molblock: str
        molblock string
    smart: str

    :return: rdkit molecule or None
    """
    if identifier is not None:
        if isinstance(identifier, str):
            try:
                molecule = Chem.MolFromSmiles(identifier)
                if molecule:
                    return molecule
            except Exception:
                pass

            try:
                molecule = Chem.MolFromInchi(identifier)
                if molecule:
                    return molecule
            except Exception:
                pass

            try:
                molecule = Chem.MolFromMolBlock(identifier)
                if molecule:
                    return molecule
            except Exception:
                pass

            try:
                molecule = Chem.MolFromSmarts(identifier)
                if molecule:
                    return molecule
            except Exception:
                pass

            raise ValueError("The identifier is not valid or it is not supported")
            
        if isinstance(identifier, Chem.Mol):
            return identifier

    
    if smiles is not None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            return molecule
        else:
            raise ValueError("The smiles string is not valid")
    
    if inchi is not None:
        molecule = Chem.MolFromInchi(inchi)
        if molecule:
            return molecule
        else:
            raise ValueError("The inchi string is not valid")
    
    if molblock is not None:
        molecule = Chem.MolFromMolBlock(molblock)
        if molecule:
            return molecule
        else:
            raise ValueError("The molblock string is not valid")
    
    if smarts is not None:
        molecule = Chem.MolFromSmarts(smarts)
        if molecule:
            return molecule
        else:
            raise ValueError("The smart string is not valid")
    
    raise ValueError("No valid input is provided")
