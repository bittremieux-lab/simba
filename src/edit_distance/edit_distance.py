#from ipywidgets import interact, fixed, widgets
import pandas as pd
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem import PandasTools
import argparse
from tqdm import tqdm
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import numpy as np

import os

def get_number_of_modification_edges(mol, substructure):
    if not mol.HasSubstructMatch(substructure):
        raise ValueError("The substructure is not a substructure of the molecule.")
    
    matches = mol.GetSubstructMatch(substructure)
    intersect = set(matches)
    modification_edges = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in intersect and bond.GetEndAtomIdx() in intersect:
            continue
        if bond.GetBeginAtomIdx() in intersect or bond.GetEndAtomIdx() in intersect:
            modification_edges.append(bond.GetIdx())
        
    return modification_edges

def get_edit_distance(mol1, mol2):
    """
        Calculates the edit distance between mol1 and mol2.
        Input:
            mol1: first molecule
            mol2: second molecule
        Output:
            edit_distance: edit distance between mol1 and mol2
    """
    if mol1.GetNumAtoms() > 60 or mol2.GetNumAtoms() > 60:
        raise ValueError("The molecules are too large.")
    mcs1 = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs1.smartsString)
    if mcs_mol.GetNumAtoms() < mol1.GetNumAtoms()//2 and mcs_mol.GetNumAtoms() < mol2.GetNumAtoms()//2:
        raise ValueError("The MCS is too small.")
    if mcs_mol.GetNumAtoms() < 2:
        raise ValueError("The MCS is too small.")
    
    dist1 = get_number_of_modification_edges(mol1, mcs_mol)
    dist2 = get_number_of_modification_edges(mol2, mcs_mol)
    return len(dist1) + len(dist2)

def simba_get_edit_distance(mol1, mol2):
    """
        Calculates the edit distance between mol1 and mol2.
        Input:
            mol1: first molecule
            mol2: second molecule
        Output:
            edit_distance: edit distance between mol1 and mol2
    """
    #if mol1.GetNumAtoms() > 60 or mol2.GetNumAtoms() > 60:
    #    print("The molecules are too large.")
    #    return np.nan
    mcs1 = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs1.smartsString)
    if mcs_mol.GetNumAtoms() < mol1.GetNumAtoms()//2 and mcs_mol.GetNumAtoms() < mol2.GetNumAtoms()//2:
        print("The molecules are too small.")
        return np.nan
    if mcs_mol.GetNumAtoms() < 2:
        print("The molecules are too small.")
        return np.nan
    
    dist1 = get_number_of_modification_edges(mol1, mcs_mol)
    dist2 = get_number_of_modification_edges(mol2, mcs_mol)
    return len(dist1) + len(dist2)

def solve_pair(smiles1, smiles2):
    if type(smiles1) == str:
        mol1 = Chem.MolFromSmiles(smiles1)
    else:
        mol1 = smiles1
    
    if type(smiles2) == str:
        mol2 = Chem.MolFromSmiles(smiles2)
    else:
        mol2 = smiles2
    fpgen = AllChem.GetRDKitFPGenerator(maxPath=3,fpSize=512)
    fps = [fpgen.GetFingerprint(x) for x in [mol1, mol2]]
    tanimoto = DataStructs.TanimotoSimilarity(fps[0],fps[1])
    if tanimoto < 0.2:
        raise ValueError("The Tanimoto similarity is too low.")
    
    distance = get_edit_distance(mol1, mol2)

    is_sub = mol1.HasSubstructMatch(mol2) or mol2.HasSubstructMatch(mol1)
    return distance, tanimoto, is_sub



def simba_solve_pair(smiles1, smiles2, low_similarity=5):
    if type(smiles1) == str:
        mol1 = Chem.MolFromSmiles(smiles1)
    else:
        mol1 = smiles1
    
    if type(smiles2) == str:
        mol2 = Chem.MolFromSmiles(smiles2)
    else:
        mol2 = smiles2
    fpgen = AllChem.GetRDKitFPGenerator(maxPath=3,fpSize=512)
    fps = [fpgen.GetFingerprint(x) for x in [mol1, mol2]]
    tanimoto = DataStructs.TanimotoSimilarity(fps[0],fps[1])
    if tanimoto < 0.2:
        print("The Tanimoto similarity is too low.")
        distance = 5
    else:
        distance = simba_get_edit_distance(mol1, mol2)
    return distance, tanimoto

def main(data, out_dir, data_y):
    mols = [Chem.MolFromSmiles(data_y.iloc[i]['Smiles']) for i in range(len(data_y))]

    with open(out_dir, 'w') as f:
        f.write('smiles1,smiles2,distance,tanimoto,is_sub,inchi1,inchi2\n')
        for i, row1 in tqdm(data.iterrows(), total=len(data), ascii=True):
            moli = Chem.MolFromSmiles(row1['Smiles'])
            if moli.GetNumAtoms() > 60:
                continue
            for j, row2 in data_y.iterrows():
                try:
                    distance, tanimoto, is_sub = solve_pair(moli, mols[j])
                    f.write(f"{row1['Smiles']},{row2['Smiles']},{distance},{tanimoto},{is_sub},\"{row1['INCHI']}\",\"{row2['INCHI']}\"\n")
                    # print("here")
                except Exception as e:
                    # print(e)
                    continue
        
        # if 10 percent of progress is made, save the file
        if i % (len(data) // 10) == 0:
            f.flush()
            os.fsync(f.fileno())

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate edit distance and Tanimoto similarity between pairs of molecules.')
    parser.add_argument('data', type=str, help='Path to the input data file.')
    parser.add_argument('out_dir', type=str, help='Path to the output directory.')
    parser.add_argument('--batch_x', type=int, help='Number of rows in the batch.', default=1)
    parser.add_argument('--batch_y', type=int, help='Number of columns in the batch.', default=1)
    parser.add_argument('--batch_index', type=int, help='Index of the batch to process.', default=0)
    parser.add_argument('--cached_dir', type=str, help='Path to the directory containing cached data.', default=None)
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.out_dir)):
        os.makedirs(os.path.dirname(args.out_dir))
    
    if args.cached_dir is not None:
        print("Checking cache...")
        file_name = os.path.basename(args.out_dir)
        print("file_name", file_name)
        cached_file = os.path.join(args.cached_dir, file_name)
        print(cached_file)
        if os.path.exists(cached_file):
            print("Cache found!.")
            # rewrite the file in the output directory
            print(args.out_dir, "is being written.")
            with open(cached_file, 'r') as f:
                with open(args.out_dir, 'w') as f2:
                    f2.write(f.read())
            exit(0)
    else:
        print("No cache directory provided.")
    

    data = pd.read_csv(args.data)
    batch_x = args.batch_x
    batch_y = args.batch_y
    batch_index = args.batch_index
    out_dir = args.out_dir

    index_x = batch_index // batch_y
    index_y = batch_index % batch_y

    data_y = get_data(data, index_y, batch_y)
    data = get_data(data, index_x, batch_x)

    # print(len(data_y), len(data))
    # print(data_y.index)

    main(data, out_dir, data_y)