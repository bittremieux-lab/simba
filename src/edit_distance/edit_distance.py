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

class EditDistance:

    @staticmethod
    def create_input_df(smiles, indexes_0, indexes_1):
        df=pd.DataFrame()
        print(f'Length of spectrums: {len(smiles)}')

        df['smiles_0']= [smiles[int(r)]  for r in indexes_0]
        df['smiles_1']= [smiles[int(r)]  for r in indexes_1]

        return df


    def compute_edit_distance(smiles,sampled_index, size_batch, id, random_sampling, config):

        # where to save results
        indexes_np = np.zeros((int(size_batch), 3),)
        # initialize randomness
        if random_sampling:
            np.random.seed(id)
            indexes_np[:,0] = np.random.randint(0,len(smiles), int(size_batch))
            indexes_np[:,1] = np.random.randint(0,len(smiles), int(size_batch))
        else:
            indexes_np[:,0] = sampled_index
            indexes_np[:,1]= np.arange(0, size_batch)

        fpgen = AllChem.GetRDKitFPGenerator(maxPath=3,fpSize=512)

        #df['edit_distance'] = df.apply(lambda x:EditDistance.simba_solve_pair_edit_distance(x['smiles_0'], x['smiles_1'], fpgen, ), axis=1)
        distances=[]

        fps=[]
        mols= []

        for index in range(0, indexes_np.shape[0]):
            #print('')
            #print(f'id: {id}, {index}, S0: {s0}, S1: {s1}')
            row = indexes_np[index]

            s0 = smiles[int(row[0])] 
            s1 = smiles[int(row[1])] 
            dist, tanimoto = EditDistance.simba_solve_pair_edit_distance(s0,s1,fpgen)
            #print(dist)
            distances.append(dist)
        
        indexes_np[:,2]= distances
        return indexes_np 

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
            #print("The molecules are too small.")
            return np.nan
        if mcs_mol.GetNumAtoms() < 2:
            
            #print("The molecules are too small.")
            return np.nan
        
        dist1 = EditDistance.get_number_of_modification_edges(mol1, mcs_mol)
        dist2 =  EditDistance.get_number_of_modification_edges(mol2, mcs_mol)

        if (dist1 is not None) and (dist2 is not None):
            return len(dist1) + len(dist2)
        else:
            return np.nan





    def simba_solve_pair_edit_distance(s0, s1, fpgen, low_similarity=5):
        mol1 = Chem.MolFromSmiles(s0)

        
        mol2 = Chem.MolFromSmiles(s1)

        

        
        #try:
        fps = [fpgen.GetFingerprint(x) for x in [mol1, mol2]]

        #except:
        #    print('')
        #    print(f'1: {mol1}')
        #    print(f'2: {mol1}')
        #    raise ValueError('None values')
        tanimoto = DataStructs.TanimotoSimilarity(fps[0],fps[1])
        if tanimoto < 0.2:
            #print("The Tanimoto similarity is too low.")
            distance = np.nan
        else:
            distance = EditDistance.simba_get_edit_distance(mol1, mol2)
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

