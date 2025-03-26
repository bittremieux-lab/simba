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
import src.edit_distance.mol_utils as mu
import os
#from myopic_mces import MCES 

class EditDistance:

    @staticmethod
    def create_input_df(smiles, indexes_0, indexes_1):
        df=pd.DataFrame()
        print(f'Length of spectrums: {len(smiles)}')

        df['smiles_0']= [smiles[int(r)]  for r in indexes_0]
        df['smiles_1']= [smiles[int(r)]  for r in indexes_1]

        return df

    def compute_ed_or_mces(smiles, sampled_index, size_batch, identifier, random_sampling, config, fps, mols, use_edit_distance, ):
        print(f'Processing: {sampled_index}')
        #print(f'id')
        # where to save results
        indexes_np = np.zeros((int(size_batch), 3),)
        # initialize randomness
        if random_sampling:
            np.random.seed(identifier)
            indexes_np[:,0] = np.random.randint(0,len(smiles), int(size_batch))
            indexes_np[:,1] = np.random.randint(0,len(smiles), int(size_batch))
        else:
            indexes_np[:,0] = sampled_index
            indexes_np[:,1]= np.arange(0, size_batch)

        #fpgen = AllChem.GetRDKitFPGenerator(maxPath=3,fpSize=512)

        #df['edit_distance'] = df.apply(lambda x:EditDistance.simba_solve_pair_edit_distance(x['smiles_0'], x['smiles_1'], fpgen, ), axis=1)
        distances=[]

        for index in range(0, indexes_np.shape[0]):
            #print('')
            #print(f'id: {id}, {index}, S0: {s0}, S1: {s1}')
            row = indexes_np[index]

            s0 = smiles[int(row[0])] 
            s1 = smiles[int(row[1])] 
            fp0 = fps[int(row[0])] 
            fp1 = fps[int(row[1])] 
            mol0= mols[int(row[0])] 
            mol1= mols[int(row[1])] 
            if use_edit_distance:
                dist, tanimoto = EditDistance.simba_solve_pair_edit_distance(s0,s1, fp0, fp1, mol0, mol1)
            else:
                dist, tanimoto = EditDistance.simba_solve_pair_mces(s0,s1, fp0, fp1, mol0, mol1, config.THRESHOLD_MCES)
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


    def get_edit_distance_from_smiles(smiles1, smiles2, return_nans=True):
        mol1=Chem.MolFromSmiles(smiles1)
        mol2=Chem.MolFromSmiles(smiles2)
        return EditDistance.simba_get_edit_distance(mol1, mol2, return_nans=return_nans)

    def simba_get_edit_distance(mol1, mol2, return_nans=True):
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
        if return_nans:
            if mcs_mol.GetNumAtoms() < mol1.GetNumAtoms()//2 and mcs_mol.GetNumAtoms() < mol2.GetNumAtoms()//2:
                #print("The molecules are too small.")
                return np.nan
            if mcs_mol.GetNumAtoms() < 2:
                
                #print("The molecules are too small.")
                return np.nan
        
        #dist1 = EditDistance.get_number_of_modification_edges(mol1, mcs_mol)
        #dist2 =  EditDistance.get_number_of_modification_edges(mol2, mcs_mol)

        #if (dist1 is not None) and (dist2 is not None):
        #    return len(dist1) + len(dist2)
        #else:
        #    return np.nan
        # print("going to calculate edit distance")
        dist1, dist2 = mu.get_edit_distance_detailed(mol1, mol2, mcs_mol)
        
        distance = dist1 + dist2

        return distance

    from functools import lru_cache

    @lru_cache(maxsize=1000)  # Set maxsize to None for an unbounded cache or a specific integer for a bounded cache
    def return_mol(smiles):
        # Simulate some processing
        return Chem.MolFromSmiles(smiles)



    def simba_solve_pair_edit_distance(s0, s1,  fp0, fp1,mol0, mol1):
        
        tanimoto = DataStructs.TanimotoSimilarity(fp0,fp1)
        
        if tanimoto < 0.2:
                #print("The Tanimoto similarity is too low.")
                #distance = np.nan
                distance = 666 # very high number to remark that they are very different
        else:
                #mol0 = EditDistance.return_mol(s0)
                #mol1 = EditDistance.return_mol(s1)
                if (mol0.GetNumAtoms() > 60) or (mol1.GetNumAtoms() > 60):
                    #raise ValueError("The molecules are too large.")
                    return np.nan, tanimoto
                else:
                    distance = EditDistance.simba_get_edit_distance(mol0, mol1)
                    
        return distance, tanimoto

    def simba_solve_pair_mces(s0, s1,  fp0, fp1,mol0, mol1, threshold, 
                                TIME_LIMIT=2,# 2 seconds
                ):
        
        tanimoto = DataStructs.TanimotoSimilarity(fp0,fp1)
        
        if tanimoto < 0.2:
                #print("The Tanimoto similarity is too low.")
                #distance = np.nan
                distance = 666 # very high number to remark that they are very different
        else:
                #mol0 = EditDistance.return_mol(s0)
                #mol1 = EditDistance.return_mol(s1)
                #distance = MCES(s0, s1, threshold=threshold)
                if (mol0.GetNumAtoms() > 60) or (mol1.GetNumAtoms() > 60):
                    #raise ValueError("The molecules are too large.")
                    return np.nan, tanimoto
                else:
                    from myopic_mces import MCES 
                    result =     MCES(
                                        s0,
                                        s1,
                                        threshold=threshold,
                                        i=0,
                                        #solver='CPLEX_CMD',       # or another fast solver you have installed
                                        solver='PULP_CBC_CMD',
                                        solver_options={
                                                'threads': 1, 
                                                'msg': False,
                                                'timeLimit': TIME_LIMIT  # Stop CBC after 1 seconds
                                            },  
                                        #solver_options={'threads': 1, 'msg': False},  # use single thread + no console messages
                                        no_ilp_threshold=False,   # allow the ILP to stop early once the threshold is exceeded
                                        always_stronger_bound=False,  # use dynamic bounding for speed
                                        catch_errors=False        # typically raise exceptions if something goes wrong
                                    )
                    distance = result[1]
                    time_taken=result[2]
                    exact_answer= result[3]

                    if (time_taken >=(0.9*TIME_LIMIT) and (exact_answer != 1)):
                        distance= np.nan

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

