from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from rdkit.Chem.inchi import MolFromInchi 
import functools
# disable logging info
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')

class Tanimoto:

    @functools.lru_cache
    def compute_tanimoto(identifier_1, identifier_2, nbits=2048, use_inchi=False):

        if use_inchi: # which format to use
                conversion_function = MolFromInchi
        else:
                conversion_function = Chem.MolFromSmiles

        # Convert SMILES notations to RDKit molecules


        if (identifier_1 != '' and identifier_1 != 'N/A') and (identifier_2 != '' and  identifier_2 != 'N/A'):
            mol_1 = conversion_function(identifier_1)
            mol_2 = conversion_function(identifier_2)
        else:
            mol_1= None
            mol_2 = None
        
        if (mol_1 is not None) and (mol_2 is not None):
            # Generate Morgan fingerprints for the molecules
            #fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_1, 2, nBits=nbits)
            #fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_2, 2, nBits=nbits)
            fp1, fp2 = Chem.RDKFingerprint(mol_1), Chem.RDKFingerprint(mol_2)
            # Calculate the Tanimoto similarity
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            
            #print(f"Tanimoto Similarity: {similarity}")
            return similarity
        else:
            #print("Unable to generate molecular fingerprints from SMILES notations.")
            return None
        