from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdRascalMCES


class MCES:
    @staticmethod
    def calculate_mcs_similarity(smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        # Perform MCS (Maximum Common Substructure) search
        mcs = rdFMCS.FindMCS([mol1, mol2])

        # Get SMARTS pattern from MCS result
        mcs_smarts = Chem.MolToSmarts(mcs.queryMol)

        # Calculate Tanimoto-like similarity
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)
        # mcs_count = len(Chem.GetMolFrags(mcs_mol))
        mcs_count = mcs_mol.GetNumAtoms()
        similarity = mcs_count / (mol1.GetNumAtoms() + mol2.GetNumAtoms() - mcs_count)

        return similarity, mcs_mol

    @staticmethod
    def calculate_mces_sim(smiles1, smiles2, similarity_threshold=0.7):

        ad1 = Chem.MolFromSmiles(smiles1)
        ad2 = Chem.MolFromSmiles(smiles2)
        opts = rdRascalMCES.RascalOptions()
        opts.similarityThreshold = similarity_threshold
        opts.returnEmptyMCES = True
        results = rdRascalMCES.FindMCES(ad1, ad2, opts)
        if len(results) != 0:
            similarity_tier1 = results[0].tier1Sim
            similarity_tier2 = results[0].tier2Sim

            if similarity_tier2 != -1:
                return similarity_tier2  # if the lower threshold is not surpassed
            else:
                return similarity_tier1
        else:
            return None
