from rdkit import Chem

class MurckoScaffold:

    '''
    code for computing murcko scaffold for dividing train, val and test sets
    '''
    
    def get_bm_scaffold(smiles):
        try:
            scaffold = Chem.MolToSmiles(MakeScaffoldGeneric(mol=Chem.MolFromSmiles(smiles)))
        except Exception:
            #print("Raise AtomValenceException, return basic Murcko Scaffold")
            scaffold = smiles
        return scaffold
