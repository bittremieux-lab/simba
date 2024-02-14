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
rkrb.DisableLog("rdApp.error")
from numba import njit


class Tanimoto:

    # @functools.lru_cache
    @staticmethod
    def compute_tanimoto(fp1, fp2, nbits=2048, use_inchi=False):
        if (fp1 is not None) and (fp2 is not None):
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            return similarity
        else:
            return None

    @functools.lru_cache
    def compute_fingerprint(smiles):
        if smiles != "" and smiles != "N/A":
            try:
                smiles_canon = Chem.CanonSmiles(smiles)
                mol = Chem.MolFromSmiles(smiles_canon)
                fp = Chem.RDKFingerprint(mol) if mol is not None else None
            except:  # sometimes the smiles is not correctly understood by the rdkit
                fp = None
        else:
            fp = None

        return fp
