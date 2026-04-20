from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric

from simba.utils.logger_setup import logger


class MurckoScaffold:
    """
    code for computing murcko scaffold for dividing train, val and test sets
    """

    @staticmethod
    def get_bm_scaffold(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES, could not parse molecule ({smiles})")
                return ""
            scaffold_mol = MakeScaffoldGeneric(mol=mol)
            if scaffold_mol is None:
                logger.warning(f"No scaffold for given SMILES ({smiles})")
                return ""
            scaffold = Chem.MolToSmiles(scaffold_mol)
        except Exception:
            logger.warning(f"No scaffold for given SMILES ({smiles})")
            scaffold = ""
        return scaffold
