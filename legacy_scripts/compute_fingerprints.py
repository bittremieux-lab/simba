print("Initiating script")

# finderprints
import dill
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm

dataset_path = (
    "/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207.pkl"
)
output_path = "/scratch/antwerpen/209/vsc20939/data/dataset_processed_augmented_20231207_fingerprints.pkl"


def generate_fingerprint(smiles, d_size=64):
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=64)
        return np.array(list(fingerprint))
    else:
        return np.zeros((64,))


def gen_fing_for_molecule_pairs(molecule_pairs):

    for m in molecule_pairs:
        m.fingerprint_0 = generate_fingerprint(m.smiles_0)
        m.fingerprint_1 = generate_fingerprint(m.smiles_1)

    return molecule_pairs


print("load data")
# Load the dataset from the pickle file
with open(dataset_path, "rb") as file:
    dataset = dill.load(file)


dataset_train = dataset["molecule_spairs_train"]
dataset_val = dataset["molecule_pairs_val"]
dataset_test = dataset["molecule_pairs_test"]


print("generating fingerprints")
dataset_train = gen_fing_for_molecule_pairs(dataset_train)
print("finished training data")
dataset_val = gen_fing_for_molecule_pairs(dataset_val)
dataset_test = gen_fing_for_molecule_pairs(dataset_test)

dataset = {
    "molecule_spairs_train": dataset_train,
    "molecule_pairs_val": dataset_val,
    "molecule_pairs_test": dataset_test,
}

print("writing")
with open(output_path, "wb") as file:
    dill.dump(dataset, file)
