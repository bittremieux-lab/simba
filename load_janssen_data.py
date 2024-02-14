import dill
from src.load_data import LoadData
from sklearn.model_selection import train_test_split
from src.train_utils import TrainUtils
from src.preprocessor import Preprocessor
import pickle
import sys
from src.config import Config
from src.parser import Parser
from datetime import datetime
from src.loader_saver import LoaderSaver

# Get the current date and time
print("Initiating molecular pair script ...")
print(f"Current time: {datetime.now()}")

## PARAMETERS
config = Config()
parser = Parser()
config = parser.update_config(config)
janssen_path = r"/scratch/antwerpen/209/vsc20939/data/drug_plus.mgf"
use_tqdm = config.enable_progress_bar
max_number_spectra_gnps = 100000
max_combinations = 100
# load spectra
all_spectrums_janssen = LoadData.get_all_spectrums(
    janssen_path,
    max_number_spectra_gnps,
    use_tqdm=use_tqdm,
    use_janssen=True,
    config=config,
)


print(f"Number of spectra from Janssen: {len(all_spectrums_janssen)}")

all_spectrums_janssen = all_spectrums_janssen[0:100]
molecule_pairs = TrainUtils.compute_all_tanimoto_results(
    all_spectrums_janssen,
    max_combinations=max_combinations,
    use_tqdm=use_tqdm,
    max_mass_diff=config.MAX_MASS_DIFF,
    min_mass_diff=config.MIN_MASS_DIFF,
)

print(len(molecule_pairs))

for i in range(5):
    print("")
    print(molecule_pairs[i].similarity)
    print(molecule_pairs[i].smiles_0)
    print(molecule_pairs[i].smiles_1)
