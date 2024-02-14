'''
export spectrums for being used by other researchers
'''
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
from spectrum_utils.spectrum import MsmsSpectrum

# parameters
nist_file= '../data/all_spectrums_nist.pkl'
gnps_file= '../data/all_spectrums_gnps.pkl'
output_pickle= '../data/to_export_gnps_nist.pkl'
print(f'Current time: {datetime.now()}')

# load gnps_spectra
with open(gnps_file, 'rb') as file:
        all_spectrums_gnps = dill.load(file)['spectrums']


print(f'Total of GNPS spectra: {len(all_spectrums_gnps)}')
# use nist
print(f'Current time: {datetime.now()}')
with open(nist_file, 'rb') as file:
        all_spectrums_nist = dill.load(file)['spectrums']



print(f'Total of NIST spectra: {len(all_spectrums_nist)}')
print(f'Current time: {datetime.now()}')
# merge spectrums
all_spectrums = all_spectrums_gnps + all_spectrums_nist 


print(f'Total spectra before removing NAN SMILES: {len(all_spectrums)}')

spectrums_su= [MsmsSpectrum(
        s.identifier,
        s.precursor_mz,
        s.precursor_charge,
        s.mz,
        s.intensity,
        s.retention_time)  for s in all_spectrums]
smiles = [s.smiles for s in all_spectrums]

##remove nan smiles
spectrums_su = [s for s,smiles in zip(spectrums_su, smiles) if smiles != '']
smiles = [s for s in smiles if smiles != '']

print(f'Total spectra after removing NANs: {len(spectrums_su)}')
# convert from spectrumext to spectrumutils


output_data = {'spectrums': spectrums_su,
               'smiles': smiles}

# write data
with open(output_pickle, 'wb') as file:
        pickle.dump(output_data, file)