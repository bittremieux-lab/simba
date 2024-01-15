from typing import Dict, IO, Sequence, Union
from src.spectrum_ext import SpectrumExt
import numpy as np
from src.config import Config
from tqdm import tqdm 
import re
from rdkit import Chem
from rdkit.Chem import inchi
import requests
from itertools import islice

class NistLoader:
    '''
    code for loading nist data
    '''
    def __init__(self):
        self._smiles_cache = {}

    @staticmethod
    def init_spec_variables():
        spectrum_dict = {}
        spectrum_dict['identifier']='N/A'
        spectrum_dict['adduct']='',
        spectrum_dict['precursor_mz']=0
        spectrum_dict['precursor_charge']=1
        spectrum_dict['mz']=[0,]
        spectrum_dict['intensity']=[0,]
        spectrum_dict['retention_time']=np.nan
        spectrum_dict['library']=1
        spectrum_dict['inchi'] = 'N/A'
        spectrum_dict['smiles'] = 'N/A'
        spectrum_dict['ionmode']= ''
        spectrum_dict['bms']=''
        spectrum_dict['superclass']=''
        spectrum_dict['classe']=''
        spectrum_dict['subclass']=''
        spectrum_dict['inchi_key'] = ''
        spectrum_dict['params']={}
        spectrum_dict['adduct']=''
        return spectrum_dict

    @staticmethod
    def get_precursor_charge( line):
        # get the adduct info
        sufix = line.split(']')[-1]
        if '+' in sufix:
            charge_symbol='+'

        else: # negative
            charge_symbol='-'
            
        charge = sufix.split(charge_symbol)[0]

        if charge =='': #if it is not specified the charge, the default is 1
            charge=1
        return int(charge)
    

    



    @staticmethod
    def parse_file(source: Union[IO, str], num_samples=None, initial_line_number=0):
        with open(source, 'r') as file:
            spectra = []
            iterator = iter(file)

            # Move iterator to the specified line number
            #for _ in range(0,initial_line_number):
            #    next(iterator)
            # Skip initial lines
            iterator = islice(file, initial_line_number, None)

            # set the pointer to initial_line_number
            line_number = initial_line_number

            #Process the file
            for line in (iterator):
                #set the line
                if line.startswith('Name:'):
                    #initialize variables:
                    spectrum_dict = NistLoader.init_spec_variables()
                    spectrum_dict['identifier']= line.split('Name:')[-1]
                #if line.startswith('Synon: $:17'):
                #    spectrum_dict['identifier'] = spectrum_dict['identifier'] + line.split('Synon: $:17')[-1]
                
                if line.startswith('Synon: $:03'): # put the adduct
                    spectrum_dict['adduct'] = line.split('Synon: $:03')[-1].split(']')[0].split('[')[-1] #e.g [M+H]+2
                    spectrum_dict['identifier'] = spectrum_dict['identifier'] + ' ' + spectrum_dict['adduct']
                    spectrum_dict['precursor_charge']= NistLoader.get_precursor_charge(line.split('Synon: $:03')[-1])
                    
                if line.startswith('Synon: $:11'):
                    acronym = line.split('Synon: $:11')[-1].strip()
                    spectrum_dict['ionmode'] = 'Positive' if acronym == 'P' else 'Negative'
                if line.startswith('Synon: $:28'):
                    spectrum_dict['inchi_key']= line.split('Synon: $:28')[-1].strip()
                if line.startswith('PrecursorMZ:'):
                    precursor_mz = line.split('PrecursorMZ:')[-1]
                    spectrum_dict['precursor_mz'] = float(precursor_mz)

                if line.startswith('Num peaks:'):
                    number_peaks = line.split('Num peaks:')[-1]
                    number_peaks = int(number_peaks)
                    mz = np.zeros(number_peaks)
                    intensity = np.zeros(number_peaks)
                    for n_peak in range(0, number_peaks):
                        # when getting the peaks we update the pointers
                        line = next(iterator)
                        line_number= line_number + 1
                        list_mz_int = line.split(' ')

                        try:
                            mz[n_peak] = float(list_mz_int[0])
                            intensity[n_peak] = float(list_mz_int[1])
                        except:
                            print('*** Error while trying to get n peaks from spectra ***')
                            break

                    spectrum_dict['number_peaks'] = number_peaks
                    spectrum_dict['mz']=mz
                    spectrum_dict['intensity']=intensity
                if line.strip() == '': #a blank line when there is a new item 
                    # recollect info for params attribute
                    if spectrum_dict['identifier']!= 'N/A':
                        spectrum_dict['m/z array'] = spectrum_dict['mz']
                        spectrum_dict['intensity array'] = spectrum_dict['intensity']
                        spectrum_dict['params']['libraryquality'] = spectrum_dict['library']
                        spectrum_dict['params']['charge'] = [spectrum_dict['precursor_charge']]
                        spectrum_dict['params']['pepmass'] = [spectrum_dict['precursor_mz']]
                        spectrum_dict['params']['ionmode'] = spectrum_dict['ionmode']
                        spectrum_dict['params']["name"] = spectrum_dict['identifier'] 
                        spectrum_dict['params']['inchi']= spectrum_dict['inchi']
                        spectrum_dict['params']['smiles']= spectrum_dict['smiles']

                        #TODO
                        spectrum_dict["params"]["organism"]= 'nist'
                        spectrum_dict["params"]["spectrumid"]='abcd'
                        spectra.append(spectrum_dict)

                        
                        spectrum_dict = NistLoader.init_spec_variables()

                        # break the loop if we attain the maximum number of samples
                        if num_samples is not None:
                            if len(spectra)>= num_samples:
                                line_number= line_number + 1
                                break
                
                # update pointer
                line_number= line_number + 1

        # set the current line number to the next value
        #current_line_number = current_line_number + 1
        return spectra, line_number
    


    def compute_smiles(self, inchi_key):
        #base_url = f"https://cactus.nci.nih.gov/chemical/structure/{inchi_key}/smiles"
        base_url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi_key}/property/CanonicalSMILES/JSON'
        #print('smiles cache:')
        #print(self._smiles_cache)
        if inchi_key in self._smiles_cache:
            #print(f"Using cached result for {inchi_key}")
            return self._smiles_cache[inchi_key]
        
        try:
            response = requests.get(base_url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
                
            #smiles=response.text
            smiles = response.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            self._smiles_cache[inchi_key] = smiles
            # Print the content of the response
            #print(f"inchi_key: {inchi_key} SMILES: {smiles}")
            return smiles
        
        except requests.exceptions.RequestException as e:
            self._smiles_cache[inchi_key] = 'N/A'
            #print(f"Error fetching data for smiles {inchi_key}")
            return 'N/A'

    

    def compute_all_smiles(self, all_spectrums_dict, use_tqdm, verbose=1):
        iterator = tqdm(all_spectrums_dict ) if use_tqdm else all_spectrums_dict

        no_smiles_count=0
        for s in iterator:
            s['smiles'] = self.compute_smiles(s['inchi_key'])
            s['params']['smiles'] = s['smiles']

            if s['smiles']=='N/A':
                no_smiles_count= no_smiles_count+1
        if verbose!=0:
            print(f'Percentage of N/A smiles in current block: {100* no_smiles_count/len(all_spectrums_dict)}%')
        return all_spectrums_dict