import logging
from typing import Dict, IO, Iterator, Sequence, Union

from pyteomics import mgf
import pyteomics
from spectrum_utils.spectrum import MsmsSpectrum
from src.spectrum_ext import SpectrumExt
import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
import spectrum_utils as su
import numpy as np
from src.config import Config
from src.preprocessing_utils import PreprocessingUtils
from src.murcko_scaffold import MurckoScaffold
import operator
from tqdm import tqdm 
import re
from src.nist_loader import NistLoader
from src.preprocessor import Preprocessor

class LoadData:
    def _get_adduct_count(adduct: str):
        """
        Split the adduct string in count and raw adduct.
    
        Parameters
        ----------
         adduct : str

        Returns
        -------
        Tuple[int, str]
          The count of the adduct and its raw value.
        """
        # Formula and charge mapping for data cleaning and harmonization.
        formulas = {
     "AC": "CH3COO",
     "Ac": "CH3COO",
    "ACN": "C2H3N",
    "AcN": "C2H3N",
    "C2H3O2": "CH3COO",
    "C2H3OO": "CH3COO",
    "EtOH": "C2H6O",
    "FA": "CHOO",
    "Fa": "CHOO",
    "Formate": "CHOO",
    "formate": "CHOO",
    "H3C2OO": "CH3COO",
    "HAc": "CH3COOH",
    "HCO2": "CHOO",
    "HCOO": "CHOO",
    "HFA": "CHOOH",
    "MeOH": "CH4O",
    "OAc": "CH3COO",
    "Oac": "CH3COO",
    "OFA": "CHOO",
    "OFa": "CHOO",
    "Ofa": "CHOO",
    "TFA": "CF3COOH",
}
        count, adduct = re.match(r"^(\d*)([A-Z]?.*)$", adduct).groups()
        count = int(count) if count else 1
        adduct = formulas.get(adduct, adduct)
        wrong_order = re.match(r"^([A-Z][a-z]*)(\d*)$", adduct)
        # Handle multimers: "M2" -> "2M".
        if wrong_order is not None:
            adduct, count_new = wrong_order.groups()
            count = int(count_new) if count_new else count
        return count, adduct
    def _clean_adduct(adduct: str) -> str:
     """
     Consistent encoding of adducts, including charge information.

     Parameters
     ----------
     adduct : str
         The original adduct string.
 
     Returns
     -------
     str
         The cleaned adduct string.
     """
     # Keep "]" for now to handle charge as "M+Ca]2"
     new_adduct = re.sub(r"[ ()\[]", "", adduct)
     # Find out whether the charge is specified at the end.
     charge, charge_sign = 0, None
     for i in reversed(range(len(new_adduct))):
         if new_adduct[i] in ("+", "-"):
             if charge_sign is None:
                 charge, charge_sign = 1, new_adduct[i]
             else:
                 # Keep increasing the charge for multiply charged ions.
                 charge += 1
         elif new_adduct[i].isdigit():
             charge += int(new_adduct[i])
         else:
             # Only use charge if charge sign was detected;
             # otherwise no charge specified.
             if charge_sign is None:
                 charge = 0
                 # Special case to handle "M+Ca]2" -> missing sign, will remove
                 # charge and try to calculate from parts later.
                 if new_adduct[i] in ("]", "/"):
                     new_adduct = new_adduct[: i + 1]
             else:
                 # Charge detected: remove from str.
                 new_adduct = new_adduct[: i + 1]
             break
     # Now remove trailing delimiters after charge detection.
     new_adduct = re.sub("[\]/]", "", new_adduct)
 
     # Unknown adduct.
     if new_adduct.lower() in map(
         str.lower, ["?", "??", "???", "M", "M+?", "M-?", "unk", "unknown"]
     ):
         return "unknown"

     # Find neutral losses and additions.
     positive_parts, negative_parts = [], []
     for part in new_adduct.split("+"):
         pos_part, *neg_parts = part.split("-")
         positive_parts.append(LoadData._get_adduct_count(pos_part))
         for neg_part in neg_parts:
             negative_parts.append(LoadData._get_adduct_count(neg_part))
     mol = positive_parts[0]
     positive_parts = sorted(positive_parts[1:], key=operator.itemgetter(1))
     negative_parts = sorted(negative_parts, key=operator.itemgetter(1))
     # Handle weird Cat = [M]+ notation.
     if mol[1].lower() == "Cat".lower():
         mol = mol[0], "M"
         charge, charge_sign = 1, "+"
     charges = {
    # Positive, singly charged.
    "H": 1,
    "K": 1,
    "Li": 1,
    "Na": 1,
    "NH4": 1,
    # Positive, doubly charged.
    "Ca": 2,
    "Fe": 2,
    "Mg": 2,
    # Negative, singly charged.
    "AC": -1,
    "Ac": -1,
    "Br": -1,
    "C2H3O2": -1,
    "C2H3OO": -1,
    "CH3COO": -1,
    "CHO2": -1,
    "CHOO": -1,
    "Cl": -1,
    "FA": -1,
    "Fa": -1,
    "Formate": -1,
    "formate": -1,
    "H3C2OO": -1,
    "HCO2": -1,
    "HCOO": -1,
    "I": -1,
    "OAc": -1,
    "Oac": -1,
    "OFA": -1,
    "OFa": -1,
    "Ofa": -1,
    "OH": -1,
    # Neutral.
    "ACN": 0,
    "AcN": 0,
    "EtOH": 0,
    "H2O": 0,
    "HFA": 0,
    "i": 0,
    "MeOH": 0,
    "TFA": 0,
    # Misceallaneous.
    "Cat": 1,
}
     # Calculate the charge from the individual components.
     if charge_sign is None:
         charge = sum(
             [
                 count * charges.get(adduct, 0)
                 for count, adduct in positive_parts
             ]
         )  + sum(
            [
                count * -abs(charges.get(adduct, 0))
                for count, adduct in negative_parts
            ]
        )
         charge_sign = "-" if charge < 0 else "+" if charge > 0 else ""

     cleaned_adduct = ["[", f"{mol[0] if mol[0] > 1 else ''}{mol[1]}"]
     if negative_parts:
         for count, adduct in negative_parts:
             cleaned_adduct.append(f"-{count if count > 1 else ''}{adduct}")
     if positive_parts:
         for count, adduct in positive_parts:
             cleaned_adduct.append(f"+{count if count > 1 else ''}{adduct}")
     cleaned_adduct.append("]")
     cleaned_adduct.append(
         f"{abs(charge) if abs(charge) > 1 else ''}{charge_sign}"
     )
     return "".join(cleaned_adduct)

    

    def get_spectra(source: Union[IO, str], scan_nrs: Sequence[int] = None, compute_classes=False, config=None)\
            -> Iterator[SpectrumExt]:
        """
        Get the MS/MS spectra from the given MGF file, optionally filtering by
        scan number.

        Parameters
        ----------
        source : Union[IO, str]
            The MGF source (file name or open file object) from which the spectra
            are read.
        scan_nrs : Sequence[int]
            Only read spectra with the given scan numbers. If `None`, no filtering
            on scan number is performed.

        Returns
        -------
        Iterator[SpectrumExt]
            An iterator over the requested spectra in the given file.
        """
        with mgf.MGF(source) as f_in:
            # Iterate over a subset of spectra filtered by scan number.
            if scan_nrs is not None:
                def spectrum_it():
                    for scan_nr, spectrum_dict in enumerate(f_in):
                        if scan_nr in scan_nrs:
                            yield spectrum_dict
            # Or iterate over all MS/MS spectra.
            else:
                def spectrum_it():
                    yield from f_in
            

            total_results=[]
            for spectrum in spectrum_it():
                try:
                    condition, res =LoadData.is_valid_spectrum(spectrum, config)
                    total_results.append(res)
                    if condition:
                        #yield spectrum['params']['name']
                        yield LoadData._parse_spectrum(spectrum, compute_classes=compute_classes)
                except ValueError as e:
                    pass
                    # logger.warning(f'Failed to read spectrum '
                    #                f'{spectrum["params"]["title"]}: %s', e)
            #print('summary')
            #for index  in range(0, len(res)):
            #    total_positives = sum([r[index] for r in total_results])
            #    #print(total_positives)
    def is_valid_spectrum(spectrum: SpectrumExt, config):

        cond_library = int(spectrum['params']["libraryquality"]) <= 3
        cond_charge=  int(spectrum['params']["charge"][0]) in config.CHARGES
        cond_pepmass = float(spectrum['params']["pepmass"][0]) > 0
        cond_mz_array = len(spectrum['m/z array']) >= config.MIN_N_PEAKS 
        cond_ion_mode =spectrum['params']["ionmode"] == "Positive" 
        cond_name = spectrum['params']["name"].rstrip().endswith(" M+H")
        cond_centroid = PreprocessingUtils.is_centroid(spectrum['intensity array'])
        cond_inchi_smiles= (
                     #spectrum['params']["inchi"] != "N/A" or
                     spectrum['params']["smiles"] != "N/A"
                )
        ##cond_name=True
        ##cond_name=True
        dict_results= {'cond_library':cond_library, 
                        'cond_charge':cond_charge, 
                        'cond_pepmass':cond_pepmass, 
                        'cond_mz_array':cond_mz_array, 
                        'cond_ion_mode':cond_ion_mode, 
                        'cond_name':cond_name, 
                        'cond_centroid':cond_centroid, 
                        'cond_inchi_smiles':cond_inchi_smiles}
        #return cond_ion_mode and cond_mz_array
   
        total_condition = cond_library and cond_charge and cond_pepmass and cond_mz_array and cond_ion_mode and cond_name and cond_centroid and cond_inchi_smiles
        return  total_condition, dict_results
                

    def _parse_spectrum(spectrum_dict: Dict, compute_classes=False) -> SpectrumExt:
        """
        Parse the Pyteomics spectrum dict.

        Parameters
        ----------
        spectrum_dict : Dict
            The Pyteomics spectrum dict to be parsed.

        Returns
        -------
        SprectumExt
            The parsed spectrum.
        """
        #identifier = spectrum_dict['params']['title']

        params = spectrum_dict["params"]
        library = spectrum_dict["params"]["organism"]
        inchi = spectrum_dict["params"]["inchi"]
        smiles = spectrum_dict["params"]["smiles"]
        ionmode = spectrum_dict["params"]["ionmode"]
        # calculate Murcko-Scaffold class
        bms=MurckoScaffold.get_bm_scaffold(smiles)

        # classes
        if compute_classes:
            clss = PreprocessingUtils.get_class(inchi, smiles)
            superclass= clss[0]
            classe = clss[1]
            subclass = clss[2]
        else:
            superclass=None
            classe= None
            subclass=None

        
        spec = SpectrumExt(
                    identifier=spectrum_dict["params"]["spectrumid"],
                    precursor_mz=float(spectrum_dict["params"]["pepmass"][0]),
                    # Re-assign charge 0 to 1.
                    precursor_charge=max(int(spectrum_dict["params"]["charge"][0]), 1),
                    mz=np.array(spectrum_dict["m/z array"]),
                    intensity=np.array(spectrum_dict["intensity array"]),
                    retention_time=np.nan,
                    params=params,
                    library=library,
                    inchi=inchi,
                    smiles=smiles, 
                    ionmode=ionmode,
                    bms=bms, 
                    superclass=superclass,
                    classe=classe,
                    subclass=subclass,
                )
        
        # postprocessing
        spec=spec.remove_precursor_peak(0.1, "Da")
        spec=spec.filter_intensity(0.01)
        
         
        return spec


    def get_all_spectrums_gnps(file, num_samples=10, compute_classes=False, use_tqdm=True, config=None):
        spectrums=[] #to save all the spectrums
        spectra = LoadData.get_spectra(file, compute_classes=compute_classes, config=config)

        if use_tqdm:
            iterator = tqdm(range(0, num_samples))
        else:
            iterator = range(0, num_samples)
        
        #preprocessor 
        pp = Preprocessor()

        for i in iterator:
            try:
                spectrum = next(spectra)
                spectrum = pp.preprocess_spectrum(spectrum)
                spectrums.append(spectrum)
            except: #in case it is not possible to get more samples
                print(f'We reached the end of the array at index {i}')
                break
            # go to next iteration
            
        return spectrums
    

    
    def get_all_spectrums_nist(file, num_samples=10, compute_classes=False, use_tqdm=True, config=None, initial_line_number=0):
        """
        Get the MS/MS spectra from the given MGF file, optionally filtering by
        scan number.

        Parameters
        ----------
        source : Union[IO, str]
            The MGF source (file name or open file object) from which the spectra
            are read.
        scan_nrs : Sequence[int]
            Only read spectra with the given scan numbers. If `None`, no filtering
            on scan number is performed.

        Returns
        -------
        Iterator[SpectrumExt]
            An iterator over the requested spectra in the given file.
        """
        nist_loader =NistLoader()
        spectrums, current_line_number = nist_loader.parse_file(file,  num_samples=num_samples, initial_line_number=initial_line_number)
        

        # check adducts
        #print([s['identifier'] for s in spectrums])

        spectrums = nist_loader.compute_all_smiles(spectrums, use_tqdm=use_tqdm)

        # processing
        all_spectrums=[]
        pp = Preprocessor()

        for spectrum in spectrums:
                condition, res =LoadData.is_valid_spectrum(spectrum, config=config)
                #print(res)
                if condition:
                    #yield spectrum['params']['name']
                    spec= LoadData._parse_spectrum(spectrum, compute_classes=compute_classes)
                    spec = pp.preprocess_spectrum(spec)
                    all_spectrums.append(spec)

        return all_spectrums, current_line_number


    def get_all_spectrums(file, num_samples=10, compute_classes=False, use_tqdm=True, use_nist=False, config=None):
     
        if use_nist:
            spectrums = LoadData.get_all_spectrums_nist(file=file, 
                                                        num_samples=num_samples, 
                                                        compute_classes=compute_classes, 
                                                        use_tqdm=use_tqdm, 
                                                        config=config)
        else:
            spectrums = LoadData.get_all_spectrums_gnps(file=file, 
                                                        num_samples=num_samples, 
                                                        compute_classes=compute_classes, 
                                                        use_tqdm=use_tqdm,
                                                        config=config)

        return spectrums
    