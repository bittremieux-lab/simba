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

from tqdm import tqdm 

class LoadData:


    def get_spectra(source: Union[IO, str], scan_nrs: Sequence[int] = None, compute_classes=False)\
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

            for spectrum in spectrum_it():
                try:
                    
                    if LoadData.is_valid_spectrum(spectrum):
                        yield LoadData._parse_spectrum(spectrum, compute_classes=compute_classes)
                
                except ValueError as e:
                    pass
                    # logger.warning(f'Failed to read spectrum '
                    #                f'{spectrum["params"]["title"]}: %s', e)

    def is_valid_spectrum(spectrum: SpectrumExt):

        cond_library = int(spectrum['params']["libraryquality"]) <= 3
        cond_charge=  int(spectrum['params']["charge"][0]) in Config.CHARGES
        cond_pepmass = float(spectrum['params']["pepmass"][0]) > 0
        cond_mz_array = len(spectrum['m/z array']) >= Config.MIN_N_PEAKS 
        cond_ion_mode =spectrum['params']["ionmode"] == "Positive" 
        cond_name = spectrum['params']["name"].rstrip().endswith(" M+H")
        cond_centroid = PreprocessingUtils.is_centroid(spectrum['intensity array'])
        cond_inchi_smiles= (
                     #spectrum['params']["inchi"] != "N/A" or
                     spectrum['params']["smiles"] != "N/A"
                )
       

        return  cond_library and cond_charge and cond_pepmass and cond_mz_array and cond_ion_mode and cond_name and cond_centroid and cond_inchi_smiles
                

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
                    mz=spectrum_dict["m/z array"],
                    intensity=spectrum_dict["intensity array"],
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

    def get_all_spectrums(mgf_path, num_samples=10, compute_classes=False, use_tqdm=True):
        spectrums=[] #to save all the spectrums
        spectra = LoadData.get_spectra(mgf_path, compute_classes=compute_classes)

        if use_tqdm:
            iterator = tqdm(range(0, num_samples))
        else:
            iterator = range(0, num_samples)
        
        for i in iterator:
            try:
                spectrum = next(spectra)
                spectrums.append(spectrum)
            except: #in case it is not possible to get more samples
                print(f'We reached the end of the array at index {i}')
                break
            # go to next iteration
            
        return spectrums
    

