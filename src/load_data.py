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
from tqdm import tqdm 

class LoadData:


    def get_spectra(source: Union[IO, str], scan_nrs: Sequence[int] = None)\
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
                    yield LoadData._parse_spectrum(spectrum)
                except ValueError as e:
                    pass
                    # logger.warning(f'Failed to read spectrum '
                    #                f'{spectrum["params"]["title"]}: %s', e)


    def _parse_spectrum(spectrum_dict: Dict) -> SpectrumExt:
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

        
        identifier = spectrum_dict['params']['name']
        mz_array = spectrum_dict['m/z array']
        intensity_array = spectrum_dict['intensity array']
        #retention_time = float(spectrum_dict['params']['rtinseconds'])
        retention_time = 0
        precursor_mz = float(spectrum_dict['params']['pepmass'][0])
        if 'charge' in spectrum_dict['params']:
            precursor_charge = int(spectrum_dict['params']['charge'][0])
        else:
            raise ValueError('Unknown precursor charge')

        spectrum = SpectrumExt(str(identifier), precursor_mz, precursor_charge,
                                mz_array, intensity_array, None)

        # add metadata to spectrum object

        spectrum.set_params(spectrum_dict['params'])

        spectrum.set_smiles(spectrum_dict['params']['smiles'])

        # add max intensity
        spectrum.set_max_peak(np.max(spectrum_dict['intensity array']))
        
        return spectrum

    def get_all_spectrums(mgf_path, num_samples=10):
        spectrums=[] #to save all the spectrums
        spectra = LoadData.get_spectra(mgf_path)

        for i in tqdm(range(0, num_samples)):
            try:
                spectrum = next(spectra)
                spectrums.append(spectrum)
            except: #in case it is not possible to get more samples
                print(f'We reached the end of the array at index {i}')
                break
            # go to next iteration
            
        return spectrums
    

