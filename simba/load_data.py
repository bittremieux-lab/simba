import logging
from typing import Dict, IO, Iterator, Sequence, Union

from pyteomics import mgf
import pyteomics
from spectrum_utils.spectrum import MsmsSpectrum
from simba.spectrum_ext import SpectrumExt
import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
import spectrum_utils as su
import numpy as np
from simba.config import Config
from simba.preprocessing_utils import PreprocessingUtils
from simba.murcko_scaffold import MurckoScaffold

from tqdm import tqdm

from simba.nist_loader import NistLoader
from simba.preprocessor import Preprocessor
from simba.utils import spectrum_hash
import pickle

class LoadData:

    def get_spectra(
        source: Union[IO, str],
        scan_nrs: Sequence[int] = None,
        compute_classes=False,
        config=None,
        use_gnps_format=True,
    ) -> Iterator[SpectrumExt]:
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

            total_results = []
            for spectrum in spectrum_it():
                # try:
                if use_gnps_format:
                    condition, res = LoadData.is_valid_spectrum_gnps(spectrum, config)
                else:  # janssen format
                    condition, res = LoadData.is_valid_spectrum_janssen(
                        spectrum, config
                    )
                total_results.append(res)
                if condition:
                    # yield spectrum['params']['name']
                    yield LoadData._parse_spectrum(
                        spectrum,
                        compute_classes=compute_classes,
                        use_gnps_format=use_gnps_format,
                    )
            # except ValueError as e:
            #    pass

    @staticmethod
    def is_valid_spectrum_janssen(spectrum: SpectrumExt, config):

        cond_library = True  # all the library is good
        if "charge" in spectrum["params"]:
            cond_charge = int(spectrum["params"]["charge"][0]) in config.CHARGES
        else:
            cond_charge=True 

        #try:
        #    cond_pepmass = float(spectrum["params"]["pepmass"][0]) > 0
        #except:
        #    cond_pepmass = float(spectrum["params"]["pepmass"]) > 0
        cond_pepmass = True

        cond_mz_array = len(spectrum["m/z array"]) >= config.MIN_N_PEAKS

        if 'ionmode' in spectrum["params"]:
            cond_ion_mode = spectrum["params"]["ionmode"] == "Positive"
        else:
            cond_ion_mode=True
        cond_name = spectrum["params"]["adduct"] in ["M+", "[M+H]+", "M+H"]  # adduct
        cond_centroid = PreprocessingUtils.is_centroid(spectrum["intensity array"])
        cond_inchi_smiles = (
            # spectrum['params']["inchi"] != "N/A" or
            spectrum["params"]["smiles"]
            != "N/A"
        )
        ##cond_name=True
        ##cond_name=True
        dict_results = {
            "cond_library": cond_library,
            "cond_charge": cond_charge,
            "cond_pepmass": cond_pepmass,
            "cond_mz_array": cond_mz_array,
            "cond_ion_mode": cond_ion_mode,
            "cond_name": cond_name,
            "cond_centroid": cond_centroid,
            "cond_inchi_smiles": cond_inchi_smiles,
        }

        # return cond_ion_mode and cond_mz_array

        total_condition = (
            cond_library
            and cond_charge
            and cond_pepmass
            and cond_mz_array
            and cond_ion_mode
            and cond_name
            and cond_centroid
            and cond_inchi_smiles
        )

        return total_condition, dict_results

    @staticmethod
    def is_valid_spectrum_gnps(spectrum: SpectrumExt, config):
        if "libraryquality" in spectrum["params"].keys():
            cond_library = int(spectrum["params"]["libraryquality"]) <= 3
        else:
            cond_library= True
        cond_charge = int(spectrum["params"]["charge"][0]) in config.CHARGES

        # try to convert to float the pep mass
        try:
            cond_pepmass = float(spectrum["params"]["pepmass"][0]) > 0
        except:
            cond_pepmass = 0.0

        cond_mz_array = len(spectrum["m/z array"]) >= config.MIN_N_PEAKS
        cond_ion_mode = spectrum["params"]["ionmode"] == "Positive"
        cond_name = spectrum["params"]["name"].rstrip().endswith(" M+H")
        cond_centroid = PreprocessingUtils.is_centroid(spectrum["intensity array"])
        cond_inchi_smiles = (
            # spectrum['params']["inchi"] != "N/A" or
            (spectrum["params"]["smiles"]
            != "N/A") & (spectrum["params"]["smiles"] != '')
        )
        ##cond_name=True
        ##cond_name=True
        dict_results = {
            "cond_library": cond_library,
            "cond_charge": cond_charge,
            "cond_pepmass": cond_pepmass,
            "cond_mz_array": cond_mz_array,
            "cond_ion_mode": cond_ion_mode,
            "cond_name": cond_name,
            "cond_centroid": cond_centroid,
            "cond_inchi_smiles": cond_inchi_smiles,
        }
        # return cond_ion_mode and cond_mz_array

        total_condition = (
            cond_library
            and cond_charge
            and cond_pepmass
            and cond_mz_array
            and cond_ion_mode
            and cond_name
            and cond_centroid
            and cond_inchi_smiles
        )
        return total_condition, dict_results

    def _parse_spectrum(
        spectrum_dict: Dict, compute_classes=False, use_gnps_format=True
    ) -> SpectrumExt:
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
        # identifier = spectrum_dict['params']['title']
        if use_gnps_format:  # GNPS
            identifier = spectrum_dict["params"]["spectrumid"]
            inchi = spectrum_dict["params"]["inchi"]
            library = spectrum_dict["params"]["organism"]
        else:  # JANSSEN
            if 'id' in spectrum_dict["params"]:
                identifier = spectrum_dict["params"]["id"]
            else:
                identifier='none'
            inchi = ""
            library = "janssen"

        params = spectrum_dict["params"]
        library = library
        inchi = inchi
        smiles = spectrum_dict["params"]["smiles"]
        if 'ionmode' in spectrum_dict["params"]:
            ionmode = spectrum_dict["params"]["ionmode"]
        else:
            ionmode='none'

        # compute hash id value
        spectrum_hash_result = spectrum_hash(
            spectrum_dict["m/z array"], spectrum_dict["intensity array"]
        )

        # calculate Murcko-Scaffold class
        bms = MurckoScaffold.get_bm_scaffold(smiles)

        # classes
        if compute_classes:
            clss = PreprocessingUtils.get_class(inchi, smiles)
            superclass = clss[0]
            classe = clss[1]
            subclass = clss[2]
        else:
            superclass = None
            classe = None
            subclass = None

        try:
            precursor_mz= float(spectrum_dict["params"]["pepmass"][0]) 
        except:
            precursor_mz= float(spectrum_dict["params"]['precursor_mz']) 

        try:
            charge=max(int(spectrum_dict["params"]["charge"][0]), 1)
        except:
            charge = 1
        spec = SpectrumExt(
            identifier=identifier,
            precursor_mz=precursor_mz,
            # Re-assign charge 0 to 1.
            precursor_charge=charge,
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
            spectrum_hash=spectrum_hash_result,
        )

        # postprocessing
        # spec=spec.remove_precursor_peak(0.1, "Da")
        # spec=spec.filter_intensity(0.01)

        return spec

    def get_all_spectrums_mgf(
        file,
        num_samples=-1,
        compute_classes=False,
        use_tqdm=True,
        config=None,
        use_gnps_format=True,
    ):  
        
        num_samples=10**8 if num_samples == -1 else num_samples
        spectrums = []  # to save all the spectrums
        spectra = LoadData.get_spectra(
            file,
            compute_classes=compute_classes,
            config=config,
            use_gnps_format=use_gnps_format,
        )

        if use_tqdm:
            iterator = tqdm(range(0, num_samples))
        else:
            iterator = range(0, num_samples)

        # preprocessor
        pp = Preprocessor()

        for i in iterator:
            try:
                spectrum = next(spectra)
                # spectrum = pp.preprocess_spectrum(spectrum)
                spectrums.append(spectrum)
            except StopIteration:  # in case it is not possible to get more samples
                print(f"We reached the end of the array at index {i}")
                break
            # go to next iteration

        return spectrums

    def get_all_spectrums_nist(
        file,
        num_samples=10,
        compute_classes=False,
        use_tqdm=True,
        config=None,
        initial_line_number=0,
    ):
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
        nist_loader = NistLoader()
        spectrums, current_line_number = nist_loader.parse_file(
            file, num_samples=num_samples, initial_line_number=initial_line_number
        )

        # check adducts
        # print([s['identifier'] for s in spectrums])

        spectrums = nist_loader.compute_all_smiles(spectrums, use_tqdm=use_tqdm)

        # processing
        all_spectrums = []
        pp = Preprocessor()

        for spectrum in spectrums:
            # use the validation from gnps format since it is the format we are parsing
            condition, res = LoadData.is_valid_spectrum_gnps(spectrum, config=config)
            # print(res)
            if condition:
                # yield spectrum['params']['name']
                spec = LoadData._parse_spectrum(
                    spectrum, compute_classes=compute_classes
                )
                # spec = pp.preprocess_spectrum(spec)
                all_spectrums.append(spec)

        return all_spectrums, current_line_number

    def get_all_spectrums_casmi(
        file,
        num_samples=10,
        compute_classes=False,
        use_tqdm=True,
        config=None,
        initial_line_number=0,
    ):
        # open casmi file 
        with open(file, 'rb') as f:
            spectra_df = pickle.load(f)
        all_spectrums_parsed = []

        for index,spectra_row in spectra_df.iterrows():

            #initialize
            spectrum_dict={}
            spectrum_dict['params'] = {}

            # get info
            adduct = ' M+H' if spectra_row['prec_type'] =='[M+H]+'else spectra_row['prec_type']
            spectrum_dict['params']['spectrumid'] = str(spectra_row['casmi_id']) + adduct
            spectrum_dict['params']['name'] = str(spectra_row['casmi_id']) + adduct
            spectrum_dict['params']['inchi']=''
            spectrum_dict['params']['organism']='casmi'
            spectrum_dict['params']['id']=spectra_row['casmi_id']
            spectrum_dict['params']['smiles'] = spectra_row['smiles']
            ionmode = 'Positive' if spectra_row['ion_mode']=='P' else 'Negative'
            spectrum_dict['params']['ionmode']=ionmode
            spectrum_dict['params']['pepmass']=[spectra_row['prec_mz']]
            spectrum_dict['params']['charge']=[1]
            spectrum_dict['params']['libraryquality']=1
            #get peaks
            peaks = spectra_row['peaks']
            mz = np.array([p[0] for p in peaks])
            intensity = np.array([p[1] for p in peaks])

            spectrum_dict['m/z array']= mz
            spectrum_dict['intensity array']= intensity 


            all_spectrums_parsed.append(spectrum_dict)

        # processing
        all_spectrums = []
        pp = Preprocessor()

        for spectrum in all_spectrums_parsed:
            # use the validation from gnps format since it is the format we are parsing
            condition, res = LoadData.is_valid_spectrum_gnps(spectrum, config=config)
            # print(res)
            if condition:
                # yield spectrum['params']['name']
                spec = LoadData._parse_spectrum(
                    spectrum, compute_classes=compute_classes
                )
                # spec = pp.preprocess_spectrum(spec)
                all_spectrums.append(spec)

        return all_spectrums

    def get_all_spectrums(
        file,
        num_samples=10,
        compute_classes=False,
        use_tqdm=True,
        use_nist=False,
        config=None,
        use_janssen=False,
    ):

        if use_janssen:
            spectrums = LoadData.get_all_spectrums_mgf(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                config=config,
                use_gnps_format=False,
            )  # use format from Janssen
        elif use_nist:
            spectrums = LoadData.get_all_spectrums_nist(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                config=config,
            )
        else:
            spectrums = LoadData.get_all_spectrums_mgf(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                config=config,
                use_gnps_format=True,
            )

        return spectrums
