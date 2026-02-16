import pickle
from collections.abc import Iterator, Sequence
from typing import IO

import numpy as np
from pyteomics import mgf
from tqdm import tqdm

from simba.core.chemistry import chem_utils
from simba.core.chemistry.scaffolds import MurckoScaffold
from simba.core.data.nist_loader import NistLoader
from simba.core.data.preprocessing import PreprocessingUtils
from simba.core.data.spectrum import SpectrumExt
from simba.core.data.spectrum_processing import spectrum_hash
from simba.utils.logger_setup import logger


class LoadData:
    def get_spectra(
        source: IO | str,
        scan_nrs: Sequence[int] = None,
        compute_classes=False,
        cfg=None,
        use_gnps_format=True,
        use_only_protonized_adducts=True,
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
        compute_classes : bool
            Whether to compute chemical superclass, class and subclass of the molecules
            using Classyfire.
        cfg : DictConfig
            Hydra configuration object containing preprocessing parameters.
        use_gnps_format : bool
            Whether the MGF file follows the GNPS format. If `False`, it is assumed
            to follow the Janssen format.

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
                            yield scan_nr, spectrum_dict

            # Or iterate over all MS/MS spectra.
            else:

                def spectrum_it():
                    for scan_nr, spectrum_dict in enumerate(f_in):
                        yield scan_nr, spectrum_dict

            total_results = []
            for mgf_index, spectrum in spectrum_it():
                # try:
                if use_only_protonized_adducts:
                    if use_gnps_format:
                        condition, res = LoadData.is_valid_spectrum_gnps(
                            spectrum, cfg
                        )
                    else:  # janssen format
                        condition, res = LoadData.is_valid_spectrum_janssen(
                            spectrum, cfg
                        )
                else:
                    condition, res = LoadData.default_filters(spectrum, cfg)

                total_results.append(res)
                if condition:
                    # yield spectrum['params']['name']
                    spec = LoadData._parse_spectrum(
                        spectrum,
                        compute_classes=compute_classes,
                        use_gnps_format=use_gnps_format,
                    )
                    if spec and (
                        np.isnan(spec.mz).any()
                        or np.isnan(spec.intensity).any()
                        or not np.isfinite(spec.mz).all()
                        or not np.isfinite(spec.intensity).all()
                        or not np.all(spec.mz >= 0)
                        or not np.all(spec.intensity >= 0)
                    ):
                        spec = None
                    if spec is not None:
                        # Assign the original MGF index before filtering
                        spec.mgf_index = mgf_index
                        yield spec
            # except ValueError as e:
            #    pass

    @staticmethod
    def get_precursor_mz(spectrum):
        if "pepmass" in spectrum["params"]:
            if len(spectrum["params"]["pepmass"]) > 0:
                precursor_mz = float(spectrum["params"]["pepmass"][0])
            else:
                precursor_mz = float(spectrum["params"]["pepmass"])
        elif "precursor_mz" in spectrum["params"]:
            precursor_mz = float(spectrum["params"]["precursor_mz"])
        else:
            precursor_mz = None
        return precursor_mz

    @staticmethod
    def default_filters(spectrum: SpectrumExt, cfg):
        """Apply default filters to spectrum.

        Args:
            spectrum: Spectrum to filter
            cfg: DictConfig (Hydra configuration object)
        """
        cond_library = True  # all the library is good
        cond_charge = True
        precursor_mz = LoadData.get_precursor_mz(spectrum)
        cond_pepmass = precursor_mz is not None and precursor_mz > 0
        cond_mz_array = len(spectrum["m/z array"]) >= cfg.data.preprocessing.min_n_peaks
        cond_ion_mode = True
        cond_name = True
        cond_inchi_smiles = True
        cond_centroid = PreprocessingUtils.is_centroid(spectrum["intensity array"])

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
    def is_valid_spectrum_janssen(spectrum: SpectrumExt, cfg):
        """Validate Janssen format spectrum.

        Args:
            spectrum: Spectrum to validate
            cfg: DictConfig (Hydra configuration object)
        """
        cond_library = True  # all the library is good
        if "charge" in spectrum["params"]:
            cond_charge = int(spectrum["params"]["charge"][0]) in cfg.data.ms_parameters.charges
        else:
            cond_charge = True

        # try:
        #    cond_pepmass = float(spectrum["params"]["pepmass"][0]) > 0
        # except:
        #    cond_pepmass = float(spectrum["params"]["pepmass"]) > 0
        cond_pepmass = True

        cond_mz_array = len(spectrum["m/z array"]) >= cfg.data.preprocessing.min_n_peaks

        if "ionmode" in spectrum["params"]:
            cond_ion_mode = spectrum["params"]["ionmode"].lower() == "positive"
        else:
            cond_ion_mode = True

        if "adduct" in spectrum["params"]:
            cond_name = spectrum["params"]["adduct"] in [
                "M+",
                "[M+H]+",
                "M+H",
            ]
        else:
            logger.warning(
                "Adduct information not found in spectrum. Please make sure the spectra corresponds to protonized adducts [M+H]"
            )
            cond_name = True

        if "smiles" in spectrum["params"]:
            cond_inchi_smiles = (
                # spectrum['params']["inchi"] != "N/A" or
                spectrum["params"]["smiles"] != "N/A"
            )
        else:
            logger.warning("Smiles not found in spectrum.")
            cond_inchi_smiles = True

        cond_centroid = PreprocessingUtils.is_centroid(spectrum["intensity array"])

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
    def is_valid_spectrum_gnps(spectrum: SpectrumExt, cfg):
        """Validate GNPS format spectrum.

        Args:
            spectrum: Spectrum to validate
            cfg: DictConfig (Hydra configuration object)
        """
        if "libraryquality" in spectrum["params"].keys():
            cond_library = int(spectrum["params"]["libraryquality"]) <= 3
        else:
            cond_library = True

        if "charge" in spectrum["params"]:
            cond_charge = int(spectrum["params"]["charge"][0]) in cfg.data.ms_parameters.charges
        else:
            cond_charge = True
        # try to convert to float the pep mass
        try:
            cond_pepmass = float(spectrum["params"]["pepmass"][0]) > 0
        except:
            cond_pepmass = False

        cond_mz_array = len(spectrum["m/z array"]) >= cfg.data.preprocessing.min_n_peaks

        if "ionmode" in spectrum["params"]:
            cond_ion_mode = spectrum["params"]["ionmode"] == "Positive"
        else:
            cond_ion_mode = True
        cond_name = spectrum["params"]["name"].rstrip().endswith(" M+H")
        cond_centroid = PreprocessingUtils.is_centroid(spectrum["intensity array"])
        cond_inchi_smiles = (
            # spectrum['params']["inchi"] != "N/A" or
            (spectrum["params"]["smiles"] != "N/A")
            & (spectrum["params"]["smiles"] != "")
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
        spectrum_dict: dict,
        compute_classes: bool = False,
        use_gnps_format: bool = True,
    ) -> SpectrumExt:
        """
        Parse the Pyteomics spectrum dict.

        Parameters
        ----------
        spectrum_dict : Dict
            The Pyteomics spectrum dict to be parsed.
        compute_classes : bool
            Whether to compute chemical superclass, class and subclass of the molecules
            using Classyfire.
        use_gnps_format : bool
            Whether the MGF file follows the GNPS format. If `False`, it is assumed
            to follow the Janssen format.

        Returns
        -------
        SpectrumExt
            The parsed spectrum.
        """
        # identifier = spectrum_dict['params']['title']
        if use_gnps_format:  # GNPS
            identifier = spectrum_dict["params"]["spectrumid"]
            inchi = spectrum_dict["params"]["inchi"]
            library = spectrum_dict["params"]["organism"]
        else:  # JANSSEN
            if "id" in spectrum_dict["params"]:
                identifier = spectrum_dict["params"]["id"]
            elif "title" in spectrum_dict["params"]:
                identifier = spectrum_dict["params"]["title"]
            else:
                identifier = "none"
            inchi = ""
            library = "janssen"

        params = spectrum_dict["params"]

        library = library
        inchi = inchi
        smiles = params["smiles"] if "smiles" in params else ""

        precursor_mz = LoadData.get_precursor_mz(spectrum_dict)
        if precursor_mz is None:
            return None

        if "charge" in params:
            charge = int(params["charge"][0])
        else:
            charge = 1  # Default charge for positive ions

        ionmode = params["ionmode"].lower() if "ionmode" in params else "none"

        # Extract adduct as string (no mass conversion)
        if "adduct" in params:
            adduct = params["adduct"].replace(" ", "")
        else:
            adduct = None

        ce = params["ce"] if "ce" in params else None
        ia = params["ion_activation"] if "ion_activation" in params else None
        im = params["ionization_method"] if "ionization_method" in params else None

        inchi_key = params["inchikey"] if "inchikey" in params else None

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
            adduct=adduct,
            ce=ce,
            ion_activation=ia,
            ionization_method=im,
            bms=bms,
            superclass=superclass,
            classe=classe,
            subclass=subclass,
            inchi_key=inchi_key,
            spectrum_hash=spectrum_hash_result,
        )

        # postprocessing
        # spec=spec.remove_precursor_peak(0.1, "Da")
        # spec=spec.filter_intensity(0.01)

        return spec

    def get_all_spectra_mgf(
        file: IO | str,
        num_samples: int = -1,
        compute_classes: bool = False,
        use_tqdm: bool = True,
        cfg=None,
        use_gnps_format: bool = True,
        use_only_protonized_adducts=True,
    ) -> list[SpectrumExt]:
        """
        Get the MS/MS spectra from the given MGF file, optionally filtering by
        scan number.

        Parameters
        ----------
        file : Union[IO, str]
            The MGF file (file name or open file object) from which the spectra
            are read.
        num_samples : int
            The maximum number of spectra to read. If -1, all spectra are read.
        compute_classes : bool
            Whether to compute chemical superclass, class and subclass of the molecules
            using Classyfire.
        use_tqdm : bool
            Whether to display a progress bar using tqdm.
        cfg : DictConfig
            Hydra configuration object containing preprocessing parameters.
        use_gnps_format : bool
            Whether the MGF file follows the GNPS format. If `False`, it is assumed
            to follow the Janssen format.
        use_only_protonized_adducts : bool
            Whether to filter spectra to only include those with protonated adducts
            ([M+H]+).

        Returns
        -------
        List[SpectrumExt]
            A list of the parsed spectra.
        """
        num_samples = 10**8 if num_samples == -1 else num_samples
        spectra = []  # to save all the spectra
        spectra_to_process = LoadData.get_spectra(
            file,
            compute_classes=compute_classes,
            cfg=cfg,
            use_gnps_format=use_gnps_format,
            use_only_protonized_adducts=use_only_protonized_adducts,
        )

        if use_tqdm:
            iterator = tqdm(range(0, num_samples))
        else:
            iterator = range(0, num_samples)

        for i in iterator:
            try:
                spectrum = next(spectra_to_process)
                # spectrum = pp.preprocess_spectrum(spectrum)
                spectra.append(spectrum)
            except StopIteration:  # in case it is not possible to get more samples
                logger.info(f"Only {i} spectra found.")
                break
            # go to next iteration

        return spectra

    def get_all_spectra_nist(
        file,
        num_samples=10,
        compute_classes=False,
        use_tqdm=True,
        cfg=None,
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
        spectra, current_line_number = nist_loader.parse_file(
            file,
            num_samples=num_samples,
            initial_line_number=initial_line_number,
        )

        # check adducts
        # print([s['identifier'] for s in spectrums])

        spectra = nist_loader.compute_all_smiles(spectra, use_tqdm=use_tqdm)

        # processing
        all_spectra = []

        for spectrum in spectra:
            # use the validation from gnps format since it is the format we are parsing
            condition, res = LoadData.is_valid_spectrum_gnps(spectrum, cfg=cfg)
            # print(res)
            if condition:
                # yield spectrum['params']['name']
                spec = LoadData._parse_spectrum(
                    spectrum, compute_classes=compute_classes
                )
                # spec = pp.preprocess_spectrum(spec)
                if spec is not None:
                    all_spectra.append(spec)

        return all_spectra, current_line_number

    def get_all_spectra_casmi(
        file,
        num_samples=10,
        compute_classes=False,
        use_tqdm=True,
        cfg=None,
        initial_line_number=0,
    ):
        # open casmi file
        with open(file, "rb") as f:
            spectra_df = pickle.load(f)
        all_spectra_parsed = []

        for index, spectra_row in spectra_df.iterrows():
            # initialize
            spectrum_dict = {}
            spectrum_dict["params"] = {}

            # get info
            adduct = (
                " M+H"
                if spectra_row["prec_type"] == "[M+H]+"
                else spectra_row["prec_type"]
            )
            spectrum_dict["params"]["spectrumid"] = (
                str(spectra_row["casmi_id"]) + adduct
            )
            spectrum_dict["params"]["name"] = str(spectra_row["casmi_id"]) + adduct
            spectrum_dict["params"]["inchi"] = ""
            spectrum_dict["params"]["organism"] = "casmi"
            spectrum_dict["params"]["id"] = spectra_row["casmi_id"]
            spectrum_dict["params"]["smiles"] = spectra_row["smiles"]
            ionmode = "Positive" if spectra_row["ion_mode"] == "P" else "Negative"
            spectrum_dict["params"]["ionmode"] = ionmode
            spectrum_dict["params"]["pepmass"] = [spectra_row["prec_mz"]]
            spectrum_dict["params"]["charge"] = [1]
            spectrum_dict["params"]["libraryquality"] = 1
            # get peaks
            peaks = spectra_row["peaks"]
            mz = np.array([p[0] for p in peaks])
            intensity = np.array([p[1] for p in peaks])

            spectrum_dict["m/z array"] = mz
            spectrum_dict["intensity array"] = intensity

            all_spectra_parsed.append(spectrum_dict)

        # processing
        all_spectra = []

        for spectrum in all_spectra_parsed:
            # use the validation from gnps format since it is the format we are parsing
            condition, res = LoadData.is_valid_spectrum_gnps(spectrum, cfg)
            # print(res)
            if condition:
                # yield spectrum['params']['name']
                spec = LoadData._parse_spectrum(
                    spectrum, compute_classes=compute_classes
                )
                # spec = pp.preprocess_spectrum(spec)
                if spec is not None:
                    all_spectra.append(spec)

        return all_spectra

    def get_all_spectra(
        file: IO | str,
        num_samples: int = 10,
        compute_classes: bool = False,
        use_tqdm: bool = True,
        use_nist: bool = False,
        cfg=None,
        use_janssen: bool = False,
    ) -> list[SpectrumExt]:
        """
        Get the MS/MS spectra from the given MGF or NIST file, optionally filtering by
        scan number.

        Parameters
        ----------
        file : Union[IO, str]
            The MGF or NIST file (file name or open file object) from which the spectra
            are read.
        num_samples : int
            The maximum number of spectra to read. If -1, all spectra are read.
        compute_classes : bool
            Whether to compute chemical superclass, class and subclass of the molecules
            using Classyfire.
        use_tqdm : bool
            Whether to display a progress bar using tqdm.
        use_nist : bool
            Whether the file is a NIST file. If `False`, it is assumed to be an MGF file.
        cfg : DictConfig
            Hydra configuration object containing preprocessing parameters.
        use_janssen : bool
            Whether the MGF file follows the Janssen format. If `False`, it is assumed
            to follow the GNPS format.

        Returns
        -------
        List[SpectrumExt]
            A list of the parsed spectra.
        """

        if use_janssen:
            spectra = LoadData.get_all_spectra_mgf(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                cfg=cfg,
                use_gnps_format=False,
            )  # use format from Janssen
        elif use_nist:
            spectra, _ = LoadData.get_all_spectra_nist(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                cfg=cfg,
            )
        else:
            spectra = LoadData.get_all_spectra_mgf(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                cfg=cfg,
                use_gnps_format=True,
            )

        return spectra
