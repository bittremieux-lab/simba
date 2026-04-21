import copy
import functools
import json
from itertools import groupby

import numpy as np
import pandas as pd
import requests
from rdkit import Chem

from simba.core.data.spectrum import SpectrumExt


class PreprocessingUtils:
    @staticmethod
    def is_centroid(intensity):
        return np.all(intensity > 0)

    @staticmethod
    def order_by_charge(
        spectra: list[SpectrumExt],
    ) -> dict[int, list[SpectrumExt]]:
        """
        Order spectra by their precursor charge.

        Parameters
        ----------
        spectra : List[SpectrumExt]
            List of SpectrumExt objects to be ordered.

        Returns
        -------
        Dict[int, List[SpectrumExt]]
            A dictionary where keys are precursor charges and values are lists of SpectrumExt objects with that charge.
        """
        spectra_new = copy.deepcopy(spectra)
        # Sort the list based on the precursor_charge
        spectra_new.sort(key=lambda a: a.precursor_charge)
        # Group the elements based on the precursor_charge
        spectra_by_charge = {}
        for key, group in groupby(spectra_new, key=lambda a: a.precursor_charge):
            spectra_by_charge[key] = list(group)
        return spectra_by_charge

    @staticmethod
    def order_spectra_by_mz(spectra: list[SpectrumExt]) -> list[SpectrumExt]:
        """
        Order spectra by their precursor m/z.

        Parameters
        ----------
        spectra : List[SpectrumExt]
            List of SpectrumExt objects to be ordered.

        Returns
        -------
        List[SpectrumExt]
            A list of SpectrumExt objects ordered by their precursor m/z.
        """
        spectra_by_charge = PreprocessingUtils.order_by_charge(spectra)

        all_spectra = []
        for charge, spectra_group in spectra_by_charge.items():
            # order by mz
            mzs = np.array([s.precursor_mz for s in spectra_group])
            ordered_indexes = np.argsort(mzs)
            sorted_spectra = [spectra_group[r] for r in ordered_indexes]
            all_spectra = all_spectra + sorted_spectra

        return all_spectra

    def _smiles_to_mol(smiles):
        try:
            return Chem.MolFromSmiles(smiles)
        except ArgumentError:
            return None

    @functools.lru_cache
    def get_class(inchi: str, smiles: str) -> tuple[str | None, str | None, str | None]:
        """
        Get the superclass, class and subclass of a molecule using Classyfire.
        Either InChI or SMILES can be used as input.
        Parameters
        ----------
        inchi : str
            The InChI string of the molecule.
        smiles : str
            The SMILES string of the molecule.
        Returns
        -------
        tuple
            A tuple (superclass, class, subclass) if successful,
            (None, None, None) otherwise.
        """
        clss = (
            PreprocessingUtils._get_class("inchi", inchi)
            if inchi is not None and inchi != "N/A"
            else None
        )
        if clss is None and not pd.isna(smiles) and smiles != "N/A":
            mol = PreprocessingUtils._smiles_to_mol(smiles)
            clss = (
                PreprocessingUtils._get_class(
                    "smiles", Chem.MolToSmiles(mol, isomericSmiles=False)
                )
                if mol is not None
                else None
            )
        return clss if clss is not None else (None, None, None)

    @functools.lru_cache
    def _get_class(mol_type: str, mol_val: str) -> tuple[str, str, str] | None:
        """
        Get the superclass, class and subclass of a molecule using Classyfire.
        Either InChI or SMILES can be used as input.
        Parameters
        ----------
        mol_type : str
            Either "inchi" or "smiles".
        mol_val : str
            The InChI or SMILES string of the molecule.
        Returns
        -------
        tuple or None
            A tuple (superclass, class, subclass) if successful, None otherwise.
        """
        r = requests.get(
            f"https://gnps-structure.ucsd.edu/classyfire?{mol_type}={mol_val}"
        )
        if r.status_code != 200:
            return None
        try:
            classyfire_json = r.json()
            if not classyfire_json:
                return None
            if (
                "superclass" not in classyfire_json
                or "class" not in classyfire_json
                or "subclass" not in classyfire_json
            ):
                return None
            superclass = classyfire_json["superclass"]
            if superclass is not None:
                superclass = superclass["name"]
            clss = classyfire_json["class"]
            if clss is not None:
                clss = clss["name"]
            subclass = classyfire_json["subclass"]
            if subclass is not None:
                subclass = subclass["name"]
            return superclass, clss, subclass
        except json.decoder.JSONDecodeError:
            return None


import copy
import random

import numpy as np
from tqdm import tqdm

from simba.core.data.spectrum import SpectrumExt


class Preprocessor:
    def __init__(self, bin_width=1, min_mz=10, max_mz=1400):
        # Define the parameters for binning
        self.bin_width = bin_width  # Adjust as needed to control the bin size
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.num_bins = int((max_mz - min_mz) / bin_width) + 1

    def preprocess_all_spectra(
        self,
        spectrums,
        fragment_tol_mass=10,
        fragment_tol_mode="ppm",
        min_intensity=0.01,
        max_num_peaks=100,
        # max_num_peaks=40,
        scale_intensity=None,
        # scale_intensity="root",
        training=False,
        random_seed=42,
    ):
        random.seed(random_seed)
        for i, spectrum in tqdm(enumerate(spectrums)):
            # try:
            if training:
                min_intensity = 0.00
            else:
                min_intensity = 0.01

            spectrums[i] = self.preprocess_spectrum(
                spectrum,
                fragment_tol_mass=fragment_tol_mass,
                fragment_tol_mode=fragment_tol_mode,
                min_intensity=min_intensity,
                max_num_peaks=max_num_peaks,
                scale_intensity=scale_intensity,
            )
        # except:
        #    print('Error preprocessing spectrum')

        # preprocess np vectors
        # all_spectrums= self.process_all_spectrum_vectors(spectrums)
        return spectrums

    def preprocess_all_spectra_variable_max_peaks(
        self,
        spectrums,
        fragment_tol_mass=10,
        fragment_tol_mode="ppm",
        min_intensity=0.01,
        max_num_peaks=100,
        # max_num_peaks=40,
        scale_intensity=None,
        # scale_intensity="root",
        training=False,
        random_seed=42,
    ):
        random.seed(random_seed)
        for i, spectrum in tqdm(enumerate(spectrums)):
            # try:
            if training:
                min_intensity = 0.00
            else:
                min_intensity = 0.01

            spectrums[i] = self.preprocess_spectrum(
                spectrum,
                fragment_tol_mass=fragment_tol_mass,
                fragment_tol_mode=fragment_tol_mode,
                min_intensity=min_intensity,
                max_num_peaks=max_num_peaks,
                scale_intensity=scale_intensity,
            )

            length_mz = len(spectrums[i].mz)
            if length_mz > 20:
                spectrums[i] = self.preprocess_spectrum(
                    spectrum,
                    fragment_tol_mass=fragment_tol_mass,
                    fragment_tol_mode=fragment_tol_mode,
                    min_intensity=min_intensity,
                    max_num_peaks=max(20, int(length_mz / (2.5))),
                    scale_intensity=scale_intensity,
                )
        # except:
        #    print('Error preprocessing spectrum')

        # preprocess np vectors
        # all_spectrums= self.process_all_spectrum_vectors(spectrums)
        return spectrums

    def preprocess_spectrum(
        self,
        spectrum: SpectrumExt,
        fragment_tol_mass: float = 10,
        fragment_tol_mode: str = "ppm",
        min_intensity: float = 0.01,
        max_num_peaks: int = 100,
        # max_num_peaks: int =40,
        scale_intensity: str | None = None,
        # scale_intensity="root",
    ) -> SpectrumExt:
        """
        Preprocess a single spectrum
        1. remove the precursor peak
        2. filter low-intensity peaks + limit the number of peaks
        3. scale intensities

        Parameters
        ----------
        spectrum : SpectrumExt
            The spectrum to preprocess.
        fragment_tol_mass : float
            The fragment tolerance mass, by default 10.
        fragment_tol_mode : str
            The fragment tolerance mode, by default "ppm".
        min_intensity : float
            The minimum intensity, by default 0.01.
        max_num_peaks : int
            The maximum number of peaks, by default 100.
        scale_intensity : Optional[str]
            The scaling method for intensities, by default None.

        Returns
        -------
        SpectrumExt
            The preprocessed spectrum.
        """

        # Process the spectrum.
        return (
            spectrum.remove_precursor_peak(fragment_tol_mass, fragment_tol_mode)
            # .set_mz_range(min_mz=self.min_mz, max_mz=self.max_mz)
            .filter_intensity(min_intensity=min_intensity, max_num_peaks=max_num_peaks)
            .scale_intensity(scale_intensity)
        )

    def return_spectrum_vector(self, spectrum):
        # Initialize an empty numpy array for bin intensities
        binned_spectrum = np.zeros(self.num_bins, dtype=np.float64)

        # Iterate through the data and assign intensities to bins
        for mz, intensity in zip(spectrum.mz, spectrum.intensity, strict=False):
            if (mz > self.min_mz) and (mz < self.max_mz):
                bin_index = int((mz - self.min_mz) / self.bin_width)
                if intensity > binned_spectrum[bin_index]:
                    binned_spectrum[bin_index] = intensity
        return binned_spectrum

    def process_all_spectrum_vectors(self, spectrums):
        """
        save spectrum vectors and apply preprocessing
        """
        for spectrum in tqdm(spectrums):
            spectrum_vector = self.return_vector_and_preprocess(spectrum)
            spectrum.set_spectrum_vector(spectrum_vector)
        return spectrums

    def return_vector_and_preprocess(self, spectrum):
        spectrum_vector = self.return_spectrum_vector(spectrum)
        spectrum_vector = self.preprocess_vector(spectrum_vector)
        return spectrum_vector

    def preprocess_vector(self, spectrum_vector, min_intensity=0.01):
        # scale values using the maximum
        maximum = np.max(spectrum_vector)
        if maximum != 0:
            spectrum_vector = spectrum_vector / maximum
        # remove small values
        spectrum_vector[spectrum_vector < min_intensity] = 0
        return spectrum_vector

    def get_all_binned_spectrums(self, spectrums):
        all_binned_spectrums = np.zeros(
            (len(spectrums), self.num_bins), dtype=np.float64
        )
        for i, spectrum in enumerate(spectrums):
            all_binned_spectrums[i] = spectrum.spectrum_vector
        return all_binned_spectrums

    def return_valid_spectra_n_peaks(self, spectra_original, min_peaks):
        """
        return valid spectra based on the minimum number of peaks
        """
        input_spectra = [copy.deepcopy(s) for s in spectra_original]
        preprocessed_spectra = self.preprocess_all_spectra(input_spectra)
        valid_indexes = [
            i for i, s in enumerate(preprocessed_spectra) if len(s.mz) > min_peaks
        ]
        return [spectra_original[i] for i in valid_indexes]
