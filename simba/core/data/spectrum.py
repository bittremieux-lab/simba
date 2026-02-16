from collections.abc import Iterable

import numpy as np
from spectrum_utils.spectrum import MsmsSpectrum


class SpectrumExt(MsmsSpectrum):
    """'
    extended spectrum class that incorporates the binned vector
    """

    # def __init__(self, **kwargs):
    #        super().__init__(**kwargs)  # Call the constructor of the base class
    #
    #        # extra variables
    #        self.spectrum_vector = np.array([])
    #        self.smiles = ''

    def __init__(
        self,
        identifier: str,
        precursor_mz: float,
        precursor_charge: int,
        mz: np.ndarray | Iterable,
        intensity: np.ndarray | Iterable,
        retention_time: float,
        params: dict,
        library: str,
        inchi: str,
        smiles: str,
        ionmode: str,
        adduct: float,
        ce: float,
        ion_activation: str,
        ionization_method: str,
        bms: str,
        superclass: str,
        classe: str,
        subclass: str,
        inchi_key: str = None,
        spectrum_hash: str = None,
        mgf_index: int = None,
    ):

        super().__init__(
            identifier,
            precursor_mz,
            precursor_charge,
            mz,
            intensity,
            retention_time,
        )

        # extra variables
        self.params = params
        self.mz_array = None
        self.intensity_array = None
        self.retention_time = retention_time
        self.spectrum_vector = ""
        self.smiles = smiles
        self.max_peak = ""
        self.library = library
        self.inchi = inchi
        self.ionmode = ionmode
        self.adduct = adduct
        self.ce = ce
        self.ion_activation = ion_activation
        self.ionization_method = ionization_method
        # classes
        self.superclass = superclass
        self.classe = classe
        self.subclass = subclass

        # preprocessed variables
        self.murcko_scaffold = bms
        self.inchi_key = inchi_key
        self.spectrum_hash = spectrum_hash
        self.mgf_index = mgf_index  # Original index in MGF file before filtering

    def set_params(self, params):
        self.params = params

    # def set_mz_array(self, mz_array):
    #    self.mz_array = mz_array

    def __getstate__(self):
        # Get the state of the base class
        # state = super(SpectrumExt, self).__getstate__()
        state = super(SpectrumExt, self).__getstate__()
        # Add state for the derived class
        state.update(
            {
                "params": self.params,
                "intensity_array": self.intensity_array,
                "mz_array": self.mz_array,
                "spectrum_vector": self.spectrum_vector,
                "smiles": self.smiles,
                "max_peak": self.max_peak,
                "library": self.library,
                "inchi": self.inchi,
                "ionmode": self.ionmode,
                "adduct": self.adduct,
                "ce": self.ce,
                "ion_activation": self.ion_activation,
                "ionization_method": self.ionization_method,
                "retention_time": self.retention_time,
                "superclass": self.superclass,
                "classe": self.classe,
                "subclass": self.subclass,
                "murcko_scaffold": self.murcko_scaffold,
                "inchi_key": self.inchi_key,
                "spectrum_hash": self.spectrum_hash,
                "mgf_index": self.mgf_index,
            }
        )
        return state

    def __setstate__(self, state):
        # Restore base class state
        super().__setstate__(state)

        # Restore derived class state
        self.params = state["params"]
        self.intensity_array = state["intensity_array"]
        self.mz_array = state["mz_array"]
        self.spectrum_vector = state["spectrum_vector"]
        self.smiles = state["smiles"]
        self.max_peak = state["max_peak"]
        self.library = state["library"]
        self.inchi = state["inchi"]
        self.ionmode = state["ionmode"]
        self.adduct = state["adduct"] if "adduct" in state.keys() else None
        try:
            self.ce = state["ce"]
        except KeyError:
            self.ce = 0.0
        try:
            self.ion_activation = state["ion_activation"]
        except KeyError:
            self.ion_activation = ""
        try:
            self.ionization_method = state["ionization_method"]
        except KeyError:
            self.ionization_method = ""
        self.retention_time = state["retention_time"]
        self.superclass = state["superclass"]
        self.classe = state["classe"]
        self.subclass = state["subclass"]
        self.murcko_scaffold = state["murcko_scaffold"]
        try:  # in other versions, the inchi key is not present
            self.inchi_key = state["inchi_key"]
        except:
            self.inchi_key = ""

        try:
            self.spectrum_hash = state["spectrum_hash"]
        except:
            self.spectrum_hash = None

        try:
            self.mgf_index = state["mgf_index"]
        except:
            self.mgf_index = None

    # def set_intesity_array(self, intensity_array):
    #    self.intensity_array = intensity_array

    def set_spectrum_vector(self, spectrum_vector):
        self.spectrum_vector = spectrum_vector

    def set_murcko_scaffold(self, murcko_scaffold):
        self.murcko_scaffold = murcko_scaffold

    def set_smiles(self, smiles):
        self.smiles = smiles

    def set_max_peak(self, max_peak):
        """
        set the maximum amplitude in the spectrum
        """
        self.max_peak = max_peak
