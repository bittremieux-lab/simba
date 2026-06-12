from collections.abc import Iterable

import numpy as np
from spectrum_utils.spectrum import MsmsSpectrum


class SpectrumExt(MsmsSpectrum):
    """Extended spectrum class with MS/MS metadata fields."""

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
        fold: str = None,
        dataset: str = None,
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
        self.retention_time = retention_time
        self.smiles = smiles
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
        self.fold = fold  # Predefined split label (e.g. "train"/"val"/"test")
        self.dataset = (
            dataset  # Source dataset name (e.g. "MassSpecGym"), None if not tagged
        )

    def set_params(self, params):
        self.params = params

    def __getstate__(self):
        # Get the state of the base class
        state = super().__getstate__()
        # Add state for the derived class
        state.update(
            {
                "params": self.params,
                "smiles": self.smiles,
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
                "fold": self.fold,
                "dataset": self.dataset,
            }
        )
        return state

    def __setstate__(self, state):
        # Restore base class state
        super().__setstate__(state)

        # Restore derived class state
        self.params = state["params"]
        self.smiles = state["smiles"]
        self.library = state["library"]
        self.inchi = state["inchi"]
        self.ionmode = state["ionmode"]
        self.adduct = state.get("adduct")
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
        self.inchi_key = state.get("inchi_key", "")
        self.spectrum_hash = state.get("spectrum_hash")
        self.mgf_index = state.get("mgf_index")

        self.fold = state.get("fold", None)
        self.dataset = state.get("dataset", None)

    def set_murcko_scaffold(self, murcko_scaffold):
        self.murcko_scaffold = murcko_scaffold

    def set_smiles(self, smiles):
        self.smiles = smiles
