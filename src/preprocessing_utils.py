import numpy as np
from itertools import groupby
import requests
from rdkit import Chem, DataStructs, RDLogger
import json
import pandas as pd
import functools
import copy

class PreprocessingUtils:

    @staticmethod
    def is_centroid(intensity):
        return np.all(intensity > 0)

    @staticmethod
    def order_by_charge(spectrums):
        #spectrums_new = spectrums.copy()
        spectrums_new= copy.deepcopy(spectrums)
        
        # Sort the list based on the property 'x' (optional, but required for groupby)
        spectrums_new.sort(key=lambda a: a.precursor_charge)

        # Group the elements based on the 'x' property
        spectrums_by_charge = {}
        for key, group in groupby(spectrums_new, key=lambda a: a.precursor_charge):
            spectrums_by_charge[key] = list(group)
        return spectrums_by_charge

    @staticmethod
    def order_spectrums_by_mz(spectrums):
        """
        in order to take into account mass differences
        """

        spectrums_by_charge = PreprocessingUtils.order_by_charge(
            spectrums
        )  # return a dictionary

        total_spectrums = []
        for charge in spectrums_by_charge:

            # order by mz
            mzs = np.array([s.precursor_mz for s in spectrums_by_charge[charge]])
            ordered_indexes = np.argsort(mzs)
            temp_spectrums = [spectrums_by_charge[charge][r] for r in ordered_indexes]
            total_spectrums = total_spectrums + temp_spectrums

        return total_spectrums

    def _smiles_to_mol(smiles):
        try:
            return Chem.MolFromSmiles(smiles)
        except ArgumentError:
            return None

    @functools.lru_cache
    def get_class(inchi, smiles):
        clss = (
            PreprocessingUtils._get_class("inchi", inchi)
            if inchi is not None and inchi != "N/A"
            else None
        )
        if clss is None and not pd.isna(smiles) and smiles != "N/A":
            mol = PreprocessingUtils._smiles_to_mol(smiles)
            clss = (
                PreprocessingUtils._get_class("smiles", Chem.MolToSmiles(mol, False))
                if mol is not None
                else None
            )
        return clss if clss is not None else (None, None, None)

    @functools.lru_cache
    def _get_class(mol_type, mol_val):
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
