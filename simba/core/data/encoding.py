import pandas as pd

from simba.core.chemistry.chem_utils import ADDUCT_TO_MASS


ION_ACTIVATION = ["HCD", "CID"]
IONIZATION_METHODS = ["NSI", "ESI", "APCI"]


def encode_adduct(adduct: str):
    """Encode adduct string as one-hot vector.

    Args:
        adduct: Adduct string (e.g., '[M+H]+')

    Returns:
        list: One-hot encoded vector
    """
    adducts = ADDUCT_TO_MASS.keys()
    return [1 if adduct == a else 0 for a in adducts]


def encode_ion_activation(ion_activation: str):
    """Encode ion activation method as one-hot vector.

    Args:
        ion_activation: Ion activation method (e.g., 'HCD')

    Returns:
        list: One-hot encoded vector
    """
    return [1 if ion_activation == ia else 0 for ia in ION_ACTIVATION]


def encode_ionization_method(ionization_method: str):
    """Encode ionization method as one-hot vector.

    Args:
        ionization_method: Ionization method (e.g., 'ESI')

    Returns:
        list: One-hot encoded vector
    """
    return [1 if ionization_method == im else 0 for im in IONIZATION_METHODS]


class OneHotEncoding:
    def __init__(self, adduct_file_path):
        self.df_adduct = self.load_adduct_info(adduct_file_path)

    # TODO: adduct to mass dict also in simba/chem_utils.py
    def load_adduct_info(self, path_adduct_csv):
        return pd.read_csv(path_adduct_csv, delimiter=";")

    ### obtain information for adduct handling based on adduct mass
    def encode_adduct(
        self,
        adduct_mass,
        ion_mode,
    ):
        """
        adduct_mass : float
            Target adduct mass (mass shift).
        ion_mode : str
            "positive" or "negative".
        df_adduct : pd.DataFrame
            DataFrame with columns:
            - 'Ion mode'
            - 'Mass' (adduct mass shift)
            - adduct_* columns (adduct_M, adduct_H, adduct_Na, ...)

        Returns
        -------
        list_adduct_elements : list[float or int]
            Values of all adduct_* columns (including adduct_M) for the closest adduct.
            The order is the column order in df_adduct.
        """

        # Keep only rows with the requested ion mode
        mask_mode = self.df_adduct["Ion mode"].str.lower() == ion_mode.lower()
        df_mode = self.df_adduct[mask_mode]

        if df_mode.empty:
            raise ValueError(f"No adducts found for ion mode: {ion_mode}")

        # Find the row whose 'Mass' is closest to the target adduct_mass
        idx_closest = (df_mode["Mass"] - adduct_mass).abs().idxmin()
        best_row = df_mode.loc[idx_closest]

        # All columns that encode adduct composition (including adduct_M)
        adduct_cols = [c for c in self.df_adduct.columns if c.startswith("adduct_")]

        # Extract their values as a list (you can cast to int if you prefer)
        list_adduct_elements = best_row[adduct_cols].tolist()

        return list_adduct_elements

    def encode_ion_activation(self, ion_activation):
        ion_activation_upper = ion_activation.upper() if ion_activation else ""
        return [1 if ion_activation_upper == ia else 0 for ia in ION_ACTIVATION]

    def encode_ionization_method(self, ionization_method):
        ionization_method_upper = ionization_method.upper() if ionization_method else ""
        return [1 if ionization_method_upper == im else 0 for im in IONIZATION_METHODS]
