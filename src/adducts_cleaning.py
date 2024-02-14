import re
import numpy as np
import operator
from src.load_data import LoadData


class AdductsCleaning:
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
                [count * charges.get(adduct, 0) for count, adduct in positive_parts]
            ) + sum(
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
        cleaned_adduct.append(f"{abs(charge) if abs(charge) > 1 else ''}{charge_sign}")
        return "".join(cleaned_adduct)
