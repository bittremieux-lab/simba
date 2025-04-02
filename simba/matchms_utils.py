from matchms.Spectrum import Spectrum

import numpy as np


class MatchmsUtils:

    @staticmethod
    def from_su_to_matchms_metadata(original_dict):
        """
        convert from su metadata to matchms
        """

        new_dict = {
            #'ionmode': original_dict['ionmode'].lower(),
            #'charge': original_dict['charge'][0] if isinstance(original_dict['charge'], list) else original_dict['charge'],
            "precursor_mz": (
                original_dict["pepmass"][0]
                if "pepmass" in original_dict.keys()
                else original_dict["precursor_mz"]
            ),
            #'compound_name': original_dict['name'],
        }

        for k in original_dict:
            new_dict[k] = original_dict[k]

        return new_dict

    @staticmethod
    def from_su_to_matchms(spectrum_su):
        metadata_matchms = MatchmsUtils.from_su_to_matchms_metadata(spectrum_su.params)
        spectrum_matchms = Spectrum(
            mz=np.array(spectrum_su.mz),
            intensities=np.array(spectrum_su.intensity, dtype=float),
            metadata=metadata_matchms,
        )
        return spectrum_matchms
