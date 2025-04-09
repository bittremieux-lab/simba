from simba.mces.mces_computation import MCES
import numpy as np
from simba.edit_distance.edit_distance import EditDistance


class GroundTruth:

    def compute_edit_distance(spectra0, spectra1, max_value=5):
        ground_truth_ed = np.zeros((len(spectra0), len(spectra1)))
        smiles0 = [s.params["smiles"] for s in spectra0]
        smiles1 = [s.params["smiles"] for s in spectra1]

        for i, s0 in enumerate(smiles0):
            for j, s1 in enumerate(smiles1):
                ground_truth_ed[i, j] = EditDistance.get_edit_distance_from_smiles(
                    s0, s1, return_nans=True
                )

        ground_truth_ed[np.isnan(ground_truth_ed)] = max_value
        ground_truth_ed[ground_truth_ed >= max_value] = max_value
        return ground_truth_ed

    def compute_mces(spectra0, spectra1):

        ground_truth_mces = np.zeros((len(spectra0), len(spectra1)))
        smiles0 = [s.params["smiles"] for s in spectra0]
        smiles1 = [s.params["smiles"] for s in spectra1]

        for i, s0 in enumerate(smiles0):
            df_results = MCES.compute_mces_list_smiles([s0] * len(smiles1), smiles1)
            mces_result = df_results["mces"]
            ground_truth_mces[i] = mces_result
        return ground_truth_mces
