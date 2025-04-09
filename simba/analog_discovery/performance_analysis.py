import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from simba.tanimoto import Tanimoto
from simba.mces.mces_computation import MCES
from rdkit.Chem import Descriptors
import os


class PerformanceAnalysis:

    def plot_performance(
        model_results,
        target_indexes_list,
        all_spectrums_janssen,
        all_spectrums_reference,
        K,
        golden_truth,
        folder_path,
        number_samples=-1,
    ):

        for target_query_index in target_indexes_list[0:number_samples]:
            # Get query spectrum and SMILES
            spec_janssen = all_spectrums_janssen[target_query_index]
            smiles_query = spec_janssen.params["smiles"]

            # Number of models
            num_models = len(model_results)

            # Create a figure with num_models rows and 3 columns
            fig, axes = plt.subplots(num_models, 3, figsize=(15, 5 * num_models))

            # If there's only one model, axes might not be a 2D array
            if num_models == 1:
                axes = np.array([axes])

            model_idx = 0
            for model_used_key in model_results:
                model_used = model_results[model_used_key]
                # Get the results of the model
                similarities = model_used["similarities"]
                smiles_reference = model_used["smiles_reference"]
                similarities_target = similarities[target_query_index]

                # Indexes retrieved
                indexes_k_retrieved = np.argsort(similarities_target)[::-1][0:K]
                sim_k_retrieved = np.sort(similarities_target)[::-1][0:K]
                smiles_retrieved = [
                    smiles_reference[index] for index in indexes_k_retrieved
                ]
                smiles_retrieved = [Chem.CanonSmiles(s) for s in smiles_retrieved]

                # Best candidate
                best_indexes_golden_truth = np.argmax(golden_truth, axis=1)[
                    target_query_index
                ]
                best_smile = all_spectrums_reference[best_indexes_golden_truth].params[
                    "smiles"
                ]
                best_smile = Chem.CanonSmiles(best_smile)
                best_smiles_vector = [
                    10 if s == best_smile else 0 for s in smiles_retrieved
                ]

                # MCES distance results
                mces_distance_list = MCES.compute_mces_list_smiles(
                    smiles_retrieved, len(smiles_retrieved) * [smiles_query]
                )["mces"].values

                # Ranking
                x_ranking = np.arange(len(mces_distance_list))

                # Difference in m/z
                precursor_mz_retrieved = np.array(
                    [
                        Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(s))
                        for s in smiles_retrieved
                    ]
                )
                precursor_mz_query = Chem.Descriptors.ExactMolWt(
                    Chem.MolFromSmiles(smiles_query)
                )
                diff_mz = np.abs(precursor_mz_retrieved - precursor_mz_query)

                # Plotting for this model
                # First subplot: MCES distance
                axes[model_idx, 0].bar(
                    x_ranking, mces_distance_list, label="MCES Distance"
                )
                axes[model_idx, 0].plot(
                    x_ranking,
                    best_smiles_vector,
                    marker="*",
                    color="r",
                    label="Best SMILES",
                )
                axes[model_idx, 0].set_title(
                    f"Median distance: {np.median(mces_distance_list):.2f}"
                )
                axes[model_idx, 0].set_xlabel("Ranking")
                axes[model_idx, 0].set_ylabel("MCES distance - Ground Truth")
                axes[model_idx, 0].set_ylim([0, 50])
                axes[model_idx, 0].legend()

                # Second subplot: Model Score
                axes[model_idx, 1].bar(
                    x_ranking, sim_k_retrieved, label="Model Score", color="g"
                )
                axes[model_idx, 1].set_title(f" Model Score")
                axes[model_idx, 1].set_xlabel("Ranking")
                axes[model_idx, 1].set_ylabel("Model Score")
                axes[model_idx, 1].set_ylim([0, 1.1])

                # Third subplot: Diff mz
                axes[model_idx, 2].bar(x_ranking, diff_mz, color="r", label="Diff mz")
                axes[model_idx, 2].set_title(
                    f" Median diff mz: {np.median(diff_mz):.2f}"
                )
                axes[model_idx, 2].set_xlabel("Ranking")
                axes[model_idx, 2].set_ylabel("Diff mz")
                axes[model_idx, 2].set_ylim([0, 100])

                # Add a label on the left to indicate the model
                axes[model_idx, 0].set_ylabel(
                    f"{model_used_key}\n\n" + axes[model_idx, 0].get_ylabel(),
                    fontsize=12,
                )
                model_idx += 1

            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig(
                os.path.join(folder_path, f"{target_query_index}_combined_plot.png")
            )
            plt.show()

    """   
    def plot_performance(model_results, 
                        target_indexes_list,
                        all_spectrums_janssen, 
                        all_spectrums_reference,
                         K, 
                         golden_truth,
                         folder_path):

        
        
        for model_used_key in model_results:
            # create subfolder
            sub_path=folder_path + model_used_key + '/'
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
                print(f"Folder created: {sub_path}")
            else:
                print(f"Folder already exists: {sub_path}")

            model_used= model_results[model_used_key]
            for target_query_index in target_indexes_list:
                spec_janssen= all_spectrums_janssen[target_query_index]
                smiles_query= spec_janssen.params['smiles']
            
                ## query mol
                mol=Chem.MolFromSmiles(smiles_query)

                # get the results of the model
                similarities = model_used['similarities']
                smiles_reference= model_used['smiles_reference']
                similarities_target= similarities[target_query_index]

                #indexes retrieved:
                indexes_k_retrieved=np.argsort(similarities_target)[::-1][0:K]
                sim_k_retrieved= np.sort(similarities_target)[::-1][0:K]
                smiles_retrieved = [smiles_reference[index] for index in indexes_k_retrieved]
                smiles_retrieved=  [Chem.CanonSmiles(s) for s in smiles_retrieved]

                ## where is the best candidate?
                best_indexes_golden_truth=np.argmax(golden_truth, axis=1)[target_query_index]
                best_smile = all_spectrums_reference[best_indexes_golden_truth].params['smiles']
                best_smile = Chem.CanonSmiles(best_smile)
                best_smiles_vector= [10 if s==best_smile else 0 for s in smiles_retrieved]

                # get the mces distance results
                mces_distance_list = MCES.compute_mces_list_smiles(smiles_retrieved ,len(smiles_retrieved)*[smiles_query])['mces'].values

                # model score
                x_ranking= np.arange(0,len(mces_distance_list))


                # diff mz
                precursor_mz_retrieved= np.array([Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(s)) for s in smiles_retrieved])
                precursor_mz_query= Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles_query))
                diff_mz=np.abs(precursor_mz_retrieved-precursor_mz_query)
                x_ranking= np.arange(0,len(mces_distance_list))
                
                # Create a single figure with 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # First subplot: MCES distance
                axes[0].bar(x_ranking, mces_distance_list, label="MCES Distance")
                axes[0].plot(x_ranking, best_smiles_vector, marker="*", color="r", label="Best Smiles Vector")
                axes[0].set_title(f'Median distance: {np.median(mces_distance_list):.2f}')
                axes[0].set_xlabel('Ranking')
                axes[0].set_ylabel('MCES distance - Ground Truth')
                axes[0].set_ylim([0, 50])
                axes[0].legend()

                # Second subplot: Model Score
                axes[1].bar(x_ranking, sim_k_retrieved, label="Model Score", color="g")
                axes[1].set_title('Model Score')
                axes[1].set_xlabel('Ranking')
                axes[1].set_ylabel('Model Score')
                axes[1].set_ylim([0, 1.1])

                # Third subplot: Diff mz
                axes[2].bar(x_ranking, diff_mz, color='r', label="Diff mz")
                axes[2].set_title(f'Median diff mz: {np.median(diff_mz):.2f}')
                axes[2].set_xlabel('Ranking')
                axes[2].set_ylabel('Diff mz')
                axes[2].set_ylim([0, 100])

                # Adjust layout and save the figure
                plt.tight_layout()
                sub_path = "./"
                target_query_index = 1  # Example index
                plt.savefig(sub_path + str(target_query_index) + '_combined_plot.png')
                plt.show()
    """

    def obtain_precision_k_candidates(
        golden_truth,
        predicted,
        smiles_reference,
        smiles_janssen,
        all_spectrums_reference,
        all_spectrums_janssen,
        min_tanimotos=[0, 0.95],
        k_list=[10, 1000],
    ):
        # indexed by tanimoto similarity and model
        accuracy_results = {}

        for min_tanimoto in min_tanimotos:
            accuracy_results[min_tanimoto] = {}
            #### filter the similarities by precursor mass
            print(f"Size of predictions: {predicted.shape}")
            best_indexes = np.argmax(golden_truth, axis=1)
            spectra_best = [all_spectrums_reference[b] for b in best_indexes]
            smiles_best = [Chem.CanonSmiles(s.params["smiles"]) for s in spectra_best]

            ordered_predicted = np.zeros(predicted.shape)
            for i in range(0, len(all_spectrums_janssen)):
                ordered_predicted[i] = np.argsort(predicted[i, :])[::-1]

            # what is the accuracy?
            accuracy_results[min_tanimoto] = []

            for k in k_list:
                total_samples = 0
                accuracy = 0
                for i in range(0, len(all_spectrums_janssen)):

                    query_smile = all_spectrums_janssen[i].params["smiles"]
                    best_smile = all_spectrums_reference[best_indexes[i]].params[
                        "smiles"
                    ]
                    best_smile = Chem.CanonSmiles(best_smile)
                    best_tanimoto = golden_truth[i, best_indexes[i]]

                    if best_tanimoto > min_tanimoto:
                        best_retrieved_indexes = ordered_predicted[i, 0:k]

                        retrieved_smiles = [
                            smiles_reference[int(b)] for b in best_retrieved_indexes
                        ]

                        retrieved_smiles = [
                            Chem.CanonSmiles(s) for s in retrieved_smiles
                        ]

                        if best_smile in retrieved_smiles:
                            accuracy = accuracy + 1
                        total_samples = total_samples + 1
                        # print(f'query_smile: {query_smile}')
                        # print(f'best_smile:{best_smile}')
                        # print(f'tanimoto: {golden_truth[i, best_indexes[i]]}')
                        # print(f'retrieved indexes: {retrieved_smiles}')
                        # print('')
                accuracy = accuracy / total_samples
                accuracy_results[min_tanimoto].append(accuracy)

        return accuracy_results
