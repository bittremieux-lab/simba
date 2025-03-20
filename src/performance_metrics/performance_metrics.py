
import numpy as  np
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
from src.tanimoto import Tanimoto
from src.analog_discovery.mces import MCES
import os 
import pandas as pd 
import seaborn 
class PerformanceMetrics:

    def filter_by_correctness(similarities1, predictions1, similarities2, predictions2, good_predictions=True):
        sim_np_1= np.array(similarities1)
        pred_np_1=np.array(predictions1)

        sim_np_2= np.array(similarities2)
        pred_np_2=np.array(predictions2)

        #return np.argwhere((sim_np==pred_np)&(sim_np>0.1))
        equal_edit_distance= (sim_np_1==pred_np_1)

        equal_mces= (np.abs(sim_np_2- pred_np_2)<0.2)

        low_mces = (sim_np_2>0.8) & (sim_np_2!=1)



        if good_predictions:
            return np.argwhere(equal_edit_distance&equal_mces & low_mces)
        else:
            return np.argwhere(~(equal_edit_distance) & ~equal_mces)
    

    @staticmethod
    def get_correct_predictions(similarities1, predictions1, similarities2, predictions2):
        '''
        return the correct predictions 
        '''
        return PerformanceMetrics.filter_by_correctness(similarities1, predictions1, similarities2, predictions2, good_predictions=True)

    @staticmethod
    def get_bad_predictions(similarities1, predictions1, similarities2, predictions2):
        '''
        return the bad predictions 
        '''
        return PerformanceMetrics.filter_by_correctness(similarities1, predictions1, similarities2, predictions2, good_predictions=False)

    


    @staticmethod 
    def generate_csv_file(total_df, smiles0, smiles1, tanimoto, sim_ed, pred_ed, sim_mces, pred_mces, spec0, spec1, prefix):
        new_sample = {
            'Smiles 0': [smiles0],
            'Smiles 1': [smiles1],
            'tanimoto': [tanimoto],
            'Edit distance (ground truth)': [sim_ed],
            'Edit distance (predicted)': [pred_ed],
            'MCES (ground truth)': [sim_mces],
            'MCES (predicted)': [pred_mces],
            'Spectrum 0-mz': [spec0.mz],  # Ensure these are lists
            'Spectrum 0-intensity': [spec0.intensity],
            'Spectrum 1-mz': [spec1.mz],
            'Spectrum 1-intensity': [spec1.intensity],
            'Metadata 0':[spec0.params],
            'Metadata 1':[spec1.params],
            'Correct?': [True if prefix == 'good' else False],
        }
        
        # Convert the dictionary into a DataFrame
        df = pd.DataFrame(new_sample)
        
        # Concatenate with the existing DataFrame
        return pd.concat([total_df, df], axis=0, ignore_index=True)
        
    @staticmethod
    def plot_mirror_spectra(spec0, spec1, figsize=(20,7)):
        fig, ax = plt.subplots(figsize=figsize)

        # 1) Mirror plot first
        seaborn.set_context(context='poster', font_scale=0.5, rc=None)
        sup.mirror(spec0, spec1, ax=ax)

        # 2) Prepare a function for filtering important peaks
        def filter_peaks(spec):
            intensities = np.array(spec.intensity)
            mzs = np.array(spec.mz)
            # Keep only peaks with intensity >5% of max:
            threshold = 0.05 * intensities.max()

            high_mz = mzs[intensities > threshold]
            high_int = intensities[intensities > threshold]
            # Sort by intensity, keep top 5
            idx_sorted = np.argsort(high_int)[-5:]
            top5_mz = high_mz[idx_sorted]

            # Optionally filter out peaks too close in m/z
            #filtered = [top20_mz[0]]
            #for val in top20_mz[1:]:
                #if abs(val - filtered[-1]) >= 100:
                #    filtered.append(val)
            filtered = top5_mz
            return filtered

        # 3) Annotate after sup.mirror
        for mz in filter_peaks(spec0):
            max_intensity= max(spec0.intensity)
            intensity0 = spec0.intensity[list(spec0.mz).index(mz)]/max_intensity
            
            # On mirror plot, spec0 intensities are > 0
            ax.text(mz, intensity0, f"{mz:.2f}", ha='center', va='bottom',
                    fontsize=6, color='red', zorder=10)

        for mz in filter_peaks(spec1):
            max_intensity= max(spec1.intensity)
            intensity1 = spec1.intensity[list(spec1.mz).index(mz)]/max_intensity
            # On mirror plot, spec1 intensities are < 0
            ax.text(mz, -intensity1, f"{mz:.2f}", ha='center', va='top',
                    fontsize=6, color='blue', zorder=10)

        # 4) (Optional) Adjust the y-limit dynamically 
        max_int0 = max(spec0.intensity)
        max_int1 = max(spec1.intensity)
        ax.set_ylim([-1.1, 1.1])

        ax.set_title("Spec0 (Top) vs Spec1 (Bottom)")
        #plt.tight_layout()
        #sup.mirror(spec0, spec1, ax=ax)
        return fig, ax
        

    @staticmethod
    def plot_molecules(molecule_pairs, prediction_results,
                        target_indexes, config, 
                        samples=20, prefix='good'):


        similarities_ed= prediction_results['similarities_ed']
        similarities_mces= prediction_results['similarities_mces']
        predictions_ed= prediction_results['predictions_ed']
        predictions_mces= prediction_results['predictions_mces']
        pred_mod_cos = prediction_results['pred_mod_cos']
        pred_ms2= prediction_results['pred_ms2']
        output_path = config.CHECKPOINT_DIR

        # create folders for the images if they dont exist
        if not(os.path.exists(output_path + prefix)):
            os.mkdir(output_path + prefix)

        # randomize the plotting
        np.random.seed(42)
        target_indexes = np.random.choice(np.reshape(target_indexes,-1), size=target_indexes.shape[0], replace=False)

        #  get the spectrums 
        spectrums_0 = molecule_pairs.get_spectrums_from_indexes(pair_index=0)
        spectrums_1 = molecule_pairs.get_spectrums_from_indexes(pair_index=1)

        # make sure the spectrum mz is unique

        similarities_target_ed=similarities_ed.copy()
        similarities_target_mces=similarities_mces.copy()
        predictions_target_ed= predictions_ed.copy()
        predictions_target_mces= predictions_mces.copy()

        # filter based on retrieved indexes
        spectrums_0 = [spectrums_0[int(index)] for index in target_indexes]
        spectrums_1 = [spectrums_1[int(index)] for index in target_indexes]
        predictions_target_ed = [predictions_target_ed[int(index)] for index in target_indexes]
        predictions_target_mces = [predictions_target_mces[int(index)] for index in target_indexes]

        similarities_target_ed = [similarities_target_ed[int(index)] for index in target_indexes]
        similarities_target_mces = [similarities_target_mces[int(index)] for index in target_indexes]


        pred_mod_cos_filtered = [pred_mod_cos[int(index)] for index in target_indexes]
        pred_ms2_filtered = [pred_ms2[int(index)] for index in target_indexes]
        total_df=pd.DataFrame()

        samples =min(len(target_indexes), samples)
        for index, (spec0, spec1,sim_ed,sim_mces, pred_ed, pred_mces, pred_mod, pred_ms2_value) in enumerate(zip(spectrums_0[0:samples], 
                                                            spectrums_1[0:samples],
                                                        similarities_target_ed[0:samples],
                                                        similarities_target_mces[0:samples],
                                                        predictions_target_ed[0:samples],
                                                        predictions_target_mces[0:samples],
                                                        pred_mod_cos_filtered[0:samples],
                                                        pred_ms2_filtered[0:samples])):
            
            fig, ax= PerformanceMetrics.plot_mirror_spectra(spec0,spec1, figsize=None)
            plot_path = output_path   + prefix + '/' + f'{prefix}_pair_{index}_spectra.png'
            fig.savefig(plot_path)
            
            smiles_0 = spec0.params['smiles']
            smiles_1 = spec1.params['smiles']

            # let's compute tanimoto 
            tanimoto = Tanimoto.compute_tanimoto_from_smiles(smiles_0, smiles_1)

            mol_0 = Chem.MolFromSmiles(smiles_0)
            mol_1 = Chem.MolFromSmiles(smiles_1)
            

            # Draw molecules
            img_0 = Draw.MolToImage(mol_0, size=(300, 300))
            img_1 = Draw.MolToImage(mol_1, size=(300, 300))
            
            # Combine images side by side
            from PIL import Image

            combined_img = Image.new('RGB', (600, 600))
            combined_img.paste(img_0, (0, 0))
            combined_img.paste(img_1, (300, 0))
            
            sim_mces= int(config.MCES20_MAX_VALUE- config.MCES20_MAX_VALUE*sim_mces)
            pred_mces=int(config.MCES20_MAX_VALUE- config.MCES20_MAX_VALUE*pred_mces)
            sim_ed = (config.EDIT_DISTANCE_N_CLASSES-1) -sim_ed
            pred_ed = (config.EDIT_DISTANCE_N_CLASSES-1) -pred_ed
            # Add title to the image using PIL
            title = f'SMILES: {smiles_0}\nSMILES: {smiles_1}\n \
                                \n \
                                \n Tanimoto: {tanimoto:.2f} \
                                \n Modified cosine: {pred_mod:.2f} \
                                \n MS2: {pred_ms2_value:.2f} \
                                \n \
                                \n Edit distance (ground truth): {sim_ed if sim_ed<5 else ">5"}  \
                                \n Edit distance (pred).: {pred_ed if pred_ed<5 else ">5"} \
                                \n \
                                \n MCES (ground truth): {sim_mces}  \
                                \n MCES(pred): {pred_mces}'

            title_img = Image.new('RGB', (600, 300), (255, 255, 255))
            draw = ImageDraw.Draw(title_img)
            font = ImageFont.load_default()
            draw.text((10, 10), title, fill=(0, 0, 0), font=font)
            
            # Draw the paragraph (wrapped text)
            #paragraph_lines = paragraph.split('. ')
            y_text = 30
                
            # Combine title and molecule images
            final_img = Image.new('RGB', (600, 600))
            
            final_img.paste(combined_img, (0, 0))
            final_img.paste(title_img, (0, 300))
            
            # Save the image
            final_img.save(output_path + prefix + '/' + f'{prefix}_pair_{index}_molecule.png')

            
            #plt.close(fig)

            fig, ax = plt.subplots()

            from PIL import Image

            # After saving the molecule image and spectra plot
            final_img_path = output_path + prefix + '/' + f'{prefix}_pair_{index}_molecule.png'
            spectra_img_path = plot_path

            # Open the two images
            molecule_img = Image.open(final_img_path)
            spectra_img = Image.open(spectra_img_path)

            # Resize spectra image to match the height of the molecule image if needed
            spectra_img = spectra_img.resize((molecule_img.width, molecule_img.height))

            # Combine the two images side by side
            combined_width = molecule_img.width + spectra_img.width
            combined_height = molecule_img.height
            combined_graph = Image.new('RGB', (combined_width, combined_height))

            # Paste the two images
            combined_graph.paste(molecule_img, (0, 0))
            combined_graph.paste(spectra_img, (molecule_img.width, 0))

            # Save the combined graph
            combined_graph_path = output_path + prefix + '/' + f'{prefix}_pair_{index}_combined.png'
            combined_graph.save(combined_graph_path)
            print(combined_graph_path)
            # save data
            total_df= PerformanceMetrics.generate_csv_file(total_df, smiles_0, smiles_1, tanimoto, sim_ed, pred_ed, sim_mces, pred_mces, spec0, spec1, prefix)

        total_df.to_csv(output_path + '/total_examples_' + prefix + '.csv')