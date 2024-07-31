
import numpy as  np
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
from src.tanimoto import Tanimoto
from src.analog_discovery.mces import MCES

class PerformanceMetrics:

    @staticmethod
    def get_correct_predictions(similarities, predictions):
        '''
        return the correct predictions for pairs with similarity>0
        '''
        sim_np= np.array(similarities)
        pred_np=np.array(predictions)
        return np.argwhere((sim_np==pred_np)&(sim_np>0.1))

    @staticmethod
    def get_bad_predictions(similarities, predictions):
        '''
        return the correct predictions for pairs with similarity>0
        '''
        sim_np= np.array(similarities)
        pred_np=np.array(predictions)
        return np.argwhere((sim_np!= pred_np)&(sim_np>0.1))




    @staticmethod
    def plot_molecules(molecule_pairs, similarities, predictions, target_indexes, config, samples=20, prefix='good'):
        output_path = config.CHECKPOINT_DIR

        # randomize the plotting
        target_indexes = target_indexes[np.random.randint(0, target_indexes.shape[0],target_indexes.shape[0] )]
        #  get the spectrums 
        spectrums_0 = molecule_pairs.get_spectrums_from_indexes(pair_index=0)
        spectrums_1 = molecule_pairs.get_spectrums_from_indexes(pair_index=1)
        mces_distances = molecule_pairs.indexes_tani[:,2]
        similarities_target=similarities.copy()
        predictions_target= predictions.copy()

        # filter based on retrieved indexes
        spectrums_0 = [spectrums_0[int(index)] for index in target_indexes]
        spectrums_1 = [spectrums_1[int(index)] for index in target_indexes]
        mces_distances = [mces_distances[int(index)] for index in target_indexes]
        predictions_target = [predictions_target[int(index)] for index in target_indexes]
        similarities_target = [similarities_target[int(index)] for index in target_indexes]

        for index, (spec0, spec1, mces,sim, pred) in enumerate(zip(spectrums_0[0:samples], 
                                                            spectrums_1[0:samples],
                                                        mces_distances[0:samples],
                                                        similarities_target[0:samples],
                                                        predictions_target[0:samples])):
            

            smiles_0 = spec0.params['smiles']
            smiles_1 = spec1.params['smiles']

            # let's compute tanimoto 
            tanimoto = Tanimoto.compute_tanimoto_from_smiles(smiles_0, smiles_1)
            mces_similarity,_= MCES.calculate_mcs_similarity(smiles_0, smiles_1)

            mol_0 = Chem.MolFromSmiles(smiles_0)
            mol_1 = Chem.MolFromSmiles(smiles_1)
            

            # Draw molecules
            img_0 = Draw.MolToImage(mol_0, size=(300, 300))
            img_1 = Draw.MolToImage(mol_1, size=(300, 300))
            
            # Combine images side by side
            combined_img = Image.new('RGB', (600, 300))
            combined_img.paste(img_0, (0, 0))
            combined_img.paste(img_1, (300, 0))
            
            # Add title to the image using PIL
            title = f'SMILES: {smiles_0}\nSMILES: {smiles_1}\nClass truth: {sim}  tanimoto: {tanimoto} mces sim. {mces_similarity} \nClass pred.: {pred}'
            title_img = Image.new('RGB', (600, 100), (255, 255, 255))
            draw = ImageDraw.Draw(title_img)
            font = ImageFont.load_default()
            draw.text((10, 10), title, fill=(0, 0, 0), font=font)
            
            # Draw the paragraph (wrapped text)
            #paragraph_lines = paragraph.split('. ')
            y_text = 30
                
            # Combine title and molecule images
            final_img = Image.new('RGB', (600, 350))
            final_img.paste(title_img, (0, 0))
            final_img.paste(combined_img, (0, 70))
            
            # Save the image
            final_img.save(output_path + f'{prefix}_pair_{index}_molecule.png')

            # plot spectra
            fig, ax = plt.subplots()
            sup.mirror(spec0, spec1, ax=ax)
            plot_path = output_path + f'{prefix}_pair_{index}_spectra.png'
            plt.savefig(plot_path)
            plt.close(fig)