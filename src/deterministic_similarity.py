from src.molecule_pair import MoleculePair
from typing import List
from  src.similarity import cosine, modified_cosine, neutral_loss
from src.config import Config
from tqdm import tqdm 
import pandas as pd
from src.tanimoto import Tanimoto
from src.transformers.load_data import LoadData
from torch.utils.data import DataLoader
from src.transformers.embedder import Embedder
import lightning.pytorch as pl
import numpy as np
from src.config import Config
from scipy.stats import spearmanr
from src.plotting import Plotting
#from src.ml_model import MlModel

class DetSimilarity:
    '''
    class for computing similarity for cosine distance
    '''
    @staticmethod
    def compute_deterministic_similarity(molecule_pairs: List[MoleculePair], similarity_metric='cosine', config=None):
        '''
        compute cosine ('cosine'), modified cosine ('modified_cosine') or neutral loss ('neutral_loss')
        '''
        computing_function = DetSimilarity.select_function(similarity_metric)
        total_scores=[]
        for m in tqdm(molecule_pairs):
            spectra_0 = m.spectrum_object_0
            spectra_1 = m.spectrum_object_1
            scores = computing_function(spectra_0, spectra_1, config.FRAGMENT_MZ_TOLERANCE)
            # set score
            m.set_det_similarity_score(scores, similarity_metric)
            total_scores.append(scores)
        return molecule_pairs, total_scores
    
    @staticmethod
    def select_function(similarity_metric):
        if similarity_metric=='cosine':
            return cosine 
        elif similarity_metric == 'modified_cosine':
            return modified_cosine
        elif similarity_metric == 'neutral_loss':
            return neutral_loss
        
    @staticmethod
    def call_saved_model(molecule_pairs, model_file):
        # siamese
        model =  MlModel(input_dim=molecule_pairs[0].vector_0.shape[0])
        model.load_best_model(model_file) 
        return model.predict(molecule_pairs)

    @staticmethod
    def call_saved_transformer_model(molecule_pairs, model_file, d_model=64, n_layers=2):
        # transformer
        dataset_test = LoadData.from_molecule_pairs_to_dataset(molecule_pairs)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
        best_model = Embedder.load_from_checkpoint(model_file, d_model=d_model, n_layers=n_layers)
        trainer = pl.Trainer(max_epochs=1,)
        pred_test = trainer.predict(best_model, dataloader_test)
        pred_test = np.array([float(p[0]) for p in pred_test])

        # clip to 0 and 1
        pred_test = np.clip(pred_test, 0, 1)
        return pred_test

    @staticmethod
    def compute_all_scores(molecule_pairs, write=False, write_file= '"./gnps_libraries.parquet"',
                            model_file='./best_model.h5', config=None):
        scores = []
        #model_scores = DetSimilarity.call_saved_model(molecule_pairs, model_file)
        model_scores = DetSimilarity.call_saved_transformer_model(molecule_pairs, model_file, d_model=config.D_MODEL, n_layers=config.N_LAYERS)

        for i,m in tqdm(enumerate(molecule_pairs)):
            spectra_0 = m.spectrum_object_0
            spectra_1 = m.spectrum_object_1
            
            cos = cosine(spectra_0, spectra_1, config.FRAGMENT_MZ_TOLERANCE)
            mod_cos = modified_cosine(
                spectra_0, spectra_1, config.FRAGMENT_MZ_TOLERANCE
            )
            nl = neutral_loss(
                spectra_0, spectra_1, config.FRAGMENT_MZ_TOLERANCE
            )

            #model_score= model_scores[i,0]   #for sieamese network 
            model_score = model_scores[i]

            tan = Tanimoto.compute_tanimoto(m.params_0["smiles"], m.params_1["smiles"])
            scores.append(
                (cos[0], cos[1], mod_cos[0], mod_cos[1], nl[0], nl[1], model_score, 0, tan)
            )

        # Compute the spearman correlation
        # Calculate Spearman correlation
        tanimoto_temp = np.array([s[8] for s in scores])
        mod_cosine_temp= np.array([s[2] for s in scores])
        model_temp = np.array([s[6] for s in scores])

        corr_mod_cos, p_value_mod_cos = spearmanr(tanimoto_temp, mod_cosine_temp)
        corr_model_temp, p_value_model_temp = spearmanr(tanimoto_temp, model_temp)

        # Print the correlation coefficient and p-value
        print("Spearman correlation coefficient for modified cosine:", corr_mod_cos)
        print("P-value:", p_value_mod_cos)
        print("Spearman correlation coefficient for model:", corr_model_temp)
        print("P-value:", p_value_model_temp)

        #roc curves
        x_class = tanimoto_temp.copy()
        x_class[tanimoto_temp<config.threshold_class] = 0
        x_class[tanimoto_temp>=config.threshold_class]= 1

        Plotting.plot_roc_curve_comparison(x_class, [model_temp, mod_cosine_temp], 
                                            title='ROC Curve', roc_file_path=config.CHECKPOINT_DIR + f'roc_curve_comparison_{config.MODEL_CODE}.png',
                                            labels = ['model', 'mod_cosine'],
                                            colors= ['r','b'])
         
        similarities = pd.DataFrame(
            {
                #"pair1": pairs[:, 0],
                #"pair2": pairs[:, 1],
                
                "id1": [m.spectrum_object_0.params['spectrumid'] for m in molecule_pairs],
                "id2": [m.spectrum_object_1.params['spectrumid'] for m in molecule_pairs],
                "class1" : [m.spectrum_object_0.classe for m in molecule_pairs],
                "class2" :[m.spectrum_object_1.classe for m in molecule_pairs],
                "superclass1":[m.spectrum_object_0.superclass for m in molecule_pairs],
                "superclass2":[m.spectrum_object_1.superclass for m in molecule_pairs],
                "subclass1":[m.spectrum_object_0.subclass for m in molecule_pairs],
                "subclass2":[m.spectrum_object_1.subclass for m in molecule_pairs],
                "smiles1": [m.spectrum_object_0.params['smiles'] for m in molecule_pairs], 
                "smiles2": [m.spectrum_object_1.params['smiles'] for m in molecule_pairs],
                "charge1": [m.spectrum_object_0.params['charge'][0] for m in molecule_pairs],
                "charge2": [m.spectrum_object_1.params['charge'][0] for m in molecule_pairs],
                "mz1": [m.spectrum_object_0.precursor_mz for m in molecule_pairs], 
                "mz2": [m.spectrum_object_1.precursor_mz for m in molecule_pairs],
            }
        )
        similarities[
            [
                "cosine",
                "cosine_explained",
                "modified_cosine",
                "modified_cosine_explained",
                "neutral_loss",
                "neutral_loss_explained",
                "model_score",
                "model_score_explained",
                "tanimoto",
            ]
        ] = scores

        similarities["tanimoto_interval"] = pd.cut(
            similarities["tanimoto"],
            5,
            labels=["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"],
        )
        similarities_tanimoto = pd.melt(
            similarities,
            id_vars="tanimoto_interval",
            value_vars=["cosine", "neutral_loss", "modified_cosine","model_score"],
        )
        return similarities, similarities_tanimoto
        #if write:
        #    #to_parquet(write_file)
