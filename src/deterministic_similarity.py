from src.molecule_pair import MoleculePair
from typing import List
from  src.similarity import cosine, modified_cosine, neutral_loss
from src.config import Config
from tqdm import tqdm 
import pandas as pd
from src.tanimoto import Tanimoto
class DetSimilarity:
    '''
    class for computing similarity for cosine distance
    '''
    @staticmethod
    def compute_deterministic_similarity(molecule_pairs: List[MoleculePair], similarity_metric='cosine'):
        '''
        compute cosine ('cosine'), modified cosine ('modified_cosine') or neutral loss ('neutral_loss')
        '''
        computing_function = DetSimilarity.select_function(similarity_metric)
        total_scores=[]
        for m in tqdm(molecule_pairs):
            spectra_0 = m.spectrum_object_0
            spectra_1 = m.spectrum_object_1
            scores = computing_function(spectra_0, spectra_1, Config.FRAGMENT_MZ_TOLERANCE)
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
    def compute_all_scores(molecule_pairs, write=False, write_file= '"./gnps_libraries.parquet"'):
        scores = []
        for m in tqdm(molecule_pairs):
            spectra_0 = m.spectrum_object_0
            spectra_1 = m.spectrum_object_1
            
            cos = cosine(spectra_0, spectra_1, Config.FRAGMENT_MZ_TOLERANCE)
            mod_cos = modified_cosine(
                spectra_0, spectra_1, Config.FRAGMENT_MZ_TOLERANCE
            )
            nl = neutral_loss(
                spectra_0, spectra_1, Config.FRAGMENT_MZ_TOLERANCE
            )

            tan = Tanimoto.compute_tanimoto(m.params_0["smiles"], m.params_1["smiles"])
            scores.append(
                (cos[0], cos[1], mod_cos[0], mod_cos[1], nl[0], nl[1], tan)
            )

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
            value_vars=["cosine", "neutral_loss", "modified_cosine"],
        )
        return similarities, similarities_tanimoto
        #if write:
        #    #to_parquet(write_file)
