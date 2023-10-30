from src.molecule_pair import MoleculePair
from typing import List
from  src.similarity import cosine, modified_cosine, neutral_loss
from src.config import Config
from tqdm import tqdm 
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
