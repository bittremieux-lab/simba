


class MoleculePair:

    def __init__(self, vector_0=None, 
                        vector_1=None, 
                        smiles_0=None, 
                        smiles_1=None, 
                        similarity=None, 
                        global_feats_0=None, 
                        global_feats_1=None,
                        index_in_spectrum_0=None,
                        index_in_spectrum_1=None,
                        spectrum_object_0=None,
                        spectrum_object_1=None,
                        params_0=None,
                        params_1=None):
        
        self.spectrum_object_0=spectrum_object_0
        self.spectrum_object_1 = spectrum_object_1
        self.vector_0 = vector_0
        self.vector_1 = vector_1
        self.global_feats_0 = global_feats_0
        self.global_feats_1 = global_feats_1
        self.smiles_0 = smiles_0
        self.smiles_1 = smiles_1
        self.similarity= similarity
        self.index_in_spectrum_0 = index_in_spectrum_0
        self.index_in_spectrum_1 = index_in_spectrum_1
        self.params_0 = params_0
        self.params_1= params_1

        self.cosine_score={}

    def set_det_similarity_score(self,score, similarity_score):
        self.deterministic_similarity[similarity_score]=score