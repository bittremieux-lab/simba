from src.ordinal_classification.embedder_multitask import EmbedderMultitask
import lightning.pytorch as pl
import pandas as pd
import numpy as np
from src.molecule_pairs_opt import MoleculePairsOpt
from src.ordinal_classification.load_data_multitasking import LoadDataMultitasking
from torch.utils.data import DataLoader
from tqdm import tqdm 

class ADMultitask:
    '''
    using edit distance as a way to filter not good candidates
    '''

    def __init__(self, model_path, config):
        self.trainer = pl.Trainer(max_epochs=2, enable_progress_bar=True)
        self.config=config
        self.multitask_model =EmbedderMultitask.load_from_checkpoint(
                model_path,
                d_model=int(config.D_MODEL),
                n_layers=int(config.N_LAYERS),
                weights=None,
                n_classes=config.EDIT_DISTANCE_N_CLASSES,
                use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
                lr=config.LR,
                use_cosine_distance=config.use_cosine_distance,
            )
    def compute_entropy(self, vector):
        # Add a small value to avoid log(0)
        vector = vector + 1e-12
        return -np.sum(vector * np.log(vector), axis=1)
        
    def create_input_data_from_spectrums(self,spectrum_query, spectrums_candidates):
        '''
        from the spectrums create the data object expected by the model
        '''

        df_smiles=pd.DataFrame()
        df_smiles.index = [index for index in range(0, len(spectrums_candidates)+1)]
        df_smiles['indexes']= [[index] for index in df_smiles.index ]

        # make all the possible pairs between spectrum query and candidates
        indexes_tani = np.array( [[0, index, 0] for index in range(1,len(spectrums_candidates)+1)])

        pair_temp = MoleculePairsOpt(spectrums_original= [spectrum_query] + spectrums_candidates, 
                 spectrums_unique=[spectrum_query] + spectrums_candidates, 
                 df_smiles=df_smiles, 
                 indexes_tani_unique=indexes_tani, 
                             tanimotos= np.zeros(len(spectrums_candidates)))
        return pair_temp
    

    def create_data_loader(self,spectrum_query, spectrums_candidates, ):
        pair_temp = self.create_input_data_from_spectrums(spectrum_query, spectrums_candidates
                                                          )
        dataset_test = LoadDataMultitasking.from_molecule_pairs_to_dataset(pair_temp)
        return DataLoader(dataset_test, batch_size=self.config.BATCH_SIZE, shuffle=False)
    
    def softmax(self,x):
        e_x = np.exp(x)  # Subtract max(x) for numerical stability
        return e_x / e_x.sum(axis=1)[:, np.newaxis]

    def get_edit_distance(self,spectrum_query, spectrums_candidates, ):
        '''
        return the probability that the pair has a edit distance>5
        '''
        dataloader_test= self.create_data_loader(spectrum_query, spectrums_candidates, )
        pred_test = self.trainer.predict(
            self.multitask_model,
            dataloader_test,
        )

        softmax_edit_distance= pred_test[0][0].numpy()
        softmax_edit_distance= self.softmax(softmax_edit_distance)

        #return self.compute_entropy(softmax_edit_distance)
        #return np.array([s[0] for s in softmax_edit_distance])
        return softmax_edit_distance
    
    def get_edit_distance_all(self, spectrum_query, spectrums_candidates,):
        '''
        receive a list of candidates and queries
        '''
        edit_distance_all=[]
        for index, spec_query in tqdm(enumerate(spectrum_query)):
                edit_distance= self.get_edit_distance(spec_query, spectrums_candidates[index], )
                edit_distance_all.append(edit_distance)

        return edit_distance_all