
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask
import lightning.pytorch as pl
from simba.transformers.encoder import Encoder
from simba.transformers.load_data_encoder import LoadDataEncoder
from simba.analog_discovery.fc_layers_analog_discovery import FcLayerAnalogDiscovery
from torch.utils.data import DataLoader
from simba.load_mces.load_mces import LoadMCES
import numpy as np
from simba.edit_distance.edit_distance import EditDistance
class Simba:


    def __init__(self, file_path, config):
        self.file_path = file_path
        self.config= config
        self.trainer = pl.Trainer(max_epochs=2, enable_progress_bar=True)
        self.encoder = self.load_encoder(file_path )
        self.model=  self.load_model(file_path)
        

    def load_encoder(self, filepath):
        return Encoder(filepath, D_MODEL=int(self.config.D_MODEL),
                 N_LAYERS=int(self.config.N_LAYERS), 
                 multitasking=True, 
                 config=self.config)

    def load_model(self, file_path,):
        model= EmbedderMultitask.load_from_checkpoint(
            file_path,
            d_model=int(self.config.D_MODEL),
            n_layers=int(self.config.N_LAYERS),
            n_classes=self.config.EDIT_DISTANCE_N_CLASSES,
            use_gumbel=self.config.EDIT_DISTANCE_USE_GUMBEL,
            use_element_wise=True,
            use_cosine_distance=self.config.use_cosine_distance,
            use_edit_distance_regresion=self.config.USE_EDIT_DISTANCE_REGRESSION,
        )
        model.eval()
        return model

    def predict(self, spectra0, spectra1, ):
        
        #create the dataloaders
        dataloader0= self.generate_data_loader( spectra0)
        dataloader1= self.generate_data_loader( spectra1)

        # get dataloader_test
        embeddings0= self.encoder.get_embeddings(dataloader0)
        embeddings1= self.encoder.get_embeddings(dataloader1)

        similarities_ed, similarities_mces=FcLayerAnalogDiscovery.compute_all_combinations(self.file_path, 
                                                                                 embeddings0, 
                                                                                 embeddings1, 
                                                                                 self.config)

        # denormilize
        similarities_ed = (self.config.EDIT_DISTANCE_N_CLASSES-1) - np.argmax(similarities_ed, axis=-1)
        
        similarities_mces= self.config.MCES20_MAX_VALUE*(1-similarities_mces)
        similarities_mces = np.round(similarities_mces)
        return similarities_ed, similarities_mces


    def generate_data_loader(self, spectrums):
        dataset= LoadDataEncoder.from_spectrums_to_dataset(spectrums, max_num_peaks=int(self.config.TRANSFORMER_CONTEXT),)
        dataloader= DataLoader(dataset, batch_size=self.config.BATCH_SIZE,  num_workers=0)
        return dataloader