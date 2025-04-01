
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask
import lightning.pytorch as pl
from simba.transformers.encoder import Encoder
from simba.transformers.load_data_encoder import LoadDataEncoder
from simba.analog_discovery.fc_layers_analog_discovery import FcLayerAnalogDiscovery
from torch.utils.data import DataLoader
from simba.load_mces.load_mces import LoadMCES
import numpy as np
from simba.edit_distance.edit_distance import EditDistance

import time 
class Simba:


    def __init__(self, file_path, config,
                    device='gpu', cache_embeddings=True):
        self.file_path = file_path
        self.config= config
        self.trainer = pl.Trainer(max_epochs=2, enable_progress_bar=True, accelerator=device,)
        self.encoder = self.load_encoder(file_path )
        self.model=  self.load_model(file_path)
        self.cache_embeddings=cache_embeddings
        self.device=device
        self._embedding_cache = {}


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

    def get_dataloader_key(self, dataloader):
        
        # Example: based on dataset size and transform config
        dataset = dataloader.dataset
        print('hash:')
        print( dir(dataset))


        hash_string=""
        for attr in dir(dataset):
            if not attr.startswith("_"):  # Skip private/internal attributes
                hash_string =hash_string + str(dataset.attr)
        try:
            value = getattr(dataset, attr)
            print(f"{attr}: {value}")
        except Exception:
            pass
        try:
            return hash((
                len(dataset),
                hash_string,
            ))
        except Exception:
            return None

    def encoder_embeddings(self, dataloader):
        print('running')
        cache_key = self.get_dataloader_key(dataloader)
        if self.cache_embeddings and (cache_key in self._embedding_cache):
            embeddings= self._embedding_cache[cache_key]
            print('Using CACHE embeddings')
        else:
            print('Processing embeddings ...')
            embeddings = self.encoder.get_embeddings(dataloader, device= self.device)
            self._embedding_cache[cache_key] = embeddings
        return embeddings

    def predict(self, spectra0, spectra1, ):
        
        #create the dataloaders
        dataloader0= self.generate_data_loader( spectra0)
        dataloader1= self.generate_data_loader( spectra1)

        
        embeddings0= self.encoder_embeddings(dataloader0)
        embeddings1= self.encoder_embeddings(dataloader1)

        start= time.time()
        similarities_ed, similarities_mces=FcLayerAnalogDiscovery.compute_all_combinations(self.file_path, 
                                                                                 embeddings0, 
                                                                                 embeddings1, 
                                                                                 self.config)
        end=time.time()
        elapsed_time =end-start
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        # denormilize
        similarities_ed = (self.config.EDIT_DISTANCE_N_CLASSES-1) - np.argmax(similarities_ed, axis=-1)
        
        similarities_mces= self.config.MCES20_MAX_VALUE*(1-similarities_mces)
        #similarities_mces = np.round(similarities_mces)
        return similarities_ed, similarities_mces


    def generate_data_loader(self, spectrums):
        dataset= LoadDataEncoder.from_spectrums_to_dataset(spectrums, max_num_peaks=int(self.config.TRANSFORMER_CONTEXT),)
        dataloader= DataLoader(dataset, batch_size=self.config.BATCH_SIZE,  num_workers=0)
        return dataloader