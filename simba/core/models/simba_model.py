import time

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader

from simba.analog_discovery.fc_layers_analog_discovery import (
    FcLayerAnalogDiscovery,
)
from simba.core.models.ordinal.embedder_multitask import EmbedderMultitask
from simba.core.models.transformers.encoder import Encoder
from simba.core.models.transformers.load_data_encoder import LoadDataEncoder


class Simba:
    def __init__(self, file_path, config, device="gpu", cache_embeddings=True):
        self.file_path = file_path
        self.config = config
        self.device = device
        self.trainer = pl.Trainer(
            max_epochs=2,
            enable_progress_bar=True,
            accelerator=device,
        )
        self.encoder = self.load_encoder(file_path)
        self.model = self.load_model(file_path)
        self.cache_embeddings = cache_embeddings
        self._embedding_cache = {}

    def load_encoder(self, filepath):
        d_model = int(self.config.model.transformer.d_model)
        n_layers = int(self.config.model.transformer.n_layers)

        return Encoder(
            filepath,
            D_MODEL=d_model,
            N_LAYERS=n_layers,
            multitasking=True,
            config=self.config,
        )

    def load_model(
        self,
        file_path,
    ):
        d_model = int(self.config.model.transformer.d_model)
        n_layers = int(self.config.model.transformer.n_layers)
        n_classes = self.config.model.tasks.edit_distance.n_classes
        use_gumbel = self.config.model.tasks.edit_distance.use_gumbel
        use_cosine_distance = (
            self.config.model.tasks.cosine_similarity.use_cosine_distance
        )
        use_edit_distance_regression = (
            self.config.model.tasks.edit_distance.use_regression
        )
        use_fingerprint = self.config.model.tasks.fingerprints.enabled
        use_learnable_multitask = self.config.model.multitasking.learnable

        model = EmbedderMultitask.load_from_checkpoint(
            file_path,
            d_model=d_model,
            n_layers=n_layers,
            n_classes=n_classes,
            use_gumbel=use_gumbel,
            use_element_wise=True,
            use_cosine_distance=use_cosine_distance,
            use_edit_distance_regresion=use_edit_distance_regression,
            use_fingerprints=use_fingerprint,
            USE_LEARNABLE_MULTITASK=use_learnable_multitask,
            strict=False,
        )
        model.eval()

        if self.device == "cpu":
            model = model.cpu()
        elif self.device == "gpu":
            model = model.cuda()

        return model

    def get_dataloader_key(self, dataloader):
        # Example: based on dataset size and transform config
        dataset = dataloader.dataset

        hash_string = ""
        for attr in dir(dataset):
            if not attr.startswith("_"):  # Skip private/internal attributes
                value = getattr(dataset, attr)
                hash_string += str(value)

        try:
            value = getattr(dataset, hash_string)
            print(f"{attr}: {value}")
        except Exception:
            pass
        try:
            return hash(
                (
                    len(dataset),
                    hash_string,
                )
            )
        except Exception:
            return None

    def encoder_embeddings(self, dataloader):
        print("running")
        cache_key = self.get_dataloader_key(dataloader)
        if self.cache_embeddings and (cache_key in self._embedding_cache):
            embeddings = self._embedding_cache[cache_key]
            print("Using CACHE embeddings")
        else:
            print("Processing embeddings ...")
            embeddings = self.encoder.get_embeddings(dataloader, device=self.device)
            self._embedding_cache[cache_key] = embeddings
        return embeddings

    def predict(
        self,
        spectra0,
        spectra1,
    ):
        # create the dataloaders
        dataloader0 = self.generate_data_loader(spectra0)
        dataloader1 = self.generate_data_loader(spectra1)

        embeddings0 = self.encoder_embeddings(dataloader0)
        embeddings1 = self.encoder_embeddings(dataloader1)

        start = time.time()
        similarities_ed, similarities_mces = (
            FcLayerAnalogDiscovery.compute_all_combinations(
                self.file_path, embeddings0, embeddings1, self.config
            )
        )
        end = time.time()
        elapsed_time = end - start
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        edit_distance_n_classes = self.config.model.tasks.edit_distance.n_classes
        mces20_max_value = self.config.model.tasks.mces.max_value

        # denormilize
        similarities_ed = (edit_distance_n_classes - 1) - np.argmax(
            similarities_ed, axis=-1
        )

        similarities_mces = mces20_max_value * (1 - similarities_mces)
        # similarities_mces = np.round(similarities_mces)
        return similarities_ed, similarities_mces

    def generate_data_loader(self, spectrums):
        transformer_context = int(self.config.model.transformer.context_length)
        batch_size = self.config.training.batch_size

        dataset = LoadDataEncoder.from_spectrums_to_dataset(
            spectrums,
            max_num_peaks=transformer_context,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        return dataloader
