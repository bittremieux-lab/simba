
from src.transformers.embedder import Embedder
from src.ordinal_classification import EmbedderMultitask
import torch.nn as nn
import lightning.pytorch as pl
import numpy as np 
import torch
class Encoder(pl.LightningModule):

    def __init__(self, model_path, D_MODEL, N_LAYERS, multitasking=False):
        super().__init__()
        self.multitasking=multitasking
        self.model= self.load_twin_network(model_path,D_MODEL, N_LAYERS).spectrum_encoder
        self.relu = nn.ReLU()

    def load_twin_network(self, model_path, D_MODEL, N_LAYERS, LR=0.001, use_cosine_distance=True):
        embedder_class = EmbedderMultitask if self.multitasking else Embedder
        return embedder_class.load_from_checkpoint(
        model_path,
        d_model=int(D_MODEL),
        n_layers=int(N_LAYERS),
        weights=None,
        lr=LR,
        use_cosine_distance=use_cosine_distance,

    )
    
    def forward(self, batch):
        """The inference pass"""

        # extra data
        kwargs = {
            "precursor_mass": batch["precursor_mass"].float(),
            "precursor_charge": batch["precursor_charge"].float(),
        }

        emb, _ = self.model(
            mz_array=batch["mz"].float(),
            intensity_array=batch["intensity"].float(),
            **kwargs,
        )

        emb = emb[:, 0, :]
        emb = self.relu(emb)

        return emb
    
    def get_embeddings(self, dataloader_spectrums):
        predictor = pl.Trainer(max_epochs=0, enable_progress_bar=True)
        embeddings = predictor.predict(
            self,
            dataloader_spectrums,
        )
        return self.flat_predictions(embeddings)
    
    def flat_predictions(self,preds):
        # flat the results
        concatenated_tensor = torch.cat(preds, dim=0)
        return concatenated_tensor.detach().numpy()