import random

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
# import ppx
# import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from depthcharge.data import AnnotatedSpectrumDataset
from depthcharge.tokenizers import PeptideTokenizer
from depthcharge.transformers import (
    SpectrumTransformerEncoder,
    PeptideTransformerEncoder,
)
from src.transformers.spectrum_transformer_encoder_custom import (
    SpectrumTransformerEncoderCustom,
)
import torch
from src.config import Config
# from dadaptation import DAdaptAdam
# Set our plotting theme:
# sns.set_style("ticks")

# Set random seeds
pl.seed_everything(42, workers=True)


class EmbedderMaldi(pl.LightningModule):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embed spectra."""

    def __init__(self, d_model, n_layers, dropout=0.1, weights=None, lr=None, 
                 use_element_wise=True, use_cosine_distance=False, #element wise instead of concat for mixing info between embeddings
                 ):
        """Initialize the CCSPredictor"""
        super().__init__()
        self.MAX_N_PEAKS=100
        self.weights = weights

        # Add a linear layer for projection
        self.use_element_wise = use_element_wise

        #self.linear = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.linear_output = nn.Linear(d_model, self.MAX_N_PEAKS)
    
        self.relu = nn.ReLU()
        self.spectrum_encoder = SpectrumTransformerEncoderCustom(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.cross_entropy= nn.CrossEntropyLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss(0.5)
        self.regression_loss = nn.MSELoss(reduction="none")
        self.dropout = nn.Dropout(p=dropout)
        # self.regression_loss = weighted_MSELoss()

        # Lists to store training and validation loss
        self.train_loss_list = []
        self.val_loss_list = []
        self.lr = lr
        self.use_cosine_distance=use_cosine_distance
        

    def forward(self, batch):
        """The inference pass"""

        # extra data
        kwargs_0 = {
            "precursor_mass": batch["precursor_mass_0"].float(),
            "precursor_charge": batch["precursor_charge_0"].float(),
        }
        emb, _ = self.spectrum_encoder(
            mz_array=batch["sampled_mz"].float(),
            intensity_array=batch["sampled_intensity"].float(),
            **kwargs_0
        )


        emb = emb[:, 0, :]
        emb = self.linear(emb)
        emb = self.dropout(emb)
        emb = self.relu(emb)
        emb = self.linear_output(emb)
        emb = self.relu(emb)
        
        return emb

    def step(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        spec = self(batch)

        # Calculate the loss efficiently:
        #target = torch.tensor(batch["similarity"]).to(self.device)
        target = torch.tensor(batch["flips"]).to(self.device)
        target = target.view(-1,100)

        # print('to compute loss')
        loss = self.cross_entropy(spec.float(), target.view(-1, 100).float()).float()

        # print(loss)
        return loss.float()

    def training_step(self, batch, batch_idx):
        """A training step"""
        loss = self.step(batch, batch_idx)
        # self.train_loss_list.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """A validation step"""
        loss = self.step(batch, batch_idx)
        # self.val_loss_list.append(loss.item())
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """A predict step"""
        spec = self(batch)
        return spec

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        # optimizer = DAdaptAdam(self.parameters(), lr=1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.RAdam(self.parameters(), lr=1e-3)
        return optimizer
