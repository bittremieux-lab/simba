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


class Embedder(pl.LightningModule):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embed spectra."""

    def __init__(self, d_model, n_layers, dropout=0.1, weights=None, lr=None, 
                 use_element_wise=True, use_cosine_distance=False, #element wise instead of concat for mixing info between embeddings
                 ):
        """Initialize the CCSPredictor"""
        super().__init__()
        self.weights = weights

        # Add a linear layer for projection
        self.use_element_wise = use_element_wise
        if self.use_element_wise:
            self.linear = nn.Linear(d_model, d_model)
            self.linear_regression = nn.Linear(d_model, 1)
        else:
            self.linear = nn.Linear(d_model * 2 + 4, 32)
            self.linear_regression = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
        
        self.spectrum_encoder = SpectrumTransformerEncoderCustom(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.regression_loss = nn.MSELoss(reduction="none")
        self.dropout = nn.Dropout(p=dropout)
        # self.regression_loss = weighted_MSELoss()

        # Lists to store training and validation loss
        self.train_loss_list = []
        self.val_loss_list = []
        self.lr = lr
        self.use_cosine_distance=use_cosine_distance
        if self.use_cosine_distance:
            self.linear_cosine = nn.Linear(d_model, d_model)
    def normalized_dot_product(self,a, b):
        # Normalize inputs
        a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)
        
        # Compute dot product
        dot_product = torch.sum(a_norm * b_norm, dim=-1)
        return dot_product

    def forward(self, batch):
        """The inference pass"""

        # extra data
        kwargs_0 = {
            "precursor_mass": batch["precursor_mass_0"].float(),
            "precursor_charge": batch["precursor_charge_0"].float(),
        }
        kwargs_1 = {
            "precursor_mass": batch["precursor_mass_1"].float(),
            "precursor_charge": batch["precursor_charge_1"].float(),
        }
        emb0, _ = self.spectrum_encoder(
            mz_array=batch["mz_0"].float(),
            intensity_array=batch["intensity_0"].float(),
            **kwargs_0
        )
        emb1, _ = self.spectrum_encoder(
            mz_array=batch["mz_1"].float(),
            intensity_array=batch["intensity_1"].float(),
            **kwargs_1
        )

        emb0 = emb0[:, 0, :]
        emb1 = emb1[:, 0, :]

        if self.use_cosine_distance:
            # Normalize input tensors
            #input0_normalized = F.normalize(emb0, p=2, dim=1)
            #input1_normalized = F.normalize(emb1, p=2, dim=1)

            # Compute cosine similarity
            #emb = F.cosine_similarity(input0_normalized, input1_normalized)

            # apply a linear function to the embeddings

            emb = self.normalized_dot_product(emb0, emb1)
            emb = (emb + 1)/2 # change range from  -1-1 to 0-1
        else:
            emb = emb0 + emb1
            emb = self.linear(emb)
            emb = self.dropout(emb)
            emb = self.relu(emb)
            emb = self.linear_regression(emb)

        return emb

    def step(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        spec = self(batch)

        # Calculate the loss efficiently:
        

        target = torch.tensor(batch["similarity"]).to(self.device)
        target = target.view(-1)

    
        # apply weight loss
        weight = 1
        loss = self.regression_loss(spec.float(), target.view(-1, 1).float()).float()
        loss = torch.mean(torch.mul(loss, weight))
        

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


    def load_weights(self):
        weights={}
        for name, param in self.named_parameters():
                    weights[name]= np.array(param.data)
        return weights
    
    def load_pretrained_maldi_embedder(self, model_path):
        # original weights
        original_weights = self.load_weights()

        # Load weights from the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu',)

        # Load weights into model B from the checkpoint
        checkpoint_keys = checkpoint['state_dict'].keys()
        original_embedder_keys = self.state_dict().keys()  # Assuming `model` is your target model

        # Load weights for shared layers
        for key in checkpoint_keys:
            if key in original_embedder_keys:
                self.state_dict()[key].copy_(checkpoint['state_dict'][key])

        # new weights
        new_weights = self.load_weights()
        
        ## sanity check (the weights of the model changed?):
        if not(self.are_weights_changed(original_weights, new_weights)):
            print('INFO: Correctly loaded pretrained Maldi Model')
        else:
            raise ValueError('ERROR!!!: Error loading Maldi model')

    def are_weights_changed(self,original_weights, new_weights, layer_test='spectrum_encoder.transformer_encoder.layers.0.norm2.bias'):
        return np.array_equal(original_weights[layer_test], new_weights[layer_test])
    


