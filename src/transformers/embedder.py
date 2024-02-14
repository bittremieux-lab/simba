import random

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import ppx
#import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from depthcharge.data import AnnotatedSpectrumDataset
from depthcharge.tokenizers import PeptideTokenizer
from depthcharge.transformers import (
    SpectrumTransformerEncoder, 
    PeptideTransformerEncoder,
)
from src.transformers.spectrum_transformer_encoder_custom import SpectrumTransformerEncoderCustom
import torch
from src.config import Config 
#from dadaptation import DAdaptAdam
# Set our plotting theme:
#sns.set_style("ticks")

# Set random seeds
pl.seed_everything(42, workers=True)


class Embedder(pl.LightningModule):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embed spectra."""
    def __init__(self, d_model, n_layers, dropout=0.1, weights=None, lr=None):
        """Initialize the CCSPredictor"""
        super().__init__()
        self.weights=weights

        # Add a linear layer for projection
        self.linear = nn.Linear(d_model*2+4, 32)
        self.relu= nn.ReLU()
        self.linear_regression = nn.Linear(32, 1)
        
        self.spectrum_encoder = SpectrumTransformerEncoderCustom(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.cosine_loss = nn.CosineEmbeddingLoss(0.5)
        self.regression_loss = nn.MSELoss(reduction='none')
        #self.regression_loss = weighted_MSELoss()

        # Lists to store training and validation loss
        self.train_loss_list = []
        self.val_loss_list = []
        self.lr = lr

    def forward(self, batch):
        """The inference pass"""

        
        # extra data
        kwargs_0 = {'precursor_mass':  batch['precursor_mass_0'].float(), 'precursor_charge': batch['precursor_charge_0'].float()}
        kwargs_1 = {'precursor_mass':  batch['precursor_mass_1'].float(), 'precursor_charge': batch['precursor_charge_1'].float()} 
        emb0,  _ =self.spectrum_encoder(mz_array=batch['mz_0'].float(), intensity_array=batch['intensity_0'].float(),**kwargs_0)
        emb1,  _ =self.spectrum_encoder(mz_array=batch['mz_1'].float(), intensity_array=batch['intensity_1'].float(),**kwargs_1)
        
        emb0 = emb0[:, 0, :]
        emb1 = emb1[:, 0, :]
        
        
        emb = torch.cat((emb0, emb1), dim=1)

        
        # stack global features
        mass_0 = batch['precursor_mass_0'].float()
        charge_0 = batch['precursor_charge_0'].float()
        mass_1 = batch['precursor_mass_1'].float()
        charge_1 = batch['precursor_charge_1'].float()



        emb = torch.cat((emb, mass_0, charge_0, mass_1, charge_1), dim=1)
        emb = self.linear(emb)
        emb = self.relu(emb)
        emb=  self.linear_regression(emb) 
        
        return emb
    
    def step(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        spec = self(batch)

        
        # Calculate the loss efficiently:
        target = torch.tensor(batch['similarity']).to(self.device)
        target = target.view(-1)
        
        # apply weight loss


        #weight = 2*torch.abs(target.view(-1, 1)-0.5)
        weight=1

        #print('to compute loss')
        loss = self.regression_loss(spec.float(), target.view(-1, 1).float()).float()
        loss = torch.mean(torch.mul(loss, weight))
        
        #print(loss)
        return loss.float()

    def training_step(self, batch, batch_idx):
        """A training step"""
        loss = self.step(batch, batch_idx)
        #self.train_loss_list.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """A validation step"""
        loss = self.step(batch, batch_idx)
        #self.val_loss_list.append(loss.item())
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        """A predict step"""
        spec = self(batch)
        return spec

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        #optimizer = DAdaptAdam(self.parameters(), lr=1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.RAdam(self.parameters(), lr=1e-3)
        return optimizer
    

