
from src.transformers.embedder import Embedder
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
import torch


class EmbedderFingerprint(Embedder):
    def __init__(self, d_model, n_layers, dropout=0.3, weights=None, fingerprint_size=128):
        """Initialize the CCSPredictor"""
        super().__init__(d_model, n_layers, dropout, weights)
        self.weights=weights
        self.d_model = d_model
        self.n_layers=n_layers
        # Add a linear layer for projection
        self.linear = nn.Linear(d_model*2+4, 32)
        self.relu= nn.ReLU()
        self.linear_regression = nn.Linear(32, fingerprint_size)

        self.spectrum_encoder = SpectrumTransformerEncoder(
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

    def forward(self, batch):
        """The inference pass"""



        emb0,  _ =self.spectrum_encoder(mz_array=batch['mz_0'].float(), intensity_array=batch['intensity_0'].float())
        emb1,  _ =self.spectrum_encoder(mz_array=batch['mz_1'].float(), intensity_array=batch['intensity_1'].float())

        emb0 = emb0[:, 0, :]
        emb1 = emb1[:, 0, :]


        emb = torch.cat((emb0, emb1), dim=1)


        # stack global features
        #mass_0 = batch['precursor_mass_0'].float()
        #charge_0 = batch['precursor_charge_0'].float()
        #mass_1 = batch['precursor_mass_1'].float()
        #charge_1 = batch['precursor_charge_1'].float()


        
        #emb = torch.cat((emb, mass_0, charge_0, mass_1, charge_1), dim=1)
        #emb = self.linear(emb)
        #emb = self.relu(emb)
        #emb=  self.linear_regression(emb)

        return emb

    def step(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        spec = self(batch)


        # Calculate the loss efficiently:
        target = torch.tensor(batch['fingerprint']).to(self.device)
        target = target.view(-1)

        # apply weight loss


        #weight = 2*torch.abs(target.view(-1, 1)-0.5)
        weight=1

        #print('to compute los
        loss = F.cross_entropy(spec.float(), target.view(-1,128).float())
        #loss = self.regression_loss(spec.float(), target.view(-1, 1).float()).float()
        #loss = torch.mean(torch.mul(loss, weight))

        #print(loss)
        return loss.float()

     

