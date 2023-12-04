import random

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ppx
import seaborn as sns
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

# Set our plotting theme:
sns.set_style("ticks")

# Set random seeds
pl.seed_everything(42, workers=True)

class Embedder(pl.LightningModule):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embed spectra."""
    def __init__(self, d_model, n_layers):
        """Initialize the CCSPredictor"""
        super().__init__()
        self.d_model = d_model
        self.n_layers=n_layers
        # Add a linear layer for projection
        self.linear_regression = nn.Linear(d_model * 2, 1)
        
        self.spectrum_encoder = SpectrumTransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
        )

        self.cosine_loss = nn.CosineEmbeddingLoss(0.5)
        self.regression_loss = nn.MSELoss()
        
        # Lists to store training and validation loss
        self.train_loss_list = []
        self.val_loss_list = []
        
    def forward(self, batch):
        """The inference pass"""

        #emb, _ = self.spectrum_encoder(mz_array= batch[0][:,:,0].float(), intensity_array = batch[0][:,:,1].float())
        #print(batch)

        #print('inputs')
        #print(batch['intensity_0'].size())
        #print(batch['intensity_1'].size())
        emb0,  _ =self.spectrum_encoder(mz_array=batch['mz_0'].float(), intensity_array=batch['intensity_0'].float())
        emb1,  _ =self.spectrum_encoder(mz_array=batch['mz_1'].float(), intensity_array=batch['intensity_1'].float())
        
        #concatenate
        #print('embeddings')
        #print(emb0.size())
        #print(emb1.size())
        # select the dimension where the data is embedded
        emb0 = emb0[:, 0, :]
        emb1 = emb1[:, 0, :]
        
        
        emb = torch.cat((emb0, emb1), dim=1)

        ## for regression problem
        emb= self.linear_regression(emb)
        
        return emb
    
    def step(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        spec = self(batch)

        
        # Calculate the loss efficiently:
        '''
        spec = spec.expand(spec.shape).reshape(-1, *spec.shape[1:])
        
        #print(spec.shape)
        # obtain the embedding for the first molecule and the corresponding second molecule per pair
        #print('spec shape')
        #print(spec.shape)
        spec_0 = spec[:, :int(spec.shape[1]/2)]
        spec_1 = spec[:, int(spec.shape[1]/2):]
        
       
        
        # thresholding
        target[target > threshold] = 1
        target[target <= threshold] = -1
        
        #print('spec0_size')
        #print(spec_0.size())
        #print('spec1_size')
        #print(spec_1.size())
        #print(target.size())
        
        loss_cos = self.cosine_loss(spec_0, spec_1, target)
        '''
        target = torch.tensor(batch['similarity']).to(self.device)
        target = target.view(-1)
        
        #print('to compute loss')
        loss = self.regression_loss(spec.float(), target.view(-1, 1).float()).float()
        #print(loss)
        return loss.float()

    def training_step(self, batch, batch_idx):
        """A training step"""
        loss = self.step(batch, batch_idx)
        self.train_loss_list.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """A validation step"""
        loss = self.step(batch, batch_idx)
        self.val_loss_list.append(loss.item())
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    

    def predict_step(self, batch, batch_idx):
        """A predict step"""
        spec = self(batch)
        return spec

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def plot_loss(self):
        # Reshape the array into a 2D array with 10 columns (adjust batch_per_epoch as needed)
        reshaped_tr = np.array(self.train_loss_list).reshape(-1, 1)
        reshaped_val = np.array(self.val_loss_list).reshape(-1, 1)

        # Calculate the mean along axis 1 (across columns)
        average_tr = np.mean(reshaped_tr, axis=1)
        average_val = np.mean(reshaped_val, axis=1)

        plt.plot(average_tr, label='train')
        plt.plot(average_val, label='val')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.grid()
        plt.show()