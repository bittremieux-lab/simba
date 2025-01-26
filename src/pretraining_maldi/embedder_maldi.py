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
    #PeptideTransformerEncoder,
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


class SharedLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(SharedLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        # inputs shape: (batch_size, inputs, dimension)
        batch_size, num_inputs, dimension = inputs.size()
        # Reshape input tensor to combine batch and input dimensions
        # print(batch_size)
        # print(num_inputs)
        # print(dimension)
        reshaped_inputs = inputs.reshape(-1, dimension)
        # Apply linear layer
        outputs = self.linear(reshaped_inputs)
        # Reshape output tensor back to original shape
        outputs = outputs.view(batch_size, num_inputs)
        return outputs


class EmbedderMaldi(pl.LightningModule):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embed spectra."""

    def __init__(
        self,
        d_model,
        n_layers,
        dropout=0.1,
        weights=None,
        lr=None,
        use_element_wise=True,
        use_cosine_distance=False,  # element wise instead of concat for mixing info between embeddings
    ):
        """Initialize the CCSPredictor"""
        super().__init__()
        self.MAX_N_PEAKS = 100
        self.weights = weights

        # Add a linear layer for projection
        self.use_element_wise = use_element_wise

        # self.linear = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.linear_output = nn.Linear(d_model, self.MAX_N_PEAKS)
        self.shared_linear = SharedLinear(d_model, 1)

        self.relu = nn.ReLU()
        self.spectrum_encoder = SpectrumTransformerEncoderCustom(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.cosine_loss = nn.CosineEmbeddingLoss(0.5)
        self.regression_loss = nn.MSELoss(reduction="none")
        self.dropout = nn.Dropout(p=dropout)
        # self.regression_loss = weighted_MSELoss()

        # Lists to store training and validation loss
        self.train_loss_list = []
        self.val_loss_list = []
        self.lr = lr
        self.use_cosine_distance = use_cosine_distance

    def forward(self, batch):
        """The inference pass"""

        # extra data
        kwargs_0 = {
            # "precursor_mass": batch["precursor_mass_0"].float(),
            # "precursor_charge": batch["precursor_charge_0"].float(),
            "precursor_mass": 0
            * batch[
                "precursor_mass_0"
            ].float(),  # put to zero to not use precursor information
            "precursor_charge": 0 * batch["precursor_charge_0"].float(),
        }
        emb, _ = self.spectrum_encoder(
            mz_array=batch["sampled_mz"].float(),
            intensity_array=batch["sampled_intensity"].float(),
            **kwargs_0
        )

        # get peak embeddings
        emb = emb[:, 1:, :]

        # it returns an embedding with 100 values as output
        emb = self.shared_linear(emb)

        return emb

    def step(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        spec = self(batch)

        # Calculate the loss efficiently:
        # target = torch.tensor(batch["similarity"]).to(self.device)

        target = torch.tensor(batch["flips"]).to(self.device)
        target = target.view(-1, 100)

        # for the no sampled peaks, do not contribute to the error

        # print('spec')
        # print(spec)
        # spec[batch["sampled_mz"]==0]=0
        # print('after zeroing')
        # print(spec)

        # print('original target')
        # print(target)
        # change the no fliped peak to 0, the fliped to 1

        # mask the mz that are 0s
        mask = torch.ones((spec.shape[0], spec.shape[1])).to(self.device)
        mask[batch["sampled_mz"] == 0] = 0
        mask[target == 0] = 0

        # mask the mz that are higher than 0s but the target=0, to not take them into account
        spec = spec * mask

        # rescoring
        target[target == 1] = 0
        target[target == 2] = 1

        # print('after adjustment of target')
        # print('')
        # print('target')
        # print(target)

        # print('spec')
        # print(spec)
        # print('')
        # print('to compute loss')
        # loss = self.cross_entropy(spec.float(), target.view(-1, 100).float()).float()
        loss = F.binary_cross_entropy_with_logits(
            spec.float(), target.view(-1, 100).float()
        ).float()

        # print('loss')
        # print(loss)
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
