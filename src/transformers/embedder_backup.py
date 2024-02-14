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
    """Embed spectra and peptides in the same space."""

    def __init__(self, d_model, n_layers):
        """Initialize the CCSPredictor"""
        super().__init__()

        self.spectrum_encoder = SpectrumTransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
        )

        self.cosine_loss = nn.CosineEmbeddingLoss(0.5)

    def forward(self, batch):
        """The inference pass"""
        emb, _ = self.spectrum_encoder(batch[0].float())

        return emb[:, 0, :]

    def step(self, batch, batch_idx):
        """A training/validation/inference step."""
        spec = self(batch)

        # Calculate the loss efficiently:
        spec = spec.expand(spec.shape).reshape(-1, *spec.shape[1:])
        target = torch.ones(spec.shape[0]).to(self.device)
        target[spec.shape[0] :] = -1
        loss = self.cosine_loss(spec, spec, target)
        return loss

    def training_step(self, batch, batch_idx):
        """A training step"""
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """A validation step"""
        loss = self.step(batch, batch_idx)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """A predict step"""
        spec = self(batch, is_spectra=True)
        return spec

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
