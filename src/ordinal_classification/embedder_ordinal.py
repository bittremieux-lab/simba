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

from src.transformers.embedder import Embedder


class EmbedderOrdinal(Embedder):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embed spectra."""

    def __init__(
        self,
        d_model,
        n_layers,
        n_classes, 
        dropout=0.1,
        weights=None,
        lr=None,
        use_element_wise=True,
        use_cosine_distance=True,  # element wise instead of concat for mixing info between embeddings
        # Number of classes for classification
    ):
        """Initialize the CCSPredictor"""
        super().__init__(
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        weights=weights,
        lr=lr,
        use_element_wise=use_element_wise,
        use_cosine_distance=use_cosine_distance,)  # element wise instead of concat for mixing info between embeddings)
        self.weights = weights

        # Add a linear layer for projection
        self.classifier = nn.Linear(d_model, n_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=dropout)

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
            **kwargs_0,
        )
        emb1, _ = self.spectrum_encoder(
            mz_array=batch["mz_1"].float(),
            intensity_array=batch["intensity_1"].float(),
            **kwargs_1,
        )

        emb0 = emb0[:, 0, :]
        emb1 = emb1[:, 0, :]

        emb0 = self.relu(emb0)
        emb1 = self.relu(emb1)

        emb = emb0 + emb1
        emb = self.classifier(emb)
        return emb

    def step(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        logits = self(batch)

        target = torch.tensor(batch["similarity"]).to(self.device)
        target = target.view(-1).long()  # Ensure targets are in the right shape and type for classification

        #loss = self.loss_fn(logits, target)
        #loss = self.cumulative_logits_loss(logits, target)
        loss= self.ordinal_cross_entropy(logits, target)
        return loss
    
    def cumulative_logits_loss(self, outputs, targets):
        # Outputs shape: (batch_size, num_classes)
        # Targets shape: (batch_size,)
        
        num_classes = outputs.size(1)
        targets = targets.view(-1, 1)  # Reshape targets to (batch_size, 1)
        
        # Generate matrix of cumulative logits
        M = torch.arange(num_classes, device=outputs.device).view(1, -1)
        
        # Compute cumulative logits loss
        loss = torch.sum(torch.log(1 + torch.exp(outputs - M)), dim=1)
        
        return torch.mean(loss)

    def ordinal_cross_entropy(self, pred, target):
        """
        pred: Tensor of shape (batch_size, num_classes)
              The predicted probabilities for each class.
        target: Tensor of shape (batch_size,)
                The target ordinal labels (0 to num_classes-1).

        Returns the loss value as a scalar Tensor.
        """

        # Calculate the ordinal target matrix
        batch_size = pred.size(0)
        num_classes = pred.size(1)
        
        target_matrix = torch.zeros_like(pred, dtype=torch.float)
        for i in range(batch_size):
            target_matrix[i, :target[i]+1] = 1.0
        
        # Compute the ordinal cross-entropy loss
        loss = -torch.sum(target_matrix * F.log_softmax(pred, dim=1), dim=1).mean()

        return loss
    
    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
  
