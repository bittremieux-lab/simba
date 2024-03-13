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


class FixedLinearRegression(nn.Module):
    '''
    linear layer for computing sum of dot product
    '''
    def __init__(self, d_model):
        super(FixedLinearRegression, self).__init__()
        self.weight = nn.Parameter(
            torch.ones(1, d_model)
        )  # Fixed weight initialized to 1
        self.bias = nn.Parameter(torch.zeros(1))  # Bias initialized to 0

        # Freeze the parameters
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


class Embedder(pl.LightningModule):
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
        self.weights = weights

        # Add a linear layer for projection
        self.use_element_wise = use_element_wise
        self.linear = nn.Linear(d_model, d_model)
        self.linear_regression = nn.Linear(d_model, 1)
        self.fixed_linear_regression = FixedLinearRegression(d_model)

        self.relu = nn.ReLU()

        self.spectrum_encoder = SpectrumTransformerEncoderCustom(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.regression_loss = nn.MSELoss()
        self.dropout = nn.Dropout(p=dropout)

        self.train_loss_list = []
        self.val_loss_list = []
        self.lr = lr
        self.use_cosine_distance = use_cosine_distance
        if self.use_cosine_distance:
            self.linear_cosine = nn.Linear(d_model, d_model)

        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def normalized_dot_product(self, a, b):
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
            emb0_l2 = torch.norm(emb0, p=2, dim=-1, keepdim=True)
            emb1_l2 = torch.norm(emb1, p=2, dim=-1, keepdim=True)
            emb = (emb0 * emb1) / (emb0_l2 * emb1_l2)
            emb = self.fixed_linear_regression(emb)
            #emb = self.cosine_similarity(emb0,emb1)
            emb = (emb+1)/2
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

        target = torch.tensor(batch["similarity"]).to(self.device)
        target = target.view(-1)

        # adjust scale
        #target = 2*(target-0.5)
        loss = self.regression_loss(spec.float(), target.view(-1, 1).float()).float()

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
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = np.array(param.data)
        return weights

    def load_pretrained_maldi_embedder(self, model_path):
        # original weights
        original_weights = self.load_weights()

        # Load weights from the checkpoint
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

        # Load weights into model B from the checkpoint
        checkpoint_keys = checkpoint["state_dict"].keys()
        original_embedder_keys = (
            self.state_dict().keys()
        )  # Assuming `model` is your target model

        # Load weights for shared layers
        for key in checkpoint_keys:
            if key in original_embedder_keys:
                self.state_dict()[key].copy_(checkpoint["state_dict"][key])

        # new weights
        new_weights = self.load_weights()

        ## sanity check (the weights of the model changed?):
        if not (self.are_weights_changed(original_weights, new_weights)):
            print("INFO: Correctly loaded pretrained Maldi Model")
        else:
            raise ValueError("ERROR!!!: Error loading Maldi model")

    def are_weights_changed(
        self,
        original_weights,
        new_weights,
        layer_test="spectrum_encoder.transformer_encoder.layers.0.norm2.bias",
    ):
        return np.array_equal(original_weights[layer_test], new_weights[layer_test])

    def set_freeze_layers(self, layer_names_to_freeze, freeze):
        # Freeze specified layers
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names_to_freeze):
                param.requires_grad = not (freeze)
            else:
                param.requires_grad = True

    def get_maldi_embedder_keys(self, model_path):
        # Load weights from the checkpoint
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

        # Load weights into model B from the checkpoint
        return checkpoint["state_dict"].keys()

    def get_all_keys(self):
        return self.state_dict().keys()
