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
from src.ordinal_classification.ordinal_classification import OrdinalClassification


class CustomizedCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes=6):
        super(CustomizedCrossEntropyLoss, self).__init__()
        penalty_matrix = [[20, 0, 0, 0, 0, 0,],
                          
                          [0, 20, 4, 3, 2, 1,] ,

                          [0, 4, 20, 4, 3, 2,],

                          [0, 3, 4,20, 4, 3,],

                          [0, 2, 3, 4, 20, 4,],

                          [0, 1, 2, 3, 4, 20,]]
        
        #normalize:
        #penalty_matrix = penalty_matrix/(np.sum(penalty_matrix, axis=0))
        
        #penalty_matrix = [[0, 1, 2, 3, 4, 5,],
        #                  [1, 0, 1, 2, 3, 4,],
        #                  [2, 1 ,0, 1, 2, 3,],
        #                  [3, 2, 1, 0, 1, 2,],
        #                  [4, 3, 2, 1, 0, 1,],
        #                  [5, 4, 3, 2, 1, 0,]]
        
        self.n_classes=n_classes
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.penalty_matrix = torch.tensor(penalty_matrix).to(self.device)/20

    def forward(self, logits, target):
        batch_size = logits.size(0)
        
        # Calculate softmax
        #exp_logits = torch.exp(logits)
        #softmax = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
        softmax = F.softmax(logits, dim=-1).to(self.device)
        # Convert target to one-hot encoded tensor
        #one_hot_target = torch.zeros_like(softmax)
        #one_hot_target[torch.arange(batch_size).to(torch.int64), target.to(torch.int64)] = 1
        
        #one_hot_target = torch.nn.functional.one_hot(target.to(torch.int64), num_classes=self.n_classes)
        # Calculate cross entropy loss

        new_hot_target = self.penalty_matrix[target.to(torch.int64).to(self.device)]
        cross_entropy_loss = -torch.sum(new_hot_target * torch.log(softmax + 1e-10)) / batch_size
        
        return cross_entropy_loss

    #def forward(self, y_pred, y_true):
    #    # Ensure probabilities are not zero to avoid log(0)
    #    #y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
        
        # Convert the true labels to one-hot encoding
        #print(y_true.shape)
    #    y_true = y_true.view(-1).to(torch.int64)
        
        
    #    y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=self.n_classes)
        
    #    # Compute the customized cross entropy loss
    #    #loss = -torch.sum(y_true_one_hot.to(torch.float32) * torch.log(y_pred) * self.penalty_matrix, dim=1).mean()
    #    loss = -torch.sum(y_true_one_hot.to(torch.float32) * torch.log(y_pred) * self.penalty_matrix, dim=1).mean()
    #    # Average the loss over all samples
    #    return loss
    

class EmbedderMultitask(Embedder):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embed spectra."""

    def __init__(
        self,
        d_model,
        n_layers,
        n_classes, 
        use_gumbel,
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
        self.regression_loss = nn.MSELoss()
        self.customised_ce = CustomizedCrossEntropyLoss()

        self.dropout = nn.Dropout(p=dropout)
        self.use_gumbel=use_gumbel

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

        # for cosine similarity, tanimoto
        emb_cosine = self.cosine_similarity(emb0, emb1)
        emb = emb0 + emb1
        emb = self.linear(emb)
        emb = self.dropout(emb)
        emb = self.relu(emb)
        emb= self.classifier(emb)

        #if self.gumbel_softmax:
        #    emb = self.gumbel_softmax(emb)
        #else:
        #    emb = F.softmax(emb, dim=-1)
        return emb, emb_cosine


    def step(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        logits_list = self(batch)

        logits1= logits_list[0]
        logits2= logits_list[1]

        #probs = F.softmax(logits, dim=-1)

        # the sim data is received in the range 0-1
        target1 = torch.tensor(batch["similarity"], dtype=torch.long).to(self.device)
        target1 = target1.view(-1)  # Ensure targets are in the right shape and type for classification

        target2 = torch.tensor(batch["similarity2"], dtype=torch.long).to(self.device)
        target2 = target2.view(-1)  # Ensure targets are in the right shape and type for classification
        
        #loss = self.loss_fn(logits, target)
        loss =self.customised_ce(logits1, target1) + self.regression_loss(logits2.float(), target2.view(-1, 1).float()).float()
        # Compute the expected value (continuous) from probabilities
        #expected_value = torch.sum(probs * torch.arange(logits.size(1)).to(self.device), dim=1)

        
        # Compute the MSE between the expected value and the target
        #loss = self.regression_loss(expected_value, target)


        #loss = self.cumulative_logits_loss(logits, target)
        #loss= self.ordinal_cross_entropy(logits, target)

        #vector = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).to(self.device)
        #print('example:')
        #print(logits[0])
        #print(target[0])
        #print(loss)
        #print(vector[0])
        ## Step 2: Compute the dot product for each row
        #dot_products = logits * vector
        #dot_products= dot_products.sum(dim=-1)
        #print(dot_products[0])
        #loss = self.regression_loss(dot_products, target)
        
        return loss
    
    def step_mse(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        logits = self(batch)
        #print('***')
        #print('before softmax')
        #print(logits[0])
        logits = F.softmax(logits, dim=-1)
        #logits = self.gumbel_softmax(logits)
        
        #print('after softmax')
        #print(logits[0])
        
        target = torch.tensor(batch["similarity"]).to(self.device)
        target = target.view(-1).float()  # Ensure targets are in the right shape and type for regression
        #print('target')
        #print(target[0])
        
        # Compute the probabilities from logits using softmax
        #probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Compute the expected value (continuous) from probabilities
        predicted_value = torch.sum(logits * torch.arange(logits.size(1)).to(self.device), dim=1)
        #print('inference')
        #print(expected_value[0])
        # Compute the MSE between the expected value and the target
        #print('loss')
        loss = self.regression_loss(predicted_value, target)
        #print(loss)

        #print('example:')
        #print(logits[0])
        #print(target[0])
        #print(expected_value[0])
        #print(loss) 
        
        return loss
    
    
    def gumbel_softmax(self, logits, temperature=0.2, hard=True):
            return F.gumbel_softmax(logits, tau=temperature, hard=hard)
    

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
  
