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
from src.weight_sampling import WeightSampling

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
        softmax = F.softmax(logits, dim=-1).to(self.device)
        new_hot_target = self.penalty_matrix[target.to(torch.int64).to(self.device)]
        cross_entropy_loss = -torch.sum(new_hot_target * torch.log(softmax + 1e-10)) / batch_size
        
        return cross_entropy_loss


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
        weights_sim2=None, #weights of second similarity
):
        """Initialize the CCSPredictor"""
        super().__init__(
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        weights=weights,
        lr=lr,
        use_element_wise=use_element_wise,
        use_cosine_distance=use_cosine_distance,
)  # element wise instead of concat for mixing info between embeddings)
        self.weights = weights

        # Add a linear layer for projection
        self.classifier = nn.Linear(d_model, n_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.customised_ce = CustomizedCrossEntropyLoss()

        self.dropout = nn.Dropout(p=dropout)
        self.use_gumbel=use_gumbel
        self.weights_sim2=weights_sim2

        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear2_cossim = nn.Linear(d_model, d_model) #extra linear transformation in case cosine similarity i sused
    def forward(self, batch, return_spectrum_output=False):
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
        if self.use_cosine_distance:
            #emb_sim_2 = self.cosine_similarity(emb0, emb1)
            ## apply transformation before apply cosine distance

            # emb0
            emb0_transformed = self.linear2(emb0)
            emb0_transformed= self.dropout(emb0_transformed)
            emb0_transformed=self.relu(emb0_transformed)
            emb0_transformed = self.linear2_cossim(emb0_transformed)
            emb0_transformed=self.relu(emb0_transformed)

            # emb1
            emb1_transformed = self.linear2(emb1)
            emb1_transformed= self.dropout(emb1_transformed)
            emb1_transformed=self.relu(emb1_transformed)
            emb1_transformed = self.linear2_cossim(emb1_transformed)
            emb1_transformed=self.relu(emb1_transformed)

            # cos sim
            emb_sim_2 = self.cosine_similarity(emb0_transformed, emb1_transformed)
        else:
            emb_sim_2 = emb0 + emb1
            emb_sim_2 = self.linear2(emb_sim_2)
            emb_sim_2 = self.dropout(emb_sim_2)
            emb_sim_2 = self.relu(emb_sim_2)
            emb_sim_2 = self.linear_regression(emb_sim_2)
            
        emb = emb0 + emb1
        emb = self.linear1(emb)
        emb = self.dropout(emb)
        emb = self.relu(emb)
        emb= self.classifier(emb)
        
        #if self.gumbel_softmax:
        #    emb = self.gumbel_softmax(emb)
        #else:
        #    emb = F.softmax(emb, dim=-1)
        if return_spectrum_output:
            return emb, emb_sim_2, emb0, emb1
        else: 
            return emb, emb_sim_2

    def calculate_weight_loss2(self):
            '''
            weight loss for second similarity metric. normally sim 2 is smaller since it is a regression problem between 0 and 1
            '''
            #if self.weights_sim2 is not None:
            #    weight_loss=200
            #else:
            #    weight_loss2=30
            weight_loss2=200
            return weight_loss2

    def compute_adjacent_diffs(self, gumbel_probs_1, batch_size):

            #original code:
            #diff_penalty = torch.sum((gumbel_probs_1[:, 1:] - gumbel_probs_1[:, :-1]) ** 2) / batch_size

            adjacent_diffs = gumbel_probs_1[:, 1:] - gumbel_probs_1[:, :-1]

            # Compute difference for the first and second columns (to include the first column)
            first_diff = gumbel_probs_1[:, 1] - gumbel_probs_1[:, 0]

            # Compute difference for the last and second-to-last columns (to include the last column)
            last_diff = gumbel_probs_1[:, -1] - gumbel_probs_1[:, -2]

            # Square all differences (element-wise)
            squared_adjacent_diffs = adjacent_diffs ** 2
            squared_first_diff = first_diff ** 2
            squared_last_diff = last_diff ** 2

            # Sum the squared differences
            diff_penalty = (
                torch.sum(squared_adjacent_diffs)  # Sum of differences between adjacent columns
                + torch.sum(squared_first_diff)    # Include first column difference
                + torch.sum(squared_last_diff)     # Include last column difference
            ) / batch_size

            return diff_penalty

    def step(self, batch, batch_idx, threshold=0.5, 
                weight_loss2=None, #loss2 (regresion) is 100 times less than loss1 (classification)
                ):
        """A training/validation/inference step."""
        logits_list = self(batch)

        logits1= logits_list[0]
        logits2= logits_list[1]

        # the sim data is received in the range 0-1
        target1 = torch.tensor(batch["similarity"], dtype=torch.long).to(self.device)
        target1 = target1.view(-1)  # Ensure targets are in the right shape and type for classification

        #print(f'batch["similarity2"] right before conversion to target2: {batch["similarity2"]}')
        target2 = torch.tensor(batch["similarity2"], dtype=torch.float32).to(self.device)
        target2 = target2.view(-1)  # Ensure targets are in the right shape and type for classification
        
        # Apply Gumbel softmax
        if self.use_gumbel:
            gumbel_probs_1 = F.gumbel_softmax(logits1, tau=5.0, hard=True)
            #gumbel_probs_1 = F.gumbel_softmax(logits1, tau=0.0, hard=True)

            # Compute the expected value (continuous) from probabilities
            expected_classes = torch.arange(gumbel_probs_1.size(1)).to(self.device)
            predicted_value = torch.sum(gumbel_probs_1 * expected_classes, dim=1)
            # Compute the MSE loss
            loss1 = self.regression_loss(predicted_value.float(), target1.float())
             # Regularization term to penalize large differences between adjacent probabilities
            batch_size=batch["similarity"].size(0)
            #diff_penalty = torch.sum((gumbel_probs_1[:, 1:] - gumbel_probs_1[:, :-1]) ** 2) / batch_size

            # exclude the first index because the >5 is very different than class 4
            diff_penalty = torch.sum((gumbel_probs_1[:, 2:] - gumbel_probs_1[:, 1:-1]) ** 2) / batch_size

            #modified to include the first and last index
            # Compute differences for adjacent columns (excluding the first and last initially)
            #adjacent_diffs = gumbel_probs_1[:, 1:] - gumbel_probs_1[:, :-1]

            # Compute difference for the first and second columns (to include the first column)
            #diff_penalty  =self.compute_adjacent_diffs(gumbel_probs_1, batch_size)
            reg_weight=0.1
            loss1 = loss1 + reg_weight * diff_penalty
        else:
            loss1 =self.customised_ce(logits1, target1) 

        if self.weights_sim2 is not None: # if there are sample weights used 
            # Calculate the squared difference for loss2
            squared_diff = (logits2.view(-1,1).float() - target2.view(-1, 1).float()) ** 2
            # remove the impact of sim=1 by making target2 ==0 when it is equal to 1
            #squared_diff[target2 >= 1]=0
            #target2[target2 >= 1] = 0
            #weighting the loss function
            weight_mask = WeightSampling.compute_sample_weights(molecule_pairs=None, 
                                                                weights=self.weights_sim2, 
                                                                use_molecule_pair_object=False,
                                                                bining_sim1=True,
                                                                targets=target2.cpu().numpy(),
                                                                normalize=False,)


            weight_mask = torch.tensor(weight_mask).to(self.device)

            loss2 = (squared_diff.view(-1, 1) * weight_mask.view(-1, 1).float()).mean()
        else:
            squared_diff = (logits2.view(-1,1).float() - target2.view(-1, 1).float()) ** 2
            # remove the impact of sim=1 by making target2 ==0 when it is equal to 1
            #squared_diff[target2 >= 1]=0
            #target2[target2 >= 1] = 0
            loss2 = squared_diff.view(-1, 1).mean()

        weight_loss2 = self.calculate_weight_loss2()

        loss = loss1 + (weight_loss2*loss2)

        #print(f'loss1:{loss1}')
        #print(f'loss2: {loss2}')
        #print(f'loss: {loss}')

        #print(f'loss 1 shape: {loss1.shape}')
        #print(f'loss 2 shape: {loss2.shape}')
        #print(f'logits2 size: {logits2.shape}')
        #print(f'target2 size: {target2.shape}')
        #print(f'squared_diff size: {squared_diff.shape}')
        #print(f'loss2 size: {loss2.shape}')
        #print(f'loss size: {loss.shape}')
        #print(f'weight_mask size: {weight_mask.shape}')


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
  
