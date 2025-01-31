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
import numpy as np
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

from src.transformers.embedder import Embedder
from src.ordinal_classification.ordinal_classification import OrdinalClassification
from src.weight_sampling import WeightSampling

class CustomizedCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes=6):
        super(CustomizedCrossEntropyLoss, self).__init__()
        #penalty_matrix = [[20, 4, 3, 2, 1,  0,],
        #                  
        #                  [4, 20, 4, 3, 2,  1,] ,
        #
        #                  [3, 4, 20, 4, 3,  2,],
        #                  [2, 3,  4, 20, 4,  3,],
        #                  [1, 2, 3, 4,  20,  4,],
        #                  [0, 1, 2, 3,  4,  20,]]

        penalty_matrix = [[20, 4, 0, 0, 0,  0,],
                          
                          [4, 20, 4, 0, 0,  0,] ,

                          [0, 4, 20, 4, 0,  0,],

                          [0, 0,  4, 20, 4,  0,],

                          [0, 0, 0, 4,  20,  4,],

                          [0, 0, 0, 0,  4,  20,]]


        #penalty_matrix = [[100, 4, 3, 2, 1,  0,],
                          
        #                  [5, 30, 10, 0, 0,  0,] ,

        #                  [5, 10, 30, 10, 0,  0,],

        #                  [5, 0,  10, 30, 10,  0,],

        #                  [5, 0, 0, 10,  30,  10,],

        #                  [5, 0, 0, 0,  10,  30]]

        
        self.n_classes=n_classes
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #normalize the penalty matrix
        self.penalty_matrix = torch.tensor(penalty_matrix).to(self.device)/np.max(penalty_matrix)
        #self.penalty_matrix = self.penalty_matrix / self.penalty_matrix.sum(dim=1, keepdim=True)


    def forward(self, logits, target):
        #batch_size = logits.size(0)
        #softmax = F.softmax(logits, dim=-1).to(self.device)
        #new_hot_target = self.penalty_matrix[target.to(torch.int64).to(self.device)]
        #cross_entropy_loss = -torch.sum(new_hot_target * torch.log(softmax + 1e-10)) / batch_size
        #return cross_entropy_loss
        batch_size = logits.size(0)
        log_probs = F.log_softmax(logits, dim=-1).to(self.device)
        new_hot_target = self.penalty_matrix[target.to(torch.int64).to(self.device)]
        cross_entropy_loss = -torch.sum(new_hot_target * log_probs) / batch_size
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
        use_edit_distance_regresion=False,
        use_mces20_log_loss=True, 
        use_fingerprints=False,
        use_precursor_mz_for_model=True,
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
        self.linear1_2 = nn.Linear(d_model, d_model)

        self.linear2 = nn.Linear(d_model, d_model)
        self.linear2_cossim = nn.Linear(d_model, d_model) #extra linear transformation in case cosine similarity i sused

        self.use_edit_distance_regresion =use_edit_distance_regresion

        if self.use_edit_distance_regresion:
            self.linear1_cossim = nn.Linear(d_model, d_model)

        self.use_mces20_log_loss=use_mces20_log_loss

        self.use_fingerprints=use_fingerprints 
        if self.use_fingerprints:
            print(' fingerprints  enabled! ...')
            self.linear_fingerprint_0= nn.Linear(2048, d_model)
            self.linear_fingerprint_1= nn.Linear(d_model, d_model)

        self.use_precursor_mz_for_model=use_precursor_mz_for_model

    def forward(self, batch, return_spectrum_output=False):
        """The inference pass"""
        #batch = {k:torch.tensor(batch[k]) for k in batch}
        
        if self.use_precursor_mz_for_model:
            mz_0=batch["precursor_mass_0"].float()
            mz_1 =batch["precursor_mass_1"].float()
        else:
            mz_0=torch.zeros_like(batch["precursor_mass_0"].float())
            mz_1=torch.zeros_like(batch["precursor_mass_1"].float())
        # extra data
        kwargs_0 = {
            "precursor_mass": mz_0,
            "precursor_charge": batch["precursor_charge_0"].float(),
        }
        kwargs_1 = {
            "precursor_mass": mz_1,
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

        ## if using fingerprints
        if self.use_fingerprints:
            fing_0= batch['fingerprint_0'].float()
            fing_0 = self.linear_fingerprint_0(fing_0)
            fing_0 = self.relu(fing_0)
            fing_0= self.dropout(fing_0)
            fing_0 = self.linear_fingerprint_1(fing_0)
            fing_0 = self.relu(fing_0)
            fing_0= self.dropout(fing_0)
            
            emb0 =emb0  + fing_0
            emb0 = self.relu(emb0)
            


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

            # avoid  values higher than 1
            x =self.relu(emb_sim_2-1)
            emb_sim_2 = emb_sim_2 - x


        if self.use_edit_distance_regresion:
            # emb0
            emb0_transformed_1 = self.linear1(emb0)
            emb0_transformed_1= self.dropout(emb0_transformed_1)
            emb0_transformed_1=self.relu(emb0_transformed_1)
            emb0_transformed_1 = self.linear1_cossim(emb0_transformed_1)
            emb0_transformed_1=self.relu(emb0_transformed_1)

            # emb1
            emb1_transformed_1 = self.linear1(emb1)
            emb1_transformed_1= self.dropout(emb1_transformed_1)
            emb1_transformed_1=self.relu(emb1_transformed_1)
            emb1_transformed_1 = self.linear1_cossim(emb1_transformed_1)
            emb1_transformed_1=self.relu(emb1_transformed_1)

            # cos sim
            emb = self.cosine_similarity(emb0_transformed_1, emb1_transformed_1)

            #round to integers
            emb=emb*5
            emb = emb + emb.round().detach() - emb.detach() # trick to make round differentiable
            emb=emb/5
        else:
            #emb = emb0 + emb1
            #emb = self.linear1(emb)

            emb_0_ = self.linear1(emb0)
            emb_0_ = self.relu(emb_0_)
            emb_0_ = self.linear1_2(emb_0_)

            emb_1_ = self.linear1(emb1)
            emb_1_ = self.relu(emb_1_)
            emb_1_ = self.linear1_2(emb_1_)

            emb = emb_0_ + emb_1_
            
            #emb = self.dropout(emb)
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

            if self.use_edit_distance_regresion:
                weight_loss2=1
            else:
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
    def ordinal_loss(self, logits, target):
        # logits: (batch_size, num_classes)
        # target: (batch_size,)
        batch_size, num_classes = logits.size()
        
        # Apply sigmoid to logits to get probabilities
        prob = torch.sigmoid(logits)  # (batch_size, num_classes)
        
        # Create a binary target matrix
        # For each sample, positions less than or equal to the target class are 1, others are 0
        target_matrix = torch.zeros((batch_size, num_classes)).to(logits.device)
        for i in range(batch_size):
            target_class = target[i].long()
            target_matrix[i, :target_class + 1] = 1.0
        
        # Compute the binary cross-entropy loss
        loss = F.binary_cross_entropy(prob, target_matrix, reduction='mean')
        return loss
        
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
        if self.use_edit_distance_regresion:
            # compress the range
            target1 = target1/5
            squared_diff_1 = (logits1.view(-1,1).float() - target1.view(-1, 1).float()) ** 2
            # remove the impact of sim=1 by making target2 ==0 when it is equal to 1
            #squared_diff[target2 >= 1]=0
            #target2[target2 >= 1] = 0
            loss1 = squared_diff_1.view(-1, 1).mean()
        else:
            if self.use_gumbel:
                gumbel_probs_1 = F.gumbel_softmax(logits1, tau=10.0, hard=False)
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
                #loss1 = self.ordinal_loss(logits1, target1)
                loss1 =self.customised_ce(logits1, target1) 

        if self.weights_sim2 is not None: # if there are sample weights used 

            # apply log function if needed
            if self.use_mces20_log_loss:
                #scaling_factor= (2*np.log(0.5))#divided by scaling factor just for normalizing the range between 0 and 1 again
                scaling_factor= np.log(2)
                logits2_for_loss= torch.log(2-logits2)/scaling_factor 
                target2_for_loss= torch.log(2-target2)/scaling_factor
            else:
                logits2_for_loss=logits2
                target2_for_loss = target2 

            # Calculate the squared difference for loss2
            squared_diff = (logits2_for_loss.view(-1,1).float() - target2_for_loss.view(-1, 1).float()) ** 2
            # remove the impact of sim=1 by making target2 ==0 when it is equal to 1
            #squared_diff[target2 >= 1]=0
            #target2[target2 >= 1] = 0
            #weighting the loss function
            weight_mask = WeightSampling.compute_sample_weights(molecule_pairs=None, 
                                                                weights=self.weights_sim2, 
                                                                use_molecule_pair_object=False,
                                                                bining_sim1=False,
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
  
