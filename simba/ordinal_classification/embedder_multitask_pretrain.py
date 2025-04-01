import torch
import torch.nn as nn
import torch.nn.functional as F
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask

class EmbedderMultitaskPretrain(EmbedderMultitask):
    """A class for pretraining the spectrum encoder to predict m/z values."""

    def __init__(self, d_model, n_layers, dropout=0.1, **kwargs):
        super().__init__(d_model=d_model, n_layers=n_layers, dropout=dropout, **kwargs)
        # Linear layer to predict m/z values at each position
        self.mz_predictor = nn.Linear(d_model, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        """Forward pass for pretraining."""
        # Encode the input spectrum
        emb, _ = self.spectrum_encoder(
            mz_array=batch["mz_0"].float(),  # Shape: (batch_size, seq_len)
            intensity_array=batch["intensity_0"].float(),
            precursor_mass=batch["precursor_mass_0"].float(),
            precursor_charge=batch["precursor_charge_0"].float(),
        )

        # emb has shape (batch_size, seq_len, d_model)
        # Predict m/z values at each position
        mz_pred = self.mz_predictor(emb)  # Shape: (batch_size, seq_len, 1)
        mz_pred = mz_pred.squeeze(-1)  # Shape: (batch_size, seq_len)

        mz_pred = mz_pred[:,1:]

        return mz_pred

    def step(self, batch, batch_idx, threshold=0.5, 
                weight_loss2=None, #loss2 (regresion) is 100 times less than loss1 (classification)
                ):
        """Compute the loss for pretraining."""
        
        # Mask 70% of the non-zero m/z values
        mask = (batch["mz_0"] != 0).float()  # Shape: (batch_size, seq_len)
        num_nonzero = mask.sum(dim=1, keepdim=True)
        num_to_mask = (0.7 * num_nonzero).int()
        # Create a mask for positions to predict (30% of non-zero m/z values)
        predict_mask = torch.zeros_like(mask)
        for i in range(mask.size(0)):
            indices = torch.where(mask[i] == 1)[0]
            perm = torch.randperm(indices.size(0))
            selected_indices = indices[perm[:num_to_mask[i].item()]]
            predict_mask[i, selected_indices] = 1
            
            
            
        # Forward pass

        y_target= batch["mz_0"] * predict_mask
        batch["mz_0"]= batch["mz_0"] *(1-predict_mask)

        mz_pred = self(batch) 


        # Compute loss only on the masked positions
        loss = self.loss_fn(mz_pred * predict_mask, y_target)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer