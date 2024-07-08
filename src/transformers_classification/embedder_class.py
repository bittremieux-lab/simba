from src.transformers.embedder import Embedder
import torch
import torch.nn as nn

class EmbedderClass(Embedder):

    def __init__(
        self,
        d_model,
        n_layers,
        dropout=0.1,
        weights=None,
        lr=None,
        use_element_wise=True,
        use_cosine_distance=True,  # element wise instead of concat for mixing info between embeddings
    ):
        """Initialize the CCSPredictor"""
        super().__init__()
        self.weights = weights

        # Add a linear layer for projection
        self.use_element_wise = use_element_wise
        self.linear = nn.Linear(d_model, d_model)
        self.linear_classification = nn.Linear(d_model, 6)  # 6 output classes

        self.relu = nn.ReLU()

        self.spectrum_encoder = SpectrumTransformerEncoderCustom(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.classification_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=dropout)

        self.train_loss_list = []
        self.val_loss_list = []
        self.lr = lr
        self.use_cosine_distance = use_cosine_distance
        if self.use_cosine_distance:
            self.linear_cosine = nn.Linear(d_model, d_model)

        self.cosine_similarity = nn.CosineSimilarity(dim=1)

        self.use_cosine_library = True

        print(f"Using cosine library from Pytorch?: {self.use_cosine_library}")
        
    def step(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        logits = self(batch)

        target = torch.tensor(batch["class_label"]).to(self.device)
        target = target.view(-1)

        loss = self.classification_loss(logits, target).float()

        return loss.float()

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

        if self.use_cosine_distance:

            if self.use_cosine_library:
                emb = self.cosine_similarity(emb0, emb1)

                # Reshape the tensor
                emb = emb.reshape(-1, 1)

            else:
                # ensure the embeddings are positive
                emb0_l2 = torch.norm(emb0, p=2, dim=-1, keepdim=True)
                emb1_l2 = torch.norm(emb1, p=2, dim=-1, keepdim=True)
                emb = (emb0 * emb1) / (emb0_l2 * emb1_l2)
                # Apply fixed linear regression if needed (this might need adjustment)
                emb = self.fixed_linear_regression(emb)
                emb = emb.reshape(-1, 1)

        else:
            emb = emb0 + emb1
            emb = self.linear(emb)
            emb = self.dropout(emb)
            emb = self.relu(emb)

        logits = self.linear_classification(emb)  # 6-class output

        return logits
