import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simba.core.data.sampling.weight_sampling import WeightSampling
from simba.core.models.transformers.embedder import Embedder


class CustomizedCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        # Construct the penalty matrix using absolute differences.
        # With this design, a correct prediction (i == j) gets a penalty of 0,
        # and misclassifications incur a penalty proportional to their distance |i - j|.
        n_classes = 6
        penalty_matrix = np.array(
            [[abs(i - j) for j in range(n_classes)] for i in range(n_classes)]
        )
        penalty_matrix = (n_classes - 1) - penalty_matrix
        penalty_matrix = penalty_matrix**2

        # Normalize each row so that the row sums to 1
        row_sums = penalty_matrix.sum(axis=1, keepdims=True)
        penalty_matrix = penalty_matrix / row_sums

        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalize the penalty matrix so that the maximum penalty (for the farthest misclassification)
        # becomes 1; this keeps the values in a controlled range.
        self.penalty_matrix = torch.tensor(penalty_matrix, dtype=torch.float).to(
            self.device
        )

        max_value = np.max(penalty_matrix)
        if max_value > 0:
            self.penalty_matrix = self.penalty_matrix / max_value

    def forward(self, logits, target):
        batch_size = logits.size(0)
        # Compute the log probabilities
        log_probs = F.log_softmax(logits, dim=-1).to(self.device)
        # For each sample, select the penalty row that corresponds to its true class.

        target = target.to(torch.int64).to(self.device)
        self.penalty_matrix = self.penalty_matrix.to(self.device)
        new_hot_target = self.penalty_matrix[target]
        # Compute the weighted loss by multiplying the log_probs with the penalty weights,
        # and averaging over the batch.
        cross_entropy_loss = -torch.sum(new_hot_target * log_probs) / batch_size
        return cross_entropy_loss


class EmbedderMultitask(Embedder):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embeds spectra."""

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
        weights_sim2=None,  # weights of second similarity
        use_edit_distance_regresion=False,
        use_mces20_log_loss=True,
        use_fingerprints=False,
        use_precursor_mz_for_model=True,
        tau_gumbel_softmax=10,
        gumbel_reg_weight=0.1,
        USE_LEARNABLE_MULTITASK=True,
        use_adduct=False,
        use_ce=False,
        use_ion_activation=False,
        use_ion_method=False,
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
            use_adduct=use_adduct,
            use_ce=use_ce,
            use_ion_activation=use_ion_activation,
            use_ion_method=use_ion_method,
        )
        self.weights = weights

        # Add a linear layer for projection
        self.classifier = nn.Linear(d_model, n_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.customised_ce = CustomizedCrossEntropyLoss()

        self.dropout = nn.Dropout(p=dropout)
        self.use_gumbel = use_gumbel

        if self.use_gumbel:
            self.tau_gumbel_softmax = tau_gumbel_softmax
            self.gumbel_reg_weight = gumbel_reg_weight
            print(f"Using TAU GUMBEL softmax: {self.tau_gumbel_softmax}")
            print(f"Using TAU GUMBEL reg weight: {self.gumbel_reg_weight}")
        self.weights_sim2 = weights_sim2

        self.linear1 = nn.Linear(d_model, d_model)
        self.linear1_2 = nn.Linear(d_model, d_model)

        self.linear2 = nn.Linear(d_model, d_model)
        self.linear2_cossim = nn.Linear(
            d_model, d_model
        )  # Extra linear layer if cosine similarity is used

        self.use_edit_distance_regresion = use_edit_distance_regresion
        if self.use_edit_distance_regresion:
            self.linear1_cossim = nn.Linear(d_model, d_model)

        self.use_mces20_log_loss = use_mces20_log_loss
        self.use_fingerprints = use_fingerprints
        if self.use_fingerprints:
            print("Fingerprints enabled!")
            self.linear_fingerprint_0 = nn.Linear(2048, d_model)
            self.linear_fingerprint_1 = nn.Linear(d_model, d_model)
            proj_dim = d_model // 2  # 2048 → d_model//2
            self.linear_fp0 = nn.Linear(2048, proj_dim)
            self.linear_mix = nn.Linear(d_model + proj_dim, d_model)
            self.norm_mix = nn.LayerNorm(d_model)
        self.use_precursor_mz_for_model = use_precursor_mz_for_model

        # Initialize learnable log variance parameters for each loss
        self.USE_LEARNABLE_MULTITASK = USE_LEARNABLE_MULTITASK
        if USE_LEARNABLE_MULTITASK:
            initial_log_sigma1 = 1.2490483522415161
            initial_log_sigma2 = -7.0018157958984375
            # self.log_sigma1 = nn.Parameter(torch.tensor(initial_log_sigma1))
            # self.log_sigma2 = nn.Parameter(torch.tensor(initial_log_sigma2))
            self.log_sigma1 = torch.tensor(
                float(initial_log_sigma1), dtype=torch.float32
            )
            self.log_sigma2 = torch.tensor(
                float(initial_log_sigma2), dtype=torch.float32
            )
            self.use_extra_metadata = use_adduct

    def forward(self, batch, return_spectrum_output=False):
        # … compute raw emb0, emb1, apply relu, fingerprints, etc. …

        # nans to zeros
        batch["precursor_mass_0"] = torch.nan_to_num(
            batch["precursor_mass_0"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["precursor_mass_1"] = torch.nan_to_num(
            batch["precursor_mass_1"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["precursor_charge_0"] = torch.nan_to_num(
            batch["precursor_charge_0"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["precursor_charge_1"] = torch.nan_to_num(
            batch["precursor_charge_1"], nan=0.0, posinf=0.0, neginf=0.0
        )

        """The inference pass"""
        if self.use_precursor_mz_for_model:
            mz_0 = batch["precursor_mass_0"].float()
            mz_1 = batch["precursor_mass_1"].float()
        else:
            mz_0 = torch.zeros_like(batch["precursor_mass_0"].float())
            mz_1 = torch.zeros_like(batch["precursor_mass_1"].float())
        kwargs_0 = {
            "precursor_mass": mz_0,
            "precursor_charge": batch["precursor_charge_0"].float(),
        }
        kwargs_1 = {
            "precursor_mass": mz_1,
            "precursor_charge": batch["precursor_charge_1"].float(),
        }

        if self.use_adduct:
            batch["ionmode_0"] = torch.nan_to_num(
                batch["ionmode_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["ionmode_1"] = torch.nan_to_num(
                batch["ionmode_1"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["adduct_0"] = torch.nan_to_num(
                batch["adduct_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["adduct_1"] = torch.nan_to_num(
                batch["adduct_1"], nan=0.0, posinf=0.0, neginf=0.0
            )

            kwargs_0["ionmode"] = batch["ionmode_0"].float()
            kwargs_1["ionmode"] = batch["ionmode_1"].float()
            kwargs_0["adduct"] = batch["adduct_0"].float()
            kwargs_1["adduct"] = batch["adduct_1"].float()

        if self.use_ce:
            batch["ce_0"] = torch.nan_to_num(
                batch["ce_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["ce_1"] = torch.nan_to_num(
                batch["ce_1"], nan=0.0, posinf=0.0, neginf=0.0
            )
            kwargs_0["ce"] = batch["ce_0"].float()
            kwargs_1["ce"] = batch["ce_1"].float()

        if self.use_ion_activation:
            batch["ion_activation_0"] = torch.nan_to_num(
                batch["ion_activation_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["ion_activation_1"] = torch.nan_to_num(
                batch["ion_activation_1"], nan=0.0, posinf=0.0, neginf=0.0
            )
            kwargs_0["ion_activation"] = batch["ion_activation_0"].float()
            kwargs_1["ion_activation"] = batch["ion_activation_1"].float()

        if self.use_ion_method:
            batch["ion_method_0"] = torch.nan_to_num(
                batch["ion_method_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["ion_method_1"] = torch.nan_to_num(
                batch["ion_method_1"], nan=0.0, posinf=0.0, neginf=0.0
            )

            kwargs_0["ion_method"] = batch["ion_method_0"].float()
            kwargs_1["ion_method"] = batch["ion_method_1"].float()

        # intensity and mz
        batch["intensity_0"] = torch.nan_to_num(
            batch["intensity_0"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["intensity_1"] = torch.nan_to_num(
            batch["intensity_1"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["mz_0"] = torch.nan_to_num(batch["mz_0"], nan=0.0, posinf=0.0, neginf=0.0)
        batch["mz_1"] = torch.nan_to_num(batch["mz_1"], nan=0.0, posinf=0.0, neginf=0.0)

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

        if self.use_fingerprints:
            fp0 = batch["fingerprint_0"].float()  # (B, 2048)
            fp_proj = self.dropout(self.relu(self.linear_fp0(fp0)))  # (B, d_model//2)
            joint = torch.cat([emb0, fp_proj], dim=-1)  # (B, d_model + d_model//2)
            emb0 = self.dropout(self.norm_mix(self.relu(self.linear_mix(joint))))

        if return_spectrum_output:
            emb, emb_sim_2 = self.compute_from_embeddings(emb0, emb1)
            return emb, emb_sim_2, emb0, emb1
        else:
            return self.compute_from_embeddings(emb0, emb1)

    def calculate_weight_loss2(self):
        if self.use_edit_distance_regresion:
            weight_loss2 = 1
        else:
            weight_loss2 = 200
        return weight_loss2

    def compute_adjacent_diffs(self, gumbel_probs_1, batch_size):
        adjacent_diffs = gumbel_probs_1[:, 1:] - gumbel_probs_1[:, :-1]
        first_diff = gumbel_probs_1[:, 1] - gumbel_probs_1[:, 0]
        last_diff = gumbel_probs_1[:, -1] - gumbel_probs_1[:, -2]
        squared_adjacent_diffs = adjacent_diffs**2
        squared_first_diff = first_diff**2
        squared_last_diff = last_diff**2
        diff_penalty = (
            torch.sum(squared_adjacent_diffs)
            + torch.sum(squared_first_diff)
            + torch.sum(squared_last_diff)
        ) / batch_size
        return diff_penalty

    def ordinal_loss(self, logits, target):
        batch_size, num_classes = logits.size()
        prob = torch.sigmoid(logits)
        target_matrix = torch.zeros((batch_size, num_classes)).to(logits.device)
        for i in range(batch_size):
            target_class = target[i].long()
            target_matrix[i, : target_class + 1] = 1.0
        loss = F.binary_cross_entropy(prob, target_matrix, reduction="mean")
        return loss

    def step(self, batch, batch_idx, threshold=0.5, weight_loss2=None):
        logits_list = self(batch)
        logits1 = logits_list[0]
        logits2 = logits_list[1]
        target1 = batch["ed"].to(dtype=torch.long, device=self.device)
        target1 = target1.view(-1)
        target2 = batch["mces"].to(dtype=torch.float32, device=self.device)
        target2 = target2.view(-1)

        if self.use_edit_distance_regresion:
            target1 = target1 / 5
            squared_diff_1 = (
                logits1.view(-1, 1).float() - target1.view(-1, 1).float()
            ) ** 2
            loss1 = squared_diff_1.view(-1, 1).mean()
        else:
            if self.use_gumbel:
                gumbel_probs_1 = F.gumbel_softmax(
                    logits1, tau=self.tau_gumbel_softmax, hard=False
                )
                expected_classes = torch.arange(gumbel_probs_1.size(1)).to(self.device)
                predicted_value = torch.sum(gumbel_probs_1 * expected_classes, dim=1)
                loss1 = self.regression_loss(predicted_value.float(), target1.float())
                batch_size = batch["ed"].size(0)
                diff_penalty = (
                    torch.sum((gumbel_probs_1[:, 2:] - gumbel_probs_1[:, 1:-1]) ** 2)
                    / batch_size
                )
                reg_weight = self.gumbel_reg_weight
                loss1 = loss1 + reg_weight * diff_penalty
            else:
                loss1 = self.customised_ce(logits1, target1)

        def log_conversion(x, a=5):
            scaling_factor = np.log(a + 1)
            logits2_for_loss = torch.log((a + 1) - (a * x)) / scaling_factor
            return 1 - logits2_for_loss

        if self.use_mces20_log_loss:
            logits2_for_loss = log_conversion(logits2)
            target2_for_loss = log_conversion(target2)
        else:
            logits2_for_loss = logits2
            target2_for_loss = target2

        if self.weights_sim2 is not None:
            squared_diff = (
                logits2_for_loss.view(-1, 1).float()
                - target2_for_loss.view(-1, 1).float()
            ) ** 2
            weight_mask = WeightSampling.compute_sample_weights(
                molecule_pairs=None,
                weights=self.weights_sim2,
                use_molecule_pair_object=False,
                bining_sim1=False,
                targets=target2.cpu().numpy(),
                normalize=False,
            )
            weight_mask = torch.tensor(weight_mask).to(self.device)
            loss2 = (squared_diff.view(-1, 1) * weight_mask.view(-1, 1).float()).mean()
        else:
            squared_diff = (
                logits2.view(-1, 1).float() - target2.view(-1, 1).float()
            ) ** 2
            loss2 = squared_diff.view(-1, 1).mean()

        weight_loss2 = self.calculate_weight_loss2()
        # loss = loss1 + (weight_loss2 * loss2)

        # Recompute sigma values dynamically so a fresh graph is built.
        # sigma1 = torch.nn.functional.softplus(self.sigma1_param)
        # sigma2 = torch.nn.functional.softplus(self.sigma2_param)

        # Combine the losses using learned weights:
        if self.USE_LEARNABLE_MULTITASK:
            loss = (
                torch.exp(-self.log_sigma1) * loss1
                + self.log_sigma1
                + torch.exp(-self.log_sigma2) * loss2
                + self.log_sigma2
            )

            # loss = (loss1 / (2 * sigma1 ** 2) +
            #    loss2 / (2 * sigma2 ** 2) +
            #    torch.log(sigma1) +
            #    torch.log(sigma2))
        else:
            loss = loss1 + (weight_loss2 * loss2)
        return loss

    def step_mse(self, batch, batch_idx, threshold=0.5):
        logits = self(batch)
        logits = F.softmax(logits, dim=-1)
        target = torch.tensor(batch["ed"]).to(self.device).view(-1).float()
        predicted_value = torch.sum(
            logits * torch.arange(logits.size(1)).to(self.device), dim=1
        )
        loss = self.regression_loss(predicted_value, target)
        return loss

    def gumbel_softmax(self, logits, temperature=0.2, hard=True):
        return F.gumbel_softmax(logits, tau=temperature, hard=hard)

    def ordinal_cross_entropy(self, pred, target):
        batch_size = pred.size(0)
        target_matrix = torch.zeros_like(pred, dtype=torch.float)
        for i in range(batch_size):
            target_matrix[i, : target[i] + 1] = 1.0
        loss = -torch.sum(target_matrix * F.log_softmax(pred, dim=1), dim=1).mean()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_from_embeddings(self, emb0: torch.Tensor, emb1: torch.Tensor):
        """
        Take two activated embeddings (after ReLU, fingerprint fusion, etc.)
        and run all the FC layers + similarity heads to produce:
          - emb: the class–probability (or regression) output
          - emb_sim_2: the second similarity score (cosine or regression)
        """
        # note: assume emb0/emb1 already had model.relu and fingerprint logic applied
        # now do exactly what you had in compute_emb_from_existing_embeddings:
        if self.use_cosine_distance:
            # transform for cos-sim
            def transform(x):
                x = self.linear2(x)
                x = self.dropout(x)
                x = self.relu(x)
                return self.relu(self.linear2_cossim(x))

            e0 = transform(emb0)
            e1 = transform(emb1)
            emb_sim_2 = self.cosine_similarity(e0, e1)
        else:
            x = self.relu(self.linear2(emb0 + emb1))
            # clamp to ≤1
            emb_sim_2 = x - self.relu(x - 1)

        if self.use_edit_distance_regresion:

            def transform1(x):
                x = self.dropout(self.relu(self.linear1(x)))
                return self.relu(self.linear1_cossim(x))

            e0 = transform1(emb0)
            e1 = transform1(emb1)
            emb = self.cosine_similarity(e0, e1)
            emb = (emb * 5).round() / 5
        else:
            e0 = self.linear1_2(self.relu(self.linear1(emb0)))
            e1 = self.linear1_2(self.relu(self.linear1(emb1)))
            emb = self.relu(e0 + e1)
            emb = self.classifier(emb)
        return emb, emb_sim_2
