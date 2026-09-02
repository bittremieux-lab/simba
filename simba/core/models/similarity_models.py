import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simba.core.models.spectrum_encoder import (
    SpectrumTransformerEncoderCustom,
)
from simba.utils.logger_setup import logger


class FixedLinearRegression(nn.Module):
    """
    linear layer for computing sum of dot product
    """

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(1, d_model)
        )  # Fixed weight initialized to 1
        self.bias = nn.Parameter(torch.zeros(1))  # Bias initialized to 0

        # Freeze the parameters
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


class SimilarityModel(pl.LightningModule):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embed spectra."""

    def __init__(
        self,
        d_model,
        n_layers,
        dropout=0.1,
        weights=None,
        lr=None,
        use_element_wise=True,
        use_cosine_distance=True,  # element wise instead of concat for mixing info between embeddings
        use_adduct=False,
        use_ce=False,
        use_ion_activation=False,
        use_ion_method=False,
        use_ion_mode=False,
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
        self.use_adduct = use_adduct
        self.use_ce = use_ce
        self.use_ion_activation = use_ion_activation
        self.use_ion_method = use_ion_method
        self.use_ion_mode = use_ion_mode

        self.spectrum_encoder = SpectrumTransformerEncoderCustom(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            use_adduct=use_adduct,
            use_ce=use_ce,
            use_ion_activation=use_ion_activation,
            use_ion_method=use_ion_method,
            use_ion_mode=use_ion_mode,
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

        self.use_cosine_library = True

        # print(f"Using cosine library from Pytorch?: {self.use_cosine_library}")

    def normalized_dot_product(self, a, b):
        # Normalize inputs
        a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)

        # Compute dot product
        dot_product = torch.sum(a_norm * b_norm, dim=-1)
        return dot_product

    def forward(self, batch):
        """The inference pass"""

        kwargs_0 = {
            "precursor_mass": batch["precursor_mass_0"].float(),
        }
        kwargs_1 = {
            "precursor_mass": batch["precursor_mass_1"].float(),
        }
        # extra data
        if self.use_ion_mode:
            kwargs_0["ionmode"] = batch["ionmode_0"].float()
            kwargs_1["ionmode"] = batch["ionmode_1"].float()
            kwargs_0["precursor_charge"] = batch["precursor_charge_0"].float()
            kwargs_1["precursor_charge"] = batch["precursor_charge_1"].float()
        if self.use_adduct:
            kwargs_0["ionmode"] = batch["ionmode_0"].float()
            kwargs_1["ionmode"] = batch["ionmode_1"].float()
            kwargs_0["adduct"] = batch["adduct_0"].float()
            kwargs_1["adduct"] = batch["adduct_1"].float()

        if self.use_ce:
            logger.info("Using CE in the model")
            kwargs_0["ce"] = batch["ce_0"].float()
            kwargs_1["ce"] = batch["ce_1"].float()

        if self.use_ion_activation:
            kwargs_0["ion_activation"] = batch["ion_activation_0"].float()
            kwargs_1["ion_activation"] = batch["ion_activation_1"].float()

        if self.use_ion_method:
            kwargs_0["ion_method"] = batch["ion_method_0"].float()
            kwargs_1["ion_method"] = batch["ion_method_1"].float()

        # ensure there are no nans
        batch["mz_0"] = torch.nan_to_num(batch["mz_0"], nan=0.0, posinf=0.0, neginf=0.0)
        batch["mz_1"] = torch.nan_to_num(batch["mz_1"], nan=0.0, posinf=0.0, neginf=0.0)
        batch["intensity_0"] = torch.nan_to_num(
            batch["intensity_0"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["intensity_1"] = torch.nan_to_num(
            batch["intensity_1"], nan=0.0, posinf=0.0, neginf=0.0
        )

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
                emb = self.fixed_linear_regression(emb)
                # emb = (emb+1)/2

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
        # target = 2*(target-0.5)
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
        # if self.use_cosine_library:
        # spec= (spec+1)/2
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


class SimilarityModelMultitask(SimilarityModel):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embeds spectra."""

    def __init__(
        self,
        d_model,
        n_layers,
        dropout=0.1,
        weights=None,
        lr=None,
        use_element_wise=True,
        use_cosine_distance=True,  # element wise instead of concat for mixing info between embeddings
        mces_max_value=40.0,  # must match model.tasks.mces.max_value; used by the mces_bucket head
        use_mces_bucket_head=False,  # optional second target: CORN-style ordinal classification on MCES buckets
        mces_bucket_bin_edges=None,  # required (from config) when use_mces_bucket_head=True
        mces_bucket_use_mlp=False,
        mces_bucket_loss_weight=1.0,
        use_precursor_mz_for_model=True,
        use_adduct=False,
        use_ce=False,
        use_ion_activation=False,
        use_ion_method=False,
        use_ion_mode=False,
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
            use_ion_mode=use_ion_mode,
        )
        self.weights = weights
        self.mces_max_value = mces_max_value

        self.dropout = nn.Dropout(p=dropout)

        self.use_mces_bucket_head = use_mces_bucket_head
        if self.use_mces_bucket_head:
            bucket_edges_t = torch.tensor(
                list(mces_bucket_bin_edges), dtype=torch.float32
            )
            self.register_buffer("mces_bucket_bin_edges", bucket_edges_t)
            # +1 for the open-ended top bin, +1 for the singleton "exactly 0" bin
            self.mces_bucket_n_classes = len(mces_bucket_bin_edges) + 2
            self.mces_bucket_use_mlp = mces_bucket_use_mlp
            self.mces_bucket_loss_weight = mces_bucket_loss_weight
            bucket_input_dim = d_model
            if mces_bucket_use_mlp:
                self.mces_bucket_mlp = nn.Sequential(
                    nn.Linear(bucket_input_dim, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model),
                )
                bucket_head_input_dim = d_model
            else:
                bucket_head_input_dim = bucket_input_dim
            self.mces_bucket_head = nn.Linear(
                bucket_head_input_dim, self.mces_bucket_n_classes - 1
            )

        self.use_precursor_mz_for_model = use_precursor_mz_for_model

    def forward(self, batch, return_spectrum_output=False):
        # … compute raw emb0, emb1, apply relu, etc. …

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

        if self.use_ion_mode:
            batch["ionmode_0"] = torch.nan_to_num(
                batch["ionmode_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["ionmode_1"] = torch.nan_to_num(
                batch["ionmode_1"], nan=0.0, posinf=0.0, neginf=0.0
            )
            kwargs_0["ionmode"] = batch["ionmode_0"].float()
            kwargs_1["ionmode"] = batch["ionmode_1"].float()

        if self.use_adduct:
            batch["adduct_0"] = torch.nan_to_num(
                batch["adduct_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["adduct_1"] = torch.nan_to_num(
                batch["adduct_1"], nan=0.0, posinf=0.0, neginf=0.0
            )
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

        if return_spectrum_output:
            return (*self.compute_from_embeddings(emb0, emb1), emb0, emb1)
        else:
            return self.compute_from_embeddings(emb0, emb1)

    def training_step(self, batch, batch_idx):
        logits_list = self(batch)
        logits2 = logits_list[0]  # [B] similarity
        target2 = batch["mces"].to(dtype=torch.float32, device=self.device).view(-1)

        loss = self.step(batch, batch_idx, logits_list=logits_list)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "mces_pred": logits2.detach().view(-1).cpu(),
            "mces_target": target2.cpu(),
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step — returns loss + predictions for the MCES scatter plot."""
        logits_list = self(batch)
        logits2 = logits_list[0]  # [B] similarity
        target2 = batch["mces"].to(dtype=torch.float32, device=self.device).view(-1)

        loss = self.step(batch, batch_idx, logits_list=logits_list)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # MCES MAE in raw MCES units: target2 = 1 - MCES/40, logits2 = predicted similarity
        mces_mae = (self.mces_max_value * (logits2.view(-1) - target2).abs()).mean()
        self.log("val_mces_mae", mces_mae, on_step=False, on_epoch=True, prog_bar=False)

        result = {
            "loss": loss,
            "mces_pred": logits2.view(-1).cpu(),
            "mces_target": target2.cpu(),
        }
        if self.use_mces_bucket_head:
            logits3 = logits_list[1]
            raw_mces_target = (1.0 - target2) * self.mces_max_value
            result["mces_bucket_pred"] = self._corn_decode_bin_generic(logits3).cpu()
            result["mces_bucket_target"] = self._mces_bucket_target_bins(
                raw_mces_target
            ).cpu()
        return result

    def step(
        self, batch, batch_idx, threshold=0.5, weight_loss2=None, logits_list=None
    ):
        if logits_list is None:
            logits_list = self(batch)
        logits2 = logits_list[0]
        logits3 = logits_list[1] if self.use_mces_bucket_head else None
        target2 = batch["mces"].to(dtype=torch.float32, device=self.device)
        target2 = target2.view(-1)

        squared_diff = (logits2.view(-1, 1).float() - target2.view(-1, 1).float()) ** 2
        loss2 = squared_diff.view(-1, 1).mean()

        if self.use_mces_bucket_head:
            raw_mces_target = (1.0 - target2) * self.mces_max_value
            bucket_target_bins = self._mces_bucket_target_bins(raw_mces_target)
            loss3 = self._corn_loss_generic(
                logits3, bucket_target_bins, self.mces_bucket_n_classes
            )

        use_mces_bucket = self.use_mces_bucket_head
        self.log("loss_mces", loss2, on_step=True, on_epoch=True, prog_bar=False)
        if use_mces_bucket:
            self.log(
                "loss_mces_bucket", loss3, on_step=True, on_epoch=True, prog_bar=False
            )

        loss = loss2
        if use_mces_bucket:
            loss = loss + (self.mces_bucket_loss_weight * loss3)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_from_embeddings(self, emb0: torch.Tensor, emb1: torch.Tensor):
        """
        Take two activated embeddings (after ReLU, fingerprint fusion, etc.)
        and run all the FC layers + similarity heads to produce the
        emb_sim_2 similarity score (and optional emb_sim_3 bucket logits).
        """
        emb_sim_2 = self.cosine_similarity(emb0, emb1)

        if self.use_mces_bucket_head:
            bucket_repr = torch.abs(emb0 - emb1)
            if self.mces_bucket_use_mlp:
                bucket_repr = self.mces_bucket_mlp(bucket_repr)
            emb_sim_3 = self.mces_bucket_head(
                bucket_repr
            )  # (B, mces_bucket_n_classes - 1) raw logits
            return (emb_sim_2, emb_sim_3)
        return (emb_sim_2,)

    @staticmethod
    def _corn_loss_generic(
        logits: torch.Tensor, target_bins: torch.Tensor, n_classes: int
    ) -> torch.Tensor:
        """CORN ordinal loss (Shi, Cao & Raschka, 2021/2023): for threshold j,
        only pairs whose true bin already exceeds j-1 contribute, with binary
        target 1{target_bin > j}."""
        total_loss = logits.new_tensor(0.0)
        total_count = logits.new_tensor(0.0)
        for j in range(n_classes - 1):
            mask = target_bins >= j
            if not mask.any():
                continue
            target_j = (target_bins[mask] > j).float()
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                logits[mask, j], target_j, reduction="sum"
            )
            total_count = total_count + mask.sum()
        return total_loss / total_count.clamp(min=1)

    @staticmethod
    def _corn_decode_bin_generic(logits: torch.Tensor) -> torch.Tensor:
        """Chain-rule decode (cumulative product of conditional probabilities)
        to a predicted ordinal bin index."""
        probas = torch.sigmoid(logits)
        cumprod = torch.cumprod(probas, dim=1)
        return (cumprod > 0.5).sum(dim=1)

    def _mces_bucket_target_bins(self, raw_mces: torch.Tensor) -> torch.Tensor:
        """Discretize raw MCES into bucket indices: 0 is its own class
        (self-pairs), then left-open/right-closed bins up to a final
        catch-all bin past the last edge."""
        raw = raw_mces.clamp(min=0)
        is_zero = raw == 0
        non_zero_bin = torch.bucketize(raw, self.mces_bucket_bin_edges, right=False)
        return torch.where(is_zero, torch.zeros_like(non_zero_bin), non_zero_bin + 1)


class EmbeddingExtractor(pl.LightningModule):
    def __init__(self, model_path, D_MODEL, N_LAYERS, multitasking=False, config=None):
        super().__init__()
        self.multitasking = multitasking
        self.config = config
        self.model = self.load_twin_network(
            model_path, D_MODEL, N_LAYERS
        ).spectrum_encoder
        self.relu = nn.ReLU()

    def load_twin_network(self, model_path, D_MODEL, N_LAYERS, strict=False):
        lr = self.config.optimizer.lr
        use_cosine_distance = (
            self.config.model.tasks.cosine_similarity.use_cosine_distance
        )

        if self.multitasking:
            return SimilarityModelMultitask.load_from_checkpoint(
                model_path,
                d_model=int(D_MODEL),
                n_layers=int(N_LAYERS),
                weights=None,
                lr=lr,
                use_cosine_distance=use_cosine_distance,
                strict=strict,
                use_adduct=self.config.model.features.use_adduct,
                use_ce=self.config.model.features.use_ce,
                use_ion_activation=self.config.model.features.use_ion_activation,
                use_ion_method=self.config.model.features.use_ion_method,
                use_ion_mode=self.config.model.features.use_ion_mode,
            )

        else:
            return SimilarityModel.load_from_checkpoint(
                model_path,
                d_model=int(D_MODEL),
                n_layers=int(N_LAYERS),
                weights=None,
                lr=lr,
                use_cosine_distance=use_cosine_distance,
                strict=strict,
                use_adduct=self.config.model.features.use_adduct,
                use_ce=self.config.model.features.use_ce,
                use_ion_activation=self.config.model.features.use_ion_activation,
                use_ion_method=self.config.model.features.use_ion_method,
                use_ion_mode=self.config.model.features.use_ion_mode,
            )

    def forward(self, batch):
        """The inference pass"""

        # extra data
        kwargs = {
            "precursor_mass": batch["precursor_mass"].float(),
            "precursor_charge": batch["precursor_charge"].float(),
        }

        # Add metadata fields if present in batch
        if "ionmode" in batch:
            kwargs["ionmode"] = batch["ionmode"].float()
        if "adduct" in batch:
            kwargs["adduct"] = batch["adduct"].float()
        if "ce" in batch:
            kwargs["ce"] = batch["ce"].float()
        if "ion_activation" in batch:
            kwargs["ion_activation"] = batch["ion_activation"].float()
        if "ion_method" in batch:
            kwargs["ion_method"] = batch["ion_method"].float()

        emb, _ = self.model(
            mz_array=batch["mz"].float(),
            intensity_array=batch["intensity"].float(),
            **kwargs,
        )

        emb = emb[:, 0, :]
        emb = self.relu(emb)

        return emb

    def get_embeddings(self, dataloader_spectrums, device="gpu"):
        predictor = pl.Trainer(
            max_epochs=0, enable_progress_bar=True, accelerator=device
        )
        embeddings = predictor.predict(
            self,
            dataloader_spectrums,
        )
        return self.flat_predictions(embeddings)

    def flat_predictions(self, preds):
        # flat the results
        concatenated_tensor = torch.cat(preds, dim=0)
        return concatenated_tensor.detach().numpy()
