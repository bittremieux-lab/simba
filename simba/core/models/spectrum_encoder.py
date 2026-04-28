import torch
from depthcharge.transformers import (
    SpectrumTransformerEncoder,
)  # PeptideTransformerEncoder,
from depthcharge.encoders import FloatEncoder
import torch.nn as nn

class SpectrumTransformerEncoderCustom(SpectrumTransformerEncoder):
    def __init__(
        self,
        *args,
        use_adduct: bool = False,
        use_ce: bool = False,
        use_ion_activation: bool = False,
        use_ion_method: bool = False,
        use_ion_mode:  bool =False,
        **kwargs,
    ):
        """
        Custom Spectrum Transformer Encoder with optional metadata usage.

        Attributes
        ----------
        use_adduct: bool
            Whether to include adduct information in the encoding (default: False).
        use_ce: bool
            Whether to include collision energy in the encoding (default: False).
        use_ion_activation: bool
            Whether to include ion activation information in the encoding (default: False).
        use_ion_method: bool
            Whether to include ionization method in the encoding (default: False).
        """
        self.use_encoders=True
        super().__init__(*args, **kwargs)
        self.use_adduct = use_adduct
        self.use_ce = use_ce
        self.use_ion_activation = use_ion_activation
        self.use_ion_method = use_ion_method
        self.use_ion_mode = use_ion_mode

        # Only used when self.use_encoders = True
        if self.use_encoders:
            metadata_hidden_dim= 256
            adduct_num_embeddings=128
            adduct_embedding_dim=16
            self.adduct_embedding = nn.Embedding(
                num_embeddings=adduct_num_embeddings,
                embedding_dim=adduct_embedding_dim,
            )

            # Input size to the metadata MLP
            metadata_input_dim = 1  # precursor_mass always included

            if self.use_ion_mode:
                metadata_input_dim += 1  # precursor_charge
                metadata_input_dim += 1  # ionmode

            if self.use_adduct:
                metadata_input_dim += adduct_embedding_dim

            if self.use_ce:
                metadata_input_dim += 1

            # These sizes assume ion_activation and ion_method are vectors already
            # projected as raw numeric features. If they are one-hot vectors, this works.
            # If you know their dimensions beforehand, set them explicitly.
            self._use_ia_raw = self.use_ion_activation
            self._use_im_raw = self.use_ion_method

            # LazyLinear avoids having to know ia/im dims at __init__ time
            self.metadata_fc1 = nn.LazyLinear(metadata_hidden_dim)
            self.metadata_fc2 = nn.Linear(metadata_hidden_dim, self.d_model)
            self.metadata_activation = nn.ReLU()

    def precursor_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ):
        device = mz_array.device
        dtype = mz_array.dtype
        batch_size = mz_array.shape[0]
        # ============================================================
        # Original behavior: keep exactly the previous placeholder logic
        # ============================================================
        if not self.use_encoders:
            placeholder = torch.zeros(
                (batch_size, self.d_model), dtype=dtype, device=device
            )

            precursor_mass = kwargs["precursor_mass"].float().to(device).view(batch_size)
            placeholder[:, 0] = precursor_mass

            precursor_charge = kwargs["precursor_charge"].float().to(device).view(batch_size)

            # original logic preserved
            if self.use_ion_mode:
                placeholder[:, 1] = precursor_charge

            current_idx = 2

            ionmode = kwargs["ionmode"].float().to(device).view(batch_size)
            if self.use_ion_mode:
                placeholder[:, current_idx] = ionmode
            current_idx += 1

            adduct = kwargs["adduct"].float().to(device).view(batch_size, -1)
            stop_idx = current_idx + adduct.shape[1]
            if self.use_adduct:
                placeholder[:, current_idx:stop_idx] = adduct
            current_idx = stop_idx

            ce = kwargs["ce"].float().to(device).view(batch_size)
            if self.use_ce:
                placeholder[:, current_idx] = ce
            current_idx += 1

            ia = kwargs["ion_activation"].float().to(device).view(batch_size, -1)
            stop_idx = current_idx + ia.shape[1]
            if self.use_ion_activation:
                placeholder[:, current_idx:stop_idx] = ia
            current_idx = stop_idx

            im = kwargs["ion_method"].float().to(device).view(batch_size, -1)
            stop_idx = current_idx + im.shape[1]
            if self.use_ion_method:
                placeholder[:, current_idx:stop_idx] = im
            current_idx = stop_idx

            placeholder = torch.nan_to_num(
                placeholder, nan=0.0, posinf=0.0, neginf=0.0
            )
            return placeholder

        # ============================================================
        # New behavior: encode metadata using embedding + 2 FC layers
        # Output shape remains (batch_size, self.d_model)
        # ============================================================
        features = []

        # precursor_mass is always included
        precursor_mass = kwargs["precursor_mass"].float().to(device).view(batch_size, 1)
        precursor_mass = torch.nan_to_num(precursor_mass, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(precursor_mass)

        if self.use_ion_mode:
            precursor_charge = kwargs["precursor_charge"].float().to(device).view(batch_size, 1)
            precursor_charge = torch.nan_to_num(
                precursor_charge, nan=0.0, posinf=0.0, neginf=0.0
            )
            features.append(precursor_charge)

            ionmode = kwargs["ionmode"].float().to(device).view(batch_size, 1)
            ionmode = torch.nan_to_num(ionmode, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(ionmode)

        if self.use_adduct:
            adduct_raw = kwargs["adduct"].to(device)

            # Case 1: one-hot or score vector, shape (batch_size, n_adducts)
            if adduct_raw.dim() == 2:
                adduct_idx = adduct_raw.argmax(dim=1).long()

            # Case 2: already a scalar per sample, shape (batch_size,) or (batch_size, 1)
            elif adduct_raw.dim() == 1:
                adduct_idx = adduct_raw.long()
            elif adduct_raw.dim() == 3 and adduct_raw.shape[-1] == 1:
                adduct_idx = adduct_raw.view(batch_size).long()
            else:
                adduct_idx = adduct_raw.view(batch_size).long()

            adduct_idx = torch.clamp(
              adduct_idx,
                min=0,
                max=self.adduct_embedding.num_embeddings - 1,
            )

            adduct_emb = self.adduct_embedding(adduct_idx)
            adduct_emb = torch.nan_to_num(adduct_emb, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(adduct_emb)
        if self.use_ce:
            ce = kwargs["ce"].float().to(device).view(batch_size, 1)
            ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(ce)

        if self.use_ion_activation:
            ia = kwargs["ion_activation"].float().to(device).view(batch_size, -1)
            ia = torch.nan_to_num(ia, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(ia)

        if self.use_ion_method:
            im = kwargs["ion_method"].float().to(device).view(batch_size, -1)
            im = torch.nan_to_num(im, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(im)

        metadata_input = torch.cat(features, dim=1)  # (batch_size, total_metadata_dim)
        metadata_input = metadata_input.to(dtype)

        placeholder = self.metadata_fc1(metadata_input)
        placeholder = self.metadata_activation(placeholder)
        placeholder = self.metadata_fc2(placeholder)

        placeholder = torch.nan_to_num(
            placeholder, nan=0.0, posinf=0.0, neginf=0.0
        ).to(dtype)

        return placeholder
