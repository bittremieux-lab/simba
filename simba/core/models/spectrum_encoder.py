import torch
from depthcharge.encoders import FloatEncoder
from depthcharge.transformers import (
    SpectrumTransformerEncoder,
)  # PeptideTransformerEncoder,


class SpectrumTransformerEncoderCustom(SpectrumTransformerEncoder):
    def __init__(
        self,
        *args,
        use_adduct: bool = False,
        use_ce: bool = False,
        use_ion_activation: bool = False,
        use_ion_method: bool = False,
        use_ion_mode: bool = False,
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
        self.use_encoders = False
        super().__init__(*args, **kwargs)
        self.use_adduct = use_adduct
        self.use_ce = use_ce
        self.use_ion_activation = use_ion_activation
        self.use_ion_method = use_ion_method
        self.use_ion_mode = use_ion_mode

        if self.use_encoders:
            if self.use_adduct:
                self.adduct_encoder = FloatEncoder(self.d_model)
                self.ionmode_encoder = FloatEncoder(self.d_model)
            if self.use_ce:
                self.ce_encoder = FloatEncoder(
                    self.d_model,
                )

            if self.use_ion_activation:
                self.ion_activation_encoder = FloatEncoder(self.d_model)

            if self.use_ion_method:
                self.ion_method_encoder = FloatEncoder(self.d_model)
            if (
                self.use_adduct
                or self.use_ce
                or self.use_ion_activation
                or self.use_ion_method
            ):
                self.precursor_mz_encoder = FloatEncoder(self.d_model)

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ):
        device = mz_array.device
        dtype = mz_array.dtype
        batch_size = mz_array.shape[0]

        if not (self.use_encoders):
            placeholder = torch.zeros(
                (batch_size, self.d_model), dtype=dtype, device=device
            )
            precursor_mass = (
                kwargs["precursor_mass"].float().to(device).view(batch_size)
            )
            placeholder[:, 0] = precursor_mass

            precursor_charge = (
                kwargs["precursor_charge"].float().to(device).view(batch_size)
            )
            # skip the use of the precursor charge field
            if self.use_ion_mode:
                placeholder[:, 1] = precursor_charge

            current_idx = 2  # keep track of where to insert metadata

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

            # ensure there are no nans
            placeholder = torch.nan_to_num(placeholder, nan=0.0, posinf=0.0, neginf=0.0)

        else:
            precursor_mass = (
                kwargs["precursor_mass"].float().to(device).view(batch_size)
            )
            precursor_mass_rep = self.precursor_mz_encoder(
                precursor_mass[:, None]
            ).squeeze(1)
            placeholder = precursor_mass_rep + 0 * precursor_mass_rep

            if self.use_adduct:
                ionmode = kwargs["ionmode"].float().to(device).view(batch_size)
                adduct = kwargs["adduct"].float().to(device).view(batch_size, -1)
                ionmode_rep = self.ionmode_encoder(ionmode[:, None]).squeeze(1)
                adduct_rep = self.adduct_encoder(adduct).mean(dim=1)

                placeholder = placeholder + (ionmode_rep + adduct_rep)

            if self.use_ce:
                ce = kwargs["ce"].float().to(device).view(batch_size)
                ce_rep = self.ce_encoder(ce[:, None]).squeeze(1)
                placeholder = placeholder + ce_rep

            if self.use_ion_method:
                im = kwargs["ion_method"].float().to(device).view(batch_size, -1)
                im_rep = self.ion_method_encoder(im).mean(dim=1)
                placeholder = placeholder + im_rep

            if self.use_ion_activation:
                ia = kwargs["ion_activation"].float().to(device).view(batch_size, -1)
                ia_rep = self.ion_activation_encoder(ia).mean(dim=1)
                placeholder = placeholder + ia_rep

            placeholder = torch.nan_to_num(placeholder, nan=0.0, posinf=0.0, neginf=0.0)

        return placeholder
