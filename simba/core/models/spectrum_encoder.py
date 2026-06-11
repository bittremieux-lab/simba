import torch
from depthcharge.encoders import FloatEncoder
from depthcharge.transformers import SpectrumTransformerEncoder
from metabo_depthcharge.encoders.spectra import MetadataEncoder


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

        Parameters
        ----------
        use_adduct: bool
            Whether to include adduct information in the encoding (default: False).
            Adduct is encoded as a categorical index using the metabo-depthcharge
            vocabulary via MetadataEncoder.
        use_ce: bool
            Whether to include collision energy in the encoding (default: False).
            CE=0 is treated as missing and masked to zero (no bias leakage).
        use_ion_activation: bool
            Whether to include ion activation information in the encoding (default: False).
        use_ion_method: bool
            Whether to include ionization method in the encoding (default: False).
        use_ion_mode: bool
            Whether to include ion mode in the encoding (default: False).
        """
        super().__init__(*args, **kwargs)
        self.use_adduct = use_adduct
        self.use_ce = use_ce
        self.use_ion_activation = use_ion_activation
        self.use_ion_method = use_ion_method
        self.use_ion_mode = use_ion_mode

        metadata_fields = []
        if use_adduct:
            metadata_fields.append("adduct")
        if use_ce:
            metadata_fields.append("collision_energy")
        if metadata_fields:
            self.metadata_encoder = MetadataEncoder(self.d_model, metadata_fields)
        else:
            self.metadata_encoder = None

        if use_ion_activation:
            self.ion_activation_encoder = FloatEncoder(self.d_model)
        if use_ion_method:
            self.ion_method_encoder = FloatEncoder(self.d_model)

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ):
        device = mz_array.device
        dtype = mz_array.dtype
        batch_size = mz_array.shape[0]

        placeholder = torch.zeros(
            (batch_size, self.d_model), dtype=dtype, device=device
        )

        precursor_mass = kwargs["precursor_mass"].float().to(device).view(batch_size)
        placeholder[:, 0] = precursor_mass

        precursor_charge = kwargs["precursor_charge"].float().to(device).view(batch_size)
        if self.use_ion_mode:
            placeholder[:, 1] = precursor_charge

        ionmode = kwargs["ionmode"].float().to(device).view(batch_size)
        if self.use_ion_mode:
            placeholder[:, 2] = ionmode

        if self.metadata_encoder is not None:
            metadata = {}
            if self.use_adduct:
                metadata["adduct"] = kwargs["adduct"].long().to(device).view(batch_size)
            if self.use_ce:
                metadata["collision_energy"] = kwargs["ce"].float().to(device).view(batch_size)
            placeholder = placeholder + self.metadata_encoder(metadata)

        if self.use_ion_activation:
            ia = kwargs["ion_activation"].float().to(device).view(batch_size, -1)
            placeholder = placeholder + self.ion_activation_encoder(ia).mean(dim=1)

        if self.use_ion_method:
            im = kwargs["ion_method"].float().to(device).view(batch_size, -1)
            placeholder = placeholder + self.ion_method_encoder(im).mean(dim=1)

        return torch.nan_to_num(placeholder, nan=0.0, posinf=0.0, neginf=0.0)
