import torch
import torch.nn as nn
from metabo_depthcharge.encoders.spectra import MetadataEncoder, SpectrumEncoder


class SpectrumTransformerEncoderCustom(SpectrumEncoder):
    """Spectrum encoder based on metabo-depthcharge's SpectrumEncoder.

    Extends it with simba-specific fields (ion_mode) injected into the global
    token via global_token_hook. Adduct, CE, ion_activation, and
    ionization_method are handled by MetadataEncoder. Pool mode is
    configurable: "attention" (default) or "cls".

    Parameters
    ----------
    d_model : int
        Transformer hidden dimension.
    n_layers : int
        Number of transformer layers.
    dropout : float
        Dropout rate.
    pool : str
        Pooling mode passed to SpectrumEncoder: ``"attention"`` (weighted sum
        over all tokens via AttnAggregator) or ``"cls"`` (CLS token only).
    use_adduct : bool
        Adduct encoded via MetadataEncoder (categorical embedding).
    use_ce : bool
        Collision energy encoded via MetadataEncoder (linear, zero-masked).
    use_ion_activation : bool
        Ion activation encoded via MetadataEncoder (categorical embedding).
    use_ion_method : bool
        Ionization method encoded via MetadataEncoder (categorical embedding).
    use_ion_mode : bool
        Ion mode scalar (+1/-1/0), projected via nn.Linear into global token.
    """

    def __init__(
        self,
        *args,
        pool: str = "attention",
        use_adduct: bool = False,
        use_ce: bool = False,
        use_ion_activation: bool = False,
        use_ion_method: bool = False,
        use_ion_mode: bool = False,
        **kwargs,
    ):
        d_model = kwargs.get("d_model", args[0] if args else 512)

        metadata_fields = []
        if use_adduct:
            metadata_fields.append("adduct")
        if use_ce:
            metadata_fields.append("collision_energy")
        if use_ion_activation:
            metadata_fields.append("ion_activation")
        if use_ion_method:
            metadata_fields.append("ionization_method")
        metadata_enc = (
            MetadataEncoder(d_model, metadata_fields) if metadata_fields else None
        )

        super().__init__(
            *args, pool=pool, metadata_encoder=metadata_enc, **kwargs
        )

        self.use_adduct = use_adduct
        self.use_ce = use_ce
        self.use_ion_activation = use_ion_activation
        self.use_ion_method = use_ion_method
        self.use_ion_mode = use_ion_mode
        self._extra_kwargs: dict = {}

        if use_ion_mode:
            self.ion_mode_proj = nn.Linear(1, self.d_model)

    def forward(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Encode a batch of spectra.

        Accepts the simba-style keyword arguments that similarity_models passes
        (precursor_mass, adduct, ce, ionmode, ...) and returns a (B, d_model)
        tensor of spectrum embeddings via attention pooling.

        Parameters
        ----------
        mz_array : torch.Tensor of shape (B, L)
        intensity_array : torch.Tensor of shape (B, L)
        **kwargs : dict
            precursor_mass, precursor_charge, ionmode, adduct, ce,
            ion_activation, ion_method as passed by SimilarityModelMultitask.
        """
        device = mz_array.device
        batch_size = mz_array.shape[0]

        self._extra_kwargs = kwargs

        metadata = {}
        if self.use_adduct and "adduct" in kwargs:
            metadata["adduct"] = kwargs["adduct"].long().to(device).view(batch_size)
        if self.use_ce and "ce" in kwargs:
            metadata["collision_energy"] = (
                kwargs["ce"].float().to(device).view(batch_size)
            )
        if self.use_ion_activation and "ion_activation" in kwargs:
            metadata["ion_activation"] = (
                kwargs["ion_activation"].long().to(device).view(batch_size)
            )
        if self.use_ion_method and "ion_method" in kwargs:
            metadata["ionization_method"] = (
                kwargs["ion_method"].long().to(device).view(batch_size)
            )

        precursor_mz = kwargs["precursor_mass"].float().to(device).view(batch_size)

        try:
            return super().forward(
                mz=mz_array,
                intensity=intensity_array,
                precursor_mz=precursor_mz,
                metadata=metadata if metadata else None,
            )
        finally:
            self._extra_kwargs = {}

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        precursor_mzs: torch.Tensor,
    ) -> torch.Tensor:
        """Global token = parent CLS + precursor_mz + simba-specific metadata."""
        latent = super().global_token_hook(mz_array, intensity_array, precursor_mzs)

        kwargs = self._extra_kwargs
        device = mz_array.device
        batch_size = mz_array.shape[0]

        if self.use_ion_mode and "ionmode" in kwargs:
            ionmode = kwargs["ionmode"].float().to(device).view(batch_size)
            latent = latent + self.ion_mode_proj(ionmode[:, None]).squeeze(1)

        return latent
