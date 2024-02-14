import torch

from depthcharge.transformers import (
    SpectrumTransformerEncoder,
    PeptideTransformerEncoder,
)


class SpectrumTransformerEncoderCustom(SpectrumTransformerEncoder):

    def precursor_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ):
        mass_precursor = torch.tensor(kwargs["precursor_mass"].float())
        charge_precursor = torch.tensor(kwargs["precursor_charge"].float())

        # placeholder
        placeholder = torch.zeros((mz_array.shape[0], self.d_model)).type_as(mz_array)

        placeholder[:, 0:2] = torch.cat(
            (mass_precursor.view(-1, 1), charge_precursor.view(-1, 1)), dim=-1
        )
        return placeholder
