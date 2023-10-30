import numpy as np
import spectrum_utils.spectrum as sus


def spec_to_neutral_loss(spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
    """
    Convert a spectrum to a neutral loss spectrum by subtracting the peak m/z
    values from the precursor m/z.

    Parameters
    ----------
    spectrum : sus.MsmsSpectrum
        The spectrum to be converted to its neutral loss spectrum.

    Returns
    -------
    sus.MsmsSpectrum
        The converted neutral loss spectrum.
    """
    # Add ghost peak at 0 m/z to anchor the m/z range after transformation.
    mz, intensity = np.copy(spectrum.mz), np.copy(spectrum.intensity)
    mz, intensity = np.insert(mz, 0, [0]), np.insert(intensity, 0, [0])
    # Create neutral loss peaks and make sure the peaks are in ascending m/z
    # order.
    # TODO: This assumes [M+H]x charged ions.
    adduct_mass = 1.007825
    neutral_mass = (
        spectrum.precursor_mz - adduct_mass
    ) * spectrum.precursor_charge
    mz, intensity = ((neutral_mass + adduct_mass) - mz)[::-1], intensity[::-1]
    return sus.MsmsSpectrum(
        spectrum.identifier,
        spectrum.precursor_mz,
        spectrum.precursor_charge,
        np.ascontiguousarray(mz),
        np.ascontiguousarray(intensity),
        spectrum.retention_time,
    )
