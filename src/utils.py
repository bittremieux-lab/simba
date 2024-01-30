import numpy as np
import spectrum_utils.spectrum as sus
import hashlib

def spectrum_hash(mz, intensities, hash_length: int = 20,
                  mz_precision: int = 5, intensity_precision: int = 2):
    """Compute hash from mz-intensity pairs of all peaks in spectrum.
    Method is inspired by SPLASH (doi:10.1038/nbt.3689).
    """
    mz_precision_factor = 10 ** mz_precision
    intensity_precision_factor = 10 ** intensity_precision

    def format_mz(mz):
        return int(mz * mz_precision_factor)

    def format_intensity(intensity):
        return int(intensity * intensity_precision_factor)

    peak_list = [(format_mz(m), format_intensity(inten)) for m,inten in zip(mz, intensities)]

    # Sort by increasing m/z and then by decreasing intensity
    peak_list.sort(key=lambda x: (x[0], - x[1]))

    encoded = " ".join(":".join(map(str, x)) for x in peak_list).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:hash_length]

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
