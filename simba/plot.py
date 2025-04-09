import collections
import math
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus

import simba.similarity
import simba.utils


FragmentAnnotation = collections.namedtuple("FragmentAnnotation", ["ion_type"])
FragmentAnnotation.__str__ = lambda _: ""


def plot_mirror(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    score: Optional[str],
    filename: str,
    fragment_mz_tol: float = 0.1,
) -> None:
    """
    Plot mirror spectra showing peak matches.

    Parameters
    ----------
    spectrum1 : sus.MsmsSpectrum
        The first spectrum.
    spectrum2 : sus.MsmsSpectrum
        The second spectrum.
    score : Optional[str]
        The similarity score used. Valid values are `None`, "cosine",
        "modified_cosine", and "neutral_loss".
    filename : str
        Filename to save the figure.
    fragment_mz_tol : float
        The fragment mass tolerance to match peaks to each other
        (default: 0.1).
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex="all", figsize=(8, 4))

    if score is None:
        sim, title = None, " "
    elif score == "cosine":
        sim = similarity.cosine(spectrum1, spectrum2, fragment_mz_tol)
        title = f"Cosine similarity = {sim[0]:.4f}"
    elif score == "modified_cosine":
        sim = similarity.modified_cosine(spectrum1, spectrum2, fragment_mz_tol)
        title = f"Modified cosine similarity = {sim[0]:.4f}"
    elif score == "neutral_loss":
        sim = similarity.neutral_loss(spectrum1, spectrum2, fragment_mz_tol)
        title = f"Neutral loss similarity = {sim[0]:.4f}"
        spectrum1 = utils.spec_to_neutral_loss(spectrum1)
        spectrum2 = utils.spec_to_neutral_loss(spectrum2)
    else:
        raise ValueError("Unknown score specified")

    _annotate_matching_peaks(
        spectrum1,
        spectrum2,
        sim.matched_indices if sim is not None else None,
        sim.matched_indices_other if sim is not None else None,
        fragment_mz_tol,
    )

    # Draw lines between matching peaks.
    if sim is not None:
        for mz1, mz2 in zip(
            spectrum1.mz[sim.matched_indices],
            spectrum2.mz[sim.matched_indices_other],
        ):
            ion_type = "b" if abs(mz1 - mz2) < fragment_mz_tol else "y"
            axes[0].plot(
                [mz1, mz2],
                [0, -0.4],
                c=sup.colors[ion_type],
                ls="dotted",
                clip_on=False,
                zorder=10,
            )

    sup.spectrum(spectrum1, ax=axes[0])
    sup.spectrum(spectrum2, ax=axes[1], mirror_intensity=True)

    mz_margin = 20
    min_mz = min(spectrum1.mz[0], spectrum2.mz[0])
    max_mz = max(spectrum1.mz[-1], spectrum2.mz[-1])
    min_mz = max(0, math.floor(min_mz / mz_margin - 1) * mz_margin)
    max_mz = math.ceil(max_mz / mz_margin + 1) * mz_margin
    axes[0].set_xlim(min_mz, max_mz)

    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(-1.05, 0)

    axes[0].set_xlabel("")
    if score == "neutral_loss":
        axes[1].set_xlabel("Δm/z", style="italic")
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    fig.text(0.04, 0.5, "Intensity", va="center", rotation="vertical")
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0%}")
    )

    axes[0].set_title(title)

    plt.subplots_adjust(hspace=0.4)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def _annotate_matching_peaks(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    peak_matches1: np.ndarray,
    peak_matches2: np.ndarray,
    fragment_mz_tol: float,
) -> None:
    """
    Somewhat hacky way to get spectrum_utils to annotate matching peaks.

    Parameters
    ----------
    spectrum1 : sus.MsmsSpectrum
        The first spectrum.
    spectrum2 : sus.MsmsSpectrum
        The second spectrum.
    peak_matches1 : np.ndarray
        Matching peak indexes in the first spectrum.
    peak_matches2 : np.ndarray
        Matching peak indexes in the second spectrum.
    fragment_mz_tol : float
        The fragment mass tolerance to match peaks to each other.
    """
    if peak_matches1 is None or peak_matches2 is None:
        return
    spectrum1._annotation = np.full_like(spectrum1.mz, None, object)
    spectrum2._annotation = np.full_like(spectrum2.mz, None, object)
    for match1, match2 in zip(peak_matches1, peak_matches2):
        ion_type = (
            "b"
            if abs(spectrum1.mz[match1] - spectrum2.mz[match2]) < fragment_mz_tol
            else "y"
        )
        spectrum1._annotation[match1] = FragmentAnnotation(ion_type=ion_type)
        spectrum2._annotation[match2] = FragmentAnnotation(ion_type=ion_type)
