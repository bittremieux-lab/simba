#!/usr/bin/env python3
"""Create a self-contained HTML analog-discovery report from a saved results pickle.

The report shows one query per section and its top-k matches. For each match it
includes metadata, SMILES, scores, molecule image when RDKit is available, and
spectrum plots embedded as base64 PNGs.

Expected pickle structure from your analog-discovery script:
combined_results["results_by_method"][method]["spectrums_k_retrieved"]
combined_results["results_by_method"][method]["sim_k_retrieved"]
combined_results["results_by_method"][method]["arg_max_k10"]
combined_results["results_by_method"][method].get("sim_mces")
combined_results["results_by_method"][method].get("sim_ed")
combined_results["results_by_method"][method].get("top10_smiles_matches")

Important:
- Match/reference spectra are already saved in spectrums_k_retrieved.
- Query spectra are only available if you saved them in the pickle, for example:
    combined_results["query_spectra"] = all_spectrums_query
  or if you pass --query-spectra-pickle.
- If query spectra are not available, the report still works, but query spectrum
  plots and mirror plots are skipped.
"""

from __future__ import annotations

import argparse
import base64
import html
import os
import pickle
from io import BytesIO
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from simba.core.chemistry.similarity_metrics import MolecularSimilarityMetrics

try:
    import spectrum_utils.plot as sup
except Exception:
    sup = None

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except Exception:
    Chem = None
    Draw = None





try:
    from matchms import Spectrum as MatchmsSpectrum
    from matchms.similarity import ModifiedCosine
except ImportError:
    MatchmsSpectrum = None
    ModifiedCosine = None



### PARAMETERS
FILTER_BY_TANIMOTO=True

###

def _get_first_existing(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default

def _extract_peaks(spectrum):
    mz = _get_first_existing(spectrum, ["mz", "m/z", "mzs"])
    intensities = _get_first_existing(spectrum, ["intensity", "intensities"])

    if mz is None and hasattr(spectrum, "peaks"):
        mz = _get_first_existing(spectrum.peaks, ["mz", "mzs"])
    if intensities is None and hasattr(spectrum, "peaks"):
        intensities = _get_first_existing(spectrum.peaks, ["intensities", "intensity"])

    mz = np.asarray(mz, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    keep = np.isfinite(mz) & np.isfinite(intensities) & (intensities > 0)
    mz = mz[keep]
    intensities = intensities[keep]

    order = np.argsort(mz)
    return mz[order], intensities[order]
def _get_param_first(spectrum, names, default=None):
    params = getattr(spectrum, "params", {}) or {}
    metadata = getattr(spectrum, "metadata", {}) or {}
    for name in names:
        if name in params and params[name] is not None:
            return params[name]
        if name in metadata and metadata[name] is not None:
            return metadata[name]
    return default

def _extract_precursor_mz(spectrum):
    precursor = _get_first_existing(spectrum, ["precursor_mz", "precursor"])
    if precursor is None:
        precursor = _get_param_first(
            spectrum,
            ["precursor_mz", "precursor", "pepmass", "parent_mass", "precursor_mass"],
        )

    if isinstance(precursor, (list, tuple, np.ndarray)):
        precursor = precursor[0] if len(precursor) > 0 else None

    try:
        return float(precursor)
    except Exception:
        return None



def to_matchms_spectrum(spectrum):
    if MatchmsSpectrum is None:
        raise ImportError(
            "matchms is required for modified_cosine. "
            "Install with: pip install matchms"
        )

    mz, intensities = _extract_peaks(spectrum)
    metadata = dict(getattr(spectrum, "params", {}) or {})
    metadata.update(dict(getattr(spectrum, "metadata", {}) or {}))
    precursor_mz = _extract_precursor_mz(spectrum)

    if precursor_mz is not None:
        metadata["precursor_mz"] = precursor_mz

    return MatchmsSpectrum(mz=mz, intensities=intensities, metadata=metadata)

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        return normalize_value(value[0])
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)

def get_params(spec: Any) -> dict:
    params = getattr(spec, "params", None)
    if params is None:
        params = getattr(spec, "metadata", None)
    return dict(params or {})

def get_param(spec: Any, keys: list[str], default: Any = None) -> Any:
    params = get_params(spec)
    for key in keys:
        if key in params and params[key] not in (None, ""):
            return params[key]
    for key in keys:
        value = getattr(spec, key, None)
        if value not in (None, ""):
            return value
    return default

def get_smiles(spec: Any, default: str = "") -> str:
    value = get_param(
        spec,
        ["smiles", "SMILES", "canonical_smiles", "canonicalsmiles", "inchi_smiles"],
        default,
    )
    return normalize_value(value)

def get_title(spec: Any, fallback: str) -> str:
    value = get_param(
        spec,
        ["title", "compound_name", "name", "spectrumid", "scans", "scan"],
        fallback,
    )
    return normalize_value(value)

def get_peaks(spec: Any) -> tuple[np.ndarray, np.ndarray]:
    mz = getattr(spec, "mz", None)
    intensity = getattr(spec, "intensity", None)

    if mz is None:
        mz = getattr(spec, "m/z", None)
    if intensity is None:
        intensity = getattr(spec, "intensities", None)

    if (mz is None or intensity is None) and hasattr(spec, "peaks"):
        peaks = spec.peaks
        if mz is None:
            mz = getattr(peaks, "mz", getattr(peaks, "mzs", None))
        if intensity is None:
            intensity = getattr(peaks, "intensities", getattr(peaks, "intensity", None))

    if mz is None or intensity is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    keep = np.isfinite(mz) & np.isfinite(intensity)
    mz = mz[keep]
    intensity = intensity[keep]
    order = np.argsort(mz)
    return mz[order], intensity[order]

def fig_to_base64(fig, dpi: int = 100) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close(fig)
    return encoded

def plot_spectrum_base64(spec: Any, title: str, figsize=(5.5, 2.4), dpi: int = 100) -> str:
    fig, ax = plt.subplots(figsize=figsize)
    if sup is not None:
        try:
            sup.spectrum(spec, ax=ax)
            ax.set_title(title, fontsize=9)
            fig.tight_layout(pad=0.2)
            return fig_to_base64(fig, dpi=dpi)
        except Exception:
            plt.close(fig)
            fig, ax = plt.subplots(figsize=figsize)

    mz, intensity = get_peaks(spec)
    if len(mz) > 0:
        ax.vlines(mz, 0, intensity, linewidth=0.8)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
    else:
        ax.text(0.5, 0.5, "No peak data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.2)
    return fig_to_base64(fig, dpi=dpi)

def plot_mirror_base64(query_spec: Any, match_spec: Any, title: str, figsize=(6.0, 2.6), dpi: int = 100, fragment_tol_mass: float = 10,
        fragment_tol_mode: str = "ppm",
        min_intensity: float = 0.01,
        max_num_peaks: int = 100,
        # max_num_peaks: int =40,
        scale_intensity: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=figsize)
    if sup is not None:
        try:
            query_spec =  query_spec.remove_precursor_peak(fragment_tol_mass, fragment_tol_mode)
            match_spec =  match_spec.remove_precursor_peak(fragment_tol_mass, fragment_tol_mode)

            sup.mirror(query_spec, match_spec, ax=ax)
            ax.set_title(title, fontsize=9)
            fig.tight_layout(pad=0.2)
            return fig_to_base64(fig, dpi=dpi)
        except Exception:
            plt.close(fig)
            fig, ax = plt.subplots(figsize=figsize)

    q_mz, q_int = get_peaks(query_spec)
    r_mz, r_int = get_peaks(match_spec)
    if len(q_mz) > 0:
        ax.vlines(q_mz, 0, q_int / np.nanmax(q_int), linewidth=0.8, label="Query")
    if len(r_mz) > 0:
        ax.vlines(r_mz, 0, -r_int / np.nanmax(r_int), linewidth=0.8, label="Match")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.2)
    return fig_to_base64(fig, dpi=dpi)

def molecule_base64(smiles: str, size=(220, 220)) -> Optional[str]:
    if not smiles or Chem is None or Draw is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return encoded

def compute_modified_cosine(query_spec: Any, match_spec: Any, tolerance: float = 0.1) -> tuple[Optional[float], Optional[int]]:
        """Compute modified cosine directly for one query/match pair.

    Returns (score, number_of_matching_peaks). If matchms is unavailable or the
    spectra do not contain the metadata required by ModifiedCosine, returns
    (None, None).
        """
        modified_cosine = ModifiedCosine(
        tolerance=0.1,
        mz_power=0.0,
        intensity_power=0.5,
    )
        score = modified_cosine.pair(query_spec, match_spec)

        # matchms versions may return either a tuple-like score or a structured object.
        if isinstance(score, tuple):
            cosine, n_matches = score
        elif hasattr(score, "item") and getattr(score, "dtype", None) is not None:
            item = score.item()
            cosine = item[0]
            n_matches = item[1]
        else:
            try:
                cosine = score["score"]
                n_matches = score["matches"]
            except Exception:
                cosine = float(score)
                n_matches = np.nan

        if np.isnan(cosine) or (not np.isnan(n_matches) and n_matches < 1):
            return 0.0, int(0 if np.isnan(n_matches) else n_matches)

        return float(cosine), int(0 if np.isnan(n_matches) else n_matches)



def params_table(params: dict, max_fields: Optional[int] = None) -> str:
    items = list(params.items())
    if max_fields is not None:
        items = items[:max_fields]
    rows = []
    for key, value in items:
        rows.append(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(normalize_value(value))}</td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='2'>No metadata available</td></tr>")
    return "<table class='meta-table'><tbody>" + "".join(rows) + "</tbody></table>"

def peaks_table(spec: Any, max_rows: int = 120) -> str:
    mz, intensity = get_peaks(spec)
    rows = []
    for m, inten in zip(mz[:max_rows], intensity[:max_rows]):
        rows.append(f"<tr><td>{m:.6f}</td><td>{inten:.6g}</td></tr>")
    if not rows:
        rows.append("<tr><td colspan='2'>No peak data available</td></tr>")
    return f"""
    <details class="peaks-details">
      <summary>Peaks shown: {min(len(mz), max_rows)} / {len(mz)}</summary>
      <table class="peaks-table">
        <thead><tr><th>m/z</th><th>intensity</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </details>
    """

def find_query_spectra(combined: dict, query_spectra_pickle: Optional[str] = None) -> Optional[list[Any]]:
    for key in ["query_spectra", "all_spectrums_query", "all_spectra_query"]:
        if key in combined:
            return combined[key]

    if query_spectra_pickle:
        obj = load_pickle(query_spectra_pickle)
        if isinstance(obj, dict):
            for key in ["query_spectra", "all_spectrums_query", "all_spectra_query"]:
                if key in obj:
                    return obj[key]
        if isinstance(obj, list):
            return obj

    return None

def get_method_result(combined: dict, method: str) -> dict:
    results_by_method = combined.get("results_by_method", {})
    if method not in results_by_method:
        available = sorted(results_by_method.keys())
        raise KeyError(f"Method {method!r} not found. Available methods: {available}")
    return results_by_method[method]

def safe_score(array: Any, query_index: int, match_rank_index: int, ref_index: Optional[int] = None) -> Optional[float]:
    if array is None:
        return None
    try:
        arr = np.asarray(array)
        if arr.ndim == 2:
            # For sim_k_retrieved shape: query x top_k, use match_rank_index.
            # For full sim_mces/sim_ed shape: query x reference, use ref_index.
            if ref_index is not None and ref_index < arr.shape[1] and arr.shape[1] > 100:
                return float(arr[query_index, ref_index])
            return float(arr[query_index, match_rank_index])
    except Exception:
        return None
    return None

def build_html(
    combined: dict,
    method: str,
    output_html: str,
    query_spectra: Optional[list[Any]] = None,
    top_k: int = 10,
    max_queries: Optional[int] = None,
    dpi: int = 100,
    max_peak_rows: int = 120,
    max_metadata_fields: Optional[int] = None,
    show_molecules: bool = True,
    modified_cosine_tolerance: float = 0.1,
    compute_ground_truth: bool = False,
) -> None:
    method_result = get_method_result(combined, method)

    top_spectra = method_result["spectrums_k_retrieved"]
    top_scores = method_result.get("sim_k_retrieved")
    top_indices = method_result.get("arg_max_k10")
    sim_mces = method_result.get("sim_mces")
    sim_ed = method_result.get("sim_ed")
    top10_smiles_matches = method_result.get("top10_smiles_matches")
    norm_mces_distances = method_result.get("norm_mces_distances")

    n_queries = len(top_spectra)
    if max_queries is not None:
        n_queries = min(n_queries, max_queries)

    query_available = query_spectra is not None and len(query_spectra) >= n_queries

    query_blocks = []
    #for qi in tqdm(range(100), desc="Building HTML", unit="query"):
    for qi in tqdm(range(n_queries), desc="Building HTML", unit="query"):
        query_spec = query_spectra[qi] if query_available else None

        if query_spec is not None:
            query_title = get_title(query_spec, f"Query {qi}")
            query_smiles = get_smiles(query_spec)
            query_meta_html = params_table(get_params(query_spec), max_fields=max_metadata_fields)
            query_plot_b64 = plot_spectrum_base64(query_spec, title="Query spectrum", dpi=dpi)
            query_plot_html = f"<div><h4>Query spectrum</h4><img class='spectrum-img query-img' src='data:image/png;base64,{query_plot_b64}' /></div>"

            query_mol_html = ""
            if show_molecules:
                query_mol_b64 = molecule_base64(query_smiles, size=(260, 260))
                if query_mol_b64:
                    query_mol_html = (
                        "<div><h4>Query molecule</h4>"
                        f"<img class='mol-img query-mol-img' src='data:image/png;base64,{query_mol_b64}' />"
                        "</div>"
                    )
                else:
                    query_mol_html = (
                        "<div><h4>Query molecule</h4>"
                        "<p class='muted'>No valid query molecule image available.</p></div>"
                    )

            query_visuals_html = f"<div class='plot-row query-visuals'>{query_plot_html}{query_mol_html}</div>"
            query_peaks_html = peaks_table(query_spec, max_rows=max_peak_rows)
        else:
            row = top10_smiles_matches[qi] if top10_smiles_matches and qi < len(top10_smiles_matches) else {}
            query_title = f"Query {qi}"
            query_smiles = normalize_value(row.get("query_smiles", ""))
            query_meta_html = "<p class='warning'>Query spectrum object was not found in the pickle, so query metadata, query spectrum plots, and mirror plots are skipped.</p>"
            query_plot_html = ""

            query_mol_html = ""
            if show_molecules:
                query_mol_b64 = molecule_base64(query_smiles, size=(260, 260))
                if query_mol_b64:
                    query_mol_html = (
                        "<div><h4>Query molecule</h4>"
                        f"<img class='mol-img query-mol-img' src='data:image/png;base64,{query_mol_b64}' />"
                        "</div>"
                    )
                else:
                    query_mol_html = (
                        "<div><h4>Query molecule</h4>"
                        "<p class='muted'>No valid query molecule image available.</p></div>"
                    )

            query_visuals_html = f"<div class='plot-row query-visuals'>{query_mol_html}</div>"
            query_peaks_html = ""

        matches_html = []
        all_matches_for_query = list(top_spectra[qi])
        matches_for_query = all_matches_for_query[:top_k]

        def get_norm_mces_for_rank(rank0: int, ref_index: Optional[int] = None) -> Optional[float]:
            """Return normalized MCES distance for one retrieved spectrum.

            Supports both query x retrieved-rank arrays and query x full-reference arrays.
            """
            if norm_mces_distances is None:
                return None
            try:
                arr = np.asarray(norm_mces_distances)
                if arr.ndim != 2 or qi >= arr.shape[0]:
                    return None

                # If the stored row aligns directly with the retrieved spectra, index by rank.
                if arr.shape[1] == len(all_matches_for_query):
                    return float(arr[qi, rank0])

                # Otherwise treat it as a full reference-library matrix when possible.
                if ref_index is not None and 0 <= ref_index < arr.shape[1]:
                    return float(arr[qi, ref_index])

                # Fallback for top-k style matrices whose width may be larger than this row.
                if 0 <= rank0 < arr.shape[1]:
                    return float(arr[qi, rank0])
            except (TypeError, ValueError, IndexError):
                return None
            return None

        def get_ref_index(rank0: int) -> Optional[int]:
            try:
                if top_indices is not None:
                    arr = np.asarray(top_indices)
                    if arr.ndim == 2 and qi < arr.shape[0] and rank0 < arr.shape[1]:
                        return int(arr[qi, rank0])
            except (TypeError, ValueError, IndexError):
                pass
            return None

        # Compute ground-truth metrics for all retrieved matches of this query in one batch.
        ground_truth_mces_values = None
        ground_truth_ed_values = None
        if query_spec is not None and len(matches_for_query) > 0:
            #print(f'Computing ground truth for {query_spec.params["smiles"]}')
            try:
                ground_truth_ed_values = None
                ground_truth_mces_values = None
                tanimotos=None
                if compute_ground_truth and query_spec is not None:
                    all_spectrums_0 = [query_spec]
                    all_spectrums_1 = list(all_matches_for_query)

                    tanimotos = MolecularSimilarityMetrics.compute_tanimoto(all_spectrums_0, all_spectrums_1)
                    tanimotos = tanimotos.reshape(-1)
                    ground_truth_ed_values = []
                    ground_truth_mces_values = []
                    
                    #print('Computing ed')
                    #ged=[]
                    #for s in all_spectrums_1:
                    #    print(f'Computing ed for smiles: {s.params["smiles"]}')
                    #    try:
                    #        ged_sub = MolecularSimilarityMetrics.compute_edit_distance(
                    #                                            all_spectrums_0, [s]
                    #                                        )
                    #        print('Finished ed')
                    #    except:
                    #        ged_sub = None
                    #        print('Error computing ed')
                    #    ged.append(ged_sub)
                        
                    print('Computing mces')
                    gmces = MolecularSimilarityMetrics.compute_mces(
                                                        all_spectrums_0, all_spectrums_1, threshold=40,
                                                    )
                    
                    ground_truth_ed_values.append(None)
                    ground_truth_mces_values.append(gmces)
                            
                ground_truth_ed_values = np.asarray(ground_truth_ed_values).reshape(-1)
                ground_truth_mces_values = np.asarray(ground_truth_mces_values).reshape(-1)
            except Exception as exc:
                print(f"Warning: ground-truth metrics failed for query {qi}: {exc}")
                ground_truth_ed_values = None
                ground_truth_mces_values = None

        def render_match_card(rank0: int, card_label: str, is_best: bool = False) -> str:
            match_spec = all_matches_for_query[rank0]
            rank = rank0 + 1
            match_smiles = get_smiles(match_spec)
            match_title = get_title(match_spec, f"Match {rank}")
            ref_index = get_ref_index(rank0)

            pred_mces = safe_score(sim_mces, qi, rank0, ref_index=ref_index)
            pred_ed = safe_score(sim_ed, qi, rank0, ref_index=ref_index)
            norm_mces = get_norm_mces_for_rank(rank0, ref_index=ref_index)

            ground_truth_mces = None
            ground_truth_ed = None
            
            if ground_truth_mces_values is not None and rank0 < len(ground_truth_mces_values):
                try:
                    ground_truth_mces = float(ground_truth_mces_values[rank0])
                except (TypeError, ValueError):
                    pass
            if ground_truth_ed_values is not None and rank0 < len(ground_truth_ed_values):
                try:
                    ground_truth_ed = float(ground_truth_ed_values[rank0])
                except (TypeError, ValueError):
                    pass
            if tanimotos is not None and rank0 < len(tanimotos):
                            try:
                                tanimoto = float(tanimotos[rank0])
                            except (TypeError, ValueError):
                                pass
            
            modified_cosine = None
            modified_cosine_matches = None
            if query_spec is not None:
                try:
                    modified_cosine, modified_cosine_matches = compute_modified_cosine(
                        to_matchms_spectrum(query_spec),
                        to_matchms_spectrum(match_spec),
                        tolerance=modified_cosine_tolerance,
                    )
                except Exception:
                    modified_cosine = None
                    modified_cosine_matches = None

            score_rows = []
            if ref_index is not None:
                score_rows.append(("Reference index", ref_index))
            if norm_mces is not None and np.isfinite(norm_mces):
                score_rows.append(("Normalized MCES distance", f"{norm_mces:.6g}"))
            if modified_cosine is not None and np.isfinite(modified_cosine):
                score_rows.append(("Modified cosine", f"{modified_cosine:.6f}"))
            elif query_spec is not None:
                score_rows.append(("Modified cosine", "N/A"))
            if pred_mces is not None and np.isfinite(pred_mces):
                score_rows.append(("Predicted MCES", f"{pred_mces:.6g}"))
            if ground_truth_mces is not None and np.isfinite(ground_truth_mces):
                score_rows.append(("Ground-truth MCES", f"{ground_truth_mces:.6g}"))
            elif query_spec is not None:
                score_rows.append(("Ground-truth MCES", "N/A"))
            if tanimoto is not None:
                score_rows.append(("Tanimoto", f"{tanimoto:.6g}"))
            else:
                score_rows.append(("Tanimoto", "N/A"))
            #if pred_ed is not None and np.isfinite(pred_ed):
            #    score_rows.append(("Predicted edit distance", f"{pred_ed:.6g}"))
            #if ground_truth_ed is not None and np.isfinite(ground_truth_ed):
            #    score_rows.append(("Ground-truth edit distance", f"{ground_truth_ed:.6g}"))
            #elif query_spec is not None:
            #    score_rows.append(("Ground-truth edit distance", "N/A"))
            score_rows.append(("SMILES", match_smiles))

            score_table = "<table class='meta-table'><tbody>" + "".join(
                f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                for k, v in score_rows
            ) + "</tbody></table>"

            plot_name = "best match" if is_best else f"match {rank}"
            if query_spec is not None:
                plot_b64 = plot_mirror_base64(
                    query_spec, match_spec,
                    title=f"Mirror plot: query vs {plot_name}",
                    dpi=dpi,
                )
                plot_title = "Mirror plot"
            else:
                plot_b64 = plot_spectrum_base64(
                    match_spec, title=f"{card_label} spectrum", dpi=dpi
                )
                plot_title = "Match spectrum"
            plot_html = (
                f"<div><h4>{plot_title}</h4>"
                f"<img class='spectrum-img' src='data:image/png;base64,{plot_b64}' /></div>"
            )

            mol_html = ""
            if show_molecules:
                mol_b64 = molecule_base64(match_smiles)
                if mol_b64:
                    mol_html = (
                        "<div><h4>Molecule</h4>"
                        f"<img class='mol-img' src='data:image/png;base64,{mol_b64}' /></div>"
                    )
                else:
                    mol_html = "<div><h4>Molecule</h4><p class='muted'>No valid molecule image available.</p></div>"

            card_class = "match-card best-match-card" if is_best else "match-card"
            return f"""
            <details class="{card_class}"{" open" if is_best else ""}>
              <summary>{html.escape(card_label)}: {html.escape(match_title)} <span class="smiles-inline">{html.escape(match_smiles[:90])}</span></summary>
              <div class="card-body">
                {score_table}
                <div class="plot-row">{plot_html}{mol_html}</div>
                <h4>Match metadata</h4>
                {params_table(get_params(match_spec), max_fields=max_metadata_fields)}
                {peaks_table(match_spec, max_rows=max_peak_rows)}
              </div>
            </details>
            """

        # Select the retrieved spectrum with the minimum finite normalized MCES distance.
        best_match_html = ""
        best_rank0= np.argmin(ground_truth_mces_values)

        if best_rank0 is not None:
            best_match_html = render_match_card(best_rank0, "Best match", is_best=True)
        else:
            best_match_html = (
                "<p class='warning'>Best match could not be determined because no finite "
                "norm_mces_distances value was available for this query.</p>"
            )

        for rank0, _match_spec in enumerate(matches_for_query):
            matches_html.append(render_match_card(rank0, f"#{rank0 + 1}"))

        query_blocks.append(f"""
        <details class="query-card" open>
          <summary>
            <span class="query-title">{html.escape(query_title)}</span>
            <span class="query-subtitle">query_index={qi} | SMILES: {html.escape(query_smiles)}</span>
          </summary>
          <div class="card-body">
            <h3>Query information</h3>
            {query_visuals_html}
            {query_meta_html}
            {query_peaks_html}
            <h3>Best match (lowest MCES distance)</h3>
            {best_match_html}
            <h3>Top {len(matches_for_query)} matches by {html.escape(method)}</h3>
            {''.join(matches_html)}
          </div>
        </details>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Analog discovery report - {html.escape(method)}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .summary {{ margin-bottom: 18px; color: #444; }}
  .query-card, .match-card, .peaks-details {{ background: white; border: 1px solid #ddd; border-radius: 8px; margin: 12px 0; overflow: hidden; }}
  .query-card > summary, .match-card > summary, .peaks-details > summary {{ cursor: pointer; padding: 12px 14px; font-weight: 700; list-style: none; }}
  .query-card > summary {{ background: #eef3f8; }}
  .match-card > summary {{ background: #fafafa; }}
  .best-match-card {{ border: 2px solid #888; }}
  .best-match-card > summary {{ background: #f0f0f0; }}
  .peaks-details > summary {{ background: #f5f5f5; }}
  .card-body {{ padding: 14px; }}
  .query-title {{ display: block; font-size: 16px; }}
  .query-subtitle, .smiles-inline, .muted {{ display: block; color: #666; font-size: 12px; margin-top: 4px; font-weight: 400; overflow-wrap: anywhere; }}
  .warning {{ padding: 10px 12px; background: #fff8db; border: 1px solid #ead47a; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0 16px 0; font-size: 13px; background: white; }}
  th, td {{ border: 1px solid #ddd; padding: 7px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
  th {{ background: #f2f2f2; width: 26%; }}
  .plot-row {{ display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; margin: 10px 0 18px 0; }}
  .spectrum-img {{ max-width: 760px; width: 100%; border: 1px solid #ddd; border-radius: 6px; background: white; }}
  .query-img {{ max-width: 900px; }}
  .mol-img {{ max-width: 240px; border: 1px solid #ddd; border-radius: 6px; background: white; padding: 8px; }}
  .query-mol-img {{ max-width: 280px; }}
  .query-visuals {{ padding: 10px; background: #fbfdff; border: 1px solid #dbe7f2; border-radius: 8px; }}
</style>
</head>
<body>
<h1>Analog discovery report</h1>
<div class="summary">
  <div><strong>Method:</strong> {html.escape(method)}</div>
  <div><strong>Queries shown:</strong> {n_queries}</div>
  <div><strong>Top matches per query:</strong> {top_k}</div>
  <div><strong>Query spectra available:</strong> {query_available}</div>
</div>
{''.join(query_blocks)}
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML report written to: {os.path.abspath(output_html)}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pickle", default="/data/simba_files_/analog_discovery_simba_modified_cosine_results_tf_unique_compounds.pkl")
    #parser.add_argument("--input-pickle", default="/data/simba_files_/analog_discovery_ground_truth.pkl")
    parser.add_argument("--method", default="simba", help="Example: simba or modified_cosine")
    #parser.add_argument("--method", default="ground_truth_mces", help="Example: simba or modified_cosine")
    parser.add_argument("--output-html", default="/data/simba_files_/analog_discovery_report_tf_unique_compounds_filter_tanimoto.html")
    #parser.add_argument("--output-html", default="/data/simba_files_/analog_discovery_ground_truth.html")
    parser.add_argument("--query-spectra-pickle", default=None, help="Optional pickle with all_spectrums_query/query_spectra/list of query spectra.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=None, help="Use a small value first to test, e.g. 5.")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--max-peak-rows", type=int, default=120)
    parser.add_argument("--max-metadata-fields", type=int, default=None)
    parser.add_argument("--no-molecules", action="store_true")
    parser.add_argument("--modified-cosine-tolerance", type=float, default=0.1,
                        help="Fragment m/z tolerance in Da for ModifiedCosineGreedy (default: 0.1).")
    parser.add_argument(
        "--compute-ground-truth",
        action="store_true",
        default=True,
        help="Compute and display ground-truth MCES and edit distance for each retrieved match.",
    )
    args = parser.parse_args()

    combined = load_pickle(args.input_pickle)
    query_spectra = find_query_spectra(combined, args.query_spectra_pickle)

    build_html(
        combined=combined,
        method=args.method,
        output_html=args.output_html,
        query_spectra=query_spectra,
        top_k=args.top_k,
        max_queries=args.max_queries,
        dpi=args.dpi,
        max_peak_rows=args.max_peak_rows,
        max_metadata_fields=args.max_metadata_fields,
        show_molecules=not args.no_molecules,
        modified_cosine_tolerance=args.modified_cosine_tolerance,
        compute_ground_truth=True,
    )

if __name__ == "__main__":
    main()

