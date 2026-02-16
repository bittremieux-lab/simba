"""Workflow for analog discovery using SIMBA predictions."""

import json
import os
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from simba.analog_discovery.simba_analog_discovery import AnalogDiscovery
from simba.core.data.preprocessing_simba import PreprocessingSimba
from simba.core.models.simba_model import Simba
from simba.utils.logger_setup import logger


def run_analog_discovery(cfg: DictConfig) -> dict:
    """Run analog discovery workflow.

    Args:
        cfg: Hydra configuration object with paths and analog_discovery settings

    Returns:
        Dictionary with results_summary and statistics
    """
    # Setup paths
    model_path = Path(cfg.paths.model_path)
    query_spectra = Path(cfg.paths.query_spectra)
    reference_spectra = Path(cfg.paths.reference_spectra)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir) + os.sep

    # Load spectra using Hydra config
    all_spectrums_query = PreprocessingSimba.load_spectra(
        str(query_spectra), cfg, use_gnps_format=cfg.analog_discovery.use_gnps_format
    )

    all_spectrums_reference = PreprocessingSimba.load_spectra(
        str(reference_spectra),
        cfg,
        use_gnps_format=cfg.analog_discovery.use_gnps_format,
    )

    if len(all_spectrums_query) == 0:
        raise ValueError("No query spectra found! Check your input file.")
    if len(all_spectrums_reference) == 0:
        raise ValueError("No reference spectra found! Check your input file.")

    # Initialize SIMBA model
    simba_model = Simba(
        str(model_path),
        config=cfg,
        device=cfg.analog_discovery.device,
        cache_embeddings=cfg.analog_discovery.cache_embeddings,
    )

    # Run predictions
    sim_ed, sim_mces = simba_model.predict(
        all_spectrums_query,
        all_spectrums_reference,
    )

    # Compute rankings
    ranking = AnalogDiscovery.compute_ranking(sim_mces, sim_ed)

    # Save full rankings if requested
    if cfg.analog_discovery.save_rankings:
        np.save(output_path + "similarity_ed.npy", sim_ed)
        np.save(output_path + "similarity_mces.npy", sim_mces)
        np.save(output_path + "rankings.npy", ranking)

    # Generate distribution plots
    plot_distributions(sim_ed, sim_mces, ranking, output_path)

    # Process queries
    results_summary = []
    query_index = cfg.analog_discovery.query_index
    top_k = cfg.analog_discovery.top_k
    compute_ground_truth = cfg.analog_discovery.compute_ground_truth
    save_individual_plots = cfg.analog_discovery.save_individual_plots

    if query_index is not None:
        # Process single query
        if query_index >= len(all_spectrums_query):
            raise ValueError(
                f"Query index {query_index} is out of range! "
                f"Available indices: 0-{len(all_spectrums_query) - 1}"
            )

        result = process_single_query(
            query_index,
            all_spectrums_query,
            all_spectrums_reference,
            sim_ed,
            sim_mces,
            ranking,
            top_k,
            compute_ground_truth,
            output_path,
            save_plots=True,
        )
        results_summary.append(result)
    else:
        # Process all queries
        for idx in range(len(all_spectrums_query)):
            result = process_single_query(
                idx,
                all_spectrums_query,
                all_spectrums_reference,
                sim_ed,
                sim_mces,
                ranking,
                top_k,
                compute_ground_truth,
                output_path,
                save_plots=save_individual_plots,
            )
            results_summary.append(result)

    # Save results summary
    results_file = output_path + "analog_discovery_results.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    # Compute statistics
    statistics = compute_summary_statistics(results_summary, compute_ground_truth)

    return {
        "results_summary": results_summary,
        "statistics": statistics,
        "output_dir": str(output_dir),
    }


def process_single_query(
    query_index: int,
    all_spectrums_query,
    all_spectrums_reference,
    sim_ed,
    sim_mces,
    ranking,
    top_k: int,
    compute_ground_truth: bool,
    output_path: str,
    save_plots: bool = True,
) -> dict:
    """Process a single query spectrum and find its top matches."""
    from simba.core.data.ground_truth import GroundTruth

    spectra_query = all_spectrums_query[query_index]

    # Get top K matches
    best_matches_indices = np.argsort(ranking[query_index])[-top_k:][::-1]
    spectra_matches = [all_spectrums_reference[ind] for ind in best_matches_indices]

    # Get predictions for top matches
    top_rankings = ranking[query_index, best_matches_indices]
    top_ed_predictions = sim_ed[query_index, best_matches_indices]
    top_mces_predictions = sim_mces[query_index, best_matches_indices]

    # Build result dictionary
    result = {
        "query_index": int(query_index),
        "query_smiles": spectra_query.params.get("smiles", "N/A"),
        "query_precursor_mz": float(spectra_query.precursor_mz),
        "top_matches": [],
    }

    # Compute ground truth if requested
    ground_truth_mces = None
    ground_truth_ed = None

    if compute_ground_truth and "smiles" in spectra_query.params:
        try:
            ground_truth_mces = GroundTruth.compute_mces(
                [spectra_query], spectra_matches
            )
            ground_truth_ed = GroundTruth.compute_edit_distance(
                [spectra_query], spectra_matches
            )
            ground_truth_mces = ground_truth_mces[0]  # Extract first row
            ground_truth_ed = ground_truth_ed[0]
        except Exception:
            # If ground truth computation fails, continue without it
            logger.exception("Failed computing ground truth for query spectrum")

    # Add match information
    for i, match_idx in enumerate(best_matches_indices):
        match_info = {
            "rank": i + 1,
            "reference_index": int(match_idx),
            "ranking_score": float(top_rankings[i]),
            "predicted_edit_distance": float(top_ed_predictions[i]),
            "predicted_mces_distance": float(top_mces_predictions[i]),
            "reference_smiles": spectra_matches[i].params.get("smiles", "N/A"),
            "reference_precursor_mz": float(spectra_matches[i].precursor_mz),
        }

        if ground_truth_mces is not None:
            match_info["ground_truth_mces_distance"] = float(ground_truth_mces[i])
            match_info["ground_truth_edit_distance"] = float(ground_truth_ed[i])

        result["top_matches"].append(match_info)

    # Save plots if requested
    if save_plots:
        save_query_plots(
            query_index,
            spectra_query,
            spectra_matches,
            best_matches_indices,
            top_rankings,
            top_ed_predictions,
            top_mces_predictions,
            ground_truth_mces,
            ground_truth_ed,
            output_path,
        )

    return result


def plot_distributions(sim_ed, sim_mces, ranking, output_path: str) -> None:
    """Plot distributions of predictions and rankings."""
    import matplotlib.pyplot as plt

    # Sample random predictions
    flat_ed = sim_ed.ravel()
    flat_mces = sim_mces.ravel()
    flat_ranking = ranking.ravel()

    n_samples = min(10000, len(flat_ed))
    idx = np.random.choice(len(flat_ed), size=n_samples, replace=False)

    samples_ed = flat_ed[idx]
    samples_mces = flat_mces[idx]
    samples_ranking = flat_ranking[idx]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Edit distance distribution
    axes[0].hist(samples_ed, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Substructure Edit Distance")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Edit Distance Distribution")
    axes[0].grid(True, alpha=0.3)

    # MCES distribution
    axes[1].hist(samples_mces, bins=20, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("MCES Distance")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("MCES Distance Distribution")
    axes[1].grid(True, alpha=0.3)

    # Ranking distribution
    axes[2].hist(samples_ranking, bins=20, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("Ranking Score")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Ranking Score Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path + "distributions.png", dpi=300)
    plt.close()


def save_query_plots(
    query_index: int,
    spectra_query,
    spectra_matches,
    best_matches_indices,
    top_rankings,
    top_ed_predictions,
    top_mces_predictions,
    ground_truth_mces,
    ground_truth_ed,
    output_path: str,
) -> None:
    """Save visualization plots for a specific query."""
    import matplotlib.pyplot as plt
    import spectrum_utils.plot as sup
    from rdkit import Chem

    query_dir = output_path + f"query_{query_index}/"
    os.makedirs(query_dir, exist_ok=True)

    # Save query molecule structure
    if "smiles" in spectra_query.params:
        mol_query = Chem.MolFromSmiles(spectra_query.params["smiles"])
        if mol_query is not None:
            from rdkit.Chem import Draw

            img = Draw.MolToImage(mol_query, size=(400, 400))
            img.save(query_dir + "query_molecule.png")

    # Save query spectrum
    fig, ax = plt.subplots(figsize=(10, 6))
    sup.spectrum(spectra_query, ax=ax)
    plt.title(f"Query Spectrum (Index: {query_index})")
    plt.tight_layout()
    plt.savefig(query_dir + "query_spectrum.png", dpi=300)
    plt.close()

    # Save top match molecules
    for i, match_spectrum in enumerate(spectra_matches[:5]):  # Save top 5
        if "smiles" in match_spectrum.params:
            mol_match = Chem.MolFromSmiles(match_spectrum.params["smiles"])
            if mol_match is not None:
                from rdkit.Chem import Draw

                img = Draw.MolToImage(mol_match, size=(400, 400))
                img.save(query_dir + f"match_{i + 1}_molecule.png")

        # Save mirror plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sup.mirror(spectra_query, match_spectrum, ax=ax)
        title = f"Match {i + 1} - Ranking: {top_rankings[i]:.3f}, "
        title += f"ED: {top_ed_predictions[i]:.1f}, MCES: {top_mces_predictions[i]:.1f}"
        if ground_truth_mces is not None:
            title += f"\nGT ED: {ground_truth_ed[i]:.1f}, GT MCES: {ground_truth_mces[i]:.1f}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(query_dir + f"match_{i + 1}_mirror.png", dpi=300)
        plt.close()


def compute_summary_statistics(
    results_summary: list, compute_ground_truth: bool
) -> dict:
    """Compute summary statistics of analog discovery results."""
    # Collect all values
    all_rankings = []
    all_ed_predictions = []
    all_mces_predictions = []

    for result in results_summary:
        for match in result["top_matches"]:
            all_rankings.append(match["ranking_score"])
            all_ed_predictions.append(match["predicted_edit_distance"])
            all_mces_predictions.append(match["predicted_mces_distance"])

    statistics = {
        "total_queries": len(results_summary),
        "total_matches": len(all_rankings),
        "ranking_scores": {
            "mean": float(np.mean(all_rankings)),
            "median": float(np.median(all_rankings)),
            "min": float(np.min(all_rankings)),
            "max": float(np.max(all_rankings)),
        },
        "predicted_edit_distance": {
            "mean": float(np.mean(all_ed_predictions)),
            "median": float(np.median(all_ed_predictions)),
        },
        "predicted_mces_distance": {
            "mean": float(np.mean(all_mces_predictions)),
            "median": float(np.median(all_mces_predictions)),
        },
    }

    if compute_ground_truth:
        all_gt_mces = []
        all_gt_ed = []

        for result in results_summary:
            for match in result["top_matches"]:
                if "ground_truth_mces_distance" in match:
                    all_gt_mces.append(match["ground_truth_mces_distance"])
                    all_gt_ed.append(match["ground_truth_edit_distance"])

        if all_gt_mces:
            statistics["ground_truth_edit_distance"] = {
                "mean": float(np.mean(all_gt_ed)),
                "median": float(np.median(all_gt_ed)),
            }
            statistics["ground_truth_mces_distance"] = {
                "mean": float(np.mean(all_gt_mces)),
                "median": float(np.median(all_gt_mces)),
            }

    return statistics
