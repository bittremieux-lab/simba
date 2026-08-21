import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem


# =============================================================================
# Configuration
# =============================================================================

sufix = "_unique_compounds"

SIMBA_MODCOS_PICKLE = (
    f"/data/simba_files_/analog_discovery_simba_modified_cosine_results_tf{sufix}.pkl"
)

MS2DEEPSCORE_PICKLE = (
    f"/data/simba_files_/analog_discovery_ms2deepscore_results_tf{sufix}.pkl"
)

GROUND_TRUTH_PICKLE = "/data/simba_files_/analog_discovery_ground_truth.pkl"

SIMBA_JSON = (
    "/data/simba_files_/training_files_new_encoding/"
    "ms2_merged_ref_auto_20260629_ad/analog_discovery_results.json"
)

OUTPUT_PLOT = (
    f"norm_mces_distance_ground_truth_simba_modified_cosine_"
    f"ms2deepscore_tf{sufix}.png"
)

SIMBA_SOURCE = "pickle"
VALID_SIMBA_SOURCES = {"pickle", "json"}

if SIMBA_SOURCE not in VALID_SIMBA_SOURCES:
    raise ValueError(
        f"Invalid SIMBA_SOURCE={SIMBA_SOURCE!r}. "
        f"Expected one of {VALID_SIMBA_SOURCES}."
    )

SIMBA_REPOSITORY_ROOT = "/home/spiedrahita/simba"
CLIP_NORMALIZED_VALUES = False
JSON_TOP_K = 10
JSON_QUERY_SMILES_FIELD = "query_smiles"
JSON_REFERENCE_SMILES_FIELD = "reference_smiles"


def get_mces_class():
    if SIMBA_REPOSITORY_ROOT not in sys.path:
        sys.path.insert(0, SIMBA_REPOSITORY_ROOT)

    try:
        from legacy.old_scripts.simba.analog_discovery.mces import MCES
    except ImportError as error:
        raise ImportError(
            "Could not import MCES. Set SIMBA_REPOSITORY_ROOT correctly."
        ) from error

    return MCES


# Dictionary order determines plot order.
method_sources = {
    "Best MCES match": {
        "source_type": "ground_truth_pickle",
        "file": GROUND_TRUTH_PICKLE,
    },
    "modified cosine": {
        "source_type": "pickle",
        "file": SIMBA_MODCOS_PICKLE,
        "pickle_method": "modified_cosine",
    },
    "MS2DeepScore": {
        "source_type": "pickle",
        "file": MS2DEEPSCORE_PICKLE,
        "pickle_method": "ms2deepscore",
    },
    "SIMBA": {
        "source_type": SIMBA_SOURCE,
        "file": SIMBA_MODCOS_PICKLE if SIMBA_SOURCE == "pickle" else SIMBA_JSON,
        "pickle_method": "simba",
    },
}


def load_pickle_norm_mces_distances(pickle_file, method_name):
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")

    with open(pickle_file, "rb") as file:
        results = pickle.load(file)

    key = "norm_mces_distances_by_method"

    if key not in results:
        raise KeyError(
            f"{pickle_file!r} does not contain {key!r}. "
            f"Available top-level keys: {list(results.keys())}"
        )

    distances_by_method = results[key]

    if method_name not in distances_by_method:
        raise KeyError(
            f"Method {method_name!r} was not found in {pickle_file!r}. "
            f"Available methods: {list(distances_by_method.keys())}"
        )

    values = np.asarray(distances_by_method[method_name], dtype=float)
    return values[np.isfinite(values)]


def load_ground_truth_norm_mces_distances(pickle_file):
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Ground-truth pickle not found: {pickle_file}")

    with open(pickle_file, "rb") as file:
        results = pickle.load(file)

    key = "norm_mces_distances_by_method"

    if key not in results:
        raise KeyError(
            f"Ground-truth pickle does not contain {key!r}. "
            f"Available top-level keys: {list(results.keys())}"
        )

    distances_by_method = results[key]
    available_methods = list(distances_by_method.keys())

    print("\nGround-truth pickle methods:", available_methods)

    candidate_names = [
        "ground_truth_mces",
        "ground_truth",
        "best_mces",
        "best_mces_match",
        "best_match",
        "mces",
    ]

    selected_method = next(
        (name for name in candidate_names if name in distances_by_method),
        None,
    )

    if selected_method is None:
        if len(available_methods) == 1:
            selected_method = available_methods[0]
            print(
                "Automatically using the only method in ground-truth pickle: "
                f"{selected_method!r}"
            )
        else:
            raise KeyError(
                "Could not automatically identify the Best MCES method. "
                f"Available methods: {available_methods}"
            )

    print(f"Using ground-truth method: {selected_method!r}")

    values = np.asarray(distances_by_method[selected_method], dtype=float)
    values = values[np.isfinite(values)]

    print(
        f"Ground truth loaded: n={len(values)}, "
        f"median={np.median(values):.4f}, mean={np.mean(values):.4f}"
    )

    return values


def safe_mces_similarity(smiles1, smiles2, default=np.nan):
    if smiles1 is None or smiles2 is None:
        return default

    smiles1 = str(smiles1)
    smiles2 = str(smiles2)

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        print(f"Invalid SMILES pair: {smiles1!r}, {smiles2!r}")
        return default

    try:
        mces_class = get_mces_class()
        similarity = float(mces_class.calculate_mces_sim(smiles1, smiles2))
    except Exception as error:
        print(f"MCES failed for {smiles1!r}, {smiles2!r}: {error}")
        return default

    return similarity if np.isfinite(similarity) else default


def load_json_norm_mces_distances(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    if not isinstance(json_data, list):
        raise TypeError("The top-level JSON object must be a list.")

    normalized_distances = []
    skipped_queries = 0

    for row_number, query_record in enumerate(json_data):
        if not isinstance(query_record, dict):
            skipped_queries += 1
            continue

        query_smiles = query_record.get(JSON_QUERY_SMILES_FIELD)
        if query_smiles is None:
            skipped_queries += 1
            continue

        top_matches = query_record.get("top_matches", [])
        if not isinstance(top_matches, list):
            skipped_queries += 1
            continue

        if JSON_TOP_K is not None:
            top_matches = top_matches[:JSON_TOP_K]

        mces_similarities = []

        for match in top_matches:
            if not isinstance(match, dict):
                continue

            reference_smiles = match.get(JSON_REFERENCE_SMILES_FIELD)
            similarity = safe_mces_similarity(reference_smiles, query_smiles)

            if np.isfinite(similarity):
                mces_similarities.append(similarity)

        if not mces_similarities:
            skipped_queries += 1
            continue

        best_similarity = float(np.max(mces_similarities))
        normalized_distances.append(1.0 - best_similarity)

    values = np.asarray(normalized_distances, dtype=float)
    values = values[np.isfinite(values)]

    print(
        f"Computed JSON MCES data: {len(values)} valid queries, "
        f"{skipped_queries} skipped."
    )

    return values


def load_method_values(method_label, source_config):
    source_type = source_config["source_type"]
    source_file = source_config["file"]

    if source_type == "ground_truth_pickle":
        return load_ground_truth_norm_mces_distances(source_file)

    if source_type == "pickle":
        return load_pickle_norm_mces_distances(
            pickle_file=source_file,
            method_name=source_config["pickle_method"],
        )

    if source_type == "json":
        return load_json_norm_mces_distances(source_file)

    raise ValueError(
        f"Unsupported source type {source_type!r} for {method_label!r}."
    )


# =============================================================================
# Load data
# =============================================================================

plot_data = []
plot_labels = []

for method_label, source_config in method_sources.items():
    try:
        values = load_method_values(method_label, source_config)
        if method_label.startswith('Best MCES match'):
            values= [v for v in values if v < 1]
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"\nERROR loading {method_label}: {error}")
        continue

    if len(values) == 0:
        print(f"Skipping {method_label}: no valid normalized MCES distances.")
        continue

    plot_data.append(values)
    plot_labels.append(method_label)

    print(
        f"Loaded {method_label}: n={len(values)}, "
        f"median={np.median(values):.4f}"
    )

if len(plot_data) == 0:
    raise ValueError("No valid normalized MCES distances were found.")

print("\nMethods that WILL be plotted:")
for label in plot_labels:
    print(f"  - {label}")


# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(max(9, 1.8 * len(plot_data)), 5.5))
positions = np.arange(1, len(plot_data) + 1)

# Explicitly different colors for each violin.
violin_colors = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#B279A2",
    "#72B7B2",
]

# Draw each violin separately to guarantee distinct colors.
for i, (values, position) in enumerate(zip(plot_data, positions)):

    # remove bad queries that were not possible to process
    violin = ax.violinplot(
        [values],
        positions=[position],
        widths=0.85,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    body = violin["bodies"][0]
    body.set_facecolor(violin_colors[i % len(violin_colors)])
    body.set_edgecolor("black")
    body.set_linewidth(0.8)
    body.set_alpha(0.8)

boxplot = ax.boxplot(
    plot_data,
    positions=positions,
    widths=0.16,
    showfliers=False,
    patch_artist=True,
)

for box in boxplot["boxes"]:
    box.set_facecolor("white")
    box.set_edgecolor("black")
    box.set_alpha(0.85)

for median in boxplot["medians"]:
    median.set_color("black")
    median.set_linewidth(1.6)

for whisker in boxplot["whiskers"]:
    whisker.set_color("black")

for cap in boxplot["caps"]:
    cap.set_color("black")

ax.set_xticks(positions)
ax.set_xticklabels(plot_labels, rotation=20, ha="right")
ax.set_ylabel("Normalized MCES distance")
ax.set_title(
    "Normalized MCES distance: Best MCES match vs retrieval methods"
)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

if CLIP_NORMALIZED_VALUES:
    ax.set_ylim(0, 1)
else:
    ax.set_ylim(0, 0.8)

plt.tight_layout()

fig.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"\nSaved plot to: {OUTPUT_PLOT}")


# =============================================================================
# Summary statistics
# =============================================================================

print("\nSummary statistics:")

for method_label, values in zip(plot_labels, plot_data):
    print(
        f"{method_label}: "
        f"n={len(values)}, "
        f"median={np.median(values):.4f}, "
        f"mean={np.mean(values):.4f}"
    )
