import json
import os
import pickle
import sys


import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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


SIMBA_JSON = (
    "/data/simba_files_/training_files_new_encoding/"
    "ms2_merged_ref_auto_20260629_ad/analog_discovery_results.json"
)


OUTPUT_PLOT = (
    f"norm_mces_distance_by_adduct_simba_modified_cosine_ms2deepscore_tf{sufix}.png"
)


# -----------------------------------------------------------------------------
# Adduct evaluation
# -----------------------------------------------------------------------------
#
# Only QUERY spectra are filtered/grouped by adduct.
# The reference library is NOT filtered.
#
# Add/remove adducts here without changing the rest of the script.
#
# Number of most common query adducts to evaluate.
# The actual adduct names are detected automatically from the query spectra.
TOP_N_ADDUCTS = 4


# Populated automatically before loading the per-method results.
#ADDUCTS_TO_EVALUATE = ['M+H', 'M-H', '[M+Na]+', 'M+FA-H']
ADDUCTS_TO_EVALUATE= None

# The pickle must contain the query spectra in the same order as the arrays in
# results["norm_mces_distances_by_method"][method].
#
# If your pickle uses another key, change this value.
QUERY_SPECTRA_KEY = "query_spectra"


# Alternative keys that are automatically tried if QUERY_SPECTRA_KEY is absent.
QUERY_SPECTRA_FALLBACK_KEYS = (
    "queries",
    "query_spectra",
    "spectra_query",
    "query_spectrum",
)


ADDUCT_PARAM = "adduct"


# Plot modes:
#   "combined"       -> one figure with algorithm/adduct violins side-by-side
#   "per_algorithm"  -> one figure per algorithm, with one violin per adduct
PLOT_MODE = "combined"


# Select where SIMBA values come from.
SIMBA_SOURCE = "pickle"
VALID_SIMBA_SOURCES = {"pickle", "json"}


if SIMBA_SOURCE not in VALID_SIMBA_SOURCES:
    raise ValueError(
        f"Invalid SIMBA_SOURCE={SIMBA_SOURCE!r}. "
        f"Expected one of {VALID_SIMBA_SOURCES}."
    )


# -----------------------------------------------------------------------------
# JSON/MCES settings
# -----------------------------------------------------------------------------


SIMBA_REPOSITORY_ROOT = "/home/spiedrahita/simba"
CLIP_NORMALIZED_VALUES = False


# Evaluate only the first JSON_TOP_K retrieved matches for each query.
# Set to None to evaluate every entry in top_matches.
JSON_TOP_K = 10


JSON_QUERY_SMILES_FIELD = "query_smiles"
JSON_REFERENCE_SMILES_FIELD = "reference_smiles"


# For JSON mode, the query adduct must be available in each query record.
# Change this if your JSON uses another field name.
JSON_QUERY_ADDUCT_FIELD = "adduct"




method_sources = {
    "modified_cosine": {
        "source_type": "pickle",
        "file": SIMBA_MODCOS_PICKLE,
        "pickle_method": "modified_cosine",
    },
    "ms2deepscore": {
        "source_type": "pickle",
        "file": MS2DEEPSCORE_PICKLE,
        "pickle_method": "ms2deepscore",
    },
    "simba": {
        "source_type": SIMBA_SOURCE,
        "file": SIMBA_MODCOS_PICKLE if SIMBA_SOURCE == "pickle" else SIMBA_JSON,
        "pickle_method": "simba",
    },
}




# =============================================================================
# Helpers
# =============================================================================


def get_mces_class():
    """Import and return the legacy MCES implementation when JSON mode is used."""
    if SIMBA_REPOSITORY_ROOT not in sys.path:
        sys.path.insert(0, SIMBA_REPOSITORY_ROOT)


    try:
        from legacy.old_scripts.simba.analog_discovery.mces import MCES
    except ImportError as error:
        raise ImportError(
            "Could not import MCES. Set SIMBA_REPOSITORY_ROOT to the root of "
            "your SIMBA repository before running this script."
        ) from error


    return MCES




def normalize_adduct(adduct):
    """
    Return the exact query-adduct label if it belongs to the automatically
    selected ADDUCTS_TO_EVALUATE list.


    Missing adducts and adducts outside the top-N are ignored.
    """
    if adduct is None:
        return None


    raw = str(adduct).strip()


    if ADDUCTS_TO_EVALUATE is None:
        raise RuntimeError(
            "ADDUCTS_TO_EVALUATE has not been initialized yet."
        )


    #return raw if raw in ADDUCTS_TO_EVALUATE else None
    for target_adduct in ADDUCTS_TO_EVALUATE:
        if target_adduct in raw:
            return target_adduct

    return None



def get_spectrum_adduct(spectrum):
    """Read the adduct from spectrum.params['adduct'] safely."""
    params = getattr(spectrum, "params", None)


    if params is None:
        return None


    try:
        return params.get(ADDUCT_PARAM)
    except AttributeError:
        try:
            return params[ADDUCT_PARAM]
        except (KeyError, TypeError):
            return None




def find_query_spectra(results):
    """
    Find the query spectra stored in a results pickle.


    The score arrays MUST correspond to these spectra in the same order.
    """
    if QUERY_SPECTRA_KEY in results:
        return results[QUERY_SPECTRA_KEY]


    for key in QUERY_SPECTRA_FALLBACK_KEYS:
        if key in results:
            print(
                f"Using query spectra key {key!r} "
                f"(configured key {QUERY_SPECTRA_KEY!r} was absent)."
            )
            return results[key]


    raise KeyError(
        "Could not find query spectra in the pickle. "
        f"Tried {QUERY_SPECTRA_KEY!r} and {QUERY_SPECTRA_FALLBACK_KEYS}. "
        f"Available top-level keys: {list(results.keys())}"
    )




def get_most_common_query_adducts(method_sources, top_n=4):
    """
    Detect the most common adducts among the query spectra.


    The function uses the first available PICKLE source because the different
    algorithms are expected to evaluate the same query set.


    Adducts are counted exactly as stored in:
        spectrum.params["adduct"]


    Returns
    -------
    list[str]
        The top-N adduct labels ordered from most to least common.
    """
    from collections import Counter

    if ADDUCTS_TO_EVALUATE is not None:
        return ADDUCTS_TO_EVALUATE
    
    for method_label, source_config in method_sources.items():
        if source_config.get("source_type") != "pickle":
            continue


        pickle_file = source_config["file"]


        if not os.path.exists(pickle_file):
            continue


        with open(pickle_file, "rb") as file:
            results = pickle.load(file)


        try:
            query_spectra = find_query_spectra(results)
        except KeyError:
            continue


        counts = Counter()


        for spectrum in query_spectra:
            adduct = get_spectrum_adduct(spectrum)


            if adduct is None:
                continue


            adduct = str(adduct).strip()


            if adduct:
                counts[adduct] += 1


        if not counts:
            continue


        most_common = counts.most_common(top_n)


        print(
            f"Detected most common query adducts using "
            f"{method_label!r} ({pickle_file}):"
        )


        for rank, (adduct, count) in enumerate(most_common, start=1):
            print(
                f"  {rank}. {adduct}: {count} query spectra"
            )


        return [adduct for adduct, _ in most_common]


    raise ValueError(
        "Could not determine the most common query adducts. "
        "No usable pickle source with query spectra was found."
    )




def safe_mces_similarity(smiles1, smiles2, default=np.nan):
    """Compute MCES similarity while safely handling invalid SMILES/errors."""
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




# =============================================================================
# Loading / grouping functions
# =============================================================================


def load_pickle_values_by_adduct(
    pickle_file,
    method_name,
    query_spectra_fallback_pickle=SIMBA_MODCOS_PICKLE,
):
    """
    Load one normalized MCES distance per query and group it by QUERY adduct.

    The score values are always loaded from ``pickle_file``.

    Query spectra are loaded from the same pickle when available. If they are
    absent (for example in the MS2DeepScore results pickle), query spectra are
    loaded from ``query_spectra_fallback_pickle`` instead. By default this is
    SIMBA_MODCOS_PICKLE.

    The reference library is untouched.

    Critical requirement:
        len(query_spectra) == len(norm_mces_distances_by_method[method_name])

    This function deliberately does NOT remove NaNs before matching values to
    queries, because doing so would destroy the index correspondence.
    """
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")

    # -------------------------------------------------------------------------
    # Load the algorithm-specific results/scores.
    # -------------------------------------------------------------------------
    with open(pickle_file, "rb") as file:
        results = pickle.load(file)

    distance_key = "norm_mces_distances_by_method"

    if distance_key not in results:
        raise KeyError(
            f"{pickle_file!r} does not contain {distance_key!r}."
        )

    distances_by_method = results[distance_key]

    if method_name not in distances_by_method:
        raise KeyError(
            f"Method {method_name!r} was not found in {pickle_file!r}. "
            f"Available methods: {list(distances_by_method)}"
        )

    values = np.asarray(
        distances_by_method[method_name],
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Find query spectra.
    #
    # First try the algorithm's own pickle. If it does not contain query
    # spectra, use the SIMBA/modified-cosine results pickle as the fallback.
    # -------------------------------------------------------------------------
    try:
        query_spectra = find_query_spectra(results)

        print(
            f"{method_name}: using query spectra from its own pickle: "
            f"{pickle_file}"
        )

    except KeyError:
        fallback_file = query_spectra_fallback_pickle

        # Avoid an unhelpful second attempt on exactly the same file.
        if os.path.abspath(fallback_file) == os.path.abspath(pickle_file):
            raise KeyError(
                f"Query spectra were not found in {pickle_file!r}, and the "
                "configured fallback query-spectra pickle is the same file."
            )

        if not os.path.exists(fallback_file):
            raise FileNotFoundError(
                f"{method_name}: query spectra were not found in "
                f"{pickle_file!r}, and fallback pickle does not exist: "
                f"{fallback_file!r}"
            )

        with open(fallback_file, "rb") as file:
            fallback_results = pickle.load(file)

        try:
            query_spectra = find_query_spectra(fallback_results)
        except KeyError as error:
            raise KeyError(
                f"{method_name}: query spectra were not found in either "
                f"{pickle_file!r} or fallback pickle {fallback_file!r}."
            ) from error

        print(
            f"{method_name}: query spectra not found in {pickle_file}. "
            f"Using query spectra from fallback SIMBA pickle: "
            f"{fallback_file}"
        )

    # -------------------------------------------------------------------------
    # Alignment check.
    #
    # This is essential when query spectra come from another pickle.
    # -------------------------------------------------------------------------
    if len(query_spectra) != len(values):
        raise ValueError(
            f"Cannot align query spectra with {method_name!r} values. "
            f"Query spectra: {len(query_spectra)}; "
            f"performance values: {len(values)}. "
            f"Score pickle: {pickle_file!r}. "
            f"Query-spectra fallback: "
            f"{query_spectra_fallback_pickle!r}."
        )

    grouped = {
        adduct: []
        for adduct in ADDUCTS_TO_EVALUATE
    }

    unknown_counts = {}

    for spectrum, value in zip(query_spectra, values):
        raw_adduct = get_spectrum_adduct(spectrum)
        canonical_adduct = normalize_adduct(raw_adduct)

        if canonical_adduct is None:
            label = (
                "<missing>"
                if raw_adduct is None
                else str(raw_adduct)
            )
            unknown_counts[label] = (
                unknown_counts.get(label, 0) + 1
            )
            continue

        if np.isfinite(value):
            grouped[canonical_adduct].append(
                float(value)
            )

    grouped = {
        adduct: np.asarray(
            adduct_values,
            dtype=float,
        )
        for adduct, adduct_values in grouped.items()
    }

    if unknown_counts:
        print(
            f"{method_name}: ignored query adducts not requested/mapped: "
            f"{unknown_counts}"
        )

    return grouped


def load_json_values_by_adduct(json_file):
    """
    Compute normalized MCES distance per JSON query and group by QUERY adduct.


    This requires each JSON query record to contain JSON_QUERY_ADDUCT_FIELD.
    The references in top_matches are NOT filtered by adduct.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")


    with open(json_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)


    if not isinstance(json_data, list):
        raise TypeError("The top-level JSON object must be a list of query records.")


    grouped = {adduct: [] for adduct in ADDUCTS_TO_EVALUATE}
    skipped_queries = 0


    for row_number, query_record in enumerate(json_data):
        if not isinstance(query_record, dict):
            skipped_queries += 1
            continue


        query_index = query_record.get("query_index", row_number)
        query_smiles = query_record.get(JSON_QUERY_SMILES_FIELD)
        raw_adduct = query_record.get(JSON_QUERY_ADDUCT_FIELD)
        canonical_adduct = normalize_adduct(raw_adduct)


        # Skip queries whose adduct is not one of the requested groups.
        if canonical_adduct is None:
            continue


        if query_smiles is None:
            print(
                f"Skipping query {query_index}: "
                f"missing {JSON_QUERY_SMILES_FIELD!r}."
            )
            skipped_queries += 1
            continue


        top_matches = query_record.get("top_matches", [])


        if not isinstance(top_matches, list):
            print(f"Skipping query {query_index}: 'top_matches' is not a list.")
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
            print(
                f"Skipping query {query_index}: no valid MCES similarities "
                "could be computed from its top matches."
            )
            skipped_queries += 1
            continue


        best_similarity = float(np.max(mces_similarities))
        normalized_distance = 1.0 - best_similarity
        grouped[canonical_adduct].append(normalized_distance)


    grouped = {
        adduct: np.asarray(values, dtype=float)
        for adduct, values in grouped.items()
    }


    print(
        f"Computed JSON MCES data from {json_file}; "
        f"{skipped_queries} requested-adduct queries skipped."
    )


    return grouped




def load_method_values_by_adduct(method_label, source_config):
    """Load values for one algorithm, grouped by query adduct."""
    source_type = source_config["source_type"]
    source_file = source_config["file"]


    if source_type == "pickle":
        pickle_method = source_config.get("pickle_method", method_label)
        return load_pickle_values_by_adduct(
            pickle_file=source_file,
            method_name=pickle_method,
        )


    if source_type == "json":
        return load_json_values_by_adduct(source_file)


    raise ValueError(
        f"Unsupported source type {source_type!r} "
        f"for method {method_label!r}."
    )




# =============================================================================
# Detect the most common query adducts
# =============================================================================


ADDUCTS_TO_EVALUATE = get_most_common_query_adducts(
    method_sources=method_sources,
    top_n=TOP_N_ADDUCTS,
)


print(
    f"\nEvaluating the {len(ADDUCTS_TO_EVALUATE)} most common query adducts: "
    f"{ADDUCTS_TO_EVALUATE}\n"
)




# =============================================================================
# Load all algorithms
# =============================================================================


results_by_method = {}


for method_label, source_config in method_sources.items():
    try:
        grouped_values = load_method_values_by_adduct(
            method_label,
            source_config,
        )
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"Skipping {method_label}: {error}")
        continue


    results_by_method[method_label] = grouped_values


    counts = {
        adduct: len(values)
        for adduct, values in grouped_values.items()
    }


    print(
        f"Loaded {method_label} from {source_config['source_type']}: "
        f"{counts}"
    )


if not results_by_method:
    raise ValueError("No valid algorithm results were loaded.")




# =============================================================================
# Plotting
# =============================================================================


def plot_per_algorithm(results_by_method):
    """
    Create one figure per algorithm.
    Each figure contains one violin + boxplot for every requested query adduct.
    """
    output_files = []


    for method_label, grouped_values in results_by_method.items():
        plot_data = []
        labels = []


        for adduct in ADDUCTS_TO_EVALUATE:
            values = grouped_values.get(adduct, np.array([]))


            if len(values) == 0:
                print(
                    f"{method_label}: no valid queries for adduct {adduct!r}; "
                    "not plotting this group."
                )
                continue


            plot_data.append(values)
            labels.append(adduct)


        if not plot_data:
            print(f"{method_label}: no adduct groups available to plot.")
            continue


        fig, ax = plt.subplots(
            figsize=(max(7, 1.6 * len(plot_data)), 5)
        )


        positions = np.arange(1, len(plot_data) + 1)


        ax.violinplot(
            plot_data,
            positions=positions,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )


        ax.boxplot(
            plot_data,
            positions=positions,
            widths=0.18,
            showfliers=False,
        )


        # Explicit median markers and labels.
        medians = [float(np.median(values)) for values in plot_data]
        ax.scatter(positions, medians, zorder=3)


        for x, median, values in zip(positions, medians, plot_data):
            ax.annotate(
                f"median={median:.3f}\nn={len(values)}",
                xy=(x, median),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )


        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Normalized MCES distance")
        ax.set_xlabel("Query adduct")
        ax.set_title(
            f"{method_label}: analog-search performance by query adduct"
        )
        ax.grid(axis="y", alpha=0.3)


        if CLIP_NORMALIZED_VALUES:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, 0.8)


        plt.tight_layout()


        root, ext = os.path.splitext(OUTPUT_PLOT)
        output_file = f"{root}_{method_label}{ext or '.png'}"


        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )


        output_files.append(output_file)
        plt.show()
        plt.close(fig)


        print(f"Saved plot to: {output_file}")


    return output_files




def plot_combined(results_by_method):
    """
    Create ONE figure containing all algorithms.

    Main x-axis groups = algorithms.
    Within every algorithm group = one violin per query adduct.

    IMPORTANT:
    Each adduct is assigned one color, and that same color is reused across
    every algorithm. This makes comparison of the same adduct across methods
    visually straightforward.
    """
    methods = list(results_by_method.keys())
    n_methods = len(methods)
    n_adducts = len(ADDUCTS_TO_EVALUATE)

    if n_adducts == 0:
        raise ValueError("ADDUCTS_TO_EVALUATE cannot be empty.")

    fig, ax = plt.subplots(
        figsize=(max(10, 2.5 * n_methods), 6)
    )

    # -------------------------------------------------------------------------
    # One fixed color per adduct.
    #
    # tab10 supports up to 10 visually distinct categorical colors.
    # The mapping is based only on ADDUCTS_TO_EVALUATE, so the same adduct
    # receives exactly the same color for modified cosine, MS2DeepScore, SIMBA,
    # and any additional algorithms.
    # -------------------------------------------------------------------------
    cmap = plt.get_cmap("tab10")

    adduct_colors = {
        adduct: cmap(i % cmap.N)
        for i, adduct in enumerate(ADDUCTS_TO_EVALUATE)
    }

    # Center of each algorithm group.
    method_centers = np.arange(n_methods, dtype=float)

    # Position adduct violins symmetrically around each algorithm center.
    if n_adducts == 1:
        adduct_offsets = np.array([0.0])
    else:
        total_group_width = 0.7
        adduct_offsets = np.linspace(
            -total_group_width / 2,
            total_group_width / 2,
            n_adducts,
        )

    violin_width = min(
        0.22,
        0.60 / max(n_adducts, 1),
    )


    all_values = []

    for method_idx, method_label in enumerate(methods):
        grouped_values = results_by_method[method_label]

        for adduct_idx, adduct in enumerate(ADDUCTS_TO_EVALUATE):
            values = np.asarray(
                grouped_values.get(adduct, np.array([])),
                dtype=float,
            )
            values = values[np.isfinite(values)]

            if len(values) == 0:
                print(
                    f"{method_label}: no valid values for "
                    f"query adduct {adduct!r}."
                )
                continue

            all_values.append(values)

            position = (
                method_centers[method_idx]
                + adduct_offsets[adduct_idx]
            )

            color = adduct_colors[adduct]

            # -----------------------------------------------------------------
            # Violin distribution.
            # -----------------------------------------------------------------
            violin = ax.violinplot(
                [values],
                positions=[position],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            # Force the violin color to correspond to the adduct.
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.65)

            # -----------------------------------------------------------------
            # Small boxplot inside the violin.
            # -----------------------------------------------------------------
            box = ax.boxplot(
                [values],
                positions=[position],
                widths=violin_width * 0.38,
                showfliers=False,
                patch_artist=True,
                boxprops={
                    "facecolor": color,
                    "edgecolor": "black",
                    "alpha": 0.85,
                },
                medianprops={
                    "color": "black",
                    "linewidth": 1.5,
                },
                whiskerprops={
                    "color": "black",
                },
                capprops={
                    "color": "black",
                },
            )

            # Explicit median marker.
            median = float(np.median(values))

            ax.scatter(
                [position],
                [median],
                marker="o",
                s=18,
                facecolor="black",
                edgecolor="black",
                zorder=4,
            )

            # Median and number of queries.
            ax.annotate(
                f"med={median:.3f}\n"
                f"n={len(values)}",
                xy=(position, median),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    if not all_values:
        raise ValueError(
            "No valid values were available for any algorithm/adduct pair."
        )

    # -------------------------------------------------------------------------
    # Algorithm labels on the main x-axis.
    # -------------------------------------------------------------------------
    ax.set_xticks(method_centers)
    ax.set_xticklabels(
        methods,
        rotation=15,
        ha="right",
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Normalized MCES distance")
    ax.set_title(
        "Analog-search performance by algorithm and query adduct"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    # Separators between algorithm groups.
    for boundary in method_centers[:-1] + 0.5:
        ax.axvline(
            boundary,
            linestyle="--",
            linewidth=0.8,
            alpha=0.25,
        )

    ax.set_xlim(
        method_centers[0] - 0.65,
        method_centers[-1] + 0.65,
    )

    if CLIP_NORMALIZED_VALUES:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, 0.8)

    # -------------------------------------------------------------------------
    # Legend: one entry per adduct, using exactly the same colors as violins.
    # -------------------------------------------------------------------------
    legend_handles = [
        Patch(
            facecolor=adduct_colors[adduct],
            edgecolor="black",
            alpha=0.65,
            label=adduct,
        )
        for adduct in ADDUCTS_TO_EVALUATE
    ]

    ax.legend(
        handles=legend_handles,
        title="Query adduct",
        loc="upper right",
        frameon=True,
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_PLOT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Saved combined plot to: {OUTPUT_PLOT}")

    return [OUTPUT_PLOT]


if PLOT_MODE == "per_algorithm":
    output_files = plot_per_algorithm(results_by_method)
elif PLOT_MODE == "combined":
    output_files = plot_combined(results_by_method)
else:
    raise ValueError(
        f"Invalid PLOT_MODE={PLOT_MODE!r}. "
        "Use 'per_algorithm' or 'combined'."
    )




# =============================================================================
# Summary statistics
# =============================================================================


print("\nSummary by algorithm and query adduct")
print("=" * 72)


for method_label, grouped_values in results_by_method.items():
    print(f"\n{method_label}")


    for adduct in ADDUCTS_TO_EVALUATE:
        values = grouped_values.get(adduct, np.array([]))


        if len(values) == 0:
            print(f"  {adduct}: n=0")
            continue


        print(
            f"  {adduct}: "
            f"n={len(values)}, "
            f"median={np.median(values):.4f}, "
            f"mean={np.mean(values):.4f}"
        )








