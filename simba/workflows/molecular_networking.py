"""Workflow for molecular networking using SIMBA predictions."""

import os
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from simba.core.models.simba_model import Simba
from simba.utils.logger_setup import logger
from simba.workflows.utils import load_spectra


def mces_to_similarity(mces: np.ndarray, mces_max: float = 40.0) -> np.ndarray:
    """Convert an [N, N] MCES distance matrix to similarity scores in [0, 1].

    Uses ``sim = 1 - mces / mces_max``.
    """
    return np.clip(1.0 - mces / mces_max, 0.0, 1.0)


def _build_scores(node_ids: list[str], similarity_matrix: np.ndarray, score_name: str):
    """Wrap a precomputed [N, N] similarity matrix into a matchms.Scores object."""
    from matchms import Scores, Spectrum

    nodes = [
        Spectrum(
            mz=np.array([], dtype=float),
            intensities=np.array([], dtype=float),
            metadata={"spectrum_id": nid},
            metadata_harmonization=False,
        )
        for nid in node_ids
    ]
    scores = Scores(references=nodes, queries=nodes, is_symmetric=True)
    scores._scores.add_dense_matrix(similarity_matrix.T, score_name)
    return scores, nodes


def _plot_network(
    graph_path: str,
    label_key: str | None = None,
    score_cutoff: float = 0.0,
) -> None:
    """Load ``graph_path`` (GraphML) and render it to ``network_plot.png`` in the same directory."""
    import math

    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import networkx as nx

    matplotlib.use("Agg")

    G = nx.read_graphml(graph_path)

    output_path = str(Path(graph_path).parent / "network_plot.png")

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    component_map = {n: i for i, comp in enumerate(components) for n in comp}
    PADDING = 1.5
    cols = math.ceil(math.sqrt(len(components)))

    comp_layouts: list[dict] = []
    comp_extents: list[float] = []
    for comp in components:
        sub = G.subgraph(comp)
        n = len(comp)
        if n == 1:
            (nd,) = comp
            lpos: dict = {nd: np.zeros(2)}
            extent = 0.5
        elif n == 2:
            ns = list(comp)
            lpos = {ns[0]: np.array([-0.5, 0.0]), ns[1]: np.array([0.5, 0.0])}
            extent = 0.8
        else:
            lpos = nx.kamada_kawai_layout(sub)
            # Normalise to unit circle so scale factor is consistent
            arr = np.array(list(lpos.values()))
            center = arr.mean(axis=0)
            lpos = {nd: np.array(p) - center for nd, p in lpos.items()}
            max_r = max(np.linalg.norm(p) for p in lpos.values()) or 1.0
            lpos = {nd: p / max_r for nd, p in lpos.items()}
            extent = 1.0
        # Scale so that the cluster footprint grows with sqrt(n)
        scale = max(1.0, math.sqrt(n))
        lpos = {nd: p * scale for nd, p in lpos.items()}
        comp_layouts.append(lpos)
        comp_extents.append(extent * scale)

    # Pack rows left-to-right with variable-size cells
    pos: dict = {}
    y_cursor = 0.0
    for ri in range(math.ceil(len(components) / cols)):
        row_slice = slice(ri * cols, (ri + 1) * cols)
        row_comps = components[row_slice]
        row_layouts = comp_layouts[row_slice]
        row_extents = comp_extents[row_slice]
        row_height = max(row_extents) * 2 + PADDING
        x_cursor = 0.0
        for comp, lpos, extent in zip(row_comps, row_layouts, row_extents):
            offset = np.array([x_cursor + extent, -y_cursor - extent])
            for nd, p in lpos.items():
                pos[nd] = p + offset
            x_cursor += extent * 2 + PADDING
        y_cursor += row_height

    node_list = list(G.nodes())
    node_colors = [cm.tab20(component_map[n] % 20) for n in node_list]
    degrees = dict(G.degree())
    # Shrink nodes in large clusters so they don't overlap
    comp_size_map = {nd: len(comp) for comp in components for nd in comp}
    node_sizes = [
        max(15, 60 / max(1.0, math.sqrt(comp_size_map[nd])) + degrees[nd] * 4)
        for nd in node_list
    ]
    weights = (
        np.array([d["weight"] for _, _, d in G.edges(data=True)])
        if G.number_of_edges()
        else np.array([])
    )

    if label_key:
        labels = {n: G.nodes[n].get(label_key, n) for n in node_list}
    else:
        labels = {n: n for n in node_list}

    fig, ax = plt.subplots(figsize=(22, 18))
    ax.set_facecolor("#111318")
    fig.patch.set_facecolor("#111318")

    if len(weights):
        edge_colors = cm.YlOrRd(weights / weights.max())
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, alpha=0.6, width=0.8)

    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=node_list,
        node_color=node_colors, node_size=node_sizes,
        linewidths=0.3, edgecolors="white", alpha=0.95,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=4.5, font_color="white")

    ax.set_title(
        f"SIMBA Molecular Network\n"
        f"score_cutoff={score_cutoff} · {G.number_of_nodes()} nodes · "
        f"{G.number_of_edges()} edges · {len(components)} clusters",
        color="white", fontsize=12, pad=14,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_molecular_networking(cfg: DictConfig) -> dict:
    """Run molecular networking workflow.

    Loads a set of spectra, runs all-vs-all SIMBA predictions, converts the
    predicted MCES distances to normalised similarity scores, and builds a
    spectral network using ``matchms.networking.SimilarityNetwork``.

    Args:
        cfg: Hydra configuration object with paths and molecular_network settings.

    Returns:
        Dictionary with keys ``"output_dir"``, ``"n_nodes"``, ``"n_edges"``,
        ``"output_file"``.
    """
    from matchms.networking import SimilarityNetwork

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir) + os.sep

    mn_cfg = cfg.molecular_network

    if mn_cfg.filter_spectra:
        all_spectra = load_spectra(
            str(cfg.paths.input_spectra),
            cfg,
            use_gnps_format=mn_cfg.use_gnps_format,
        )
    else:
        all_spectra = load_spectra(
            str(cfg.paths.input_spectra),
            cfg,
            use_gnps_format=mn_cfg.use_gnps_format,
            min_peaks=0,
            use_only_protonized_adducts=False,
        )
    logger.info(f"Loaded {len(all_spectra)} spectra.")

    node_ids = [str(s.mgf_index) for s in all_spectra]

    precomputed = mn_cfg.precomputed_mces
    if precomputed:
        logger.info(f"Loading precomputed MCES from {precomputed}")
        sim_mces = np.load(precomputed)
    else:
        simba_model = Simba(
            str(cfg.paths.model_path),
            config=cfg,
            device=mn_cfg.device,
            cache_embeddings=True,
        )
        _, sim_mces = simba_model.predict(all_spectra, all_spectra)
        np.save(output_path + "similarity_mces.npy", sim_mces)
        logger.info(f"Saved raw MCES matrix to {output_path}similarity_mces.npy")

    similarity = mces_to_similarity(sim_mces, mces_max=float(cfg.model.tasks.mces.max_value))
    logger.info(
        f"Similarity — min: {similarity.min():.3f}, "
        f"mean: {similarity.mean():.3f}, max: {similarity.max():.3f}"
    )

    score_name = "simba_similarity"
    scores, _ = _build_scores(node_ids, similarity, score_name)

    network = SimilarityNetwork(
        identifier_key="spectrum_id",
        top_n=mn_cfg.top_n,
        max_links=mn_cfg.max_links,
        score_cutoff=mn_cfg.score_cutoff,
        link_method=mn_cfg.link_method,
        keep_unconnected_nodes=mn_cfg.keep_unconnected_nodes,
    )
    network.create_network(scores, score_name=score_name)
    G = network.graph

    # Annotate ALL metadata fields from each spectrum onto the graph node so
    # the exported GraphML is self-contained (readable in Cytoscape, etc.).
    for s in all_spectra:
        nid = str(s.mgf_index)
        if nid in G:
            for key, val in s.params.items():
                if val is not None:
                    G.nodes[nid][key] = str(val)

    plot_label_key = mn_cfg.plot_label_key
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    logger.info(f"Network: {n_nodes} nodes, {n_edges} edges.")

    graph_format = mn_cfg.graph_format
    output_file = output_path + f"molecular_network.{graph_format}"
    network.export_to_file(output_file, graph_format=graph_format)
    logger.info(f"Saved to {output_file}")

    if mn_cfg.plot:
        _plot_network(
            graph_path=output_file,
            label_key=plot_label_key,
            score_cutoff=mn_cfg.score_cutoff,
        )

    return {
        "output_dir": str(output_dir),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "output_file": output_file,
    }
