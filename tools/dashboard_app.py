"""Streamlit dashboard for monitoring SIMBA training experiments.

Reads the artifacts ValMetricsCallback (simba/core/training/callbacks.py)
writes into an experiment's checkpoint dir: metrics.csv (Lightning's
CSVLogger) for the loss/MAE curves, and, per validation check,
mces_binned_box_{val_name}_step*.png (pre-rendered box plot) plus
val_pairs_{val_name}_step*.csv (per-pair data) -- both saved at the same
step, used here to build the fine-grained box plot and the mass heatmap on
demand from the per-pair data.

Run with:
    uv run --extra dashboard streamlit run tools/dashboard_app.py
"""

import re
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "training"
SLURM_DIR = Path(__file__).resolve().parent / "slurm"

_BOX_PLOT_RE = re.compile(r"mces_binned_box_(?P<val_name>.+)_step(?P<step>\d+)\.png$")
_CSV_RE = re.compile(r"val_pairs_(?P<val_name>.+)_step(?P<step>\d+)\.csv$")
_MCES_MAX = 40.0
_BIN_ORDER = [
    "self (MCES=0)",
    "(0,5]",
    "(5,10]",
    "(10,15]",
    "(15,20]",
    "(20,25]",
    "(25,30]",
    "(30,35]",
    "(35,40]",
]
_TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def list_experiments() -> list[Path]:
    """Experiments that used the current ValMetricsCallback -- identified by
    having at least one binned-box PNG (older runs' metrics.csv has a
    different, incompatible schema: Spearman columns instead of val_mae_mces,
    no binned-box plots)."""
    if not EXPERIMENTS_DIR.is_dir():
        return []
    return sorted(
        d
        for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir() and any(d.glob("mces_binned_box_*_step*.png"))
    )


def load_metrics(exp_dir: Path) -> pd.DataFrame:
    path = exp_dir / "metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def infer_limit_train_batches(exp_dir: Path) -> int | None:
    """Best-effort: read training.limit_train_batches from the matching SLURM
    script (tools/slurm/{experiment_name}.slurm.sh), so curves can use
    fractional epoch as x. Returns None if it can't be determined -- callers
    should fall back to plotting against raw step."""
    script = SLURM_DIR / f"{exp_dir.name}.slurm.sh"
    if not script.exists():
        return None
    m = re.search(r"training\.limit_train_batches=(\d+)", script.read_text())
    return int(m.group(1)) if m else None


def list_box_plot_steps(exp_dir: Path, val_name: str) -> list[int]:
    steps = []
    for p in exp_dir.glob(f"mces_binned_box_{val_name}_step*.png"):
        m = _BOX_PLOT_RE.match(p.name)
        if m:
            steps.append(int(m.group("step")))
    return sorted(steps)


def list_val_names(exp_dir: Path) -> list[str]:
    names = set()
    for p in exp_dir.glob("mces_binned_box_*_step*.png"):
        m = _BOX_PLOT_RE.match(p.name)
        if m:
            names.add(m.group("val_name"))
    return sorted(names)


def list_csv_steps(exp_dir: Path, val_name: str) -> list[int]:
    steps = []
    for p in exp_dir.glob(f"val_pairs_{val_name}_step*.csv"):
        m = _CSV_RE.match(p.name)
        if m:
            steps.append(int(m.group("step")))
    return sorted(steps)


@st.cache_data(show_spinner="Loading per-pair CSV ...")
def load_pair_csv(path_str: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """Cached by (path, columns) -- each validation check's CSV is written
    exactly once and never touched again, so it's immutable once it exists
    and safe to cache by path alone."""
    return pd.read_csv(path_str, usecols=list(columns))


def render_loss_tab(df: pd.DataFrame, x_col: str, x_label: str):
    fig = go.Figure()
    train = df[df["train_loss_step"].notna()]
    fig.add_trace(
        go.Scatter(
            x=train[x_col],
            y=train["train_loss_step"],
            mode="lines",
            name="train_loss",
            line={"width": 1, "color": "#1f77b4"},
        )
    )
    if "validation_loss_epoch" in df.columns:
        val = df[df["validation_loss_epoch"].notna()]
        fig.add_trace(
            go.Scatter(
                x=val[x_col],
                y=val["validation_loss_epoch"],
                mode="lines+markers",
                name="validation_loss",
                line={"width": 2, "color": "#d62728"},
                marker={"size": 6},
            )
        )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="loss",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        height=500,
        margin={"t": 40},
    )
    st.plotly_chart(fig, width="stretch")


@st.cache_data(show_spinner="Counting pairs per MCES bin ...")
def bin_counts_for_val(exp_dir_str: str, val_name: str, step: int) -> dict[str, int]:
    """Per-bin pair counts, read from one validation check's per-pair CSV.
    With resampling off, every check evaluates the same full val set, so
    these counts are identical at every step (verified directly: steps
    1000/70000/143000 all gave the exact same per-bin counts for this
    experiment) -- one step's CSV is enough to label every point on the
    curve, no need to recompute per step."""
    path = Path(exp_dir_str) / f"val_pairs_{val_name}_step{step:06d}.csv"
    df = pd.read_csv(path, usecols=["mces_bin"])
    return df["mces_bin"].value_counts().to_dict()


def _parse_bin_range(label: str) -> tuple[float, float]:
    """'(0,5]' -> (0.0, 5.0)."""
    lo, hi = label.strip("()[]").split(",")
    return float(lo), float(hi)


@st.cache_data(show_spinner="Computing molecule mass per index ...")
def mol_idx_mass_lookup(exp_dir_str: str, val_name: str, step: int) -> dict[int, float]:
    """mol_idx -> RDKit ExactMolWt, built once from one step's CSV. Molecule
    identity (and therefore mass) doesn't depend on model weights or
    training step, so one step's smiles_0/smiles_1<->mol_idx_0/mol_idx_1
    correspondence is enough for the whole run."""
    path = Path(exp_dir_str) / f"val_pairs_{val_name}_step{step:06d}.csv"
    df = pd.read_csv(path, usecols=["mol_idx_0", "mol_idx_1", "smiles_0", "smiles_1"])
    idx_to_smiles = dict(zip(df["mol_idx_0"], df["smiles_0"]))
    idx_to_smiles.update(zip(df["mol_idx_1"], df["smiles_1"]))
    unique_smiles = tuple(sorted(set(idx_to_smiles.values())))
    smiles_to_mass = mass_lookup_for_smiles(unique_smiles)
    return {idx: smiles_to_mass[smi] for idx, smi in idx_to_smiles.items()}


def _mass_diff_masked_bin_stats(
    pair_df: pd.DataFrame, mol_mass: dict, mass_lo: float, mass_hi: float
) -> dict:
    """{label: (mae, n)} for every _BIN_ORDER label present in this step,
    plus "__overall__", restricted to pairs whose |mass_0 - mass_1| falls in
    [mass_lo, mass_hi]. Self-pairs have mass_diff==0 by construction (same
    molecule), so they're naturally included/excluded by the range alone."""
    mass_0 = pair_df["mol_idx_0"].map(mol_mass).to_numpy()
    mass_1 = pair_df["mol_idx_1"].map(mol_mass).to_numpy()
    diff = np.abs(mass_0 - mass_1)
    mask = (diff >= mass_lo) & (diff <= mass_hi) & ~np.isnan(diff)
    if not mask.any():
        return {}
    gt = pair_df["gt_mces"].to_numpy()[mask]
    pred = pair_df["pred_mces"].to_numpy()[mask]
    bins = pair_df["mces_bin"].to_numpy()[mask]
    abs_err = np.abs(pred - gt)

    out = {"__overall__": (float(abs_err.mean()), int(mask.sum()))}
    for label in _BIN_ORDER:
        bmask = bins == label
        n = int(bmask.sum())
        if n:
            out[label] = (float(abs_err[bmask].mean()), n)
    return out


def build_mass_filtered_metrics_figure(
    exp_dir: Path,
    val_name: str,
    steps: list[int],
    mol_mass: dict,
    mass_range: tuple[float, float],
    include_self: bool,
    mces_range: tuple[float, float],
) -> go.Figure:
    per_step = []
    progress = st.progress(0.0, text="Reading per-pair CSVs ...")
    for i, step in enumerate(steps):
        path = exp_dir / f"val_pairs_{val_name}_step{step:06d}.csv"
        pair_df = load_pair_csv(
            str(path), ("mol_idx_0", "mol_idx_1", "gt_mces", "pred_mces", "mces_bin")
        )
        per_step.append(
            _mass_diff_masked_bin_stats(pair_df, mol_mass, mass_range[0], mass_range[1])
        )
        progress.progress(
            (i + 1) / len(steps), text=f"Read {i + 1}/{len(steps)} checks"
        )
    progress.empty()

    fig = go.Figure()
    overall_xy = [
        (s, stats["__overall__"])
        for s, stats in zip(steps, per_step)
        if "__overall__" in stats
    ]
    if overall_xy:
        xs, vals = zip(*overall_xy)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[v[0] for v in vals],
                mode="lines+markers",
                name=f"overall MAE (n={vals[-1][1]:,})",
                line={"width": 3, "color": "black", "dash": "dot"},
            )
        )
    for i, label in enumerate(_BIN_ORDER):
        is_self_label = label.startswith("self")
        if is_self_label and not include_self:
            continue
        if not is_self_label:
            lo, hi = _parse_bin_range(label)
            if hi <= mces_range[0] or lo >= mces_range[1]:
                continue
        xy = [(s, stats[label]) for s, stats in zip(steps, per_step) if label in stats]
        if not xy:
            continue
        xs, vals = zip(*xy)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[v[0] for v in vals],
                mode="lines+markers",
                name=f"{label} [{val_name}], n={vals[-1][1]:,}",
                line={"width": 1.5, "color": _TAB10[i % len(_TAB10)]},
                marker={"size": 4},
            )
        )
    fig.update_layout(
        xaxis_title="step (sampled)",
        yaxis_title="MAE (raw MCES units)",
        legend={"orientation": "h", "yanchor": "top", "y": -0.2},
        height=600,
        margin={"t": 40},
        title=f"Molecule mass difference in [{mass_range[0]:g}, {mass_range[1]:g}] Da",
    )
    return fig


def _overlap_coefficient(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
    """Overlapping coefficient between two samples' distributions: sum of
    min(p_a, p_b) over a shared histogram, normalized to each sum to 1.
    0 = fully separated, 1 = identical distributions. Not fooled by large n
    the way a rank/AUC test can be -- it measures actual overlap, not
    statistical significance of ordering."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(np.minimum(pa, pb).sum())


def _active_bin_labels(
    include_self: bool, mces_range: tuple[float, float]
) -> list[str]:
    """_BIN_ORDER labels surviving the include-self / GT-MCES-range filters,
    in order -- used both to pick which MAE lines to draw and which adjacent
    pairs to compare for the overlap-coefficient metric."""
    active = []
    for label in _BIN_ORDER:
        is_self_label = label.startswith("self")
        if is_self_label and not include_self:
            continue
        if not is_self_label:
            lo, hi = _parse_bin_range(label)
            if hi <= mces_range[0] or lo >= mces_range[1]:
                continue
        active.append(label)
    return active


def _mass_diff_masked_pred_by_bin(
    pair_df: pd.DataFrame, mol_mass: dict, mass_lo: float, mass_hi: float
) -> dict[str, np.ndarray]:
    """{label: predicted-MCES array}, restricted to pairs whose
    |mass_0 - mass_1| falls in [mass_lo, mass_hi] -- the raw material for
    the overlap-coefficient metric (unlike MAE, this needs each bin's full
    array, not just an aggregate, so there's no pre-logged fast path for it
    at all)."""
    mass_0 = pair_df["mol_idx_0"].map(mol_mass).to_numpy()
    mass_1 = pair_df["mol_idx_1"].map(mol_mass).to_numpy()
    diff = np.abs(mass_0 - mass_1)
    mask = (diff >= mass_lo) & (diff <= mass_hi) & ~np.isnan(diff)
    if not mask.any():
        return {}
    pred = pair_df["pred_mces"].to_numpy()[mask]
    bins = pair_df["mces_bin"].to_numpy()[mask]
    return {label: pred[bins == label] for label in _BIN_ORDER if (bins == label).any()}


def build_overlap_coefficient_figure(
    exp_dir: Path,
    val_name: str,
    steps: list[int],
    mol_mass: dict,
    mass_range: tuple[float, float],
    include_self: bool,
    mces_range: tuple[float, float],
    skip: int = 0,
) -> go.Figure:
    active_labels = _active_bin_labels(include_self, mces_range)
    # skip=0 -> immediate neighbor (i, i+1), skip=1 -> one bin in between
    # (i, i+2), etc. -- e.g. skip=1 compares self directly against (5,10],
    # passing over (0,5].
    adjacent_pairs = list(zip(active_labels, active_labels[skip + 1 :]))

    per_step = []
    progress = st.progress(0.0, text="Reading per-pair CSVs ...")
    for i, step in enumerate(steps):
        path = exp_dir / f"val_pairs_{val_name}_step{step:06d}.csv"
        pair_df = load_pair_csv(
            str(path), ("mol_idx_0", "mol_idx_1", "pred_mces", "mces_bin")
        )
        pred_by_bin = _mass_diff_masked_pred_by_bin(
            pair_df, mol_mass, mass_range[0], mass_range[1]
        )
        step_result = {}
        for a, b in adjacent_pairs:
            if a in pred_by_bin and b in pred_by_bin:
                step_result[(a, b)] = (
                    _overlap_coefficient(pred_by_bin[a], pred_by_bin[b]),
                    len(pred_by_bin[a]),
                    len(pred_by_bin[b]),
                )
        per_step.append(step_result)
        progress.progress(
            (i + 1) / len(steps), text=f"Read {i + 1}/{len(steps)} checks"
        )
    progress.empty()

    fig = go.Figure()
    for i, (a, b) in enumerate(adjacent_pairs):
        xy = [(s, res[(a, b)]) for s, res in zip(steps, per_step) if (a, b) in res]
        if not xy:
            continue
        xs, vals = zip(*xy)
        ovl, n_a, n_b = zip(*vals)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ovl,
                mode="lines+markers",
                name=f"{a} vs {b} (n={n_a[-1]:,}/{n_b[-1]:,})",
                line={"width": 1.5, "color": _TAB10[i % len(_TAB10)]},
                marker={"size": 4},
            )
        )
    skip_desc = "next-highest bin" if skip == 0 else f"bin {skip} step(s) further out"
    fig.update_layout(
        xaxis_title="step (sampled)",
        yaxis_title="Overlap coefficient (0=separated, 1=identical)",
        yaxis={"range": [0, 1]},
        legend={"orientation": "h", "yanchor": "top", "y": -0.2},
        height=600,
        margin={"t": 40},
        title=(
            f"Predicted-MCES overlap with {skip_desc} -- "
            f"mass diff in [{mass_range[0]:g}, {mass_range[1]:g}] Da"
        ),
    )
    return fig


def render_metrics_tab(df: pd.DataFrame, x_col: str, x_label: str, exp_dir: Path):
    col0, col1, col2 = st.columns([1.2, 1, 3])
    with col0:
        metric_choice = st.radio(
            "Metric",
            ["MAE", "Overlap coefficient (vs next bin)"],
            key="metrics_metric_choice",
        )
    with col1:
        include_self = st.checkbox("Include self-pairs (MCES=0)", value=True)
    with col2:
        mces_range = st.slider(
            "GT MCES difference range (numeric bins) -- any difference allowed by default",
            min_value=0.0,
            max_value=_MCES_MAX,
            value=(0.0, _MCES_MAX),
            step=1.0,
        )

    val_names_present = sorted(
        {
            c.rsplit("/", 1)[-1]
            for label in _BIN_ORDER
            for c in df.columns
            if c.startswith(f"val_mae_mces/{label}/")
        }
    )
    bin_counts = {}
    for val_name in val_names_present:
        steps = list_csv_steps(exp_dir, val_name)
        bin_counts[val_name] = (
            bin_counts_for_val(str(exp_dir), val_name, steps[-1]) if steps else {}
        )

    # Molecule mass-difference filter. Unlike the GT-MCES filter above (a
    # free reshuffle of already-logged per-bin scalars), this dimension was
    # never logged at training time -- honoring it means recomputing directly
    # from each validation check's per-pair CSV (mol mass via RDKit, joined
    # against that step's gt/pred), so *every* curve reflects only
    # mass-difference-matching pairs. That's multiple large file reads
    # instead of one cheap metrics.csv read, so it's opt-in behind a button
    # rather than reactive like the rest of this tab. The overlap-coefficient
    # metric needs each bin's full predicted-value array (not an aggregate
    # scalar), so it has no pre-logged fast path at all -- it always uses
    # this same heavier per-pair-CSV route, mass filter or not.
    primary_val = val_names_present[0] if val_names_present else None
    steps_avail = list_csv_steps(exp_dir, primary_val) if primary_val else []

    st.divider()
    mass_diff_range, max_diff = None, 1.0
    if not steps_avail:
        st.info("No val_pairs_*.csv found yet -- this needs those.")
    else:
        mol_mass = mol_idx_mass_lookup(str(exp_dir), primary_val, steps_avail[-1])
        masses = np.array(list(mol_mass.values()))
        max_diff = float(masses.max() - masses.min()) if len(masses) else 1.0

        st.markdown(
            "**Filter by molecule mass difference** (|mass_0 - mass_1|, Da) -- "
            "applies to *all* curves, not just an extra line"
        )
        mass_diff_range = st.slider(
            "Mass difference between the two molecules in a pair (Da) -- any difference allowed by default",
            min_value=0.0,
            max_value=max_diff,
            value=(0.0, max_diff),
            step=max(1.0, max_diff / 100),
        )

    mass_filter_active = mass_diff_range is not None and mass_diff_range != (
        0.0,
        max_diff,
    )
    needs_heavy_path = metric_choice != "MAE" or mass_filter_active

    if needs_heavy_path:
        if not steps_avail:
            return
        effective_mass_range = mass_diff_range or (0.0, max_diff)

        skip = 0
        if metric_choice != "MAE":
            n_active = len(_active_bin_labels(include_self, mces_range))
            max_skip = max(0, n_active - 2)
            skip = st.number_input(
                "Bins to skip between comparisons (0 = adjacent, e.g. self vs (0,5]; "
                "1 = skip one, e.g. self vs (5,10]; 2 = self vs (10,15]; ...)",
                min_value=0,
                max_value=max_skip,
                value=0,
                step=1,
            )

        max_checks = st.slider(
            "Validation checks to sample (fewer = faster; each one is a full per-pair CSV read)",
            min_value=min(5, len(steps_avail)),
            max_value=len(steps_avail),
            value=min(30, len(steps_avail)),
        )
        stride = max(1, len(steps_avail) // max_checks)
        sampled_steps = steps_avail[::stride]
        if steps_avail[-1] not in sampled_steps:
            sampled_steps.append(steps_avail[-1])
        st.caption(
            f"Will read {len(sampled_steps)} per-pair CSVs (of {len(steps_avail)} available)."
        )
        if st.button("Build curves"):
            if metric_choice == "MAE":
                fig = build_mass_filtered_metrics_figure(
                    exp_dir,
                    primary_val,
                    sampled_steps,
                    mol_mass,
                    effective_mass_range,
                    include_self,
                    mces_range,
                )
            else:
                fig = build_overlap_coefficient_figure(
                    exp_dir,
                    primary_val,
                    sampled_steps,
                    mol_mass,
                    effective_mass_range,
                    include_self,
                    mces_range,
                    skip,
                )
            st.plotly_chart(fig, width="stretch")
        return

    # Fast path: MAE metric, mass-difference filter at its default (any
    # difference allowed) -- render straight from the pre-logged metrics.csv,
    # no per-pair CSV reads needed.
    fig = go.Figure()
    val = (
        df[df["val_mces_mae"].notna()] if "val_mces_mae" in df.columns else df.iloc[0:0]
    )
    if len(val):
        total_n = (
            sum(bin_counts[val_names_present[0]].values())
            if len(val_names_present) == 1
            else None
        )
        n_label = f" (n={total_n:,})" if total_n else ""
        fig.add_trace(
            go.Scatter(
                x=val[x_col],
                y=val["val_mces_mae"],
                mode="lines+markers",
                name=f"overall MAE{n_label}",
                line={"width": 3, "color": "black", "dash": "dot"},
            )
        )
    for i, label in enumerate(_BIN_ORDER):
        is_self_label = label.startswith("self")
        if is_self_label and not include_self:
            continue
        if not is_self_label:
            lo, hi = _parse_bin_range(label)
            if hi <= mces_range[0] or lo >= mces_range[1]:
                continue
        matching = [c for c in df.columns if c.startswith(f"val_mae_mces/{label}/")]
        for col in matching:
            sub = df[df[col].notna()]
            if not len(sub):
                continue
            val_name = col.rsplit("/", 1)[-1]
            n = bin_counts.get(val_name, {}).get(label)
            n_label = f", n={n:,}" if n is not None else ""
            fig.add_trace(
                go.Scatter(
                    x=sub[x_col],
                    y=sub[col],
                    mode="lines+markers",
                    name=f"{label} [{val_name}]{n_label}",
                    line={"width": 1.5, "color": _TAB10[i % len(_TAB10)]},
                    marker={"size": 4},
                )
            )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="MAE (raw MCES units)",
        legend={"orientation": "h", "yanchor": "top", "y": -0.2},
        height=600,
        margin={"t": 40},
    )
    st.plotly_chart(fig, width="stretch")


def build_fine_box_plot_figure(df: pd.DataFrame, bin_width: float) -> plt.Figure:
    """Same drawing code as ValMetricsCallback._plot_binned_box in
    simba/core/training/callbacks.py -- whis=(5,95), outliers hidden, pred=GT
    reference diagonal, each box n-annotated directly on the plot (not just on
    hover), self-pairs as their own box at GT=0 instead of folded into the
    lowest numeric bin. Duplicated here (not imported) so the dashboard has
    no dependency on the training pipeline's module; the only difference from
    the training-time PNG is a configurable bin_width instead of the fixed 5.
    """
    gt = df["gt_mces"].to_numpy()
    pred = df["pred_mces"].to_numpy()
    is_self = df["is_self_pair"].astype(bool).to_numpy()

    edges = np.arange(0, _MCES_MAX + bin_width, bin_width)
    labels, positions, widths, groups = [], [], [], []

    labels.append("self (MCES=0)")
    positions.append(0.0)
    widths.append(bin_width * 0.3)
    groups.append(pred[is_self])

    non_self_gt = gt[~is_self]
    non_self_pred = pred[~is_self]
    bin_idx = np.clip(np.digitize(non_self_gt, edges[1:-1]), 0, len(edges) - 2)
    lo = 0.0
    for i, hi in enumerate(edges[1:]):
        labels.append(f"({lo:g},{hi:g}]")
        positions.append((lo + hi) / 2.0)
        widths.append(bin_width * 0.8)
        groups.append(non_self_pred[bin_idx == i])
        lo = hi

    plot_groups = [g for g in groups if len(g) > 0]
    plot_positions = [p for p, g in zip(positions, groups) if len(g) > 0]
    plot_widths = [w for w, g in zip(widths, groups) if len(g) > 0]
    plot_labels = [lab for lab, g in zip(labels, groups) if len(g) > 0]
    plot_ns = [len(g) for g in plot_groups]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if not plot_groups:
        ax.set_title("No pairs")
        return fig

    ax.boxplot(
        plot_groups,
        positions=plot_positions,
        widths=plot_widths,
        whis=(5, 95),
        showfliers=False,
    )
    ax.plot(
        [0, _MCES_MAX],
        [0, _MCES_MAX],
        color="red",
        linestyle="--",
        linewidth=1,
        label="pred = GT",
    )
    ymax = max(np.percentile(g, 95) for g in plot_groups)
    label_y = ymax * 1.03
    for p, n in zip(plot_positions, plot_ns):
        ax.text(p, label_y, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_ylim(top=label_y * 1.25)
    ax.set_xticks(plot_positions)
    ax.set_xticklabels(plot_labels, rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.set_xlabel(f"GT MCES (bin width={bin_width:g}; self-pairs kept separate at 0)")
    ax.set_ylabel("Predicted MCES")
    ax.set_title(f"Predicted MCES by GT bin (n={sum(plot_ns):,} pairs)")
    fig.tight_layout()
    return fig


def render_box_plot_tab(exp_dir: Path):
    val_names = list_val_names(exp_dir)
    if not val_names:
        st.info("No binned-box plots found yet for this experiment.")
        return
    val_name = (
        st.selectbox("Val set", val_names) if len(val_names) > 1 else val_names[0]
    )
    steps = list_box_plot_steps(exp_dir, val_name)
    if not steps:
        st.info("No binned-box plots found yet for this val set.")
        return
    step = st.select_slider("Validation step", options=steps, value=steps[-1])
    st.image(str(exp_dir / f"mces_binned_box_{val_name}_step{step:06d}.png"))

    st.divider()
    st.subheader(f"Fine-grained box plot -- same step ({step:,}), custom bin width")
    csv_path = exp_dir / f"val_pairs_{val_name}_step{step:06d}.csv"
    if not csv_path.exists():
        st.info(f"No val_pairs CSV found for step {step}.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        bin_width = st.number_input(
            "Bin width (MCES units)", min_value=0.1, max_value=10.0, value=1.0, step=0.5
        )

    # Only ever touches this one step's CSV (already loaded/cheap), so unlike
    # the Validation Metrics tab's mass filter, this stays reactive -- no
    # button needed.
    pair_df = load_pair_csv(
        str(csv_path),
        ("gt_mces", "pred_mces", "is_self_pair", "mol_idx_0", "mol_idx_1"),
    )
    mol_mass = mol_idx_mass_lookup(str(exp_dir), val_name, step)
    mass_0 = pair_df["mol_idx_0"].map(mol_mass).to_numpy()
    mass_1 = pair_df["mol_idx_1"].map(mol_mass).to_numpy()
    mass_diff = np.abs(mass_0 - mass_1)
    max_diff = float(np.nanmax(mass_diff)) if len(mass_diff) else 1.0

    with col2:
        mass_range = st.slider(
            "Molecule mass difference (Da) -- any difference allowed by default",
            min_value=0.0,
            max_value=max_diff,
            value=(0.0, max_diff),
            step=max(1.0, max_diff / 100),
            key="fine_box_mass_range",
        )

    mask = (
        (mass_diff >= mass_range[0])
        & (mass_diff <= mass_range[1])
        & ~np.isnan(mass_diff)
    )
    filtered_df = pair_df[mask]
    st.caption(
        f"{len(filtered_df):,} / {len(pair_df):,} pairs match this mass-difference range."
    )
    fig = build_fine_box_plot_figure(filtered_df, bin_width)
    st.pyplot(fig)


@st.cache_data(
    show_spinner="Computing theoretical (RDKit ExactMolWt) mass per molecule ..."
)
def mass_lookup_for_smiles(unique_smiles: tuple[str, ...]) -> dict[str, float]:
    lookup = {}
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        lookup[smi] = ExactMolWt(mol) if mol is not None else float("nan")
    return lookup


def compute_mass_heatmap_grids(
    x: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    step: float,
    min_n: int,
):
    hi = float(np.ceil(max(x.max(), y.max()) / step) * step)
    edges = np.arange(0, hi + step, step)
    nx = ny = len(edges) - 1
    xi = np.clip(np.digitize(x, edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(y, edges) - 1, 0, ny - 1)
    flat_idx = xi * ny + yi
    minlen = nx * ny

    count = np.bincount(flat_idx, minlength=minlen).reshape(nx, ny)
    grids = {}
    for name, vals in [
        ("mae", np.abs(pred - gt)),
        ("bias", pred - gt),
        ("gt_mces", gt),
    ]:
        sums = np.bincount(flat_idx, weights=vals, minlength=minlen)
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = (sums / count.astype(float).ravel()).reshape(nx, ny)
        grid[count < min_n] = np.nan
        grids[name] = grid
    return edges, grids, count


_HEATMAP_METRICS = [
    ("mae", "MAE (|pred - GT MCES|)", "Viridis", False),
    ("bias", "Signed bias (pred - GT MCES)", "RdBu", True),
    ("gt_mces", "Mean GT MCES", "Viridis", False),
]


def build_mass_heatmap_figure(
    edges: np.ndarray, grids: dict, count: np.ndarray
) -> go.Figure:
    centers = (edges[:-1] + edges[1:]) / 2
    n_cols = len(_HEATMAP_METRICS)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[title for _, title, _, _ in _HEATMAP_METRICS],
        horizontal_spacing=0.1,
    )
    for col, (key, _title, colorscale, center_zero) in enumerate(
        _HEATMAP_METRICS, start=1
    ):
        grid = grids[key]
        zmid = 0 if center_zero else None
        # Colorbar x is read back from this subplot's own computed domain (not
        # guessed from col/n_cols) so it sits just to its right without
        # reaching into the next subplot's plotting area.
        axis_key = "xaxis" if col == 1 else f"xaxis{col}"
        domain_right = fig.layout[axis_key].domain[1]
        fig.add_trace(
            go.Heatmap(
                x=centers,
                y=centers,
                z=grid.T,
                colorscale=colorscale,
                zmid=zmid,
                colorbar={
                    "len": 0.85,
                    "thickness": 14,
                    "x": domain_right + 0.015,
                    "xanchor": "left",
                },
                hovertemplate="min mass=%{x:.0f}<br>max mass=%{y:.0f}<br>value=%{z:.2f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        ok = count >= 1
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if ok[i, j] and not np.isnan(grid[i, j]):
                    fig.add_annotation(
                        x=centers[i],
                        y=centers[j],
                        text=f"{grid[i, j]:.2f}<br>n={count[i, j]:,}",
                        showarrow=False,
                        font={"size": 8, "color": "white"},
                        row=1,
                        col=col,
                    )
        fig.update_xaxes(title_text="min(mass_0, mass_1) (Da)", row=1, col=col)
        if col == 1:
            fig.update_yaxes(title_text="max(mass_0, mass_1) (Da)", row=1, col=col)
    fig.update_layout(height=550, margin={"t": 60})
    return fig


def render_mass_heatmap_tab(exp_dir: Path):
    st.subheader("Mass1 x mass2 heatmaps (val-to-val pairs)")
    val_names = list_val_names(exp_dir)
    if not val_names:
        st.info("No val_pairs_*.csv found yet.")
        return
    val_name = (
        st.selectbox("Val set", val_names, key="mass_heatmap_val_name")
        if len(val_names) > 1
        else val_names[0]
    )
    steps = list_csv_steps(exp_dir, val_name)
    if not steps:
        st.info("No val_pairs_*.csv found yet for this val set.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        step = st.select_slider(
            "Validation step", options=steps, value=steps[-1], key="mass_step"
        )
    with col2:
        mass_step = st.number_input(
            "Mass bin step (Da)",
            min_value=10.0,
            max_value=500.0,
            value=100.0,
            step=10.0,
        )
    with col3:
        min_n = st.number_input("Min pairs per cell", min_value=1, value=100, step=50)
    with col4:
        include_self = st.checkbox(
            "Include self-pairs (same molecule, different spectra)",
            value=True,
        )

    path = exp_dir / f"val_pairs_{val_name}_step{step:06d}.csv"
    pair_df = load_pair_csv(
        str(path), ("smiles_0", "smiles_1", "gt_mces", "pred_mces", "is_self_pair")
    )
    if not include_self:
        pair_df = pair_df[~pair_df["is_self_pair"].astype(bool)]

    unique_smiles = tuple(
        pd.unique(pd.concat([pair_df["smiles_0"], pair_df["smiles_1"]]))
    )
    mass_lookup = mass_lookup_for_smiles(unique_smiles)
    mass_0 = pair_df["smiles_0"].map(mass_lookup).to_numpy()
    mass_1 = pair_df["smiles_1"].map(mass_lookup).to_numpy()
    x = np.minimum(mass_0, mass_1)
    y = np.maximum(mass_0, mass_1)

    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    pred = pair_df["pred_mces"].to_numpy()[valid]
    gt = pair_df["gt_mces"].to_numpy()[valid]

    st.caption(
        f"{len(gt):,} pairs ({'self-pairs included, land on the diagonal' if include_self else 'self-pairs excluded'}), "
        f"{len(unique_smiles):,} unique molecules (mass range {x.min():.0f}-{y.max():.0f} Da)"
    )
    edges, grids, count = compute_mass_heatmap_grids(
        x, y, pred, gt, mass_step, int(min_n)
    )
    fig = build_mass_heatmap_figure(edges, grids, count)
    st.plotly_chart(fig, width="stretch")


def main():
    st.set_page_config(page_title="SIMBA training dashboard", layout="wide")
    st.title("SIMBA training dashboard")

    experiments = list_experiments()
    if not experiments:
        st.warning(f"No compatible experiments found under {EXPERIMENTS_DIR}")
        return

    with st.sidebar:
        exp_name = st.selectbox("Experiment", [e.name for e in experiments])
        st.button(
            "Refresh"
        )  # any interaction reloads data fresh; this is just an explicit no-op trigger
    exp_dir = EXPERIMENTS_DIR / exp_name

    df = load_metrics(exp_dir)
    if df.empty:
        st.warning(f"No metrics.csv found (or it's empty) for {exp_name}")
        return

    limit_train_batches = infer_limit_train_batches(exp_dir)
    if limit_train_batches:
        df["_epoch_frac"] = df["step"] / limit_train_batches
        x_col, x_label = "_epoch_frac", "epoch"
    else:
        x_col, x_label = "step", "step"

    tab_loss, tab_metrics, tab_box, tab_mass = st.tabs(
        ["Loss", "Validation metrics", "Box plot", "Mass heatmap"]
    )
    with tab_loss:
        render_loss_tab(df, x_col, x_label)
    with tab_metrics:
        render_metrics_tab(df, x_col, x_label, exp_dir)
    with tab_box:
        render_box_plot_tab(exp_dir)
    with tab_mass:
        render_mass_heatmap_tab(exp_dir)


if __name__ == "__main__":
    main()
