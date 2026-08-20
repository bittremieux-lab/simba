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


def infer_preprocessing_dir(exp_dir: Path) -> str | None:
    """Best-effort: read the PREPRO_DIR shell variable from the matching
    SLURM script (tools/slurm/{experiment_name}.slurm.sh) -- every
    experiment script in this project sets `PREPRO_DIR=...` then passes
    `paths.preprocessing_dir="${PREPRO_DIR}"`, so this is more reliable than
    trying to regex the Hydra override itself. Used to locate
    val_cosine_{val_name}.parquet (tools/compute_val_cosine.py), which lives
    with the val set rather than any one experiment."""
    script = SLURM_DIR / f"{exp_dir.name}.slurm.sh"
    if not script.exists():
        return None
    m = re.search(r"^PREPRO_DIR=(\S+)$", script.read_text(), re.MULTILINE)
    return m.group(1) if m else None


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


def consolidated_parquet_path(exp_dir: Path, val_name: str) -> Path:
    return exp_dir / f"val_pairs_{val_name}_consolidated.parquet"


_PARQUET_PRED_COL_RE = re.compile(r"^pred_mces_step(\d+)$")


@st.cache_data(show_spinner=False)
def list_consolidated_steps(exp_dir_str: str, val_name: str) -> list[int]:
    """Steps available in the consolidated parquet file (one static table +
    one pred_mces_step{N} column per validation check), read from its schema
    only -- no data loaded."""
    path = consolidated_parquet_path(Path(exp_dir_str), val_name)
    if not path.exists():
        return []
    import pyarrow.parquet as pq

    steps = []
    for name in pq.ParquetFile(path).schema.names:
        m = _PARQUET_PRED_COL_RE.match(name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def list_available_steps(exp_dir: Path, val_name: str) -> list[int]:
    """Prefer the consolidated parquet file's steps when present (its whole
    point is being cheap to read regardless of how many checks it covers);
    fall back to per-step CSVs for experiments that don't have one yet."""
    consolidated = list_consolidated_steps(str(exp_dir), val_name)
    return consolidated if consolidated else list_csv_steps(exp_dir, val_name)


@st.cache_data(show_spinner="Loading per-pair data ...")
def load_pair_data_for_step(
    exp_dir_str: str, val_name: str, step: int, columns: tuple[str, ...]
) -> pd.DataFrame:
    """Same interface as load_pair_csv (a DataFrame with a plain 'pred_mces'
    column for this one step) but transparently prefers the consolidated
    parquet file when one exists -- reads only the needed static columns
    plus this step's single prediction column, columnar and fast regardless
    of how many total checks the run has had. Falls back to the per-step CSV
    for experiments without a consolidated file yet."""
    exp_dir = Path(exp_dir_str)
    parquet_path = consolidated_parquet_path(exp_dir, val_name)
    if parquet_path.exists():
        pred_col = f"pred_mces_step{step:06d}"
        wants_pred = "pred_mces" in columns
        static_needed = [c for c in columns if c != "pred_mces"]
        read_cols = static_needed + ([pred_col] if wants_pred else [])
        df = pd.read_parquet(parquet_path, columns=read_cols)
        if wants_pred:
            df = df.rename(columns={pred_col: "pred_mces"})
        return df
    csv_path = exp_dir / f"val_pairs_{val_name}_step{step:06d}.csv"
    return load_pair_csv(str(csv_path), columns)


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
    df = load_pair_data_for_step(exp_dir_str, val_name, step, ("mces_bin",))
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
    df = load_pair_data_for_step(
        exp_dir_str, val_name, step, ("mol_idx_0", "mol_idx_1", "smiles_0", "smiles_1")
    )
    idx_to_smiles = dict(zip(df["mol_idx_0"], df["smiles_0"]))
    idx_to_smiles.update(zip(df["mol_idx_1"], df["smiles_1"]))
    unique_smiles = tuple(sorted(set(idx_to_smiles.values())))
    smiles_to_mass = mass_lookup_for_smiles(unique_smiles)
    return {idx: smiles_to_mass[smi] for idx, smi in idx_to_smiles.items()}


@st.cache_data(show_spinner="Loading cosine baseline ...")
def load_val_cosine(preprocessing_dir: str, val_name: str) -> pd.DataFrame:
    """Raw spectral cosine similarity per validation pair
    (tools/compute_val_cosine.py) -- one file per val set, shared by every
    experiment built from the same preprocessing dir (empty DataFrame if it
    hasn't been computed yet)."""
    path = Path(preprocessing_dir) / f"val_cosine_{val_name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


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
    progress = st.progress(0.0, text="Reading per-pair data ...")
    for i, step in enumerate(steps):
        pair_df = load_pair_data_for_step(
            str(exp_dir),
            val_name,
            step,
            ("mol_idx_0", "mol_idx_1", "gt_mces", "pred_mces", "mces_bin"),
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
    pair_df: pd.DataFrame,
    mol_mass: dict,
    mass_lo: float,
    mass_hi: float,
    value_col: str = "pred_mces",
) -> dict[str, np.ndarray]:
    """{label: score array}, restricted to pairs whose |mass_0 - mass_1|
    falls in [mass_lo, mass_hi] -- the raw material for the overlap-
    coefficient metric (unlike MAE, this needs each bin's full array, not
    just an aggregate, so there's no pre-logged fast path for it at all).
    `value_col` defaults to SIMBA's own `pred_mces` but also works for any
    other per-pair score in `pair_df` sharing the same bin labels -- e.g.
    `cosine`, the raw spectral-cosine baseline from
    tools/compute_val_cosine.py."""
    mass_0 = pair_df["mol_idx_0"].map(mol_mass).to_numpy()
    mass_1 = pair_df["mol_idx_1"].map(mol_mass).to_numpy()
    diff = np.abs(mass_0 - mass_1)
    mask = (diff >= mass_lo) & (diff <= mass_hi) & ~np.isnan(diff)
    if not mask.any():
        return {}
    pred = pair_df[value_col].to_numpy()[mask]
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
    progress = st.progress(0.0, text="Reading per-pair data ...")
    for i, step in enumerate(steps):
        pair_df = load_pair_data_for_step(
            str(exp_dir),
            val_name,
            step,
            ("mol_idx_0", "mol_idx_1", "pred_mces", "mces_bin"),
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

    avg_xy = [
        (s, np.mean([v[0] for v in res.values()]))
        for s, res in zip(steps, per_step)
        if res
    ]
    if avg_xy:
        xs, avgs = zip(*avg_xy)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=avgs,
                mode="lines+markers",
                name="average (over currently-filtered pairs)",
                line={"width": 3, "color": "black", "dash": "dot"},
            )
        )

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


_LOGGED_MAX_SKIP = (
    4  # ValMetricsCallback logs skip 0..4 directly; beyond that needs recomputation
)


def build_overlap_fast_figure(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    skip: int,
    include_self: bool,
    mces_range: tuple[float, float],
    bin_counts: dict,
) -> go.Figure:
    """Reads val_overlap/{a}_vs_{b}_skip{skip}/{val_name} straight from the
    pre-logged metrics.csv -- same fast path as MAE, valid for skip 0-4 since
    that's what ValMetricsCallback logs directly every check."""
    active_labels = _active_bin_labels(include_self, mces_range)
    pairs = list(zip(active_labels, active_labels[skip + 1 :]))
    fig = go.Figure()

    avg_matching = [
        c for c in df.columns if c.startswith(f"val_overlap_avg/skip{skip}/")
    ]
    for col in avg_matching:
        sub = df[df[col].notna()]
        if not len(sub):
            continue
        val_name = col.rsplit("/", 1)[-1]
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[col],
                mode="lines+markers",
                name=f"average [{val_name}]",
                line={"width": 3, "color": "black", "dash": "dot"},
            )
        )

    for i, (a, b) in enumerate(pairs):
        prefix = f"val_overlap/{a}_vs_{b}_skip{skip}/"
        matching = [c for c in df.columns if c.startswith(prefix)]
        for col in matching:
            sub = df[df[col].notna()]
            if not len(sub):
                continue
            val_name = col.rsplit("/", 1)[-1]
            n_a = bin_counts.get(val_name, {}).get(a)
            n_b = bin_counts.get(val_name, {}).get(b)
            n_label = (
                f" (n={n_a:,}/{n_b:,})" if n_a is not None and n_b is not None else ""
            )
            fig.add_trace(
                go.Scatter(
                    x=sub[x_col],
                    y=sub[col],
                    mode="lines+markers",
                    name=f"{a} vs {b} [{val_name}]{n_label}",
                    line={"width": 1.5, "color": _TAB10[i % len(_TAB10)]},
                    marker={"size": 4},
                )
            )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Overlap coefficient (0=separated, 1=identical)",
        yaxis={"range": [0, 1]},
        legend={"orientation": "h", "yanchor": "top", "y": -0.2},
        height=600,
        margin={"t": 40},
        title=f"Predicted-MCES overlap, skip={skip} (adjacent bins if 0)",
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
        steps = list_available_steps(exp_dir, val_name)
        bin_counts[val_name] = (
            bin_counts_for_val(str(exp_dir), val_name, steps[-1]) if steps else {}
        )

    # Skip control (overlap coefficient only). ValMetricsCallback logs skip
    # 0-4 directly every check, so those read from metrics.csv just like
    # MAE; skip > 4 has no pre-logged fast path and needs recomputation.
    skip = 0
    if metric_choice != "MAE":
        n_active = len(_active_bin_labels(include_self, mces_range))
        max_skip = max(0, n_active - 2)
        skip = st.number_input(
            "Bins to skip between comparisons (0 = adjacent, e.g. self vs (0,5]; "
            "1 = skip one, e.g. self vs (5,10]; 2 = self vs (10,15]; ... "
            f"0-{_LOGGED_MAX_SKIP} are pre-logged/instant, higher needs recomputation)",
            min_value=0,
            max_value=max_skip,
            value=0,
            step=1,
        )

    # Molecule mass-difference filter. Unlike the GT-MCES filter above (a
    # free reshuffle of already-logged per-bin scalars), this dimension was
    # never logged at training time -- honoring it means recomputing directly
    # from each validation check's per-pair data (mol mass via RDKit, joined
    # against that step's gt/pred). With the consolidated parquet file
    # (static columns once + one pred_mces_step{N} column per check), this is
    # cheap regardless of how many checks are involved. Without it (older
    # experiments, per-step CSVs only), it's still a large file read per
    # check.
    primary_val = val_names_present[0] if val_names_present else None
    steps_avail = list_available_steps(exp_dir, primary_val) if primary_val else []
    has_consolidated = (
        bool(primary_val) and consolidated_parquet_path(exp_dir, primary_val).exists()
    )

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
    # Skip 0-4 has a fast path (ValMetricsCallback logs it directly) only if
    # this run's metrics.csv actually has those columns -- older runs (like
    # any experiment before this feature existed) don't, so fall back to
    # recomputing from the per-pair data instead of silently showing an
    # empty chart.
    overlap_logged_at_all = any(c.startswith("val_overlap/") for c in df.columns)
    overlap_not_logged = metric_choice != "MAE" and not overlap_logged_at_all
    overlap_needs_recompute = metric_choice != "MAE" and (
        skip > _LOGGED_MAX_SKIP or overlap_not_logged
    )
    needs_heavy_path = mass_filter_active or overlap_needs_recompute

    if needs_heavy_path:
        if overlap_not_logged and not mass_filter_active and skip <= _LOGGED_MAX_SKIP:
            st.info(
                "Overlap coefficient isn't in this run's metrics.csv (it predates "
                "this feature) -- building it from the per-pair data instead."
            )
        if not steps_avail:
            return
        effective_mass_range = mass_diff_range or (0.0, max_diff)

        default_checks = min(10, len(steps_avail))
        max_checks = st.slider(
            "Validation checks to use (fewer = faster -- 160 available checks can "
            "take a couple minutes to build; 10-30 is usually plenty)",
            min_value=min(2, len(steps_avail)),
            max_value=len(steps_avail),
            value=default_checks,
        )
        stride = max(1, len(steps_avail) // max_checks)
        sampled_steps = steps_avail[::stride]
        if steps_avail[-1] not in sampled_steps:
            sampled_steps.append(steps_avail[-1])
        source_note = (
            "consolidated file" if has_consolidated else "per-step CSVs (slower)"
        )
        st.caption(
            f"Using {len(sampled_steps)} of {len(steps_avail)} available checks ({source_note})."
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

    if metric_choice != "MAE":
        # Fast path: skip 0-4, no mass filter -- read straight from the
        # pre-logged metrics.csv, same as MAE below.
        fig = build_overlap_fast_figure(
            df, x_col, x_label, skip, include_self, mces_range, bin_counts
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
    has_data = (
        consolidated_parquet_path(exp_dir, val_name).exists()
        or (exp_dir / f"val_pairs_{val_name}_step{step:06d}.csv").exists()
    )
    if not has_data:
        st.info(f"No per-pair data found for step {step}.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        bin_width = st.number_input(
            "Bin width (MCES units)", min_value=0.1, max_value=10.0, value=1.0, step=0.5
        )

    # Only ever touches this one step's data (already fast/cheap either way),
    # so unlike the Validation Metrics tab's mass filter, this stays
    # reactive -- no button needed.
    pair_df = load_pair_data_for_step(
        str(exp_dir),
        val_name,
        step,
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
    steps = list_available_steps(exp_dir, val_name)
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

    pair_df = load_pair_data_for_step(
        str(exp_dir),
        val_name,
        step,
        ("smiles_0", "smiles_1", "gt_mces", "pred_mces", "is_self_pair"),
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


def _last_non_null(df: pd.DataFrame, col: str) -> float | None:
    """Last non-null value logged for `col`, or None if it's missing/all-NaN.
    Train/val columns log at very different cadences (train every step, val
    only at validation checks, some val columns only once per check vs. once
    per val batch) so each metric needs its own last-logged value rather than
    assuming one "last row" has everything."""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) else None


def _pick_val_name(exp_dir: Path) -> str | None:
    """Prefer the scaffold "val" set when a run has more than one (e.g. an
    official-split val too) -- it's the one every experiment so far has."""
    names = list_val_names(exp_dir)
    if not names:
        return None
    return "val" if "val" in names else names[0]


def _pred_by_bin_last_step(exp_dir: Path, val_name: str) -> dict[str, np.ndarray]:
    """Predicted-MCES values grouped by GT-MCES bin, from this run's most
    recent validation check (unfiltered by mass) -- the raw material to
    recompute overlap-coefficient cells for runs that predate that logging
    (e.g. experiment 009, run before val_overlap* columns existed)."""
    steps = list_available_steps(exp_dir, val_name)
    if not steps:
        return {}
    pair_df = load_pair_data_for_step(
        str(exp_dir), val_name, steps[-1], ("pred_mces", "mces_bin")
    )
    pred = pair_df["pred_mces"].to_numpy()
    bins = pair_df["mces_bin"].to_numpy()
    return {label: pred[bins == label] for label in _BIN_ORDER if (bins == label).any()}


def _overlap_avg_skip_from_pairs(pred_by_bin: dict, skip: int) -> float | None:
    """Same quantity as val_overlap_avg/skip{skip}, computed from per-pair
    predictions instead of read from metrics.csv."""
    adjacent_pairs = list(zip(_BIN_ORDER, _BIN_ORDER[skip + 1 :]))
    vals = [
        _overlap_coefficient(pred_by_bin[a], pred_by_bin[b])
        for a, b in adjacent_pairs
        if a in pred_by_bin and b in pred_by_bin
    ]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else None


def _identity_overlap_at_skip(
    df: pd.DataFrame, val_name: str, skip: int, pred_by_bin_getter
) -> float | None:
    """Overlap between the self (MCES=0) bin and the bin `skip` steps further
    out (skip=0 -> (0,5], skip=1 -> (5,10], skip=2 -> (10,15]), unfiltered by
    mass. Prefers the pre-logged metrics.csv column; falls back to this run's
    most recent per-pair check for experiments that predate that logging."""
    if skip + 1 >= len(_BIN_ORDER):
        return None
    anchor, partner = _BIN_ORDER[0], _BIN_ORDER[skip + 1]
    val = _last_non_null(df, f"val_overlap/{anchor}_vs_{partner}_skip{skip}/{val_name}")
    if val is None:
        pb = pred_by_bin_getter()
        if anchor in pb and partner in pb:
            val = _overlap_coefficient(pb[anchor], pb[partner])
    return val


@st.cache_data(show_spinner=False)
def _mae_for_bin(
    exp_dir: Path,
    val_name: str,
    mass_lo: float,
    mass_hi: float,
    bin_label: str | None,
) -> float | None:
    """MAE at this run's most recent validation check, restricted to molecule
    mass difference in [mass_lo, mass_hi] -- overall (bin_label=None) or one
    specific GT-MCES bin. Always a per-pair recompute: no pre-logged
    mass-filtered metric exists in metrics.csv. Reuses the same helpers as
    the Mass heatmap tab's heavy path."""
    steps = list_available_steps(exp_dir, val_name)
    if not steps:
        return None
    step = steps[-1]
    pair_df = load_pair_data_for_step(
        str(exp_dir),
        val_name,
        step,
        ("mol_idx_0", "mol_idx_1", "gt_mces", "pred_mces", "mces_bin"),
    )
    mol_mass = mol_idx_mass_lookup(str(exp_dir), val_name, step)
    stats = _mass_diff_masked_bin_stats(pair_df, mol_mass, mass_lo, mass_hi)
    entry = stats.get("__overall__" if bin_label is None else bin_label)
    return entry[0] if entry else None


@st.cache_data(show_spinner=False)
def _overlap_for_spec(
    exp_dir: Path,
    val_name: str,
    mass_lo: float,
    mass_hi: float,
    skip: int,
    anchor_bin: str | None,
) -> float | None:
    """Overlap coefficient at this run's most recent validation check,
    restricted to molecule mass difference in [mass_lo, mass_hi]: the average
    over every adjacent bin-pair at this skip distance (anchor_bin=None), or
    one specific pair anchored at anchor_bin (paired with the bin `skip`
    further out). Always a per-pair recompute, same reason as `_mae_for_bin`."""
    steps = list_available_steps(exp_dir, val_name)
    if not steps:
        return None
    step = steps[-1]
    pair_df = load_pair_data_for_step(
        str(exp_dir),
        val_name,
        step,
        ("mol_idx_0", "mol_idx_1", "pred_mces", "mces_bin"),
    )
    mol_mass = mol_idx_mass_lookup(str(exp_dir), val_name, step)
    pred_by_bin = _mass_diff_masked_pred_by_bin(pair_df, mol_mass, mass_lo, mass_hi)
    pairs = list(zip(_BIN_ORDER, _BIN_ORDER[skip + 1 :]))
    if anchor_bin is not None:
        pairs = [(a, b) for a, b in pairs if a == anchor_bin]
    vals = [
        _overlap_coefficient(pred_by_bin[a], pred_by_bin[b])
        for a, b in pairs
        if a in pred_by_bin and b in pred_by_bin
    ]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else None


@st.cache_data(show_spinner=False)
def _overlap_for_spec_cosine(
    exp_dir: Path,
    val_name: str,
    mass_lo: float,
    mass_hi: float,
    skip: int,
    anchor_bin: str | None,
) -> float | None:
    """Same as `_overlap_for_spec`, but scored against raw spectral cosine
    similarity (tools/compute_val_cosine.py) instead of SIMBA's own
    pred_mces -- a classical, non-learned baseline. Cosine is experiment-
    independent (same val set -> same spec_idx pairing -> same cosine for
    every run), so this only depends on this run's most recent check for
    its GT-MCES bin labels/mass lookup, not for the score itself. Restricted
    to Overlap in the UI: cosine and MCES aren't on comparable scales, so an
    MAE between them wouldn't mean anything."""
    preprocessing_dir = infer_preprocessing_dir(exp_dir)
    if preprocessing_dir is None:
        return None
    cosine_df = load_val_cosine(preprocessing_dir, val_name)
    if cosine_df.empty:
        return None

    steps = list_available_steps(exp_dir, val_name)
    if not steps:
        return None
    step = steps[-1]
    pair_df = load_pair_data_for_step(
        str(exp_dir),
        val_name,
        step,
        ("mol_idx_0", "mol_idx_1", "spec_idx_0", "spec_idx_1", "mces_bin"),
    )
    merged = pair_df.merge(
        cosine_df,
        on=["mol_idx_0", "mol_idx_1", "spec_idx_0", "spec_idx_1"],
        how="inner",
    )
    if merged.empty:
        return None
    mol_mass = mol_idx_mass_lookup(str(exp_dir), val_name, step)
    pred_by_bin = _mass_diff_masked_pred_by_bin(
        merged, mol_mass, mass_lo, mass_hi, value_col="cosine"
    )
    pairs = list(zip(_BIN_ORDER, _BIN_ORDER[skip + 1 :]))
    if anchor_bin is not None:
        pairs = [(a, b) for a, b in pairs if a == anchor_bin]
    vals = [
        _overlap_coefficient(pred_by_bin[a], pred_by_bin[b])
        for a, b in pairs
        if a in pred_by_bin and b in pred_by_bin
    ]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else None


# (mass_lo, mass_hi, skip, anchor_bin) matching each fixed Overlap column's
# own SIMBA-based definition -- used to fill the cosine-baseline row's
# equivalent cells with the same skip/mass-range semantics.
COSINE_ROW_LABEL = "cosine (raw spectral baseline)"
_FIXED_OVERLAP_COSINE_SPECS = {
    "Overlap (skip0)": (0.0, np.inf, 0, None),
    "Overlap (skip2)": (0.0, np.inf, 2, None),
    "Identity overlap (skip0)": (0.0, np.inf, 0, "self (MCES=0)"),
    "Identity overlap (skip1)": (0.0, np.inf, 1, "self (MCES=0)"),
    "Identity overlap (skip2)": (0.0, np.inf, 2, "self (MCES=0)"),
    "Overlap (mass diff<30)": (0.0, 30.0, 0, None),
    "Overlap (mass diff<100)": (0.0, 100.0, 0, None),
}


def _find_cosine_reference(surviving: dict) -> tuple[Path, str] | None:
    """First (exp_dir, val_name) in `surviving` whose val_cosine_*.parquet
    already exists -- cosine is experiment-independent (same val set -> same
    spec_idx pairing -> same cosine everywhere), so any one experiment's
    per-pair mces_bin/mol_idx labeling works as the reference."""
    for exp_dir in surviving.values():
        val_name = _pick_val_name(exp_dir)
        if val_name is None:
            continue
        preprocessing_dir = infer_preprocessing_dir(exp_dir)
        if preprocessing_dir is None:
            continue
        if not load_val_cosine(preprocessing_dir, val_name).empty:
            return exp_dir, val_name
    return None


# Hit@k retrieval benchmark among same-molecule near-neighbors (see
# tools/benchmark_self_retrieval.py, which this is adapted from -- kept as a
# self-contained duplicate here rather than a cross-script import, since
# dashboard_app.py otherwise depends only on installed packages / the simba
# package, not sibling tools/*.py files).
HIT_AT_K_VALUES = (1, 5, 20)
N_DECOYS_DEFAULT = 255


def _build_pool_and_queries(ref_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(pool_mols, query_mols): every molecule in the "self (MCES=0)" bucket,
    and the subset with >=2 spectra (spec_idx_0 != spec_idx_1 on their own
    self-pair row) -- the queries for the retrieval benchmark."""
    self_df = ref_df[ref_df["mces_bin"] == "self (MCES=0)"]
    pool_mols = pd.unique(pd.concat([self_df["mol_idx_0"], self_df["mol_idx_1"]]))
    same_spec = self_df["spec_idx_0"] == self_df["spec_idx_1"]
    query_mols = self_df.loc[~same_spec, "mol_idx_0"].to_numpy()
    return pool_mols, query_mols


def _build_score_matrix(
    ref_df: pd.DataFrame, pool_mols: np.ndarray, mol_to_local: dict, value_col: str
) -> np.ndarray:
    """Dense (n_pool, n_pool) matrix of `value_col` for every cross-molecule
    pair within the pool, symmetric; NaN where absent (there shouldn't be
    any within this pool -- confirmed it's a complete graph, 3,727,815
    pairs == exactly C(2731,2))."""
    n = len(pool_mols)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    cross = ref_df[ref_df["mol_idx_0"] != ref_df["mol_idx_1"]]
    pool_set = set(pool_mols.tolist())
    within = cross[
        cross["mol_idx_0"].isin(pool_set) & cross["mol_idx_1"].isin(pool_set)
    ]
    i = within["mol_idx_0"].map(mol_to_local).to_numpy()
    j = within["mol_idx_1"].map(mol_to_local).to_numpy()
    v = within[value_col].to_numpy()
    mat[i, j] = v
    mat[j, i] = v
    return mat


def _true_match_scores(ref_df: pd.DataFrame, value_col: str) -> dict:
    """query_mol -> that method's score for the query's own "self, different
    spectrum" row (its true match: the molecule's other spectrum)."""
    self_df = ref_df[ref_df["mces_bin"] == "self (MCES=0)"]
    diff_spec = self_df["spec_idx_0"] != self_df["spec_idx_1"]
    rows = self_df.loc[diff_spec]
    return dict(zip(rows["mol_idx_0"], rows[value_col]))


def _hit_at_k(
    gt_matrix,
    score_matrix,
    true_scores,
    mol_to_local,
    query_mols,
    n_decoys,
    ks,
    higher_is_better,
) -> dict:
    hits = dict.fromkeys(ks, 0)
    n_scored = 0
    for q in query_mols:
        qi = mol_to_local[q]
        if q not in true_scores or np.isnan(true_scores[q]):
            continue
        gt_row = gt_matrix[qi].copy()
        gt_row[qi] = np.inf  # never pick the query molecule itself as a decoy
        decoy_local = np.argsort(gt_row)[:n_decoys]
        decoy_scores = score_matrix[qi, decoy_local]
        decoy_scores = decoy_scores[~np.isnan(decoy_scores)]
        candidates = np.append(decoy_scores, true_scores[q])
        true_idx = len(candidates) - 1
        order = np.argsort(-candidates if higher_is_better else candidates)
        rank = int(np.nonzero(order == true_idx)[0][0]) + 1
        n_scored += 1
        for k in ks:
            hits[k] += int(rank <= k)
    return {k: (hits[k] / n_scored if n_scored else float("nan")) for k in ks}


@st.cache_data(show_spinner="Computing Hit@k retrieval benchmark ...")
def compute_hit_at_k_all(
    exp_dir_strs: tuple[str, ...],
    val_name: str,
    cosine_parquet_path: str | None,
    n_decoys: int = N_DECOYS_DEFAULT,
    ks: tuple[int, ...] = HIT_AT_K_VALUES,
) -> dict[str, dict]:
    """{run_label: {k: hit_rate}} for every exp_dir (keyed by exp_dir.name,
    matching the Compare-runs table's index) plus, if cosine_parquet_path is
    given, one extra entry keyed COSINE_ROW_LABEL. Decoy selection (255
    nearest-GT-MCES pool molecules per query) is computed once from the
    first exp_dir's data and reused for every method, so the comparison is
    apples-to-apples -- only the ranking differs per method."""
    exp_dirs = [Path(p) for p in exp_dir_strs]
    ref_path = exp_dirs[0] / f"val_pairs_{val_name}_consolidated.parquet"
    ref_df = pd.read_parquet(
        ref_path,
        columns=[
            "mol_idx_0",
            "mol_idx_1",
            "gt_mces",
            "mces_bin",
            "spec_idx_0",
            "spec_idx_1",
        ],
    )
    pool_mols, query_mols = _build_pool_and_queries(ref_df)
    mol_to_local = {m: i for i, m in enumerate(pool_mols)}
    gt_matrix = _build_score_matrix(ref_df, pool_mols, mol_to_local, "gt_mces")

    results = {}
    for exp_dir in exp_dirs:
        steps = list_available_steps(exp_dir, val_name)
        if not steps:
            continue
        pred_col = f"pred_mces_step{steps[-1]:06d}"
        path = exp_dir / f"val_pairs_{val_name}_consolidated.parquet"
        df = pd.read_parquet(
            path,
            columns=[
                "mol_idx_0",
                "mol_idx_1",
                "mces_bin",
                "spec_idx_0",
                "spec_idx_1",
                pred_col,
            ],
        )
        score_matrix = _build_score_matrix(df, pool_mols, mol_to_local, pred_col)
        true_scores = _true_match_scores(df, pred_col)
        results[exp_dir.name] = _hit_at_k(
            gt_matrix,
            score_matrix,
            true_scores,
            mol_to_local,
            query_mols,
            n_decoys,
            ks,
            higher_is_better=False,
        )

    if cosine_parquet_path:
        cos = pd.read_parquet(cosine_parquet_path)
        df = ref_df[
            ["mol_idx_0", "mol_idx_1", "mces_bin", "spec_idx_0", "spec_idx_1"]
        ].merge(
            cos, on=["mol_idx_0", "mol_idx_1", "spec_idx_0", "spec_idx_1"], how="left"
        )
        score_matrix = _build_score_matrix(df, pool_mols, mol_to_local, "cosine")
        true_scores = _true_match_scores(df, "cosine")
        results[COSINE_ROW_LABEL] = _hit_at_k(
            gt_matrix,
            score_matrix,
            true_scores,
            mol_to_local,
            query_mols,
            n_decoys,
            ks,
            higher_is_better=True,
        )
    return results


def _compare_runs_row(exp_dir: Path, include_mass_filtered: bool) -> dict | None:
    """One comparison row for a single experiment. Returns None for
    experiments with no validation data logged yet."""
    df = load_metrics(exp_dir)
    if df.empty:
        return None
    val_name = _pick_val_name(exp_dir)
    if val_name is None:
        return None

    row = {"Run": exp_dir.name, "Last step": int(df["step"].max())}
    row["Val loss"] = _last_non_null(df, "validation_loss_epoch")
    row["Overall MAE"] = _last_non_null(df, "val_mces_mae")
    row["Identity MAE"] = _last_non_null(df, f"val_mae_mces/self (MCES=0)/{val_name}")
    if row["Val loss"] is None and row["Overall MAE"] is None:
        return None  # no validation check has landed for this run yet

    # Older runs (e.g. 009) predate val_overlap* logging entirely -- recompute
    # from per-pair data on demand rather than leaving those cells empty.
    # Loaded at most once per row, only if actually needed.
    pred_by_bin_cache = {}

    def _pred_by_bin():
        if not pred_by_bin_cache:
            pred_by_bin_cache.update(_pred_by_bin_last_step(exp_dir, val_name))
        return pred_by_bin_cache

    for skip in (0, 2):
        val = _last_non_null(df, f"val_overlap_avg/skip{skip}/{val_name}")
        if val is None:
            val = _overlap_avg_skip_from_pairs(_pred_by_bin(), skip)
        row[f"Overlap (skip{skip})"] = val

    for skip in (0, 1, 2):
        row[f"Identity overlap (skip{skip})"] = _identity_overlap_at_skip(
            df, val_name, skip, _pred_by_bin
        )

    if include_mass_filtered:
        row["Overlap (mass diff<30)"] = _overlap_for_spec(
            exp_dir, val_name, 0.0, 30.0, 0, None
        )
        row["Overlap (mass diff<100)"] = _overlap_for_spec(
            exp_dir, val_name, 0.0, 100.0, 0, None
        )

    return row


def _custom_metric_label(spec: dict) -> str:
    mass = f"[{spec['mass_lo']:g},{spec['mass_hi']:g}]"
    if spec["kind"] == "MAE":
        bin_part = "overall" if spec["bin"] is None else spec["bin"]
        return f"MAE {bin_part} mass{mass}"
    bin_part = "avg" if spec["bin"] is None else spec["bin"]
    return f"Overlap skip{spec['skip']} {bin_part} mass{mass}"


def _custom_metric_value(
    exp_dir: Path, val_name: str | None, spec: dict
) -> float | None:
    if val_name is None:
        return None
    if spec["kind"] == "MAE":
        return _mae_for_bin(
            exp_dir, val_name, spec["mass_lo"], spec["mass_hi"], spec["bin"]
        )
    return _overlap_for_spec(
        exp_dir, val_name, spec["mass_lo"], spec["mass_hi"], spec["skip"], spec["bin"]
    )


def render_compare_runs_tab():
    st.subheader("Compare runs")
    st.caption(
        "Each run's most recently logged validation check. Loss/MAE/overlap "
        "come straight from metrics.csv when logged; overlap cells for older "
        "runs that predate that logging, the mass-filtered columns, and any "
        "custom columns added below are recomputed on demand from that "
        "check's per-pair data. Overlap and MAE/loss are all lower-is-better; "
        "Hit@k (see below) is higher-is-better."
    )

    experiments = list_experiments()
    if not experiments:
        st.info("No compatible experiments found.")
        return

    include_mass_filtered = st.checkbox(
        "Include mass-difference overlap columns (<30 Da, <100 Da) "
        "(reads per-pair data, one step per run)",
        value=True,
        key="compare_include_mass_diff",
    )

    rows = [
        r
        for r in (
            _compare_runs_row(exp_dir, include_mass_filtered) for exp_dir in experiments
        )
        if r is not None
    ]
    if not rows:
        st.info("No run has a validation check logged yet.")
        return

    table = pd.DataFrame(rows).set_index("Run")
    surviving = {d.name: d for d in experiments if d.name in table.index}

    include_cosine = st.checkbox(
        "Include cosine-similarity baseline row (raw spectral cosine, "
        "non-learned -- an extra row, not a column; only Overlap cells are "
        "filled in, MAE/loss are left blank since cosine isn't on a "
        "comparable scale to MCES)",
        value=True,
        key="compare_include_cosine",
    )
    cosine_ref = _find_cosine_reference(surviving) if include_cosine else None
    if include_cosine and cosine_ref is None:
        st.caption(
            "Cosine baseline not available yet for this val set -- run "
            "tools/compute_val_cosine.py first."
        )

    st.markdown("###### Add a custom metric column")
    st.caption(
        "Pick a metric type, a molecule mass-difference range, and (for "
        "overlap) a skip distance -- then either the average over all bins "
        "at that skip, or one particular bin/pair."
    )
    with st.form("compare_custom_metric_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns(4)
        kind = c1.selectbox("Metric", ["MAE", "Overlap"], key="cm_kind")
        mass_lo = c2.number_input(
            "Mass diff ≥ (Da)", min_value=0.0, value=0.0, step=5.0, key="cm_mass_lo"
        )
        mass_hi = c3.number_input(
            "Mass diff ≤ (Da)", min_value=0.0, value=30.0, step=5.0, key="cm_mass_hi"
        )
        skip = None
        if kind == "Overlap":
            skip = c4.number_input(
                "Skip",
                min_value=0,
                max_value=len(_BIN_ORDER) - 2,
                value=0,
                step=1,
                key="cm_skip",
            )

        d1, d2 = st.columns(2)
        particular = d1.checkbox(
            "Show one particular bin/pair instead of the average",
            key="cm_particular",
        )
        bin_choice = None
        if particular:
            if kind == "Overlap":
                anchor_options = _BIN_ORDER[: len(_BIN_ORDER) - int(skip) - 1]
                bin_choice = d2.selectbox(
                    "Anchor bin (paired with the bin `skip` further out)",
                    anchor_options,
                    key="cm_bin_choice",
                )
            else:
                bin_choice = d2.selectbox("Bin", _BIN_ORDER, key="cm_bin_choice")

        submitted = st.form_submit_button("Add column")
        if submitted:
            spec = {
                "kind": kind,
                "mass_lo": float(mass_lo),
                "mass_hi": float(mass_hi),
                "skip": int(skip) if skip is not None else None,
                "bin": bin_choice,
            }
            st.session_state.setdefault("compare_custom_metrics", []).append(spec)

    custom_specs = st.session_state.get("compare_custom_metrics", [])
    if custom_specs:
        st.caption("Custom columns:")
        for i, spec in enumerate(custom_specs):
            rcol1, rcol2 = st.columns([6, 1])
            rcol1.write(_custom_metric_label(spec))
            if rcol2.button("Remove", key=f"cm_remove_{i}"):
                custom_specs.pop(i)
                st.rerun()

    custom_labels = [_custom_metric_label(spec) for spec in custom_specs]
    if custom_specs:
        build_custom = st.button(
            "Build custom columns",
            key="compare_build_custom",
            help="Computes each custom column for every run below -- reads "
            "per-pair data and recomputes molecule masses on any cache miss, "
            "so this can take a while with several runs/columns. Not "
            "recomputed automatically on every interaction.",
        )
        cached = st.session_state.setdefault("compare_custom_metric_values", {})
        if build_custom:
            n_runs = max(len(surviving), 1) + (1 if cosine_ref is not None else 0)
            progress = st.progress(0.0, text="Computing custom columns ...")
            total_work = max(len(custom_specs) * n_runs, 1)
            done = 0
            for spec, label in zip(custom_specs, custom_labels):
                value_by_run = {}
                for name, exp_dir in surviving.items():
                    value_by_run[name] = _custom_metric_value(
                        exp_dir, _pick_val_name(exp_dir), spec
                    )
                    done += 1
                    progress.progress(min(done / total_work, 1.0))
                if cosine_ref is not None and spec["kind"] == "Overlap":
                    ref_exp_dir, ref_val_name = cosine_ref
                    value_by_run[COSINE_ROW_LABEL] = _overlap_for_spec_cosine(
                        ref_exp_dir,
                        ref_val_name,
                        spec["mass_lo"],
                        spec["mass_hi"],
                        spec["skip"],
                        spec["bin"],
                    )
                    done += 1
                    progress.progress(min(done / total_work, 1.0))
                cached[label] = value_by_run
            progress.empty()
        for label in custom_labels:
            if label in cached:
                table[label] = table.index.map(cached[label])
            else:
                table[label] = np.nan
        if any(label not in cached for label in custom_labels):
            st.info(
                "Click **Build custom columns** to compute the new column(s) "
                "-- showing blank until then."
            )

    if cosine_ref is not None:
        ref_exp_dir, ref_val_name = cosine_ref
        cosine_row = dict.fromkeys(table.columns, np.nan)
        for col, (mlo, mhi, skip, anchor) in _FIXED_OVERLAP_COSINE_SPECS.items():
            if col in cosine_row:
                cosine_row[col] = _overlap_for_spec_cosine(
                    ref_exp_dir, ref_val_name, mlo, mhi, skip, anchor
                )
        cached = st.session_state.get("compare_custom_metric_values", {})
        for label in custom_labels:
            if label in cosine_row and label in cached:
                cosine_row[label] = cached[label].get(COSINE_ROW_LABEL, np.nan)
        table.loc[COSINE_ROW_LABEL] = cosine_row

    st.markdown(
        "###### Retrieval benchmark: Hit@1/5/20 among same-molecule near-neighbors"
    )
    st.caption(
        "For every validation molecule with ≥2 spectra (2,010 of them), ranks "
        "it against 255 decoys -- the pool's own lowest-GT-MCES neighbor "
        "molecules -- plus its own other spectrum (the true match). "
        "Hit@k = is the true match in the top k, sorted by predicted MCES "
        "ascending (SIMBA) or cosine similarity descending. Fully computed "
        "from existing per-pair data, no re-inference. See "
        "tools/benchmark_self_retrieval.py."
    )
    include_hit_at_k = st.checkbox(
        "Compute Hit@1/5/20 (reads full per-pair data, builds one "
        "molecule×molecule matrix per run -- a few seconds per run)",
        value=True,
        key="compare_include_hit_at_k",
    )
    if include_hit_at_k and surviving:
        hit_cosine_ref = _find_cosine_reference(surviving)
        cosine_parquet_path = None
        if hit_cosine_ref is not None:
            ref_exp_dir, ref_val_name = hit_cosine_ref
            preprocessing_dir = infer_preprocessing_dir(ref_exp_dir)
            if preprocessing_dir is not None:
                cosine_parquet_path = str(
                    Path(preprocessing_dir) / f"val_cosine_{ref_val_name}.parquet"
                )
        exp_dir_strs = tuple(str(d) for d in surviving.values())
        val_name_for_hitk = _pick_val_name(next(iter(surviving.values())))
        hit_results = compute_hit_at_k_all(
            exp_dir_strs, val_name_for_hitk, cosine_parquet_path
        )
        for run_label, ks_dict in hit_results.items():
            for k, v in ks_dict.items():
                table.loc[run_label, f"Hit@{k}"] = v

    default_cols = [
        c
        for c in [
            "Last step",
            "Val loss",
            "Overall MAE",
            "Identity MAE",
            "Overlap (skip0)",
            "Overlap (skip2)",
            "Identity overlap (skip0)",
            "Identity overlap (skip1)",
            "Identity overlap (skip2)",
            "Overlap (mass diff<30)",
            "Overlap (mass diff<100)",
            "Hit@1",
            "Hit@5",
            "Hit@20",
            *custom_labels,
        ]
        if c in table.columns
    ]
    selected = st.multiselect(
        "Metrics to show", options=list(table.columns), default=default_cols
    )
    if not selected:
        st.info("Pick at least one metric.")
        return

    shown = table[selected]
    # Excludes all-NaN columns (e.g. a custom column added but not yet built)
    # -- Styler.background_gradient's np.nanmax/np.nanmin on an all-NaN slice
    # segfaults under this environment's pandas/numpy/matplotlib versions
    # rather than raising, so this isn't just a cosmetic skip.
    # Hit@k is higher-is-better, unlike everything else in this table
    # (loss/MAE/overlap are all lower-is-better) -- needs the opposite
    # color-gradient direction or it would show high hit rates as red.
    higher_is_better_cols = {f"Hit@{k}" for k in HIT_AT_K_VALUES}
    color_cols = [c for c in selected if c != "Last step" and shown[c].notna().any()]
    lower_better_cols = [c for c in color_cols if c not in higher_is_better_cols]
    higher_better_cols = [c for c in color_cols if c in higher_is_better_cols]
    styled = shown.style.format(precision=4, na_rep="—")
    if lower_better_cols:
        styled = styled.background_gradient(
            cmap="RdYlGn_r", axis=0, subset=lower_better_cols
        )
    if higher_better_cols:
        styled = styled.background_gradient(
            cmap="RdYlGn", axis=0, subset=higher_better_cols
        )
    st.dataframe(styled, width="stretch")

    bar_options = [c for c in selected if c != "Last step"]
    if bar_options:
        st.markdown("###### Bar chart")
        metric_for_bar = st.selectbox("Metric", bar_options, key="compare_bar_metric")
        st.bar_chart(shown[metric_for_bar])


def main():
    st.set_page_config(page_title="SIMBA training dashboard", layout="wide")
    st.title("SIMBA training dashboard")

    experiments = list_experiments()
    if not experiments:
        st.warning(f"No compatible experiments found under {EXPERIMENTS_DIR}")
        return

    with st.sidebar:
        exp_name = st.selectbox("Experiment", [e.name for e in experiments])
        if st.button("Refresh"):
            # Rerunning the script alone does NOT invalidate @st.cache_data
            # (e.g. list_consolidated_steps, load_pair_data_for_step) --
            # those cache by arguments for the server process's lifetime, so
            # a long-running experiment's newly-appended validation checks
            # would otherwise stay invisible until the dashboard is
            # restarted. Explicitly drop all cached data so this button does
            # what it says.
            st.cache_data.clear()
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

    tab_loss, tab_metrics, tab_box, tab_mass, tab_compare = st.tabs(
        ["Loss", "Validation metrics", "Box plot", "Mass heatmap", "Compare runs"]
    )
    with tab_loss:
        render_loss_tab(df, x_col, x_label)
    with tab_metrics:
        render_metrics_tab(df, x_col, x_label, exp_dir)
    with tab_box:
        render_box_plot_tab(exp_dir)
    with tab_mass:
        render_mass_heatmap_tab(exp_dir)
    with tab_compare:
        render_compare_runs_tab()


if __name__ == "__main__":
    main()
