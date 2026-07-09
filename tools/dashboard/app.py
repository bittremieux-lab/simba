"""
SIMBA experiment dashboard.

Run: uv run streamlit run tools/dashboard/app.py --server.port 8505
"""

import io
from pathlib import Path

import streamlit as st
from PIL import Image


st.set_page_config(page_title="SIMBA Experiments", layout="wide")

ASSETS = Path(__file__).parent / "assets"

DATA = Path("/mnt/data/nkubrakov/experiments_3_dataset/training")
DATA2 = Path("/mnt/data2/nkubrakov/experiments_3_dataset/training")

EXPERIMENTS = [
    {
        "label": "clip-40 (reference)",
        "dir": DATA2 / "msg_scaffold_split_mces40",
        "inference_dir": DATA / "msg_scaffold_split_mces40/val_hexbin_step22k",
        "job": "7614",
        "status": "done",
        "data": "MSG HDF5",
        "val_sets": "official val + scaffold val + test",
        "note": "MSG · MCES only · clip at 40 · n_classes=11 · best scaffold ρ=0.660 @ step 23k",
    },
    {
        "label": "clip-40 + metadata",
        "dir": DATA / "msg_scaffold_split_mces40_metadata",
        "inference_dir": DATA / "msg_scaffold_split_mces40_metadata/val_hexbin_step22k",
        "job": "7636",
        "status": "done",
        "data": "MSG HDF5",
        "val_sets": "official val + scaffold val + test",
        "note": "MSG · MCES only · clip at 40 · adduct + CE + ion mode · best scaffold ρ=0.653 @ step 22k",
    },
    {
        "label": "Gaetan lb_matrix · clip-40",
        "dir": DATA / "msg_gaetan_official_mces40",
        "inference_dir": DATA / "msg_gaetan_official_mces40/val_hexbin_step35k",
        "job": "7656",
        "status": "done",
        "data": "Gaetan lb_matrix",
        "val_sets": "official val + scaffold val + test",
        "note": "MSG · Gaetan tighter lower bounds · clip at 40 · best scaffold ρ=0.383 @ step 35k",
    },
    {
        "label": "MSG scaffold-v2 · MCES only",
        "dir": DATA / "scaffold_v2_mces_only",
        "inference_dir": DATA / "scaffold_v2_mces_only/val_hexbin_step67k",
        "job": "7637",
        "status": "done",
        "data": "MSG HDF5 (scaffold v2 split)",
        "val_sets": "val only",
        "note": "MSG · Murcko scaffold split v2 · MCES only · ED-based sampling · lr=3.33e-5 · val ρ @ step 67k",
    },
    {
        "label": "MSG scaffold-v2 · ED + MCES",
        "dir": DATA / "scaffold_v2_both",
        "inference_dir": DATA / "scaffold_v2_both/val_hexbin_step67k",
        "job": "7638",
        "status": "done",
        "data": "MSG HDF5 (scaffold v2 split)",
        "val_sets": "val only",
        "note": "MSG · Murcko scaffold split v2 · ED + MCES objectives · lr=3.33e-5 · best val ρ=0.804 @ step 67k",
    },
]

STATUS_COLOR = {"done": "🟢", "running": "🔵", "pending": "🟡"}


def placeholder(label: str):
    st.markdown(
        f"<div style='background:#f5f5f5;padding:20px;text-align:center;"
        f"color:#aaa;border-radius:6px;font-size:0.82em'>not yet available<br>{label}</div>",
        unsafe_allow_html=True,
    )


def show(path: Path | None, caption: str = "", w: int | None = None):
    if path and path.exists():
        kw = {"width": w} if w else {"use_container_width": True}
        st.image(str(path), caption=caption, **kw)
    else:
        placeholder(caption or (path.name if path else ""))


def show_hexbin(path: Path | None, caption: str = ""):
    """Show only the top half (linear scale) of a hexbin image."""
    if not (path and path.exists()):
        placeholder(caption or (path.name if path else ""))
        return
    img = Image.open(path)
    w, h = img.size
    cropped = img.crop((0, 0, w, h // 2))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    st.image(buf.getvalue(), caption=caption, use_container_width=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("SIMBA Dashboard")
    for exp in EXPERIMENTS:
        icon = STATUS_COLOR.get(exp["status"], "⚪")
        st.markdown(f"{icon} {exp['label']}  `{exp['job']}`")
    st.markdown("---")
    st.caption("Refresh to pick up new checkpoints.")


# ── Header ───────────────────────────────────────────────────────────────────
st.title("SIMBA · MSG Scaffold Split Experiments")

# ── Background ───────────────────────────────────────────────────────────────
with st.expander("📜 Background", expanded=False):
    st.markdown("""
### The goal

We want SIMBA to predict MCES-based molecular similarity from spectra alone.
The paper trained on both edit distance (ED) and MCES with a joint dataset and
got strong results on the MassSpecGym benchmark. We're trying to reproduce and
then improve those results using the official MSG splits.
    """)
    show(ASSETS / "paper_result.png", w=500)

    st.markdown("---")
    st.markdown("""
### First attempt: discard-20

We trained on MSG only, with MCES as the sole objective, using scaffold-based
train/val splits (official val and test kept as-is). Pairs with MCES > 20 were
discarded. In the 0–20 range the results look comparable to the paper — scaffold
val even shows better separation in 0–10. Splitting strategy does not make much
difference to performance.
    """)
    show(ASSETS / "discard20_hexbins.png", "official val · test · scaffold val", w=800)

    st.markdown("---")
    st.markdown("""
### Distance source check

Before going further we checked how the MCES distances in the MSG HDF5 compare
to independently computed values (from SIMBA's own joint preprocessing). Spearman
ρ = 0.808, MAE = 5.6 — decent correlation but a clear systematic gap, especially
above MCES 20 where HDF5 values cluster. This raised questions about HDF5 reliability
for the higher-distance regime.
    """)
    show(ASSETS / "joint_vs_hdf5_scatter.png", w=700)

    st.markdown("---")
    st.markdown("""
### Why the gap? HDF5 has a solver threshold at 10

The MSG HDF5 was computed with `threshold=10, always_stronger_bound=False`.
This means values below 10 are **exact**, but values ≥ 10 are only **lower bounds**
produced by an ILP solver that stops early. The distribution below confirms
there is no artificial spike at 10 — the solver gives smooth lower bounds —
but the tail is heavily underestimated.

Juan Sebastián suggested the model might be learning a mean prediction because
so many pairs pile up at the clip boundary. The distribution shows that is not
the issue here (no spike), so the natural next step was to clip at 40 instead
of discarding, keeping all pairs as a learning signal.
    """)
    show(ASSETS / "mces_distribution.png", w=800)

    st.markdown("---")
    st.markdown("""
### clip-40: keep all pairs, clip at 40

Instead of discarding pairs with MCES > 20 we now clip all values at 40 and use
11 output classes. Performance in the 0–20 range is slightly worse — with > 95 %
of pairs having MCES > 20 the model spends more capacity on the dissimilar tail —
but the full-range coverage is much better and correlation improves considerably
on official val and test.
    """)
    show(ASSETS / "clip40_hexbins.png", "official val · test · scaffold val", w=800)

    st.markdown("---")
    st.markdown("""
### Better lower bounds: Gaetan's lb_matrix

The HDF5 lower bounds are weak for MCES ≥ 10 because the solver stops early.
Gaetan computed a 108 GB lb_matrix with `threshold=0, always_stronger_bound=True`
covering MSG + SpectraVerse + Enveda — a much tighter lower bound for the
high-distance regime. For pairs with exact MCES < 10 the HDF5 is exact and Gaetan
underestimates (below diagonal); for MCES ≥ 10 Gaetan is consistently tighter
(above diagonal). Spearman ρ = 0.851, MAE = 7.49 vs HDF5.

The next round of experiments uses Gaetan's bounds as the distance source.
    """)
    show(ASSETS / "hdf5_vs_gaetan_scatter.png", w=550)

st.markdown("---")

# ── Experiments ──────────────────────────────────────────────────────────────
st.header("Experiments")

for exp in EXPERIMENTS:
    icon = STATUS_COLOR.get(exp["status"], "⚪")
    with st.expander(f"{icon} {exp['label']}  ·  job {exp['job']}", expanded=True):
        st.caption(f"**Data:** {exp['data']}  ·  **Eval:** {exp['val_sets']}")
        st.caption(exp["note"])
        d = exp["dir"]

        # Loss + metrics
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Loss**")
            show(d / "loss_plot.png")
        with c2:
            st.markdown("**Metrics**")
            show(d / "metrics_curves.png")

        # Inference results
        inf_dir = exp.get("inference_dir")
        inf_dir = Path(inf_dir) if inf_dir else None
        if inf_dir and inf_dir.exists():
            step_label = inf_dir.name.replace("val_hexbin_", "")
            st.markdown(f"**Inference · {step_label}**")
            balanced = inf_dir / "mces_hexbin_balanced.png"
            if balanced.exists():
                show_hexbin(balanced)
            else:
                panels = sorted(inf_dir.glob("*_linear.png"))
                if panels:
                    cols = st.columns(len(panels))
                    for col, p in zip(cols, panels):
                        with col:
                            show(p, p.stem)
                else:
                    placeholder("no plots yet")
        else:
            st.info("No inference results yet.")
