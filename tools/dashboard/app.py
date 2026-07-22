"""
SIMBA experiment dashboard.

Run: uv run streamlit run tools/dashboard/app.py --server.port 8505
"""

import io
from pathlib import Path

import pandas as pd
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
        "inference_dirs": [DATA / "msg_scaffold_split_mces40/val_hexbin_step22k"],
        "job": "7614",
        "status": "done",
        "data": "MSG HDF5",
        "val_sets": "official val + scaffold val + test",
        "note": "MSG · MCES only · clip at 40 · n_classes=11 · best scaffold ρ=0.660 @ step 23k",
        "retrieval_tsv": Path(
            "/home/nkubrakov/simba/results/simba_retrieval_clip40_step22k.tsv"
        ),
    },
    {
        "label": "clip-40 + metadata",
        "dir": DATA / "msg_scaffold_split_mces40_metadata",
        "inference_dirs": [
            DATA / "msg_scaffold_split_mces40_metadata/val_hexbin_step22k"
        ],
        "job": "7636",
        "status": "done",
        "data": "MSG HDF5",
        "val_sets": "official val + scaffold val + test",
        "note": "MSG · MCES only · clip at 40 · adduct + CE + ion mode · best scaffold ρ=0.653 @ step 22k",
        "retrieval_tsv": Path(
            "/home/nkubrakov/simba/results/simba_retrieval_clip40_metadata_step22k.tsv"
        ),
    },
    {
        "label": "Gaetan lb_matrix · clip-40",
        "dir": DATA / "msg_gaetan_official_mces40",
        "inference_dirs": [DATA / "msg_gaetan_official_mces40/val_hexbin_step35k"],
        "job": "7656",
        "status": "done",
        "data": "Gaetan lb_matrix",
        "val_sets": "official val + scaffold val + test",
        "note": "MSG · Gaetan tighter lower bounds · clip at 40 · best scaffold ρ=0.383 @ step 35k",
        "retrieval_tsv": Path(
            "/home/nkubrakov/simba/results/simba_retrieval_gaetan_official_step35k.tsv"
        ),
    },
    {
        "label": "max(Gaetan lb, HDF5) · MCES only",
        "dir": DATA / "msg_max_lb_hdf5_mces40",
        "inference_dirs": [
            DATA / "msg_max_lb_hdf5_mces40/val_hexbin_step22k",
            DATA / "msg_max_lb_hdf5_mces40/val_hexbin_step70k",
        ],
        "job": "7759",
        "status": "done",
        "data": "max(Gaetan lb, HDF5)",
        "val_sets": "official val + scaffold val + test",
        "note": "MSG · max(lb_matrix, HDF5) distances · MCES only · clip at 40 · n_classes=11 · scaffold ρ≈0.388 @ step 70k",
        "retrieval_tsvs": [
            {
                "label": "step 22k",
                "path": Path(
                    "/home/nkubrakov/simba/results/simba_retrieval_msg_max_lb_hdf5_step22k.tsv"
                ),
            },
            {
                "label": "step 70k",
                "path": Path(
                    "/home/nkubrakov/simba/results/simba_retrieval_msg_max_lb_hdf5_step70k.tsv"
                ),
            },
        ],
    },
    {
        "label": "max(Gaetan lb, HDF5) · Tani≥0.2",
        "dir": DATA / "msg_max_lb_hdf5_tani02_mces40",
        "inference_dirs": [
            DATA / "msg_max_lb_hdf5_tani02_mces40/val_hexbin_step45k",
            DATA / "msg_max_lb_hdf5_tani02_mces40/val_hexbin_step100k",
        ],
        "job": "7772",
        "status": "done",
        "data": "max(Gaetan lb, HDF5) · Tani≥0.2 filter",
        "val_sets": "official val + scaffold val + test",
        "note": "MSG · max(lb_matrix, HDF5) distances · Tanimoto≥0.2 + ≤40 atoms filter · MCES only · clip at 40 · n_classes=11 · 14 epochs (~130k steps)",
        "retrieval_tsv": Path(
            "/home/nkubrakov/simba/results/simba_retrieval_tani02_step100k.tsv"
        ),
    },
    {
        "label": "clip-40 + metadata (CE fix + norm/100)",
        "dir": DATA / "msg_scaffold_split_mces40_metadata_ce_v2",
        "inference_dirs": [
            DATA / "msg_scaffold_split_mces40_metadata_ce_v2/val_hexbin_step15k"
        ],
        "job": "7955",
        "status": "running",
        "data": "MSG HDF5",
        "val_sets": "official val + scaffold val + test",
        "note": "MSG · MCES only · clip at 40 · adduct + CE + ion mode · CE fix (reads collision_energy, float-safe) · CE/100 normalization in encoder · MCES sampling bug fixed · 8 epochs ≈ 80k steps",
        "retrieval_tsvs": [
            {
                "label": "step 15k",
                "path": Path(
                    "/home/nkubrakov/simba/results/simba_retrieval_ce_v2_step15k.tsv"
                ),
            },
            {
                "label": "step 60k",
                "path": Path(
                    "/home/nkubrakov/simba/results/simba_retrieval_ce_v2_step60k.tsv"
                ),
            },
            {
                "label": "step 15k · CE=0 🧪",
                "path": Path(
                    "/home/nkubrakov/simba/results/simba_retrieval_ce_v2_step15k_cezero.tsv"
                ),
            },
            {
                "label": "step 60k · CE=0 🧪",
                "path": Path(
                    "/home/nkubrakov/simba/results/simba_retrieval_ce_v2_step60k_cezero.tsv"
                ),
            },
        ],
    },
    {
        "label": "MSG scaffold-v2 · MCES only",
        "dir": DATA / "scaffold_v2_mces_only",
        "inference_dirs": [DATA / "scaffold_v2_mces_only/val_hexbin_step67k"],
        "job": "7637",
        "status": "done",
        "data": "MSG HDF5 (scaffold v2 split)",
        "val_sets": "val only",
        "note": "MSG · Murcko scaffold split v2 · MCES only · ED-based sampling · lr=3.33e-5 · val ρ @ step 67k",
    },
    {
        "label": "MSG scaffold-v2 · ED + MCES",
        "dir": DATA / "scaffold_v2_both",
        "inference_dirs": [DATA / "scaffold_v2_both/val_hexbin_step67k"],
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
with st.expander("📖 Research log", expanded=False):
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
    st.markdown("""
### Training pair MCES distribution · max(Gaetan lb, HDF5) · job 7759

238,372,695 training pairs. The distribution peaks around MCES 22–23 (~9.75M pairs per unit).
The [10, 20] range contains **62,647,072 pairs (26.28%)**. The spike at 40 (33.5M, 14.05%)
is the clip artifact — all pairs with true MCES ≥ 40 are collapsed to exactly 40.
    """)
    show(ASSETS / "mces_pair_distribution_7759.png", w=900)

    st.markdown("---")
    st.markdown("""
### Resolving the [10, 20] range: exact MCES computation

The 26.28 % of training pairs with `max(Gaetan lb, HDF5) ∈ [10, 20]` were only
lower-bounded — the ILP solver stopped before finding the exact value. For these
62,647,072 pairs (plus 323k val-scaffold, 602k val-official, 555k test) we ran a full
exact computation with `threshold=20, always_stronger_bound=True` on asimov2,
parallelised as 1,000 SLURM array tasks of ~62k pairs each (16 CPUs per task, ~4.5 h
for block 0 of 313k pairs as the benchmark).

**Results across all four splits:**

| Split | Pairs in [10, 20] | Mean lb → Mean exact | % increased | % capped at 20 | Spearman ρ (lb vs exact) |
|---|---|---|---|---|---|
| Train | 62,647,072 | 16.24 → 18.97 (+2.73) | 87.1 % | 66.9 % | 0.636 |
| Val (scaffold) | 323,253 | 16.35 → 19.34 (+3.00) | 89.0 % | 76.1 % | 0.567 |
| Val (official) | 602,786 | 17.10 → 19.93 (+2.82) | 88.4 % | **96.4 %** | 0.263 |
| Test | 555,937 | 17.15 → 19.93 (+2.78) | 88.3 % | **96.4 %** | 0.263 |

The low Spearman ρ for val-official and test (0.26) and the 96 % cap rate reveals that
most of these pairs actually have true MCES > 20 — the lower bounds were in [10, 20]
but the molecules are dissimilar enough that the exact edit distance exceeds the
threshold. Only 3,843 train pairs (0.006 %) were unresolvable (watchdog timeout) and
retain their original lower bound.

All [< 10] and [> 20] pairs are unchanged — below 10 the HDF5 values are already exact,
above 20 the clip-at-40 mechanism handles them.
    """)
    show(ASSETS / "mces_exact_vs_lb_scatter.png", w=950)

    st.markdown("---")
    st.markdown("""
### MCES distance distribution across splits

Distribution of ground-truth MCES values across all four splits in
`preprocessing_msg_exact_mces_1020`. The dominant [20, 25) bin (red) is largely
composed of pairs where the ILP solver hit `threshold=20` — the exact MCES is ≥ 20
but stored as 20.0. The orange line on the train panel shows the inverse-frequency
**sampler weight** each bin receives: rare bins like [0, 2.5) are upweighted ~500×
relative to the crowded [20, 25) bin so the model sees a balanced distribution of
similarities during training.

Note: there is **no mechanism to exclude pairs at MCES = 20** — they are included
in training with the weight of the [15, 20) sampler bin (because `searchsorted` with
`side='left'` maps the value 20.0 exactly onto edge index 5, i.e. the [15, 20) bin).
This means the ~42M threshold-capped pairs at exactly 20.0 get the same weight as
the 17.5M pairs with true MCES in [15, 20), inflating that bin and slightly
reducing its per-pair sampling probability.
    """)
    show(ASSETS / "mces_distribution_splits.png", w=1000)

    st.markdown("---")
    st.markdown("""
### Job 8104 · own val weights · fixed val seed · bs=2048 · lr=0.0001

Same dataset as job 8041 (`preprocessing_msg_exact_mces_1020`, own per-split inverse-frequency
weights, no metadata, no early stopping, 8 epochs). Changes vs. job 8041:

- **batch_size=2048** — matches scaffold-v2 reference throughput.
- **Fixed val sampler seed=0** — val/val_official samplers now draw the same batches every
  epoch, so Spearman ρ traces are comparable across checkpoints without sampling noise.
- **limit_val_batches=100** — keeps each validation pass fast.
- **Histogram binning fix** — MCES display bins now use `searchsorted(side='left')` matching
  the weight computation; boundary integers (5, 10, …, 35) were previously assigned to the
  wrong bin in the log output.

Results appear below as they arrive.
    """)
    _d8104 = DATA / "msg_exact_mces_1020_no_meta_own_val_weights_bs2048_v2"
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Loss · job 8104**")
        show(_d8104 / "loss_plot.png")
    with c2:
        st.markdown("**Metrics · job 8104**")
        show(_d8104 / "metrics_curves.png")

    _inf8104 = _d8104 / "val_hexbin_step44k"
    if _inf8104.exists():
        st.markdown("**Inference hexbin · step 44k**")
        _balanced = _inf8104 / "mces_hexbin_balanced.png"
        if _balanced.exists():
            show_hexbin(_balanced)
        else:
            _panels = sorted(_inf8104.glob("*_linear.png"))
            if _panels:
                _cols = st.columns(len(_panels))
                for _col, _p in zip(_cols, _panels):
                    with _col:
                        show(_p, _p.stem)

    _ret8104 = Path(
        "/home/nkubrakov/simba/results/simba_retrieval_bs2048_v2_step44k.tsv"
    )
    if _ret8104.exists():
        st.markdown("**Retrieval benchmark · step 44k (all 194k train spectra)**")
        _df = pd.read_csv(_ret8104, sep="\t")
        _row = _df.iloc[0]
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("SIMBA hit@1", f"{_row['hit@1'] * 100:.2f}%")
        _c2.metric("SIMBA hit@5", f"{_row['hit@5'] * 100:.2f}%")
        _c3.metric("SIMBA hit@20", f"{_row['hit@20'] * 100:.2f}%")
        st.caption(
            f"n={int(_row['n'])} · cosine-NN → Morgan FP transfer → Tanimoto ranking"
        )

    _cos_hex = Path(
        "/home/nkubrakov/simba/results/cosine_hexbins_bs2048_v2_step44k.png"
    )
    if _cos_hex.exists():
        st.markdown(
            "**Spectral cosine hexbins · step 44k** — GT MCES / pred MCES / calibration error vs spectral cosine (GT-balanced, with marginals)"
        )
        show(_cos_hex)

    _ret_cos_hex = Path(
        "/home/nkubrakov/simba/results/retrieval_cosine_hexbins_bs2048_v2_step44k.png"
    )
    if _ret_cos_hex.exists():
        st.markdown(
            "**Retrieval cosine hexbins · step 44k** — test vs SIMBA-picked training spectrum: GT MCES / pred MCES / error vs spectral cosine"
        )
        show(_ret_cos_hex)

    _hits_json = Path(
        "/home/nkubrakov/simba/results/retrieval_diagnostics_bs2048_v2_step44k_dedup_hits.json"
    )
    if _hits_json.exists():
        import json as _json

        _hits = _json.loads(_hits_json.read_text())
        _ora, _sim = _hits.get("oracle", {}), _hits.get("simba_cosine_nn", {})
        st.markdown("**Oracle vs SIMBA hit rates · step 44k dedup**")
        _cols = st.columns(6)
        for _i, _k in enumerate(("hit@1", "hit@5", "hit@20")):
            _cols[_i].metric(f"Oracle {_k}", f"{_ora.get(_k, 0) * 100:.2f}%")
            _cols[_i + 3].metric(f"SIMBA {_k} (diag)", f"{_sim.get(_k, 0) * 100:.2f}%")
        st.caption(
            "Oracle = best training molecule by GT MCES; SIMBA = cosine-NN pick; Tanimoto candidate ranking"
        )

    _diag8104 = Path(
        "/home/nkubrakov/simba/results/retrieval_diagnostics_bs2048_v2_step44k_dedup.png"
    )
    _diag8104_mol = Path(
        "/home/nkubrakov/simba/results/retrieval_diagnostics_bs2048_v2_step44k_dedup_molprops.png"
    )
    if not _diag8104.exists():
        _diag8104 = Path(
            "/home/nkubrakov/simba/results/retrieval_diagnostics_bs2048_v2_step44k.png"
        )
        _diag8104_mol = Path(
            "/home/nkubrakov/simba/results/retrieval_diagnostics_bs2048_v2_step44k_molprops.png"
        )
    if _diag8104.exists():
        st.markdown(
            "**Retrieval diagnostics · step 44k** — oracle mol rank, calibration errors, oracle GT MCES"
        )
        show(_diag8104)
    if _diag8104_mol.exists():
        st.markdown(
            "**Molecular property analysis · step 44k** — Tanimoto, SIMBA GT vs oracle GT"
        )
        show(_diag8104_mol)

    _cal = Path(
        "/home/nkubrakov/simba/results/calibration_analysis_bs2048_v2_step44k.png"
    )
    _cal_pop = Path(
        "/home/nkubrakov/simba/results/calibration_analysis_bs2048_v2_step44k_pop.png"
    )
    if _cal.exists():
        st.markdown(
            "**Calibration error anatomy · step 44k** — cosine sim distributions, error vs mass/atoms/Tanimoto"
        )
        show(_cal)
    if _cal_pop.exists():
        st.markdown(
            "**Calibration error population analysis** — high-error vs low-error pair properties"
        )
        show(_cal_pop)

st.markdown("---")

# ── Oracle retrieval upper bound ──────────────────────────────────────────────
st.header("Oracle retrieval upper bound")
st.markdown("""
This section establishes a **theoretical ceiling** for SIMBA's retrieval performance.
The oracle answers the question: *if the model had access to the true ground-truth MCES
distances at inference time, what is the best hit@k it could achieve using the same
Morgan FP transfer + Tanimoto ranking pipeline?*

**Method.**  For every test molecule (n = 17,556) the oracle:
1. Computes the true MCES distance to every training molecule, using the best available
   ground truth: `max(Gaetan lb_matrix, MSG HDF5)`.
2. Selects the nearest training molecule by that distance (nearest-neighbor transfer).
3. Uses that training molecule's Morgan fingerprint as the predicted fingerprint.
4. Ranks the retrieval candidates by Tanimoto similarity to the predicted fingerprint —
   exactly as SIMBA does at inference.

Any learned model that predicts molecular similarity from spectra is bounded by these
numbers: even perfect MCES prediction cannot exceed them, because the ceiling is set by
the retrieval candidates available in the training set and the Tanimoto ranking of
Morgan FPs.
""")

_oracle_tsv = Path("/home/nkubrakov/simba/results/oracle_retrieval_max_lb_hdf5.tsv")
if _oracle_tsv.exists():
    _odf = pd.read_csv(_oracle_tsv, sep="\t")
    _or = _odf.iloc[0]
    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("Oracle hit@1", f"{_or['hit@1'] * 100:.2f}%")
    _c2.metric("Oracle hit@5", f"{_or['hit@5'] * 100:.2f}%")
    _c3.metric("Oracle hit@20", f"{_or['hit@20'] * 100:.2f}%")
else:
    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("Oracle hit@1", "—")
    _c2.metric("Oracle hit@5", "—")
    _c3.metric("Oracle hit@20", "—")

st.markdown("""
*Script: `tools/oracle_retrieval_max_lb_hdf5.py` · Result: `results/oracle_retrieval_max_lb_hdf5.tsv`*
""")

_spec_cos_tsv = Path(
    "/home/nkubrakov/simba/results/oracle_retrieval_spectral_cosine_bs2048_v2_step44k.tsv"
)
if _spec_cos_tsv.exists():
    st.markdown(
        "**Spectral cosine oracle** — picks training spectrum with highest peak-based cosine sim to test spectrum"
    )
    _scdf = pd.read_csv(_spec_cos_tsv, sep="\t")
    _scr = _scdf.iloc[0]
    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("Spectral-cos oracle hit@1", f"{_scr['hit@1'] * 100:.2f}%")
    _c2.metric("Spectral-cos oracle hit@5", f"{_scr['hit@5'] * 100:.2f}%")
    _c3.metric("Spectral-cos oracle hit@20", f"{_scr['hit@20'] * 100:.2f}%")
    st.caption(
        f"n={int(_scr['n'])} · spectral cosine NN → Morgan FP transfer → Tanimoto ranking"
    )

st.markdown("---")

# ── Experiments ──────────────────────────────────────────────────────────────
st.header("Experiments")

for exp in EXPERIMENTS:
    icon = STATUS_COLOR.get(exp["status"], "⚪")
    with st.expander(f"{icon} {exp['label']}  ·  job {exp['job']}", expanded=False):
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

        # Inference results (supports multiple checkpoints)
        inf_dirs = [Path(d) for d in exp.get("inference_dirs", [])]
        any_inf = False
        for inf_dir in inf_dirs:
            if not inf_dir.exists():
                continue
            any_inf = True
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
        if not any_inf:
            st.info("No inference results yet.")

        # Retrieval benchmark (official splits only)
        # Supports both legacy `retrieval_tsv` (single path) and `retrieval_tsvs` (list of dicts).
        retrieval_entries = []
        if exp.get("retrieval_tsvs"):
            retrieval_entries = [(e["label"], e["path"]) for e in exp["retrieval_tsvs"]]
        elif exp.get("retrieval_tsv"):
            retrieval_entries = [("", Path(exp["retrieval_tsv"]))]

        for step_label, tsv_path in retrieval_entries:
            if Path(tsv_path).exists():
                header = f"**Retrieval benchmark · SIMBA NN transfer (test set){' · ' + step_label if step_label else ''}**"
                st.markdown(header)
                df = pd.read_csv(tsv_path, sep="\t")
                row = df.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("hit@1", f"{row['hit@1'] * 100:.2f}%")
                c2.metric("hit@5", f"{row['hit@5'] * 100:.2f}%")
                c3.metric("hit@20", f"{row['hit@20'] * 100:.2f}%")
                st.caption(
                    f"n={int(row['n'])} · SIMBA embedding NN → Morgan FP transfer → Tanimoto candidate ranking"
                )
