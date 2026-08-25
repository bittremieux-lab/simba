"""Val-vs-test-to-test binned-box comparison, side by side, using the
EXACT SAME binning/styling AND the exact same molecule-level pairing
protocol for both (val's own callbacks.py::_plot_binned_box convention:
self-pairs get a dedicated box at GT=0, non-self pairs use 5-unit-wide
bins up to 40; ood_generalization_check.score_test_to_test_molecule_level
for the test side, matching multitask_dataset.py's own validation-pair
resolution rule) -- genuinely comparable, unlike a naive exhaustive
spectrum-vs-spectrum test-to-test computation would be.

Produces a 2x2 grid: rows are {raw regression, CORN-corrected}, columns are
{val, test-to-test} -- so the classification (bucket) head's effect is
visible on both sides, not just retrieval ranking.

Val side: raw per-pair gt_mces/pred_mces_step{N:06d}/pred_mces_bucket_step{N:06d}/
is_self_pair columns straight from val_pairs_val_consolidated.parquet (the
exact same data callbacks.py plotted live during training).

Test-to-test side: recomputed from scratch via
score_test_to_test_molecule_level_with_corn (this module), a thin wrapper
around the shared ood_generalization_check.score_test_to_test_molecule_level
that additionally runs the CORN bucket head pairwise on each pair's own
representative-spectrum embeddings (a small, cheap forward pass -- only
~3.7M pairs at this granularity, not the ~200M an exhaustive spectrum-level
computation would need). Requires re-embedding test spectra with
return_raw=True (the intermediates dir only ever saved the normalized
embeddings) since the bucket head needs raw, magnitude-sensitive embeddings.

Usage:
    uv run python tools/plot_val_vs_test_to_test_binned_box.py \\
        --val_parquet /path/to/val_pairs_val_consolidated.parquet \\
        --val_step 229000 \\
        --checkpoint /path/to/014_2/checkpoint.ckpt \\
        --mgf /path/to/gaetan_test.mgf \\
        --test_to_test_prepro_dir /path/to/preprocessing_gaetan_split_max_lb_hdf5_v2 \\
        --output_dir /path/to/output
"""

import argparse
import pickle
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ood_generalization_check import score_test_to_test_molecule_level
from simba_retrieval import embed_spectra, load_model, load_spectra
from simba_retrieval_iceberg import (
    CORN_BUCKET_EDGES,
    _corn_corrected_mces,
    _corn_decode_bucket,
)


_MCES_MAX = 40.0
_BIN_EDGES = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
_SELF_LABEL = "self (MCES=0)"


def _bin_labels() -> list[str]:
    labels = [_SELF_LABEL]
    lo = 0.0
    for hi in _BIN_EDGES:
        labels.append(f"({lo:g},{hi:g}]")
        lo = hi
    return labels


def plot_binned_box_on_ax(ax, gt_mces, pred_mces, is_self, title):
    """Exact port of callbacks.py's ValMetricsCallback._plot_binned_box,
    parameterized to draw onto a given ax instead of a standalone figure."""
    labels = _bin_labels()
    edges = _BIN_EDGES
    groups, positions, widths, ns = [], [], [], []

    self_vals = pred_mces[is_self]
    groups.append(self_vals)
    positions.append(0.0)
    widths.append(1.5)
    ns.append(len(self_vals))

    non_self_gt = gt_mces[~is_self]
    non_self_pred = pred_mces[~is_self]
    bin_idx = np.clip(np.digitize(non_self_gt, edges[:-1]), 0, len(edges) - 1)
    lo = 0.0
    for i, hi in enumerate(edges):
        vals = non_self_pred[bin_idx == i]
        groups.append(vals)
        positions.append((lo + hi) / 2.0)
        widths.append((hi - lo) * 0.8)
        ns.append(len(vals))
        lo = hi

    plot_groups = [g for g in groups if len(g) > 0]
    plot_positions = [p for p, g in zip(positions, groups) if len(g) > 0]
    plot_widths = [w for w, g in zip(widths, groups) if len(g) > 0]
    plot_labels = [lab for lab, g in zip(labels, groups) if len(g) > 0]
    if not plot_groups:
        ax.set_title(f"{title}\n(no pairs)")
        return
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
    for p, n in zip(plot_positions, [n for n in ns if n > 0]):
        ax.text(
            p, label_y, f"n={n:,}", ha="center", va="bottom", fontsize=7, rotation=90
        )
    ax.set_ylim(top=label_y * 1.25)
    ax.set_xticks(plot_positions)
    ax.set_xticklabels(plot_labels, rotation=30, ha="right")
    ax.set_xlabel("GT MCES")
    ax.set_ylabel("Predicted MCES")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def score_test_to_test_molecule_level_with_corn(
    gt_pairs: np.ndarray,
    idx_to_smiles: list[str],
    test_smiles_raw: list[str],
    test_embs_raw: torch.Tensor,
    model,
    device: torch.device,
    mces_max_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Thin wrapper around ood_generalization_check.score_test_to_test_molecule_level
    (the canonical, validation-matching molecule-level scoring -- see that
    function's docstring for the full protocol/rationale) that additionally
    computes the CORN-corrected score, which needs a genuine pairwise
    bucket-head forward pass on the raw (pre-normalize, magnitude-sensitive)
    embeddings at each pair's own representative spectrum positions -- not
    something the shared regression-only function does, since none of its
    other callers (mces_calibration_plots.py) need a bucket head at all.

    Returns (pred_raw, pred_corn, gt, is_self, same_spectrum) -- all 1-D,
    same length, self-pairs first then cross-pairs (order inherited from the
    shared function).
    """
    pred_raw, gt, is_self, same_spectrum, _mol_a, _mol_b, idx0, idx1 = (
        score_test_to_test_molecule_level(
            gt_pairs, idx_to_smiles, test_smiles_raw, test_embs_raw, mces_max_value
        )
    )
    print(f"  {len(gt):,} molecule-level pairs, {int(is_self.sum()):,} self-pairs")

    emb0_raw = test_embs_raw[idx0].to(device)
    emb1_raw = test_embs_raw[idx1].to(device)
    with torch.no_grad():
        _, _, emb_sim_3 = model.compute_from_embeddings(emb0_raw, emb1_raw)
    bucket = _corn_decode_bucket(emb_sim_3).cpu().numpy()
    pred_corn = _corn_corrected_mces(pred_raw, bucket)

    return pred_raw, pred_corn, gt, is_self, same_spectrum


def run(
    val_parquet: str,
    val_step: int,
    checkpoint: str,
    mgf: str,
    test_to_test_prepro_dir: str,
    output_dir: str,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading val pairs (raw regression + bucket columns) ...")
    pred_col = f"pred_mces_step{val_step:06d}"
    bucket_col = f"pred_mces_bucket_step{val_step:06d}"
    val_df = pd.read_parquet(
        val_parquet,
        columns=["gt_mces", pred_col, bucket_col, "is_self_pair", "same_spectrum"],
    )
    val_gt = val_df["gt_mces"].to_numpy()
    val_pred_raw = val_df[pred_col].to_numpy()
    val_bucket = val_df[bucket_col].to_numpy()
    val_pred_corn = _corn_corrected_mces(val_pred_raw, val_bucket)
    val_is_self = val_df["is_self_pair"].to_numpy().astype(bool)
    val_same_spectrum = val_df["same_spectrum"].to_numpy().astype(bool)
    print(f"  {len(val_gt):,} val pairs (step {val_step})")

    print("\nLoading + embedding test spectra (raw, for pairwise bucket head) ...")
    test_smiles_raw, test_spectra = load_spectra(mgf, "test", "theoretical")
    model = load_model(
        checkpoint,
        device,
        head_mode="cosine_no_head",
        use_mces_bucket_head=True,
        mces_bucket_use_mlp=True,
    )
    _, test_embs_raw = embed_spectra(
        model, test_spectra, batch_size=512, device=device, return_raw=True
    )

    print("\nLoading test-to-test mined molecule-level GT pairs ...")
    prepro = Path(test_to_test_prepro_dir)
    with open(prepro / "mapping.pkl", "rb") as fh:
        mapping = pickle.load(fh)
    idx_to_smiles = mapping["df_smiles_test"]["canon_smiles"].tolist()
    gt_pairs = np.load(
        prepro / "ed_mces_indexes_tani_incremental_test_node0_chunk0.npy"
    )
    print(
        f"  {len(gt_pairs):,} mined molecule-pairs (same i<j convention as val's own file)"
    )

    print(f"\nCORN bucket edges: {CORN_BUCKET_EDGES.tolist()}")
    print("Scoring test-to-test at molecule level (val's exact protocol) ...")
    tt_pred_raw, tt_pred_corn, tt_gt, tt_is_self, tt_same_spectrum = (
        score_test_to_test_molecule_level_with_corn(
            gt_pairs,
            idx_to_smiles,
            test_smiles_raw,
            test_embs_raw,
            model,
            device,
            mces_max_value=_MCES_MAX,
        )
    )

    print(
        "\nPlotting 2x2 grid (rows: raw regression / CORN-corrected; cols: val / test-to-test) ..."
    )
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plot_binned_box_on_ax(
        axes[0, 0],
        val_gt,
        val_pred_raw,
        val_is_self,
        f"[VAL, raw regression] step {val_step:,}",
    )
    plot_binned_box_on_ax(
        axes[0, 1],
        tt_gt,
        tt_pred_raw,
        tt_is_self,
        "[TEST-TO-TEST, raw regression]",
    )
    plot_binned_box_on_ax(
        axes[1, 0],
        val_gt,
        val_pred_corn,
        val_is_self,
        f"[VAL, CORN-corrected] step {val_step:,}",
    )
    plot_binned_box_on_ax(
        axes[1, 1],
        tt_gt,
        tt_pred_corn,
        tt_is_self,
        "[TEST-TO-TEST, CORN-corrected]",
    )
    fig.tight_layout()
    out_path = out_dir / "val_vs_test_to_test_binned_box_with_corn.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")

    print(
        "\n=== Self-pair (GT=0) predicted-MCES summary (all self-pairs, trivial+genuine mixed, same as val) ==="
    )
    for name, pred, is_self in [
        ("VAL raw", val_pred_raw, val_is_self),
        ("VAL corn", val_pred_corn, val_is_self),
        ("TEST-TO-TEST raw", tt_pred_raw, tt_is_self),
        ("TEST-TO-TEST corn", tt_pred_corn, tt_is_self),
    ]:
        vals = pred[is_self]
        print(
            f"  {name:20s} n={is_self.sum():,}  median={np.median(vals):.3f}  mean={vals.mean():.3f}"
        )

    print(
        "\n=== Trivial (same-spectrum) vs genuine (different-spectrum) self-pair mix, both sides ==="
    )
    for name, pred, is_self, same_spec in [
        ("VAL", val_pred_raw, val_is_self, val_same_spectrum),
        ("TEST-TO-TEST", tt_pred_raw, tt_is_self, tt_same_spectrum),
    ]:
        trivial = is_self & same_spec
        genuine = is_self & ~same_spec
        print(
            f"  {name}: {int(is_self.sum()):,} self-pairs total -- "
            f"{int(trivial.sum()):,} trivial ({trivial.sum() / max(is_self.sum(), 1) * 100:.1f}%, "
            f"median={np.median(pred[trivial]) if trivial.any() else float('nan'):.3f}), "
            f"{int(genuine.sum()):,} genuine ({genuine.sum() / max(is_self.sum(), 1) * 100:.1f}%, "
            f"median={np.median(pred[genuine]) if genuine.any() else float('nan'):.3f})"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--val_parquet", required=True)
    p.add_argument("--val_step", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mgf", required=True)
    p.add_argument("--test_to_test_prepro_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
