"""
Run inference on scaffold + official val sets from a trained SIMBA checkpoint.
Saves per-set CSVs (mces_pred, mces_target) and hexbin plots with marginal histograms.

Usage (via Hydra overrides + explicit args):
    uv run python tools/run_val_hexbin.py \
        --checkpoint /path/to/best_model.ckpt \
        --output_dir /path/to/output \
        [hydra overrides for model / paths]
"""

import argparse
import os
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from scipy.stats import spearmanr
from tqdm import tqdm


def plot_hexbin(gt_mces, pred_mces, val_name, step, output_dir, scale="linear"):
    """
    Hexbin with marginal histograms.
    Both axes show raw MCES [0, 40]. step=2.5 bins.
    """
    r, _ = spearmanr(gt_mces, pred_mces)
    mse = float(np.mean((pred_mces - gt_mces) ** 2))

    bins = np.arange(0, 42.5, 2.5)
    color = "#4E9A7A"

    fig = plt.figure(figsize=(6.5, 6.5))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.04,
        wspace=0.04,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    fig.add_subplot(gs[0, 1]).set_visible(False)  # empty corner

    # Main hexbin
    bins_arg = "log" if scale == "log" else None
    ax_main.hexbin(
        gt_mces,
        pred_mces,
        gridsize=16,
        cmap="Greens",
        mincnt=1,
        bins=bins_arg,
        extent=[0, 40, 0, 40],
    )
    ax_main.plot([0, 40], [0, 40], "r--", lw=1)
    ax_main.set_xlabel("Ground truth MCES", fontsize=11)
    ax_main.set_ylabel("Predicted MCES", fontsize=11)
    ax_main.set_xlim(0, 40)
    ax_main.set_ylim(0, 40)
    ax_main.grid(True, alpha=0.2)

    # Top marginal (GT distribution)
    ax_top.hist(gt_mces, bins=bins, color=color, edgecolor="none")
    ax_top.set_xlim(0, 40)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_ylabel("count", fontsize=8)
    ax_top.grid(True, alpha=0.2)
    ax_top.tick_params(axis="both", labelsize=7)

    # Right marginal (Pred distribution)
    ax_right.hist(
        pred_mces, bins=bins, orientation="horizontal", color=color, edgecolor="none"
    )
    ax_right.set_ylim(0, 40)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.set_xlabel("count", fontsize=8)
    ax_right.grid(True, alpha=0.2)
    ax_right.tick_params(axis="both", labelsize=7)

    scale_label = "Log scale" if scale == "log" else "Linear scale"
    fig.suptitle(
        f"MCES hexbin [{val_name}] — step {step:,}   "
        f"ρ={r:.3f}  MSE={mse:.3f}   [{scale_label}]",
        fontsize=10,
    )

    fname = f"mces_hexbin_{val_name}_step{step:06d}_{scale}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def spectral_cosine_batch(
    mz0, mz1, int0, int1, bin_size: float = 1.0, n_bins: int = 2000
) -> np.ndarray:
    """
    Peak-based spectral cosine similarity for a batch of spectrum pairs.
    mz0/mz1/int0/int1: (B, max_peaks) float32 tensors (intensities already sqrt+L2-norm by augmentation).
    Returns cosine similarity per pair (B,) as float32 numpy array.
    """
    B = mz0.shape[0]
    # Bin peaks by m/z; padding peaks have mz=0 and int=0 so they're harmless
    bins0 = (mz0 / bin_size).long().clamp(0, n_bins - 1)
    bins1 = (mz1 / bin_size).long().clamp(0, n_bins - 1)
    v0 = torch.zeros(B, n_bins, dtype=torch.float32)
    v1 = torch.zeros(B, n_bins, dtype=torch.float32)
    v0.scatter_add_(1, bins0, int0)
    v1.scatter_add_(1, bins1, int1)
    # Re-L2-normalize after binning (multiple peaks may land in same bin)
    v0 = torch.nn.functional.normalize(v0, p=2, dim=1)
    v1 = torch.nn.functional.normalize(v1, p=2, dim=1)
    return (v0 * v1).sum(dim=1).numpy()


def run_inference_on_loader(model, dataloader, device):
    """Iterate dataloader, return (pred_similarity, gt_similarity, spectral_cosine) arrays."""
    model.eval()
    all_pred, all_gt, all_spec_cos = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  batches", leave=False):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits_list = model(batch)
            logits2 = logits_list[1].view(-1)  # cosine sim prediction
            target2 = batch["mces"].float().view(-1)  # MCES similarity GT
            all_pred.append(logits2.cpu().numpy())
            all_gt.append(target2.cpu().numpy())
            # Spectral cosine on CPU (batch tensors already on device; move peaks to CPU)
            mz0 = batch["mz_0"].cpu()
            mz1 = batch["mz_1"].cpu()
            int0 = batch["intensity_0"].cpu()
            int1 = batch["intensity_1"].cpu()
            all_spec_cos.append(spectral_cosine_batch(mz0, mz1, int0, int1))
    return (
        np.concatenate(all_pred),
        np.concatenate(all_gt),
        np.concatenate(all_spec_cos),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prepro_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=3072)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Extra Hydra overrides forwarded as positional trailing args
    args, overrides = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load Hydra config ──────────────────────────────────────────────────────
    from simba.utils.config_utils import get_config_path

    config_path = get_config_path()
    base_overrides = [
        f"paths.preprocessing_dir={args.prepro_dir}",
        f"paths.preprocessing_dir_train={args.prepro_dir}",
        "paths.preprocessing_pickle_file=mapping.pkl",
        f"paths.checkpoint_dir={args.output_dir}",
    ] + overrides

    with initialize_config_dir(
        config_dir=str(config_path.absolute()), version_base=None
    ):
        cfg = compose(config_name="config", overrides=base_overrides)

    # ── Load dataset ───────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader

    from simba.core.data.datasets.multitask_dataset_builder import MultitaskDataBuilder
    from simba.workflows.training import load_dataset, prepare_data

    print("Loading dataset...")
    mol_train, mol_val, mol_val_official, mol_test, _ = load_dataset(cfg)

    print("Preparing pair data...")
    (
        _ds_train,
        _ts,
        ds_val,
        _vs,
        ds_val_official,
        _vos,
        _weights_ed,
        _bins_ed,
    ) = prepare_data(
        mol_train,
        mol_val,
        mol_test,
        None,
        cfg,
        molecule_pairs_val_official=mol_val_official,
    )

    # Build test dataset — same logic as prepare_data does for val_official
    ds_test = None
    if mol_test is not None:
        import numpy as np

        from simba.core.chemistry.mces_loader.load_mces import LoadMCES

        print("Loading test pairs from npy files...")
        test_arr = LoadMCES.merge_numpy_arrays(
            args.prepro_dir,
            prefix="ed_mces_indexes_tani_incremental_test",
            use_edit_distance=cfg.model.tasks.edit_distance.enabled,
            use_multitask=cfg.model.multitasking.enabled,
            add_high_similarity_pairs=False,
        )
        mces_col = cfg.model.data_columns.mces20
        ed_col = cfg.model.data_columns.edit_distance
        # Apply mol_idx_remap if spectra were remapped during load
        remap = getattr(mol_test, "_mol_idx_remap", None)
        if remap is not None:
            col0, col1 = test_arr[:, 0].astype(int), test_arr[:, 1].astype(int)
            mask = np.array([c in remap and d in remap for c, d in zip(col0, col1)])
            test_arr = test_arr[mask].copy()
            test_arr[:, 0] = [remap[int(x)] for x in test_arr[:, 0]]
            test_arr[:, 1] = [remap[int(x)] for x in test_arr[:, 1]]
        mol_test.pair_distances = test_arr[:, [0, 1, ed_col]]
        mol_test.extra_distances = test_arr[:, mces_col]
        print(f"  Test pairs: {len(mol_test):,}")
        ds_test = MultitaskDataBuilder.from_molecule_pairs_to_dataset(
            mol_test,
            max_num_peaks=int(cfg.model.transformer.context_length),
            use_adduct=cfg.model.features.use_adduct,
            use_ce=cfg.model.features.use_ce,
            use_ion_activation=cfg.model.features.use_ion_activation,
            use_ion_method=cfg.model.features.use_ion_method,
            use_ion_mode=cfg.model.features.use_ion_mode,
        )

    # Unweighted loaders — iterate ALL pairs
    def make_loader(dataset):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )

    loaders = {"scaffold": make_loader(ds_val)}
    if ds_val_official is not None:
        loaders["official"] = make_loader(ds_val_official)
    if ds_test is not None:
        loaders["test"] = make_loader(ds_test)

    # ── Load model ─────────────────────────────────────────────────────────────

    print("Loading model checkpoint...")
    model_kwargs = {
        "d_model": cfg.model.transformer.d_model,
        "n_layers": cfg.model.transformer.n_layers,
        "n_classes": cfg.model.tasks.edit_distance.n_classes,
        "use_gumbel": cfg.model.tasks.edit_distance.use_gumbel,
        "use_element_wise": cfg.model.features.use_element_wise,
        "use_cosine_distance": cfg.model.tasks.cosine_similarity.use_cosine_distance,
        "use_edit_distance_regresion": cfg.model.tasks.edit_distance.use_regression,
        "use_fingerprints": cfg.model.tasks.fingerprints.enabled,
        "USE_LEARNABLE_MULTITASK": cfg.model.multitasking.learnable,
        "use_mces20_log_loss": cfg.model.tasks.mces.use_log_loss,
        "tau_gumbel_softmax": cfg.model.tasks.edit_distance.tau_gumbel_softmax,
        "gumbel_reg_weight": cfg.model.tasks.edit_distance.gumbel_reg_weight,
        "weights": None,
        "lr": cfg.optimizer.lr,
        "use_adduct": cfg.model.features.use_adduct,
        "use_precursor_mz_for_model": cfg.model.features.use_precursor_mz,
        "use_ce": cfg.model.features.use_ce,
        "use_ion_activation": cfg.model.features.use_ion_activation,
        "use_ion_method": cfg.model.features.use_ion_method,
        "use_ion_mode": cfg.model.features.use_ion_mode,
        "use_edit_distance": cfg.model.tasks.edit_distance.enabled,
    }
    from simba.core.models.similarity_models import SimilarityModelMultitask

    model = SimilarityModelMultitask.load_from_checkpoint(
        args.checkpoint,
        **model_kwargs,
        strict=False,
        map_location=args.device,
    )
    model = model.to(args.device)
    model.eval()

    # Get step from checkpoint filename
    ckpt_name = Path(args.checkpoint).stem
    step = 0
    for part in ckpt_name.replace("-", "=").split("="):
        if part.isdigit():
            step = int(part)

    # ── Run inference + save ───────────────────────────────────────────────────
    for val_name, loader in loaders.items():
        print(f"\nRunning inference: {val_name} ({len(loader.dataset):,} pairs)...")
        pred_sim, gt_sim, spec_cos = run_inference_on_loader(model, loader, args.device)

        # Convert similarity → raw MCES
        pred_mces = (1.0 - pred_sim) * 40.0
        gt_mces = (1.0 - gt_sim) * 40.0

        # Save CSV
        df = pd.DataFrame(
            {
                "mces_pred": pred_sim,
                "mces_target": gt_sim,
                "mces_pred_raw": pred_mces,
                "mces_target_raw": gt_mces,
                "cosine_spectral": spec_cos,
            }
        )
        csv_path = os.path.join(args.output_dir, f"val_predictions_{val_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}  ({len(df):,} rows)")

        r, _ = spearmanr(gt_sim, pred_sim)
        mse = float(np.mean((pred_sim - gt_sim) ** 2))
        print(f"  Spearman ρ={r:.4f}  MSE={mse:.6f}")

        for scale in ["linear", "log"]:
            plot_hexbin(gt_mces, pred_mces, val_name, step, args.output_dir, scale)

    print("\nDone.")


if __name__ == "__main__":
    main()
