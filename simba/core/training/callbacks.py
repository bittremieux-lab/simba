import os

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.callbacks import Callback

from simba.utils.logger_setup import logger


class LossCallback(Callback):
    """Saves loss_plot.png to checkpoint dir after every validation run."""

    def __init__(self, file_path, n_val_sanity_checks=2, plot_every_n_steps: int = 100):
        self.file_path = file_path
        self.plot_every_n_steps = plot_every_n_steps

        # per-step train tracking
        self.train_steps: list[int] = []
        self.train_loss: list[float] = []
        self.train_loss_ed: list[float] = []
        self.train_loss_mces: list[float] = []
        self.train_sigma1: list[float] = []
        self.train_sigma2: list[float] = []

        # per-validation tracking (by global step)
        self.val_steps: list[int] = []
        self.val_loss: list[float] = []
        self.val_loss_ed: list[float] = []
        self.val_loss_mces: list[float] = []

    def _get(self, metrics, key):
        v = metrics.get(key)
        return float(v) if v is not None else None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.plot_every_n_steps != 0:
            return
        m = trainer.callback_metrics
        loss = self._get(m, "train_loss")
        if loss is None:
            return
        self.train_steps.append(trainer.global_step)
        self.train_loss.append(loss)
        self.train_loss_ed.append(self._get(m, "loss_ed"))
        self.train_loss_mces.append(self._get(m, "loss_mces"))
        self.train_sigma1.append(self._get(m, "log_sigma1"))
        self.train_sigma2.append(self._get(m, "log_sigma2"))

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        loss = self._get(m, "validation_loss_epoch")
        if loss is None:
            return
        self.val_steps.append(trainer.global_step)
        self.val_loss.append(loss)
        self.val_loss_ed.append(self._get(m, "loss_ed_epoch"))
        self.val_loss_mces.append(self._get(m, "loss_mces_epoch"))
        self.plot_loss()

    def plot_loss(self):
        has_components = any(v is not None for v in self.train_loss_ed)
        has_sigma = any(v is not None for v in self.train_sigma1)

        n_cols = 1 + (2 if has_components else 0) + (1 if has_sigma else 0)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        col = 0

        def _plot_series(ax, steps, values, nones_ok=True, **kw):
            s = [x for x, v in zip(steps, values) if v is not None]
            v = [v for v in values if v is not None]
            if s:
                ax.plot(s, v, **kw)

        # Total loss
        ax = axes[col]
        col += 1
        _plot_series(
            ax,
            self.train_steps,
            self.train_loss,
            lw=1.0,
            color="steelblue",
            label="train",
        )
        if self.val_steps:
            _plot_series(
                ax,
                self.val_steps,
                self.val_loss,
                marker="o",
                ms=6,
                lw=1.5,
                color="darkorange",
                label="val",
            )
        ax.set_title("Total loss")
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if has_components:
            # ED loss
            ax = axes[col]
            col += 1
            _plot_series(
                ax,
                self.train_steps,
                self.train_loss_ed,
                lw=1.0,
                color="steelblue",
                label="train",
            )
            _plot_series(
                ax,
                self.val_steps,
                self.val_loss_ed,
                marker="o",
                ms=6,
                lw=1.5,
                color="darkorange",
                label="val",
            )
            ax.set_title("Edit distance loss (CE)")
            ax.set_xlabel("step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # MCES loss
            ax = axes[col]
            col += 1
            _plot_series(
                ax,
                self.train_steps,
                self.train_loss_mces,
                lw=1.0,
                color="steelblue",
                label="train",
            )
            _plot_series(
                ax,
                self.val_steps,
                self.val_loss_mces,
                marker="o",
                ms=6,
                lw=1.5,
                color="darkorange",
                label="val",
            )
            ax.set_title("MCES similarity loss (MSE)")
            ax.set_xlabel("step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        if has_sigma:
            ax = axes[col]
            col += 1
            _plot_series(
                ax,
                self.train_steps,
                self.train_sigma1,
                lw=1.0,
                color="royalblue",
                label="log σ1 (ED)",
            )
            _plot_series(
                ax,
                self.train_steps,
                self.train_sigma2,
                lw=1.0,
                color="tomato",
                label="log σ2 (MCES)",
            )
            ax.set_title("Learnable uncertainty weights")
            ax.set_xlabel("step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="gray", lw=0.5, ls="--")

        last_step = self.train_steps[-1] if self.train_steps else 0
        fig.suptitle(
            f"simba training — step {last_step} "
            f"({len(self.train_steps)} train points, {len(self.val_steps)} val points)",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(self.file_path, dpi=130)
        plt.close(fig)


class ProgressLogCallback(Callback):
    """Logs training progress to the Python logger (visible in SLURM .err file)."""

    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps == 0:
            m = trainer.callback_metrics
            loss = m.get("train_loss")
            loss_ed = m.get("loss_ed")
            loss_mces = m.get("loss_mces")
            log_s1 = m.get("log_sigma1")
            log_s2 = m.get("log_sigma2")
            if loss is not None:
                msg = (
                    f"[TRAIN] epoch={trainer.current_epoch} step={trainer.global_step} "
                    f"train_loss={float(loss):.4f}"
                )
                if loss_ed is not None:
                    msg += f" loss_ed={float(loss_ed):.4f}"
                if loss_mces is not None:
                    msg += f" loss_mces={float(loss_mces):.6f}"
                if log_s1 is not None:
                    msg += f" σ1={float(log_s1):.3f}"
                if log_s2 is not None:
                    msg += f" σ2={float(log_s2):.3f}"
                logger.info(msg)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        val_loss = m.get("validation_loss_epoch")
        train_loss = m.get("train_loss_epoch")
        val_ed = m.get("loss_ed_epoch")
        val_mces = m.get("loss_mces_epoch")
        msg = f"[VAL]   epoch={trainer.current_epoch} step={trainer.global_step}"
        if val_loss is not None:
            msg += f" val_loss={float(val_loss):.4f}"
        if train_loss is not None:
            msg += f" train_loss={float(train_loss):.4f}"
        if val_ed is not None:
            msg += f" val_loss_ed={float(val_ed):.4f}"
        if val_mces is not None:
            msg += f" val_loss_mces={float(val_mces):.6f}"
        logger.info(msg)


class ValMetricsCallback(Callback):
    """
    After each validation epoch: binned MAE and predicted-distribution
    overlap for the MCES regression head (GT bin 0 = self-pairs, then
    bin_edges), a boxplot of predicted MCES per GT bin, a Hit@k retrieval
    benchmark (see _hit_at_k), and -- when the mces_bucket CORN head is
    enabled -- its balanced accuracy and confusion matrix, plus a
    CORN-corrected variant of the MAE/overlap/Hit@k metrics (see
    _corn_corrected_mces). All plots are saved to output_dir and, when a
    TensorBoardLogger is attached, also logged as figures.
    """

    _SELF_LABEL = "self (MCES=0)"

    def __init__(
        self,
        output_dir: str,
        mae_bin_edges,
        mces_max_value: float = 40.0,
        max_skip: int = 2,
        hit_at_k_n_decoys: int = 255,
        hit_at_k_ks=(1, 5, 20),
    ):
        self.output_dir = output_dir
        self.bin_edges = np.array(mae_bin_edges, dtype=float)
        self.mces_max_value = mces_max_value
        self.max_skip = max_skip
        self.hit_at_k_n_decoys = hit_at_k_n_decoys
        self.hit_at_k_ks = tuple(hit_at_k_ks)
        self._mces_preds = []
        self._mces_targets = []
        self._bucket_preds = []
        self._bucket_targets = []
        self._mol_idx_0 = []
        self._mol_idx_1 = []
        self._spec_idx_0 = []
        self._spec_idx_1 = []

    def _bin_labels(self) -> list[str]:
        return _bin_labels(self.bin_edges)

    def _bin_index(self, gt_mces: np.ndarray, is_self: np.ndarray) -> np.ndarray:
        return _bin_index(gt_mces, is_self, self.bin_edges)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.sanity_checking or not isinstance(outputs, dict):
            return
        self._mces_preds.append(outputs["mces_pred"].float().numpy())
        self._mces_targets.append(outputs["mces_target"].float().numpy())
        if "mces_bucket_pred" in outputs:
            self._bucket_preds.append(outputs["mces_bucket_pred"].numpy())
            self._bucket_targets.append(outputs["mces_bucket_target"].numpy())
        if "mol_idx_0" in batch:
            self._mol_idx_0.append(batch["mol_idx_0"].cpu().numpy())
            self._mol_idx_1.append(batch["mol_idx_1"].cpu().numpy())
            self._spec_idx_0.append(batch["spec_idx_0"].cpu().numpy())
            self._spec_idx_1.append(batch["spec_idx_1"].cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._mces_preds:
            return

        mces_pred_sim = np.concatenate(self._mces_preds)
        mces_target_sim = np.concatenate(self._mces_targets)
        self._mces_preds.clear()
        self._mces_targets.clear()

        pred_mces = (1.0 - mces_pred_sim) * self.mces_max_value
        gt_mces = (1.0 - mces_target_sim) * self.mces_max_value
        abs_err = np.abs(pred_mces - gt_mces)
        is_self = gt_mces <= 1e-6
        bin_idx = self._bin_index(gt_mces, is_self)
        step = trainer.global_step
        tb_logger = self._tb_logger(trainer)

        self._log_binned_mae(pl_module, abs_err, bin_idx, prefix="val_mae_mces")
        self._log_overlap_coefficients(
            pl_module, pred_mces, bin_idx, prefix="val_overlap"
        )
        self._plot_binned_box(pred_mces, bin_idx, step, tb_logger)

        corrected_mces = None
        if self._bucket_preds:
            bucket_pred = np.concatenate(self._bucket_preds)
            bucket_target = np.concatenate(self._bucket_targets)
            self._bucket_preds.clear()
            self._bucket_targets.clear()
            self._log_and_plot_bucket_confusion(
                pl_module, bucket_pred, bucket_target, step, tb_logger
            )

            edges = pl_module.mces_bucket_bin_edges.cpu().numpy()
            corrected_mces = _corn_corrected_mces(pred_mces, bucket_pred, edges)
            corrected_abs_err = np.abs(corrected_mces - gt_mces)
            self._log_binned_mae(
                pl_module, corrected_abs_err, bin_idx, prefix="val_mae_mces_corrected"
            )
            self._log_overlap_coefficients(
                pl_module, corrected_mces, bin_idx, prefix="val_overlap_corrected"
            )

        if self._mol_idx_0:
            mol_idx_0 = np.concatenate(self._mol_idx_0)
            mol_idx_1 = np.concatenate(self._mol_idx_1)
            spec_idx_0 = np.concatenate(self._spec_idx_0)
            spec_idx_1 = np.concatenate(self._spec_idx_1)
            self._mol_idx_0.clear()
            self._mol_idx_1.clear()
            self._spec_idx_0.clear()
            self._spec_idx_1.clear()
            self._log_hit_at_k(
                pl_module,
                mol_idx_0,
                mol_idx_1,
                spec_idx_0,
                spec_idx_1,
                gt_mces,
                pred_mces,
                corrected_mces,
            )

    @staticmethod
    def _tb_logger(trainer):
        from lightning.pytorch.loggers import TensorBoardLogger

        for lg in trainer.loggers:
            if isinstance(lg, TensorBoardLogger):
                return lg
        return None

    def _log_binned_mae(self, pl_module, abs_err, bin_idx, prefix):
        bucket_maes = []
        for li, label in enumerate(self._bin_labels()):
            mask = bin_idx == li
            if not mask.any():
                continue
            mae = float(abs_err[mask].mean())
            pl_module.log(
                f"{prefix}/{label}",
                mae,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            bucket_maes.append(mae)
        if bucket_maes:
            pl_module.log(
                f"{prefix}_bucket_avg",
                float(np.mean(bucket_maes)),
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )

    def _log_overlap_coefficients(self, pl_module, pred_mces, bin_idx, prefix):
        labels = self._bin_labels()
        pairwise, skip_avg = _overlap_metrics(pred_mces, bin_idx, labels, self.max_skip)
        for suffix, ovl in pairwise.items():
            pl_module.log(
                f"{prefix}/{suffix}",
                ovl,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        for skip, avg in skip_avg.items():
            pl_module.log(
                f"{prefix}_avg/skip{skip}",
                avg,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )

    def _plot_binned_box(self, pred_mces, bin_idx, step, tb_logger):
        labels = self._bin_labels()
        edges = self.bin_edges
        data, positions, widths, ns = [], [], [], []
        for li in range(len(labels)):
            mask = bin_idx == li
            n = int(mask.sum())
            if n == 0:
                continue
            data.append(pred_mces[mask])
            ns.append(n)
            if li == 0:
                positions.append(0.0)
                widths.append(2.0)
            else:
                lo = 0.0 if li == 1 else edges[li - 2]
                hi = edges[li - 1]
                positions.append((lo + hi) / 2)
                widths.append((hi - lo) * 0.8)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(
            data, positions=positions, widths=widths, whis=(5, 95), showfliers=False
        )
        ax.plot(
            [0, float(edges[-1])], [0, float(edges[-1])], "r--", lw=1, label="pred = GT"
        )
        ylim = ax.get_ylim()
        for pos, n in zip(positions, ns):
            ax.text(pos, ylim[1] * 0.98, f"n={n}", ha="center", va="top", fontsize=7)
        ax.set_xlabel("GT MCES bin")
        ax.set_ylabel("Predicted MCES")
        ax.set_title(f"Predicted MCES per GT bin — step {step}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"mces_binned_box_step{step:06d}.png")
        fig.savefig(path, dpi=130)
        if tb_logger is not None:
            tb_logger.experiment.add_figure(
                "val_plots/mces_binned_box", fig, global_step=step
            )
        plt.close(fig)

    def _log_and_plot_bucket_confusion(self, pl_module, pred, target, step, tb_logger):
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix

        edges = pl_module.mces_bucket_bin_edges.cpu().numpy()
        labels = ["0"]
        prev = 0.0
        for edge in edges:
            labels.append(f"({prev:g},{edge:g}]")
            prev = edge
        labels.append(f"({prev:g},inf)")
        n = len(labels)

        bal_acc = balanced_accuracy_score(target, pred)
        pl_module.log(
            "val_mces_bucket_balanced_acc",
            float(bal_acc),
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

        cm = confusion_matrix(target, pred, labels=list(range(n)))
        cm_row_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, mat, title, fmt in [
            (axes[0], cm, "counts", "{:,}"),
            (axes[1], cm_row_pct, "row %", "{:.1f}%"),
        ]:
            im = ax.imshow(mat, cmap="Blues")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Predicted bucket")
            ax.set_ylabel("True bucket")
            ax.set_title(
                f"mces_bucket confusion — {title} (step {step}, bal_acc={bal_acc:.3f})"
            )
            vmax = mat.max() if mat.max() > 0 else 1
            for i in range(n):
                for j in range(n):
                    ax.text(
                        j,
                        i,
                        fmt.format(mat[i, j]),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if mat[i, j] > 0.5 * vmax else "black",
                    )
        plt.tight_layout()
        path = os.path.join(
            self.output_dir, f"mces_bucket_confusion_step{step:06d}.png"
        )
        fig.savefig(path, dpi=130)
        if tb_logger is not None:
            tb_logger.experiment.add_figure(
                "val_plots/mces_bucket_confusion", fig, global_step=step
            )
        plt.close(fig)

    def _log_hit_at_k(
        self,
        pl_module,
        mol_idx_0,
        mol_idx_1,
        spec_idx_0,
        spec_idx_1,
        gt_mces,
        pred_mces,
        corrected_mces,
    ):
        is_self = mol_idx_0 == mol_idx_1
        pool_mols, query_mols = _build_pool_and_queries(
            mol_idx_0, spec_idx_0, spec_idx_1, is_self
        )
        if len(pool_mols) == 0 or len(query_mols) == 0:
            return
        local_of = _local_index_lookup(pool_mols)
        gt_matrix = _build_score_matrix(
            mol_idx_0, mol_idx_1, gt_mces, pool_mols, local_of
        )

        self._log_hit_at_k_for_score(
            pl_module,
            gt_matrix,
            local_of,
            query_mols,
            mol_idx_0,
            mol_idx_1,
            spec_idx_0,
            spec_idx_1,
            is_self,
            pred_mces,
            pool_mols,
            prefix="val_hit_at",
        )
        if corrected_mces is not None:
            ranking_score = _corn_corrected_ranking_score(corrected_mces, pred_mces)
            self._log_hit_at_k_for_score(
                pl_module,
                gt_matrix,
                local_of,
                query_mols,
                mol_idx_0,
                mol_idx_1,
                spec_idx_0,
                spec_idx_1,
                is_self,
                ranking_score,
                pool_mols,
                prefix="val_hit_at_corrected",
            )

    def _log_hit_at_k_for_score(
        self,
        pl_module,
        gt_matrix,
        local_of,
        query_mols,
        mol_idx_0,
        mol_idx_1,
        spec_idx_0,
        spec_idx_1,
        is_self,
        score,
        pool_mols,
        prefix,
    ):
        score_matrix = _build_score_matrix(
            mol_idx_0, mol_idx_1, score, pool_mols, local_of
        )
        true_scores = _true_match_scores(
            mol_idx_0, spec_idx_0, spec_idx_1, is_self, score
        )
        hits = _hit_at_k(
            gt_matrix,
            score_matrix,
            true_scores,
            local_of,
            query_mols,
            self.hit_at_k_n_decoys,
            self.hit_at_k_ks,
            higher_is_better=False,
        )
        for k, v in hits.items():
            pl_module.log(
                f"{prefix}_{k}",
                v,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )


def _bin_labels(bin_edges, self_label: str = "self (MCES=0)") -> list[str]:
    labels = [self_label]
    prev = 0.0
    for edge in bin_edges:
        labels.append(f"({prev:g},{edge:g}]")
        prev = edge
    return labels


def _bin_index(gt_mces: np.ndarray, is_self: np.ndarray, bin_edges) -> np.ndarray:
    idx = np.zeros(len(gt_mces), dtype=int)
    non_self = ~is_self
    idx[non_self] = (
        np.clip(np.digitize(gt_mces[non_self], bin_edges[:-1]), 0, len(bin_edges) - 1)
        + 1
    )
    return idx


def _overlap_coefficient(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
    """Histogram overlapping coefficient: 0 = fully separated, 1 = identical."""
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


def _overlap_metrics(
    values: np.ndarray, bin_idx: np.ndarray, labels: list[str], max_skip: int
) -> tuple[dict, dict]:
    """(pairwise, skip_avg): pairwise[f"{label_i}_vs_{label_j}_skip{k}"] is
    the overlap coefficient between `values` in bin i and bin j, for every
    bin pair at skip distance k (0..max_skip); skip_avg[k] is the mean over
    all bin pairs at that skip. Shared by the live per-epoch callback and
    the standalone cosine-baseline script."""
    n_bins = len(labels)
    by_bin = {i: values[bin_idx == i] for i in range(n_bins) if (bin_idx == i).any()}
    pairwise, skip_avg = {}, {}
    for skip in range(max_skip + 1):
        skip_values = []
        for i in range(n_bins - skip - 1):
            j = i + skip + 1
            if i not in by_bin or j not in by_bin:
                continue
            ovl = _overlap_coefficient(by_bin[i], by_bin[j])
            pairwise[f"{labels[i]}_vs_{labels[j]}_skip{skip}"] = ovl
            skip_values.append(ovl)
        if skip_values:
            skip_avg[skip] = float(np.mean(skip_values))
    return pairwise, skip_avg


def _corn_corrected_mces(
    pred_mces: np.ndarray, bucket_pred: np.ndarray, bin_edges
) -> np.ndarray:
    """Combine the CORN bucket prediction with the continuous MCES
    prediction: the bucket's own [left, right] range clips the continuous
    value into it (bucket 0 clips to exactly 0)."""
    extended = np.concatenate([[0.0], bin_edges, [np.inf]])
    b = np.clip(bucket_pred.astype(np.int64), 0, len(extended) - 1)
    is_zero = b == 0
    left = np.where(is_zero, 0.0, extended[np.clip(b - 1, 0, None)])
    right = np.where(is_zero, 0.0, extended[b])
    return np.clip(pred_mces, left, right)


def _corn_corrected_ranking_score(
    corrected_mces: np.ndarray, pred_mces: np.ndarray
) -> np.ndarray:
    """corrected*1000 + raw breaks exact ties within a bucket (e.g. bucket 0,
    where every candidate corrects to exactly 0) for Hit@k ranking, while
    leaving the corrected value itself untouched for other uses (MAE/overlap)."""
    return corrected_mces * 1000.0 + pred_mces


def _build_pool_and_queries(mol_idx_0, spec_idx_0, spec_idx_1, is_self):
    """(pool_mols, query_mols): every molecule with a self-pair row, and the
    subset whose self-pair used two different spectra -- usable as
    retrieval queries."""
    pool_mols = np.unique(mol_idx_0[is_self])
    diff_spec = is_self & (spec_idx_0 != spec_idx_1)
    query_mols = np.unique(mol_idx_0[diff_spec])
    return pool_mols, query_mols


def _local_index_lookup(pool_mols: np.ndarray) -> np.ndarray:
    """Array mapping a molecule id -> its dense index in pool_mols (-1 if
    absent), for vectorized lookups over millions of pairs."""
    local_of = np.full(int(pool_mols.max()) + 1, -1, dtype=np.int64)
    local_of[pool_mols] = np.arange(len(pool_mols))
    return local_of


def _build_score_matrix(
    mol_idx_0, mol_idx_1, values, pool_mols, local_of
) -> np.ndarray:
    """Dense (n_pool, n_pool) matrix of `values` for every cross-molecule
    pair within the pool, symmetric; NaN where absent."""
    n = len(pool_mols)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    max_id = local_of.shape[0] - 1
    cross = (mol_idx_0 != mol_idx_1) & (mol_idx_0 <= max_id) & (mol_idx_1 <= max_id)
    i = local_of[mol_idx_0[cross]]
    j = local_of[mol_idx_1[cross]]
    in_pool = (i >= 0) & (j >= 0)
    i, j = i[in_pool], j[in_pool]
    v = values[cross][in_pool]
    mat[i, j] = v
    mat[j, i] = v
    return mat


def _true_match_scores(mol_idx_0, spec_idx_0, spec_idx_1, is_self, values) -> dict:
    """query_mol -> its own "self, different spectrum" row's value (the
    true match: the same molecule's other spectrum)."""
    diff_spec = is_self & (spec_idx_0 != spec_idx_1)
    return dict(zip(mol_idx_0[diff_spec], values[diff_spec]))


def _hit_at_k(
    gt_matrix,
    score_matrix,
    true_scores,
    local_of,
    query_mols,
    n_decoys,
    ks,
    higher_is_better,
) -> dict:
    """Fraction of query molecules whose true match (its own other spectrum)
    ranks in the top k among n_decoys other pool molecules -- the n_decoys
    with the lowest ground-truth MCES to the query -- plus the true match
    itself, ranked by `score` (ascending unless higher_is_better)."""
    hits = dict.fromkeys(ks, 0)
    n_scored = 0
    for q in query_mols:
        qi = local_of[q]
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


class IcebergHitRateCallback(Callback):
    """Every check_every_n_val_checks-th validation check, ranks the model
    against the Gaetan test fold's real spectra vs ICEBERG-predicted
    formula-matched candidates and logs Hit@1/5/20 (raw embedding cosine,
    and CORN-corrected when the model has an mces_bucket head). Test/
    candidate data (spectra, candidate index, ICEBERG predictions) doesn't
    depend on model weights, so it's loaded once at construction; only
    re-embedding and re-ranking happen on the model's current weights each
    check. See simba.core.training.iceberg_retrieval for the underlying
    data loading, embedding, and ranking logic."""

    def __init__(
        self,
        mgf: str,
        candidates: str,
        candidate_tsv,
        iceberg_preds,
        batch_size: int = 512,
        ks=(1, 5, 20),
        check_every_n_val_checks: int = 1,
    ):
        from simba.core.training.iceberg_retrieval import load_all_iceberg_data

        self.data = load_all_iceberg_data(mgf, candidates, candidate_tsv, iceberg_preds)
        self.batch_size = batch_size
        self.ks = ks
        self.check_every_n_val_checks = max(1, check_every_n_val_checks)
        self._val_check_count = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._val_check_count += 1
        if self._val_check_count % self.check_every_n_val_checks != 0:
            return
        from simba.core.training.iceberg_retrieval import compute_iceberg_hit_rates

        raw_hits, corrected_hits = compute_iceberg_hit_rates(
            pl_module,
            pl_module.device,
            self.data,
            batch_size=self.batch_size,
            ks=self.ks,
        )
        for k, v in raw_hits.items():
            pl_module.log(
                f"iceberg_hit_at_{k}",
                v,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        if corrected_hits is not None:
            for k, v in corrected_hits.items():
                pl_module.log(
                    f"iceberg_hit_at_corrected_{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )
