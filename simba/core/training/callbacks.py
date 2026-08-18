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

        # per-validation tracking (by global step) — scaffold val (dataloader 0)
        self.val_steps: list[int] = []
        self.val_loss: list[float] = []
        self.val_loss_ed: list[float] = []
        self.val_loss_mces: list[float] = []
        # official val (dataloader 1, if present)
        self.val_steps_official: list[int] = []
        self.val_loss_official: list[float] = []
        self.val_loss_mces_official: list[float] = []

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
        # Support both single-loader (plain key) and multi-loader (/dataloader_idx_N) cases
        loss_0 = self._get(m, "validation_loss_epoch/dataloader_idx_0")
        if loss_0 is not None:
            self.val_steps.append(trainer.global_step)
            self.val_loss.append(loss_0)
            self.val_loss_ed.append(self._get(m, "loss_ed_epoch/dataloader_idx_0"))
            self.val_loss_mces.append(self._get(m, "loss_mces_epoch/dataloader_idx_0"))
            loss_1 = self._get(m, "validation_loss_epoch/dataloader_idx_1")
            if loss_1 is not None:
                self.val_steps_official.append(trainer.global_step)
                self.val_loss_official.append(loss_1)
                self.val_loss_mces_official.append(
                    self._get(m, "loss_mces_epoch/dataloader_idx_1")
                )
        else:
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
                label="val scaffold",
            )
        if self.val_steps_official:
            _plot_series(
                ax,
                self.val_steps_official,
                self.val_loss_official,
                marker="s",
                ms=5,
                lw=1.5,
                color="forestgreen",
                label="val official",
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
                label="val scaffold",
            )
            if self.val_steps_official:
                _plot_series(
                    ax,
                    self.val_steps_official,
                    self.val_loss_mces_official,
                    marker="s",
                    ms=5,
                    lw=1.5,
                    color="forestgreen",
                    label="val official",
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
        val_mces_mae = m.get("val_mces_mae")
        msg = f"[VAL]   epoch={trainer.current_epoch} step={trainer.global_step}"
        if val_loss is not None:
            msg += f" val_loss={float(val_loss):.4f}"
        if train_loss is not None:
            msg += f" train_loss={float(train_loss):.4f}"
        if val_ed is not None:
            msg += f" val_loss_ed={float(val_ed):.4f}"
        if val_mces is not None:
            msg += f" val_loss_mces={float(val_mces):.6f}"
        if val_mces_mae is not None:
            msg += f" mces_mae={float(val_mces_mae):.3f}"
        logger.info(msg)


class ValMetricsCallback(Callback):
    """
    After each validation epoch: saves a per-pair CSV of every scored validation
    pair, logs MAE (raw MCES units) broken out by GT-MCES bin (self-pairs kept
    as their own bin, not lumped into the lowest numeric bin), and saves a
    GT-binned boxplot of predicted MCES per val set (same style as the
    test_to_test_binned_box.png investigation plot).

    No Spearman anywhere (train or val) and no ED confusion matrix -- this
    callback only scores the MCES head now. Supports multiple val dataloaders;
    val_names labels each one.
    """

    _MCES_MAX = 40.0
    _BIN_EDGES = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    _SELF_LABEL = "self (MCES=0)"

    def __init__(self, output_dir: str, val_names: list | None = None):
        self.output_dir = output_dir
        self.val_names = val_names or ["val"]
        self._preds: dict = {}

    def _bin_labels(self) -> list[str]:
        labels = [self._SELF_LABEL]
        lo = 0.0
        for hi in self._BIN_EDGES:
            labels.append(f"({lo:g},{hi:g}]")
            lo = hi
        return labels

    def _buf(self, idx: int) -> dict:
        if idx not in self._preds:
            self._preds[idx] = {
                "mces": [],
                "mces_t": [],
                "mol_idx_0": [],
                "mol_idx_1": [],
                "spec_idx_0": [],
                "spec_idx_1": [],
                "smiles_0": [],
                "smiles_1": [],
            }
        return self._preds[idx]

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.sanity_checking or outputs is None:
            return
        if not isinstance(outputs, dict):
            return
        buf = self._buf(dataloader_idx)
        buf["mces"].append(outputs["mces_pred"].float().numpy())
        buf["mces_t"].append(outputs["mces_target"].float().numpy())
        buf["mol_idx_0"].append(outputs["mol_idx_0"].numpy())
        buf["mol_idx_1"].append(outputs["mol_idx_1"].numpy())
        buf["spec_idx_0"].append(outputs["spec_idx_0"].numpy())
        buf["spec_idx_1"].append(outputs["spec_idx_1"].numpy())
        buf["smiles_0"].extend(outputs["smiles_0"])
        buf["smiles_1"].extend(outputs["smiles_1"])

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._preds:
            return

        step = trainer.global_step

        for idx in sorted(self._preds.keys()):
            buf = self._preds[idx]
            if not buf["mces"]:
                continue
            mces_pred = np.concatenate(buf["mces"])
            mces_target = np.concatenate(buf["mces_t"])
            mol_idx_0 = np.concatenate(buf["mol_idx_0"])
            mol_idx_1 = np.concatenate(buf["mol_idx_1"])
            spec_idx_0 = np.concatenate(buf["spec_idx_0"])
            spec_idx_1 = np.concatenate(buf["spec_idx_1"])
            smiles_0 = buf["smiles_0"]
            smiles_1 = buf["smiles_1"]
            for k in buf:
                buf[k] = [] if isinstance(buf[k], list) else buf[k]

            val_name = self.val_names[idx] if idx < len(self.val_names) else f"val{idx}"

            # similarity (1 - MCES/max) -> raw MCES
            pred_mces = (1.0 - mces_pred) * self._MCES_MAX
            gt_mces = (1.0 - mces_target) * self._MCES_MAX
            abs_err = np.abs(pred_mces - gt_mces)
            is_self = mol_idx_0 == mol_idx_1

            bin_idx = self._bin_index(gt_mces, is_self)
            self._log_binned_mae(pl_module, abs_err, bin_idx, val_name)
            self._log_overlap_coefficients(pl_module, pred_mces, bin_idx, val_name)
            self._save_consolidated(
                step,
                val_name,
                mol_idx_0,
                mol_idx_1,
                spec_idx_0,
                spec_idx_1,
                smiles_0,
                smiles_1,
                gt_mces,
                pred_mces,
                is_self,
                bin_idx,
            )
            self._plot_binned_box(gt_mces, pred_mces, is_self, step, val_name)

    def _bin_index(self, gt_mces: np.ndarray, is_self: np.ndarray) -> np.ndarray:
        """0 = self-pair bin, 1..len(_BIN_EDGES) = the numeric GT-MCES bins
        (bin i+1 is (_BIN_EDGES[i-1], _BIN_EDGES[i]])."""
        idx = np.zeros(len(gt_mces), dtype=int)
        non_self = ~is_self
        idx[non_self] = (
            np.clip(
                np.digitize(gt_mces[non_self], self._BIN_EDGES[:-1]),
                0,
                len(self._BIN_EDGES) - 1,
            )
            + 1
        )
        return idx

    def _log_binned_mae(self, pl_module, abs_err, bin_idx, val_name):
        labels = self._bin_labels()
        for li, label in enumerate(labels):
            mask = bin_idx == li
            n = int(mask.sum())
            if n == 0:
                continue
            mae = float(abs_err[mask].mean())
            pl_module.log(
                f"val_mae_mces/{label}/{val_name}",
                mae,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )

    @staticmethod
    def _overlap_coefficient(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
        """Overlapping coefficient between two prediction samples: sum of
        min(p_a, p_b) over a shared histogram, each normalized to sum to 1.
        0 = fully separated, 1 = identical distributions -- unlike a rank/AUC
        test, this isn't fooled by large n into looking separated when there's
        still real practical overlap."""
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

    def _log_overlap_coefficients(
        self, pl_module, pred_mces, bin_idx, val_name, max_skip=4
    ):
        """Overlap coefficient between GT-MCES bins at skip distances
        0..max_skip (0 = adjacent, e.g. self vs (0,5]; 1 = one bin further,
        e.g. self vs (5,10]; ...), logged the same way as _log_binned_mae so
        the dashboard's overlap-coefficient view can read these directly for
        skip 0-4 instead of always recomputing from the per-pair data. Also
        logs val_overlap_avg/skip{k}/{val_name}, the mean over all bin-pairs
        at that skip distance -- one summary number per skip level, same
        idea as "overall MAE" alongside the per-bin MAE lines."""
        labels = self._bin_labels()
        n_bins = len(labels)
        pred_by_bin = {
            i: pred_mces[bin_idx == i] for i in range(n_bins) if (bin_idx == i).any()
        }
        for skip in range(max_skip + 1):
            skip_values = []
            for i in range(n_bins - skip - 1):
                j = i + skip + 1
                if i not in pred_by_bin or j not in pred_by_bin:
                    continue
                ovl = self._overlap_coefficient(pred_by_bin[i], pred_by_bin[j])
                pl_module.log(
                    f"val_overlap/{labels[i]}_vs_{labels[j]}_skip{skip}/{val_name}",
                    ovl,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )
                skip_values.append(ovl)
            if skip_values:
                pl_module.log(
                    f"val_overlap_avg/skip{skip}/{val_name}",
                    float(np.mean(skip_values)),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )

    def _save_consolidated(
        self,
        step,
        val_name,
        mol_idx_0,
        mol_idx_1,
        spec_idx_0,
        spec_idx_1,
        smiles_0,
        smiles_1,
        gt_mces,
        pred_mces,
        is_self,
        bin_idx,
    ):
        """Wide per-pair table: static columns (pair identity, GT, bin) once,
        one pred_mces_step{N:06d} column added per validation check, instead
        of a full fresh CSV each time (which was ~750MB x every check on the
        Gaetan-split val set -- ~120GB for 160 checks alone). Row order is
        positionally aligned across checks (not re-joined by pair identity)
        because it's guaranteed stable: the val DataLoader is sequential
        (shuffle=False, see create_dataloaders in workflows/training.py), so
        every check iterates the exact same pair order. The array_equal
        check below makes that assumption loud instead of silently
        misaligning predictions with the wrong pairs if it's ever violated
        (e.g. shuffling gets reintroduced for val).
        """
        import pandas as pd

        labels = self._bin_labels()
        path = os.path.join(
            self.output_dir, f"val_pairs_{val_name}_consolidated.parquet"
        )
        pred_col = f"pred_mces_step{step:06d}"

        if not os.path.exists(path):
            df = pd.DataFrame(
                {
                    "mol_idx_0": mol_idx_0,
                    "mol_idx_1": mol_idx_1,
                    "spec_idx_0": spec_idx_0,
                    "spec_idx_1": spec_idx_1,
                    "smiles_0": smiles_0,
                    "smiles_1": smiles_1,
                    "gt_mces": gt_mces,
                    "mces_bin": [labels[i] for i in bin_idx],
                    "is_self_pair": is_self,
                    "same_spectrum": spec_idx_0 == spec_idx_1,
                }
            )
            df[pred_col] = pred_mces.astype(np.float32)
            df.to_parquet(path, index=False, compression="snappy")
            return

        base = pd.read_parquet(path)
        if len(base) != len(mol_idx_0) or not (
            np.array_equal(base["mol_idx_0"].to_numpy(), mol_idx_0)
            and np.array_equal(base["mol_idx_1"].to_numpy(), mol_idx_1)
        ):
            raise RuntimeError(
                "val_pairs_consolidated row order/count doesn't match this check's "
                "pairs, so a positional column append isn't safe. This relies on the "
                "val DataLoader being sequential (shuffle=False) -- something must "
                "have changed that."
            )
        base[pred_col] = pred_mces.astype(np.float32)
        base.to_parquet(path, index=False, compression="snappy")

    def _plot_binned_box(self, gt_mces, pred_mces, is_self, step, val_name):
        """Same convention as mces_calibration_plots.py's binned_box_on_ax
        (test_to_test_binned_box.png): boxplot of predicted MCES per GT bin,
        boxes positioned at each bin's real GT-MCES midpoint, whis=(5,95),
        outliers hidden, pred=GT reference diagonal, each box n-annotated --
        except self-pairs get their own box at GT=0 instead of being folded
        into the lowest numeric bin.
        """
        labels = self._bin_labels()
        edges = self._BIN_EDGES
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

        fig, ax = plt.subplots(figsize=(10, 5.5))
        plot_groups = [g for g in groups if len(g) > 0]
        plot_positions = [p for p, g in zip(positions, groups) if len(g) > 0]
        plot_widths = [w for w, g in zip(widths, groups) if len(g) > 0]
        plot_labels = [lab for lab, g in zip(labels, groups) if len(g) > 0]
        if not plot_groups:
            ax.set_title(f"No pairs [{val_name}] — step {step}")
        else:
            ax.boxplot(
                plot_groups,
                positions=plot_positions,
                widths=plot_widths,
                whis=(5, 95),
                showfliers=False,
            )
            ax.plot(
                [0, self._MCES_MAX],
                [0, self._MCES_MAX],
                color="red",
                linestyle="--",
                linewidth=1,
                label="pred = GT",
            )
            ymax = max(np.percentile(g, 95) for g in plot_groups)
            label_y = ymax * 1.03
            for p, n in zip(plot_positions, [n for n in ns if n > 0]):
                ax.text(
                    p,
                    label_y,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
            ax.set_ylim(top=label_y * 1.25)
            ax.set_xticks(plot_positions)
            ax.set_xticklabels(plot_labels, rotation=30, ha="right")
            ax.legend(fontsize=8)

        ax.set_xlabel("GT MCES (binned; self-pairs kept separate at 0)")
        ax.set_ylabel("Predicted MCES")
        ax.set_title(f"Predicted MCES by GT bin [{val_name}] — step {step:,}")
        fig.tight_layout()
        path = os.path.join(
            self.output_dir, f"mces_binned_box_{val_name}_step{step:06d}.png"
        )
        fig.savefig(path, dpi=130)
        plt.close(fig)
