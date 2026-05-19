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
    After each validation epoch, saves:
      - confusion_matrix.png  (ED predicted vs true class)
      - mces_scatter.png      (MCES predicted vs true similarity)
    to the checkpoint directory.
    """

    def __init__(self, output_dir: str, n_classes: int = 6):
        self.output_dir = output_dir
        self.n_classes = n_classes
        self._ed_preds = []
        self._ed_targets = []
        self._mces_preds = []
        self._mces_targets = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.sanity_checking or outputs is None:
            return
        if not isinstance(outputs, dict):
            return
        self._ed_preds.append(outputs["ed_pred"].numpy())
        self._ed_targets.append(outputs["ed_target"].numpy())
        self._mces_preds.append(outputs["mces_pred"].float().numpy())
        self._mces_targets.append(outputs["mces_target"].float().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._ed_preds:
            return

        ed_pred = np.concatenate(self._ed_preds)
        ed_target = np.concatenate(self._ed_targets)
        mces_pred = np.concatenate(self._mces_preds)
        mces_target = np.concatenate(self._mces_targets)

        self._ed_preds.clear()
        self._ed_targets.clear()
        self._mces_preds.clear()
        self._mces_targets.clear()

        step = trainer.global_step
        self._plot_confusion(ed_pred, ed_target, step)
        self._plot_mces_scatter(mces_pred, mces_target, step)

    def _plot_confusion(self, pred, target, step):
        n = self.n_classes
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(target, pred):
            t = int(np.clip(t, 0, n - 1))
            p = int(np.clip(p, 0, n - 1))
            cm[t, p] += 1

        cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
        accuracy = (pred == target).mean()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Left: raw counts
        ax = axes[0]
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel(f"Predicted class  [acc={accuracy:.3f}]")
        ax.set_ylabel("True class")
        ax.set_title(f"ED confusion matrix — counts (step {step})")
        vmax = cm.max() if cm.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > 0.5 * vmax else "black",
                )

        # Right: row percentages
        ax = axes[1]
        im = ax.imshow(cm_pct, vmin=0, vmax=100, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel(f"Predicted class  [acc={accuracy:.3f}]")
        ax.set_ylabel("True class")
        ax.set_title(f"ED confusion matrix — row % (step {step})")
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{cm_pct[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm_pct[i, j] > 50 else "black",
                )

        plt.tight_layout()
        path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=130)
        plt.close(fig)

    def _plot_mces_scatter(self, pred, target, step):
        # subsample to at most 10k points for speed
        if len(pred) > 10_000:
            idx = np.random.choice(len(pred), 10_000, replace=False)
            pred, target = pred[idx], target[idx]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        ax.scatter(target, pred, alpha=0.15, s=4, rasterized=True)
        lo, hi = min(target.min(), pred.min()), max(target.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("True MCES similarity")
        ax.set_ylabel("Predicted MCES similarity")
        corr = np.corrcoef(target, pred)[0, 1] if len(target) > 1 else float("nan")
        ax.set_title(f"MCES scatter — step {step}\nr={corr:.3f}")
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        residuals = pred - target
        ax2.hist(residuals, bins=50, color="steelblue", edgecolor="none")
        ax2.axvline(0, color="red", lw=1, ls="--")
        ax2.set_xlabel("Residual (pred − true)")
        ax2.set_title(
            f"MCES residuals  mean={residuals.mean():.3f}  std={residuals.std():.3f}"
        )
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "mces_scatter.png")
        plt.savefig(path, dpi=130)
        plt.close(fig)
