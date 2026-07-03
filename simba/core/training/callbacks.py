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
    After each validation epoch, saves per-val-set hexbin plots and logs Spearman + MSE.
    Also tracks train Spearman/MSE from training_step outputs.
    Supports multiple val dataloaders; val_names labels each one.
    """

    _MAX_TRAIN_PAIRS = 300_000  # cap on pairs accumulated between two val runs

    def __init__(
        self, output_dir: str, n_classes: int = 6, val_names: list | None = None
    ):
        self.output_dir = output_dir
        self.n_classes = n_classes
        self.val_names = val_names or ["val"]
        self._preds: dict = {}
        self._train_buf: dict = {"mces": [], "mces_t": []}

        # History for metrics curves (one entry per validation run)
        self._curve_steps: list[int] = []
        self._curve_train_spearman: list[float] = []
        self._curve_train_mse: list[float] = []
        self._curve_val_spearman: dict[str, list[float]] = {}
        self._curve_val_mse: dict[str, list[float]] = {}

    def _buf(self, idx: int) -> dict:
        if idx not in self._preds:
            self._preds[idx] = {"ed": [], "ed_t": [], "mces": [], "mces_t": []}
        return self._preds[idx]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not isinstance(outputs, dict) or "mces_pred" not in outputs:
            return
        n_pairs = sum(len(x) for x in self._train_buf["mces"])
        if n_pairs >= self._MAX_TRAIN_PAIRS:
            return
        self._train_buf["mces"].append(outputs["mces_pred"].float().numpy())
        self._train_buf["mces_t"].append(outputs["mces_target"].float().numpy())

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.sanity_checking or outputs is None:
            return
        if not isinstance(outputs, dict):
            return
        buf = self._buf(dataloader_idx)
        buf["ed"].append(outputs["ed_pred"].numpy())
        buf["ed_t"].append(outputs["ed_target"].numpy())
        buf["mces"].append(outputs["mces_pred"].float().numpy())
        buf["mces_t"].append(outputs["mces_target"].float().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._preds:
            return

        from scipy.stats import spearmanr

        step = trainer.global_step
        self._curve_steps.append(step)

        # --- Train metrics from accumulated buffer ---
        if self._train_buf["mces"]:
            tr_pred = np.concatenate(self._train_buf["mces"])
            tr_target = np.concatenate(self._train_buf["mces_t"])
            self._train_buf = {"mces": [], "mces_t": []}
            if len(tr_pred) > 1:
                r_tr, _ = spearmanr(tr_target, tr_pred)
                mse_tr = float(np.mean((tr_pred - tr_target) ** 2))
                self._curve_train_spearman.append(float(r_tr))
                self._curve_train_mse.append(mse_tr)
                pl_module.log(
                    "train_mces_spearman",
                    float(r_tr),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        # --- Val metrics per dataloader ---
        for idx in sorted(self._preds.keys()):
            buf = self._preds[idx]
            if not buf["ed"]:
                continue
            ed_pred = np.concatenate(buf["ed"])
            ed_target = np.concatenate(buf["ed_t"])
            mces_pred = np.concatenate(buf["mces"])
            mces_target = np.concatenate(buf["mces_t"])
            buf["ed"].clear()
            buf["ed_t"].clear()
            buf["mces"].clear()
            buf["mces_t"].clear()

            val_name = self.val_names[idx] if idx < len(self.val_names) else f"val{idx}"

            if len(mces_pred) > 1:
                r, _ = spearmanr(mces_target, mces_pred)
                mse = float(np.mean((mces_pred - mces_target) ** 2))
                pl_module.log(
                    f"val_mces_spearman/{val_name}",
                    float(r),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )
                self._curve_val_spearman.setdefault(val_name, []).append(float(r))
                self._curve_val_mse.setdefault(val_name, []).append(mse)

            self._plot_confusion(ed_pred, ed_target, step, val_name)
            self._plot_mces_hexbin(mces_pred, mces_target, step, val_name)

        self._plot_metrics_curves()

    def _plot_metrics_curves(self):
        """Plot Spearman ρ and MSE curves for train + all val sets over training steps."""
        steps = self._curve_steps
        if not steps:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colors = {
            "train": "steelblue",
            "scaffold": "darkorange",
            "official": "forestgreen",
        }
        markers = {"scaffold": "o", "official": "s"}

        def _color(name):
            for k, c in colors.items():
                if k in name:
                    return c
            return "gray"

        def _marker(name):
            for k, m in markers.items():
                if k in name:
                    return m
            return "^"

        # Spearman
        ax = axes[0]
        if self._curve_train_spearman:
            n = len(self._curve_train_spearman)
            ax.plot(
                steps[-n:],
                self._curve_train_spearman,
                lw=1.0,
                color=colors["train"],
                label="train",
            )
        for name, vals in self._curve_val_spearman.items():
            n = len(vals)
            ax.plot(
                steps[-n:],
                vals,
                marker=_marker(name),
                ms=6,
                lw=1.5,
                color=_color(name),
                label=f"val {name}",
            )
        ax.set_title("MCES Spearman ρ")
        ax.set_xlabel("step")
        ax.set_ylabel("ρ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # MSE
        ax = axes[1]
        if self._curve_train_mse:
            n = len(self._curve_train_mse)
            ax.plot(
                steps[-n:],
                self._curve_train_mse,
                lw=1.0,
                color=colors["train"],
                label="train",
            )
        for name, vals in self._curve_val_mse.items():
            n = len(vals)
            ax.plot(
                steps[-n:],
                vals,
                marker=_marker(name),
                ms=6,
                lw=1.5,
                color=_color(name),
                label=f"val {name}",
            )
        ax.set_title("MCES MSE (pred vs GT similarity)")
        ax.set_xlabel("step")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "metrics_curves.png"), dpi=130)
        plt.close(fig)

    def _plot_confusion(self, pred, target, step, val_name="val"):
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
        path = os.path.join(self.output_dir, f"confusion_matrix_{val_name}.png")
        plt.savefig(path, dpi=130)
        plt.close(fig)

    def _plot_mces_hexbin(self, pred, target, step, val_name="val"):
        from scipy.stats import spearmanr

        r, _ = spearmanr(target, pred) if len(pred) > 1 else (float("nan"), None)

        # Convert similarity (1 - MCES/40) back to raw MCES [0, 20]
        target_mces = (1.0 - target) * 40.0
        pred_mces = (1.0 - pred) * 40.0
        lo = min(float(target_mces.min()), float(pred_mces.min()))
        hi = max(float(target_mces.max()), float(pred_mces.max()))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: linear scale
        ax = axes[0]
        hb = ax.hexbin(target_mces, pred_mces, gridsize=60, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="count")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("True MCES")
        ax.set_ylabel("Predicted MCES")
        ax.set_title("Linear scale")
        ax.grid(True, alpha=0.2)

        # Right: log scale
        ax = axes[1]
        hb = ax.hexbin(
            target_mces, pred_mces, gridsize=60, cmap="Blues", mincnt=1, bins="log"
        )
        plt.colorbar(hb, ax=ax, label="log10(count)")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("True MCES")
        ax.set_ylabel("Predicted MCES")
        ax.set_title("Log scale")
        ax.grid(True, alpha=0.2)

        fig.suptitle(
            f"MCES hexbin [{val_name}] — step {step:,}   Spearman ρ = {r:.3f}",
            fontsize=11,
        )
        plt.tight_layout()
        path = os.path.join(
            self.output_dir, f"mces_hexbin_{val_name}_step{step:06d}.png"
        )
        plt.savefig(path, dpi=130)
        plt.close(fig)
