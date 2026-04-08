import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class LossCallback(Callback):
    def __init__(self, file_path, n_val_sanity_checks=2):
        self.val_loss = []
        self.val_loss_step = []
        self.val_global_steps = []
        self.train_loss = []
        self.train_loss_step = []
        self.train_global_steps = []
        self.file_path = file_path
        self.history_path = str(Path(file_path).with_name("loss_history.json"))
        self.n_val_sanity_checks = n_val_sanity_checks
        self._start_time = None
        self.train_epoch_times = []
        self.val_times = []
        self.train_step_times = []

    # def on_validation_batch_end(self, trainer, pl_module, outputs):
    #    self.val_outs.append(outputs)

    def _ensure_start_time(self):
        if self._start_time is None:
            self._start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._ensure_start_time()
        if trainer.global_step <= 0:
            return
        if trainer.global_step % trainer.log_every_n_steps != 0:
            return

        loss = trainer.callback_metrics.get("train_loss")
        if loss is None:
            return

        self.train_loss_step.append(float(loss))
        self.train_global_steps.append(int(trainer.global_step))
        self.train_step_times.append(time.time() - self._start_time)

    def on_train_epoch_end(self, trainer, pl_module):
        self._ensure_start_time()
        self.train_loss.append(float(trainer.callback_metrics["train_loss_epoch"]))
        self.train_epoch_times.append(time.time() - self._start_time)

    # def on_validation_epoch_end(self, trainer, pl_module):
    #    self.val_loss.append(float(trainer.callback_metrics["validation_loss_epoch"]))
    #    self.plot_loss(file_path =  self.file_path)
    #    #self.val_outs  # <- access them here

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._ensure_start_time()
        self.val_loss.append(float(trainer.callback_metrics["validation_loss_epoch"]))
        self.val_loss_step.append(float(trainer.callback_metrics["validation_loss"]))
        self.val_global_steps.append(int(trainer.global_step))
        self.val_times.append(time.time() - self._start_time)
        self.save_history()
        self.plot_loss(file_path=self.file_path)

    def on_train_end(self, trainer, pl_module):
        self.save_history()

    def save_history(self):
        history = {
            "train_loss_epoch": self.train_loss,
            "train_loss_step": self.train_loss_step,
            "train_global_steps": self.train_global_steps,
            "validation_loss_epoch": self.val_loss,
            "validation_loss_step": self.val_loss_step,
            "validation_global_steps": self.val_global_steps,
            "train_epoch_times_seconds": self.train_epoch_times,
            "train_step_times_seconds": self.train_step_times,
            "validation_times_seconds": self.val_times,
        }
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)

    def plot_loss(self, file_path="./loss.png"):
        print("Train loss:")
        print(self.train_loss)
        print("Validation loss")
        print(self.val_loss)

        # Create subplots
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot train loss on the first subplot
        ax1.plot(self.train_loss, label="train", marker="o", color="b")
        ax1.set_title("Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid()

        # ax2.plot(self.val_loss[1:], label="val epoch", marker="o", color="r")
        # ax2.set_title("Val Loss")
        # ax2.set_xlabel("Number of Epochs")
        # ax2.set_ylabel("Loss")
        # ax2.legend()
        # ax2.grid()

        ax2.plot(self.val_loss_step, label="val step", marker="o", color="r")
        ax2.set_title("Val Loss")
        ax2.set_xlabel("Validation event")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid()

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.savefig(file_path)
        print(f"Saved loss figure to {file_path}")
        plt.close(fig)


"""
class LossCallback(Callback):
    def __init__(self, file_path, n_val_sanity_checks=2):
        super().__init__()
        self.train_loss1 = []
        self.train_loss2 = []
        self.val_loss1 = []
        self.val_loss2 = []
        self.file_path = file_path
        self.n_val_sanity_checks = n_val_sanity_checks

    def on_train_epoch_end(self, trainer, pl_module):
        # Get the logged metrics for the epoch
        self.train_loss1.append(trainer.callback_metrics.get("train_loss1").item())
        self.train_loss2.append(trainer.callback_metrics.get("train_loss2").item())

    def on_validation_end(self, trainer, pl_module):
        # Get the logged validation metrics
        self.val_loss1.append(trainer.callback_metrics.get("val_loss1").item())
        self.val_loss2.append(trainer.callback_metrics.get("val_loss2").item())
        self.plot_loss(self.file_path)

    def plot_loss(self, file_path):
        epochs = range(1, len(self.train_loss1) + 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot training vs. validation loss for the first task
        ax1.plot(epochs, self.train_loss1, label="Train Loss1", marker="o", color="b")
        ax1.plot(epochs, self.val_loss1, label="Val Loss1", marker="o", color="r")
        ax1.set_title("Loss for Multitask Output 1")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid()

        # Plot training vs. validation loss for the second task
        ax2.plot(epochs, self.train_loss2, label="Train Loss2", marker="o", color="b")
        ax2.plot(epochs, self.val_loss2, label="Val Loss2", marker="o", color="r")
        ax2.set_title("Loss for Multitask Output 2")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close(fig)
"""
