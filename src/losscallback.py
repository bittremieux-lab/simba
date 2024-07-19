import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class LossCallback(Callback):
    def __init__(self, file_path, n_val_sanity_checks=2):
        self.val_loss = []
        self.val_loss_step = []
        self.train_loss = []
        self.file_path = file_path
        self.n_val_sanity_checks = n_val_sanity_checks

    # def on_validation_batch_end(self, trainer, pl_module, outputs):
    #    self.val_outs.append(outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(float(trainer.callback_metrics["train_loss_epoch"]))

    # def on_validation_epoch_end(self, trainer, pl_module):
    #    self.val_loss.append(float(trainer.callback_metrics["validation_loss_epoch"]))
    #    self.plot_loss(file_path =  self.file_path)
    #    #self.val_outs  # <- access them here

    def on_validation_end(self, trainer, pl_module):
        self.val_loss.append(float(trainer.callback_metrics["validation_loss_epoch"]))
        self.val_loss_step.append(float(trainer.callback_metrics["validation_loss"]))
        self.plot_loss(file_path=self.file_path)

    def plot_loss(self, file_path="./loss.png"):

        print("Train loss:")
        print(self.train_loss)
        print("Validation loss")
        print(self.val_loss)

        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

        # Plot train loss on the first subplot
        ax1.plot(self.train_loss, label="train", marker="o", color="b")
        ax1.set_title("Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid()

        ax2.plot(self.val_loss[1:], label="val epoch", marker="o", color="r")
        ax2.set_title("Val Loss")
        ax2.set_xlabel("Number of Epochs")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid()


        ax3.plot(self.val_loss_step[1:], label="val step", marker="o", color="r")
        ax3.set_title("Val Loss")
        ax3.set_xlabel("Number of Steps")
        ax3.set_ylabel("Loss")
        ax3.legend()
        ax3.grid()

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.savefig(file_path)
