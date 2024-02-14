import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl


class LossPlotCallback(pl.Callback):
    def __init__(self, batch_per_epoch_tr=1, batch_per_epoch_val=2):
        super().__init__()
        self.batch_per_epoch_tr = batch_per_epoch_tr
        self.batch_per_epoch_val = batch_per_epoch_val
        self.train_loss_list = []
        self.val_loss_list = []

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # Assuming the train loss is present in the outputs dictionary
        train_loss = outputs["loss"].detach().cpu().numpy()
        self.train_loss_list.append(train_loss)
        self._update_plot()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # Assuming the validation loss is present in the outputs dictionary
        val_loss = outputs["val_loss"].detach().cpu().numpy()
        self.val_loss_list.append(val_loss)
        self._update_plot()

    def _update_plot(self):
        reshaped_tr = np.array(self.train_loss_list).reshape(
            -1, self.batch_per_epoch_tr
        )
        reshaped_val = np.array(self.val_loss_list).reshape(
            -1, self.batch_per_epoch_val
        )

        average_tr = np.mean(reshaped_tr, axis=1)
        average_val = np.mean(reshaped_val, axis=1)

        plt.clf()  # Clear the previous plot
        plt.plot(average_tr, label="train")
        plt.plot(average_val, label="val")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.grid()
        plt.pause(0.01)  # Add a small pause to allow the plot to update
