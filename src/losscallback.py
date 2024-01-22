import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

class LossCallback(Callback):
    def __init__(self, file_path, n_val_sanity_checks=2):
        self.val_loss = []
        self.train_loss=[]
        self.file_path =file_path
        self.n_val_sanity_checks=n_val_sanity_checks
    #def on_validation_batch_end(self, trainer, pl_module, outputs):
    #    self.val_outs.append(outputs)
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(float(trainer.callback_metrics["train_loss_epoch"]))

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_loss.append(float(trainer.callback_metrics["validation_loss_epoch"]))
        self.plot_loss(file_path =  self.file_path) 
        #self.val_outs  # <- access them here

    def plot_loss(self, file_path= './loss.png'):

        print('Train loss:')
        print(self.train_loss)
        print('Validation loss')
        print(self.val_loss)


        plt.figure()
        plt.plot(self.train_loss, label='train')
        plt.plot(self.val_loss[1:], label='val')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.grid()
        plt.savefig(file_path)