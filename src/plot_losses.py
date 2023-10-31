import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow.keras.callbacks import Callback

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        #self.fig, self.ax = plt.subplots(1, 1)

    def on_epoch_end(self, epoch, logs={}):

        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        clear_output(wait=True)
        plt.figure()
        plt.plot(self.x, self.losses, label="train loss")
        plt.plot(self.x, self.val_losses, label="validation loss")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid()
        plt.legend()
        plt.show()

        #plt.show()
