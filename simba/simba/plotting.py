import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt


class Plotting:

    def plot_mces(
        similarities_test2,
        flat_pred_test2,
        remove_threshold_values=True,
        threshold_mces=20,
    ):

        flat_pred_test2= np.round(flat_pred_test2)
        if remove_threshold_values:
            x = similarities_test2[similarities_test2 != threshold_mces]
            y = flat_pred_test2[similarities_test2 != threshold_mces]
        else:
            x = similarities_test2
            y = flat_pred_test2
        sns.set_theme(style="ticks")
        plot = sns.jointplot(
            x=x, y=y, kind="hex", color="#4CB391", joint_kws=dict(alpha=1, gridsize=20)
        )
        # Set x and y labels
        plot.set_axis_labels("Ground truth Similarity", "Prediction", fontsize=12)
        plot.fig.suptitle(f"MCES prediction", fontsize=16)
        # Set x-axis limits
        plot.ax_joint.set_xlim(0, 40)
        # Set x-axis limits
        plot.ax_joint.set_ylim(0, 40)

    def plot_cm(true, preds, config=None, file_name="cm.png"):
        # Compute the confusion matrix and accuracy
        cm = confusion_matrix(true, preds)
        accuracy = accuracy_score(true, preds)
        print("Accuracy:", accuracy)

        # Normalize the confusion matrix by the number of true instances per class
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create the plot
        plt.figure(figsize=(10, 7))
        labels = [">5", "4", "3", "2", "1", "0"]

        # Plot the heatmap using the 'Blues' colormap
        im = plt.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
        plt.colorbar(im)

        # Compute a threshold to decide the annotation text color
        threshold = cm_normalized.max() / 2.0

        # Annotate each cell with the percentage, using white text if the background is dark
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                text_color = "white" if cm_normalized[i, j] > threshold else "black"
                plt.text(
                    j,
                    i,
                    f"{cm_normalized[i, j]:.2%}",
                    ha="center",
                    va="center",
                    color=text_color,
                )

        # Set tick labels and increase font size for clarity
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, fontsize=12)
        plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=12)
        plt.xlabel("Predicted Labels", fontsize=14)
        plt.ylabel("True Labels", fontsize=14)
        plt.title(
            f"Confusion Matrix (Normalized), Acc: {accuracy:.2f}, Samples: {preds.shape[0]}",
            fontsize=16,
        )
