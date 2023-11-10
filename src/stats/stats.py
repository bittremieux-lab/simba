
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

class Stats:

    @staticmethod
    def calculate_roc_curve(prediction, true, 
                        min_bin=0.01,
                        threshold_positive=0.6):
        # Assuming p and t are your predicted and true values, respectively
        # Create a binary ground truth vector
        y_true = (true > threshold_positive).astype(int)

        # Choose a range of thresholds
        thresholds = np.arange(0.1, 1.1, min_bin)

        precision_scores = []
        recall_scores = []

        for th in thresholds:
            # Convert predicted values to binary predictions based on the threshold
            y_pred = (prediction > th).astype(int)

            # Calculate precision and recall
            precision, recall, _ = precision_recall_curve(y_true, y_pred)

            # Append the precision and recall scores to the lists
            precision_scores.append(precision[1])  # Precision when class 1 is the positive class
            recall_scores.append(recall[1])  # Recall when class 1 is the positive class
        
        return precision_scores, recall_scores
