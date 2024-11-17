import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, data):
        """
        Evaluates the performance of models based on their predictions.
    
        Args:
            data (pd.DataFrame): A DataFrame where the first row contains column headers,
                                 the first column contains the true labels ('expected'),
                                 and subsequent columns contain predictions from models.
    
        Returns:
            pd.DataFrame: A DataFrame containing evaluation metrics for each model.
        """
        # Extract headers and data separately
        headers = data.iloc[0]  # First row contains headers
        data = data.iloc[1:]  # Remaining rows contain actual data
    
        true_labels = data.iloc[:, 0]  # First column (true labels)
        results = []
    
        for col_index in range(1, data.shape[1]):  # Iterate through model prediction columns
            model_name = headers[col_index]
            predicted_labels = data.iloc[:, col_index]
    
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, zero_division=0, pos_label="anomaly")
            recall = recall_score(true_labels, predicted_labels, zero_division=0, pos_label="anomaly")
            f1 = f1_score(true_labels, predicted_labels, zero_division=0, pos_label="anomaly")
    
            # Compute ROC curve and AUC score
            fpr, tpr, _ = roc_curve(true_labels == "anomaly", predicted_labels == "anomaly")
            roc_auc = auc(fpr, tpr)
    
            # Append results for this model
            results.append({
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
            })
    
        # Convert results to a DataFrame
        return pd.DataFrame(results)


    def plot_roc_curve(self, data):
        """
        Plots the ROC curve for each model.

        Args:
            data (pd.DataFrame): A DataFrame where the first column contains the true labels
                                 and subsequent columns contain predictions from models.
        """
        import matplotlib.pyplot as plt

        true_labels = data.iloc[:, 0]  # First column contains the true labels
        plt.figure(figsize=(10, 6))

        for col in data.columns[1:]:  # Iterate through the model prediction columns
            predicted_labels = data[col]
            fpr, tpr, _ = roc_curve(true_labels, predicted_labels, pos_label=1)
            plt.plot(fpr, tpr, label=f"Model: {col} (AUC: {auc(fpr, tpr):.2f})")

        # Plot formatting
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="best")
        plt.grid()
        plt.show()
