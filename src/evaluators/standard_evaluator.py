import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve, 
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

class Evaluator():
    def __init__(self):
        pass

    # This function is used to evaluate the performance of the models
    # It should receive a dataframe with expected and predicted values from one or more models
    # It should return a dataframe with the evaluation metrics and the results for each model
    def evaluate(self, data):
        # calculate the accuracy for each collumn
        # first collumn is the expected values
        # the rest are all the models

        headers = data.iloc[0]  # First row contains headers
        data = data.iloc[1:]  # Remaining rows contain actual data
    
        true_labels = data.iloc[:, 0]  # First column (true labels)
        results = []
        for col in data.columns[1:]:
            model_name = headers[col]
            predicted_labels = data.iloc[:, col]
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, zero_division=0, pos_label="anomaly")
            recall = recall_score(true_labels, predicted_labels, zero_division=0, pos_label="anomaly")
            f1 = f1_score(true_labels, predicted_labels, zero_division=0, pos_label="anomaly")
    
            # Append results for this model
            results.append({
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            })
        return pd.DataFrame(results)
    

    def model_comparison(self, results):
        # Reshape the data for plotting
        df_melted = results.melt(id_vars=["model"], var_name="Metric", value_name="Value")

        # Plot the grouped bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melted, x="Metric", y="Value", hue="model", palette="viridis")

        # Add titles and labels
        plt.title("Model Comparison", fontsize=16)
        plt.ylabel("Score", fontsize=12)
        plt.xlabel("Metric", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title="Model")
        plt.tight_layout()

        # Show the plot
        plt.show()


    def roc_curve_per_model(self, df):
        # Step 1: Extract the True Labels
        y_true = df[0].values  # True labels are in the first column

        # Step 2: Convert True Labels to Binary Format
        # "normal" will be 0 and "anomaly" will be 1
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)[:, 0]

        # Step 3: Prepare to Calculate ROC for Each Model
        model_columns = df.columns[1:]
        plt.figure(figsize=(10, 6))

        for model_index in range(len(model_columns)):
            # Step 4: Extract the Predictions for the Current Model
            y_scores = df[model_columns[model_index]].values

            # Convert predictions to binary
            y_scores_bin = (y_scores == 'anomaly').astype(int)  # Convert to binary (1 for anomaly, 0 for normal)

            # Step 5: Calculate ROC Curve Data
            fpr, tpr, thresholds = roc_curve(y_true_bin, y_scores_bin)
            roc_auc = auc(fpr, tpr)

            # Step 6: Plot the ROC Curve
            plt.plot(fpr, tpr, lw=2, label='Model {} (AUC = {:.2f})'.format(model_index + 1, roc_auc))

        # Step 7: Plot the Diagonal Line
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()


    def heatmap_performance_metrics(self, results):
        # Create a heatmap for the performance metrics
        plt.figure(figsize=(8, 5))
        heatmap_data = results.set_index('model').T  # Transpose for heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', cbar=True, fmt=".2f", linewidths=.5)
        plt.title('Performance Metrics Heatmap')
        plt.xlabel('Models')
        plt.ylabel('Metrics')
        plt.tight_layout()
        plt.show()