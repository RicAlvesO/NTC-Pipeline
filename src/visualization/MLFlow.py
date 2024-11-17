import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.base import BaseEstimator


class MLFlowLogger:

    """
    MLFlow Logger Class

    Instructions to use MLFlow Logger:
    ---------------------------------
    Before using the MLFlowLogger, ensure that you have the following:
    
    1. Install MLflow:
       To install MLflow, use the following command:
       ```
       pip install mlflow
       ```

    2. MLflow Server Setup:
       MLflow requires a running server to log experiments. To run the server locally, use this command:
       ```
       mlflow server --host 127.0.0.1 --port 8080
       ```
       You can change the host to a different IP or URL if you're running the server remotely.

    3. Ensure that the `tracking_uri` in this class points to the correct MLflow server. For example:
       ```python
       tracking_uri = "http://127.0.0.1:8080"
       ```

    4. Logging a Model:
       The class logs training results, such as metrics and the trained model itself, into MLflow.

    """

    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initializes the MLFlowLogger.

        Args:
            tracking_uri (str): The URI for the MLflow tracking server.
            experiment_name (str): The name of the experiment.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log_run(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train_pred: pd.DataFrame,
        results: pd.DataFrame,
        tag: str,
        registered_model_name: str,
    ):
        """
        Logs the details of a training run to MLflow.

        Args:
            model (BaseEstimator): The trained model.
            X_train (pd.DataFrame): Training data.
            y_train_pred (pd.DataFrame): Model predictions on the training data.
            results (pd.DataFrame): Evaluation results.
            tag (str): A tag to describe the run.
            registered_model_name (str): Name for the registered model.
        """
        with mlflow.start_run():
            # Log metrics
            print("Logging metrics...")
            for _, row in results.iterrows():
                model_name = row['model']
                mlflow.log_metric(f"{model_name}_accuracy", row['accuracy'])
                mlflow.log_metric(f"{model_name}_precision", row['precision'])
                mlflow.log_metric(f"{model_name}_recall", row['recall'])
                mlflow.log_metric(f"{model_name}_f1_score", row['f1_score'])
                mlflow.log_metric(f"{model_name}_roc_auc", row['roc_auc'])

            # Set tag
            print("Setting tags...")
            mlflow.set_tag("Training Info", tag)

            # Infer the model signature
            print("Inferring model signature...")
            signature = infer_signature(X_train, y_train_pred)

            # Log the model
            print("Logging the model...")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train,
                registered_model_name=registered_model_name,
            )
