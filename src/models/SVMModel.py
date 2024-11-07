from sklearn.svm import SVC
import pandas as pd

class SVMModel():
    def __init__(self, model_id="SVMModel"):
        super().__init__()
        self.id = model_id
        self.model = SVC()
        self.online = False

    def train(self, data, target_column):
        """
        Train the SVM model.
        
        Parameters:
            data (pd.DataFrame): DataFrame containing the data, including the target column.
            target_column (str): Name of the column with target labels.
            
        Returns:
            bool: True if the model was trained successfully.
        """
        try:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            self.model.fit(X, y)
            return True
        except Exception as e:
            print(f"Training failed: {e}")
            return False

    def predict_batch(self, data):
        """
        Make batch predictions on the given data.
        
        Parameters:
            data (pd.DataFrame): DataFrame containing the features.
            
        Returns:
            pd.DataFrame: DataFrame with an added 'prediction' column.
        """
        try:
            data['prediction'] = self.model.predict(data)
            return data
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            return data

    def predict(self, data):
        """
        Make a prediction on a single data point.
        
        Parameters:
            data (tuple): Tuple containing the feature values for a single data point.
            
        Returns:
            The predicted label for the input data.
        """
        try:
            return self.model.predict([data])[0]
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None
