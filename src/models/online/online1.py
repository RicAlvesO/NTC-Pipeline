import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

class Online_RandomForest():
    def __init__(self):
        self.online = True  # Indicates the model can train online
        self.id = "Online1"
        self.model = SGDClassifier(random_state=42, loss="log_loss")  # Supports partial_fit
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_initialized = False  # To check if the model has been initialized for partial_fit
        
    def is_online(self):
        return self.online

    # This function prepares the data by selecting relevant features, scaling them, and encoding labels
    def prepare_data(self, data, training=False):
        # Selecting features related to IP, TCP, UDP, and size-related metrics
        features = [
            "ip.len", "ip.ttl", "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
            "tcp.len", "udp.length", "tcp.flags_syn", "tcp.flags_ack", "tcp.flags_fin",
            "timestamp", "size", "frame_number"
        ]
        # Check if the required features exist in the data
        available_features = [f for f in features if f in data.columns]
        if not available_features:
            raise ValueError("None of the required features are present in the dataset.")
        
        # Fill missing values
        feature_data = data[available_features].fillna(0)
        if training:
            self.scaler.partial_fit(feature_data)  # Incrementally scale data for online learning
        scaled_data = self.scaler.transform(feature_data)
        
        # Encode the labels
        labels = None
        if training:
            if "label" not in data.columns:
                raise ValueError("Data must contain a 'label' column.")
            labels = self.label_encoder.fit_transform(data["label"])
        
        return pd.DataFrame(scaled_data, columns=available_features), labels
    
    # This function trains the model incrementally using labeled data
    def train(self, data):
        try:
            X, y = self.prepare_data(data, training=True)
            
            # Initialize model with classes if not already done
            if not self.is_initialized:
                self.model.partial_fit(X, y, classes=np.unique(y))
                self.is_initialized = True
            else:
                self.model.partial_fit(X, y)
            
            return True
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    # This function predicts labels in batch and continues training incrementally
    def predict_batch(self, data):
        if self.model is None:
            raise ValueError("The model is not online.")
        
        # Prepare the features for prediction and training
        X, _ = self.prepare_data(data)
        predictions = self.model.predict(X)
        
        # Map predictions back to original label names
        data["label"] = self.label_encoder.inverse_transform(predictions)
        
        y = self.label_encoder.transform(data["label"])

        # Continue training with the new data
        if y is not None:
            self.model.partial_fit(X, y)
        
        return data["label"]
    
    # This function predicts the label for a single data point
    def predict(self, data):
        if self.model is None:
            raise ValueError("The model is not trained.")

        # Scale the single data point
        data = pd.DataFrame([data])  # Convert single data point to DataFrame
        feature_data, _ = self.prepare_data(data, training=False)
        prediction = self.model.predict(feature_data)
        
        # Map prediction to original label
        label = self.label_encoder.inverse_transform(prediction)[0]
        self.model.partial_fit(feature_data, prediction)

        return label