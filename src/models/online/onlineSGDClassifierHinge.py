from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import pickle
import time
import os

class Online_SGDClassifierHinge():
    def __init__(self):
        self.online = True  # Indicates the model supports online learning
        self.name = "OnlineSGDHinge"
        self.id = f"OnlineSGDHinge_{int(time.time())}"
        self.model = SGDClassifier(loss="hinge", random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_initialized = False

    def is_online(self):
        return self.online
    
    def get_name(self):
        return self.name

    def prepare_data(self, data, training=False):
        features = [
            "ip.len", "ip.ttl", "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
            "tcp.len", "udp.length", "tcp.flags_syn", "tcp.flags_ack", "tcp.flags_fin",
            "size", "frame_number"
        ]
        available_features = [f for f in features if f in data.columns]
        if not available_features:
            raise ValueError("None of the required features are present in the dataset.")
        
        feature_data = data[available_features].fillna(0)
        if training:
            self.scaler.partial_fit(feature_data)  # Incrementally fit the scaler
        scaled_data = self.scaler.transform(feature_data)
        
        labels = None
        if training:
            if "label" not in data.columns:
                raise ValueError("Data must contain a 'label' column.")
            labels = self.label_encoder.fit_transform(data["label"])
        
        return pd.DataFrame(scaled_data, columns=available_features), labels
    
    def train(self, data):
        try:
            X, y = self.prepare_data(data, training=True)
            
            # Initialize with classes if not done already
            if not self.is_initialized:
                self.model.partial_fit(X, y, classes=np.unique(y))
                self.is_initialized = True
            else:
                self.model.partial_fit(X, y)
            
            return X
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def predict_batch(self, data):
        X, _ = self.prepare_data(data, training=False)
        predictions = self.model.predict(X)
        data["label"] = self.label_encoder.inverse_transform(predictions)
        return data["label"]
    
    def predict(self, data):
        if self.model is None:
            raise ValueError("The model is not trained.")

        # Scale the single data point
        if isinstance(data, pd.DataFrame):
            input_data = data
        else:
            input_data = pd.DataFrame(data) if data.ndim == 2 else pd.DataFrame([data])
        feature_data, _ = self.prepare_data(input_data, training=False)
        prediction = self.model.predict(feature_data)
        
        # Map prediction to original label
        label = self.label_encoder.inverse_transform(prediction)[0]
        self.model.partial_fit(feature_data, prediction)

        return label

    def save_model(self, path):
        fpath = f"{path}/{self.name}"
        mpath = f"{fpath}/{self.id}.pkl"
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        pick = {'model': self.model, 'scaler': self.scaler, 'label_encoder': self.label_encoder}
        with open(mpath, 'wb') as f:
            pickle.dump(pick, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            pick = pickle.load(f)
            self.model = pick['model']
            self.scaler = pick['scaler']
            self.label_encoder = pick['label_encoder']
            self.is_initialized = True
