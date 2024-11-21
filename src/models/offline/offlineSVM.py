from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import time
import os

class Offline_SVM():
    def __init__(self):
        self.online = False
        self.name = "OfflineSVM"
        self.id = f"OfflineSVM_{int(time.time())}"
        self.model = SVC(kernel='rbf', probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.initialized = False
        
    def is_online(self):
        return self.online
    
    def get_name(self):
        return self.name

    # This function prepares the data by selecting relevant features, scaling them, and encoding labels
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
        scaled_data = self.scaler.fit_transform(feature_data)
        
        labels = None
        if training:
            if "label" not in data.columns:
                raise ValueError("Data must contain a 'label' column.")
            labels = self.label_encoder.fit_transform(data["label"])
        
        return pd.DataFrame(scaled_data, columns=available_features), labels
    
    # This function trains the SVM model using labeled data
    def train(self, data):
        try:
            X, y = self.prepare_data(data, training=True)
            
            # Train the SVM model
            self.model.fit(X, y)

            return X
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    # This function predicts labels in batch
    def predict_batch(self, data):
        if self.model is None:
            raise ValueError("The model is not trained.")
        
        feature_data, _ = self.prepare_data(data)
        predictions = self.model.predict(feature_data)
        data["label"] = self.label_encoder.inverse_transform(predictions)
        return data["label"]
    
    # This function predicts the label for a single data point
    def predict(self, data):
        if self.model is None:
            raise ValueError("The model is not trained.")

        if isinstance(data, pd.DataFrame):
            input_data = data
        else:
            input_data = pd.DataFrame(data) if data.ndim == 2 else pd.DataFrame([data])
        feature_data, _ = self.prepare_data(input_data, training=False)
        prediction = self.model.predict(feature_data)
        
        return self.label_encoder.inverse_transform(prediction)[0]

    def save_model(self, path):
        fpath = f"{path}/{self.name}"
        mpath = f"{fpath}/{self.id}.pkl"
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        pick={'model':self.model,'scaler':self.scaler,'label_encoder':self.label_encoder}
        with open(mpath, 'wb') as f:
            pickle.dump(pick, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            pick = pickle.load(f)
            self.model = pick['model']
            self.scaler = pick['scaler']
            self.label_encoder = pick['label_encoder']
            self.initialized = True
