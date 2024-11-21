from river.ensemble import BaggingClassifier
from river.tree import HoeffdingTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle
import time
import os

class Online_BaggingClassifier():
    def __init__(self):
        self.online = True  # Indicates the model can train online
        self.name = "OnlineBagging"
        self.id = f"OnlineBagging_{int(time.time())}"
        self.model = BaggingClassifier(
            model=HoeffdingTreeClassifier(),
            n_models=10,
            seed=42
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_initialized = False
        
    def is_online(self):
        return self.online
    
    def get_name(self):
        return self.name

    # This function prepares the data by selecting relevant features, scaling them, and encoding labels
    def prepare_data(self, data, training=False):
        # Selecting features related to IP, TCP, UDP, and size-related metrics
        features = [
            "ip.len", "ip.ttl", "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
            "tcp.len", "udp.length", "tcp.flags_syn", "tcp.flags_ack", "tcp.flags_fin",
            "size", "frame_number"
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
    
    def train(self, data):
        try:
            X, y = self.prepare_data(data, training=True)
            
            # Stream training samples to the River model
            for i in range(len(X)):
                row = X.iloc[i].to_dict()
                label = y[i]
                self.model.learn_one(row, label)
            
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def predict_batch(self, data):
        X, _ = self.prepare_data(data, training=False)
        predictions = []
        for i in range(len(X)):
            row = X.iloc[i].to_dict()
            pred = self.model.predict_one(row)
            predictions.append(pred)
        
        # Decode the predictions to labels
        data["label"] = self.label_encoder.inverse_transform(predictions)
        return data["label"]
    
    def predict(self, data):
        input_data = pd.DataFrame(data) if isinstance(data, dict) else pd.DataFrame([data])
        X, _ = self.prepare_data(input_data, training=False)
        row = X.iloc[0].to_dict()
        prediction = self.model.predict_one(row)
        return self.label_encoder.inverse_transform([prediction])[0]

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