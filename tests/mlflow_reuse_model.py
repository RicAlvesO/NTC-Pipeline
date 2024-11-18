import mlflow.sklearn
import mlflow.pyfunc
import sys
sys.path.append('..')
from src.preprocessors.pcap_preprocessor import PcapPreprocessor
from src.visualization.MLFlow import MLFlowLogger
preprocessor = PcapPreprocessor()

logger = MLFlowLogger(tracking_uri="http://127.0.0.1:8080", experiment_name="MLflow Test")

model_name = "tracking-test"
model_version = 8

# Load the model
print("Loading model...")
model = logger.reuse_model(model_name, model_version)


print("Getting data...")
base_data = preprocessor.get_all_data(cols=["ip.len", "ip.ttl", "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
            "tcp.len", "udp.length", "tcp.flags_syn", "tcp.flags_ack", "tcp.flags_fin",
            "size", "frame_number"])

# Predict using the loaded model
print("Predicting...")
predictions = model.predict(base_data)
print(predictions)