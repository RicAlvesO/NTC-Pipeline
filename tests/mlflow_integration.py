import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nest_asyncio
import time
import sys
sys.path.append('..')
from src.preprocessors.pcap_preprocessor import PcapPreprocessor
from src.models.offline.offline1 import Offline_RandomForest
from src.evaluators.standard_evaluator import Evaluator
from src.visualization.MLFlow import MLFlowLogger

nest_asyncio.apply()
run_seed=int(time.time())
base_training_percentage = 60
online_training_percentage = 20
validation_percentage = 20
preprocessor = PcapPreprocessor()
evaluator = Evaluator()


# Train a model and prepare metadata for logging
training_data,online_training_data = preprocessor.get_training_data(base_training_percentage, online_training_percentage, True, seed=run_seed)

# Train the offleine model
model = Offline_RandomForest()

X = model.train(training_data)

validation_data = preprocessor.get_validation_data(validation_percentage,seed=run_seed)


# Get the predictions from all models into a dataframe.

header = ['expected']
header.append(model.id)
validation_results = [header]

# Results
labels = validation_data['label']
validation_data = validation_data.drop(columns=['label'])
model_results = []
model_results.append(model.predict_batch(validation_data))

for i in range(len(labels)):
    row = [labels[i]]
    for model_result in model_results:
        row.append(model_result[i])
    validation_results.append(row)

df = pd.DataFrame(validation_results)

results = evaluator.evaluate(df)



# HERE STARTS CODE MLFLOW
# Infer the model signature
print("Inferring model signature...")
signature = infer_signature(X, model.predict(X))

# Log results to MLflow
logger = MLFlowLogger(tracking_uri="http://127.0.0.1:8080", experiment_name="MLflow Test")
logger.log_run(
    model=model,
    X_train= X,
    signature=signature,
    results=results,
    tag="Basic Offline RandomForest",
    registered_model_name="tracking-test",
)



# falta disconnect da bd