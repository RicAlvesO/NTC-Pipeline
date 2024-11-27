import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nest_asyncio
import time
import sys
sys.path.append('..')
from src.preprocessors.pcap_preprocessor import PcapPreprocessor
from src.evaluators.standard_evaluator import Evaluator
from src.visualization.MLFlow import MLFlowLogger

from src.models.offline.offlineRandomForest import Offline_RandomForest
from src.models.offline.offlineDecisionTree import Offline_DecisionTree
from src.models.offline.offlineSVM import Offline_SVM
from src.models.online.onlineSGDClassifierHinge import Online_SGDClassifierHinge
from src.models.online.onlineSGDClassifierLogLoss import Online_SGDClassifierLogLoss
from src.models.online.onlineBaggingClassifier import Online_BaggingClassifier


# model_list = [Offline_RandomForest(), Offline_DecisionTree(), Offline_SVM(), Online_SGDClassifierHinge(), Online_SGDClassifierLogLoss(), Online_BaggingClassifier()]


nest_asyncio.apply()
run_seed=int(time.time())
base_training_percentage = 60
online_training_percentage = 20
validation_percentage = 20
preprocessor = PcapPreprocessor()
evaluator = Evaluator()


# Train a model and prepare metadata for logging
print("Preparing training data...")
training_data,online_training_data = preprocessor.get_training_data(base_training_percentage, online_training_percentage, True, seed=run_seed)

# Train the model
model = Online_SGDClassifierHinge()

print("Training offline model...")
X = model.train(training_data)

if model.is_online():
    print("Training online model " + model.get_name())
    for index, row in online_training_data.iterrows():
        model.predict(row)

print("Preparing validation data...")
validation_data = preprocessor.get_validation_data(validation_percentage,seed=run_seed)


# Get the predictions from all models into a dataframe.

print("Model evaluation...")
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



# HERE STARTS MLFLOW CODE 
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
    tag="Basic Model",
    registered_model_name=model.get_name(),
)