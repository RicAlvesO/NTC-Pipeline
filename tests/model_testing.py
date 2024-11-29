import pandas as pd
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessors.pcap_preprocessor import PcapPreprocessor
from src.evaluators.standard_evaluator import Evaluator
from src.visualization.MLFlow import MLFlowLogger

run_seed=int(time.time())
validation_percentage = 20
page_size = 10000

preprocessor = PcapPreprocessor()
evaluator = Evaluator()

# adicionar função de ir buscar os modelos ao mlflow
model_name=["OfflineRF", "OfflineDT", "OfflineSVM", "OnlineSGDHinge", "OnlineSGDLogLoss", "OnlineBagging"]
model_list = []


logger = MLFlowLogger(tracking_uri="http://127.0.0.1:8080", experiment_name="MLflow Test")
model_version = 1
i = 0

for model in model_name:
    # Load the model
    print("Loading model" + model)
    model_list.append(logger.reuse_model(model, model_version))
    i+=1


# Header
header = ['expected']
for model in model_list:
    header.append(model.id)
validation_results = [header]

i=0
while True:
    print(f"Validation iteration {i}")
    validation_data = preprocessor.get_validation_data(validation_percentage, seed=run_seed, page_size=page_size, page_number=i, dataset="data/pcap/anomaly/injection_normal1.pcap")
    if validation_data is None:
        break
    print(f"Validation data size: {len(validation_data)}")
    labels = validation_data['label']
    validation_data = validation_data.drop(columns=['label'])
    model_results = []
    for model in model_list:
        model_results.append(model.predict_batch(validation_data))
        print(f"Model {model.id} validated for iteration {i}")

    for i in range(len(labels)):
        row = [labels[i]]
        for model_result in model_results:
            row.append(model_result[i])
        validation_results.append(row)
    
    i+=1

df = pd.DataFrame(validation_results)

results = pd.DataFrame(evaluator.evaluate(df))
print(results)
time = int(time.time())
# Save the results to a CSV file
with open(f'data/results/{time}.csv', 'w') as f:
    results.to_csv(f, index=False)
