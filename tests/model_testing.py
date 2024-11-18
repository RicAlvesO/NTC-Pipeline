import pandas as pd
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessors.pcap_preprocessor import PcapPreprocessor
from src.models.offline.offline1 import Offline_RandomForest
from src.models.online.online1 import Online_RandomForest
from src.evaluators.standard_evaluator import Evaluator


run_seed=int(time.time())
validation_percentage = 20
page_size = 10000

preprocessor = PcapPreprocessor()
evaluator = Evaluator()

# adicionar função de ir buscar os modelos ao mlflow
model_list=[]

for model_file in os.listdir('data/models/Offline1'):
    model = Offline_RandomForest()
    model.load_model(f"data/models/Offline1/{model_file}")
    model.id = model_file.split('.')[0]
    model_list.append(model)

for model_file in os.listdir('data/models/Online1'):
    model = Online_RandomForest()
    model.load_model(f"data/models/Online1/{model_file}")
    model.id = model_file.split('.')[0]
    model_list.append(model)


# Header
header = ['expected']
for model in model_list:
    header.append(model.id)
validation_results = [header]

i=0
while True:
    print(f"Validation iteration {i}")
    validation_data = preprocessor.get_validation_data(validation_percentage, seed=run_seed, page_size=page_size, page_number=i)
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
