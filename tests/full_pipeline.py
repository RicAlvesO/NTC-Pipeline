import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Prepocessores
from src.preprocessors.pcap_preprocessor import PcapPreprocessor

# Models 
from src.models.offline.offlineRandomForest import Offline_RandomForest
from src.models.offline.offlineDecisionTree import Offline_DecisionTree
from src.models.offline.offlineSVM import Offline_SVM

from src.models.online.onlineSGDClassifierHinge import Online_SGDClassifierHinge
from src.models.online.onlineSGDClassifierLogLoss import Online_SGDClassifierLogLoss
from src.models.online.onlineBaggingClassifier import Online_BaggingClassifier


total_time = 0
model_list = [Offline_RandomForest(), Offline_DecisionTree(), Offline_SVM(), Online_SGDClassifierHinge(), Online_SGDClassifierLogLoss(), Online_BaggingClassifier()]
total_model_time = [0] * len(model_list)
dataset_files = []

for file in os.listdir('data/pcap/normal'):
    dataset_files.append(('data/pcap/normal/' + file, 'normal'))
for file in os.listdir('data/pcap/anomaly'):
    dataset_files.append(('data/pcap/anomaly/' + file, 'anomaly'))

#Preprocess the data
t_pp_start = time.time()
preprocessor = PcapPreprocessor()
if len(dataset_files) > 0:
    preprocessor.load_datasets(dataset_files)
t_pp_end = time.time()
delta = t_pp_end - t_pp_start
total_time += delta
for i in range(len(total_model_time)):
    total_model_time[i] += delta
print(f"Data preprocessing took {delta} seconds")

# Get the training data
t_td_start = time.time()
base_training_percentage = 0.8
online_training_percentage = 0.1
run_seed = 0
training_data,online_training_data = preprocessor.get_training_data(base_training_percentage, online_training_percentage, True, seed=run_seed)
t_td_end = time.time()
delta = t_td_end - t_td_start
total_time += delta
for i in range(len(total_model_time)):
    total_model_time[i] += delta
print(f"Getting training data took {delta} seconds")

# Train the models
for i,model in enumerate(model_list):
    t_train_start = time.time()
    model.train(training_data, online_training_data)
    t_train_end = time.time()
    delta = t_train_end - t_train_start
    total_time += delta
    total_model_time[i] += delta
    print(f"Offline training {model.get_name()} took {delta} seconds")

for i,model in enumerate(model_list):
    if model.is_online():
        t_otrain_start = time.time()
        for index, row in online_training_data.iterrows():
            model.predict(row)
        t_otrain_end = time.time()
        delta = t_otrain_end - t_otrain_start
        total_time += delta
        total_model_time[i] += delta
        print(f"Online training {model.get_name()} took {delta} seconds")

# Get the validation data
t_vd_start = time.time()
evaluator = Evaluator()
validation_data = preprocessor.get_validation_data(validation_percentage,seed=run_seed)
t_vd_end = time.time()
delta += t_vd_end - t_vd_start
total_time += delta
for i in range(len(total_model_time)):
    total_model_time[i] += delta
print(f"Getting validation data took {delta} seconds")

# Evaluate the models
header = ['expected']
for model in model_list:
    header.append(model.id)
validation_results = [header]
labels = validation_data['label']
validation_data = validation_data.drop(columns=['label'])
model_results = []
for i,model in enumerate(model_list):
    t_pred_start = time.time()
    model_results.append(model.predict_batch(validation_data))
    t_pred_end = time.time()
    delta = t_pred_end - t_pred_start
    total_time += delta
    total_model_time[i] += delta
    print(f"Predicting with {model.get_name()} took {delta} seconds")


for i in range(len(labels)):
    row = [labels[i]]
    for model_result in model_results:
        row.append(model_result[i])
    validation_results.append(row)
df = pd.DataFrame(validation_results)

t_eval_start = time.time()
evaluator.evaluate(df)
t_eval_end = time.time()
delta = t_eval_end - t_eval_start
total_time += delta
for i in range(len(total_model_time)):
    total_model_time[i] += delta/len(model_list)
print(f"Evaluating the predictions took {delta} seconds")

print(f"Total time: {total_time}")
for i,model in enumerate(model_list):
    print(f"Total time for {model.get_name()}: {total_model_time[i]}")