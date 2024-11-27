import pandas as pd
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessors.pcap_preprocessor import PcapPreprocessor
from src.models.offline.offlineRandomForest import Offline_RandomForest
from src.models.offline.offlineDecisionTree import Offline_DecisionTree
from src.models.offline.offlineSVM import Offline_SVM
from src.models.online.onlineSGDClassifierHinge import Online_SGDClassifierHinge
from src.models.online.onlineSGDClassifierLogLoss import Online_SGDClassifierLogLoss
from src.models.online.onlineBaggingClassifier import Online_BaggingClassifier

run_seed=int(time.time())
base_training_percentage = 60
online_training_percentage = 20
page_size = 10000

preprocessor = PcapPreprocessor()
model_list = [Offline_RandomForest(), Offline_DecisionTree(), Offline_SVM(), Online_SGDClassifierHinge(), Online_SGDClassifierLogLoss(), Online_BaggingClassifier()]

train = preprocessor.get_offline_training_data(base_training_percentage, online_training_percentage, online=True, seed=run_seed)
print(f"Training data size: {len(train)}")
for model in model_list:
    model.train(train)
    print(f"Model {model.id} trained")

i=0
while True:
    print(f"Online training iteration {i}")
    online = preprocessor.get_online_training_data(base_training_percentage, online_training_percentage, online=True, seed=run_seed, page_size=page_size, page_number=i)
    if online is None:
        break
    print(f"Online data size: {len(online)}")
    for model in model_list:
        if model.is_online():
            model.predict_batch(online)
            print(f"Model {model.id} online trained for iteration {i}")
    i+=1

for model in model_list:
    model.save_model(f"data/models")