import pandas as pd
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessors.pcap_preprocessor import PcapPreprocessor
from src.models.offline.offline1 import Offline_RandomForest
from src.models.online.online1 import Online_RandomForest

run_seed=int(time.time())
base_training_percentage = 60
online_training_percentage = 20
page_size = 10000

preprocessor = PcapPreprocessor()
model_list = [Offline_RandomForest(), Online_RandomForest()]

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