import os

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessors.pcap_preprocessor import PcapPreprocessor

dataset_files = []

for file in os.listdir('data/normal'):
    dataset_files.append(('data/normal/' + file, 'normal'))
for file in os.listdir('data/anomaly'):
    dataset_files.append(('data/anomaly/' + file, 'anomaly'))

preprocessor = PcapPreprocessor()
if len(dataset_files) > 0:
    preprocessor.load_datasets(dataset_files)