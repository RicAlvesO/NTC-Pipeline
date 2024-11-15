import pyshark
import pandas as pd
from collections import OrderedDict
from src.databases.mongo_connector import Database
from concurrent.futures import ThreadPoolExecutor, as_completed

class PcapPreprocessor():
    # This class is used to preprocess the data
    def __init__(self):
        self.db = Database()
        pass

    # This function is used to load the datasets
    # It should allow to load the data from the formats bellow:
    # - PCAP
    # It should return a dataframes with the data
    def load_datasets(self, datasets=[]):
        def process_dataset(data):
            print(f"Processing dataset: {data}")
            (dataset, label) = data
            input_file = dataset
            if label.lower() not in ['normal', 'anomaly', 'unknown']:
                raise ValueError('The label must be either "allow", "anomaly" or "unknown"')

            # Open the PCAP file
            capture = pyshark.FileCapture(input_file)

            # Convert the capture to a list of packets(dict) with proper data types
            packets = []
            for i,packet in enumerate(capture,start=1):
                fields = self.extract_fields(packet)
                fields['dataset'] = dataset
                fields['label'] = label.lower()
                fields['timestamp'] = float(packet.sniff_timestamp)
                fields['size'] = packet.length
                # add the frame info
                fields['frame_number']= packet.frame_info.number
                packets.append(fields)
                if i % 1000 == 0:
                    print(f'{i} packets processed for {dataset}')
                    self.db.add_data(packets)
                    packets = []
                    if (i == 50000 and label=="normal") or (i == 10000 and label=="anomaly"):
                        break
            # Add any remaining packets
            if packets:
                self.db.add_data(packets)
            return f"{dataset} processing complete."

        # Use ThreadPoolExecutor to process datasets in parallel
        with ThreadPoolExecutor() as executor:
            # Submit each dataset to the executor for parallel processing
            futures = [executor.submit(process_dataset, data) for data in datasets]

            # Collect results as each thread completes
            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"Error processing dataset: {e}")

        return True

    # This function is used to extract the fields from the packet
    def extract_fields(self, packet):
        # convert packet to json 
        # correct the data types of the fields
        packet_dict = {}
        for layer in packet.layers:
            packet_dict[layer.layer_name] = {}
            for field in layer.field_names:
                #correct the data types of the fields
                try:
                    packet_dict[layer.layer_name][field] = int(getattr(layer, field))
                except:
                    try:
                        packet_dict[layer.layer_name][field] = float(getattr(layer, field))
                    except:
                        # verify bool
                        if getattr(layer, field).lower() in ['true', 'false']:
                            packet_dict[layer.layer_name][field] = getattr(layer, field).lower() == 'true'
                        else:
                            packet_dict[layer.layer_name][field] = getattr(layer, field)
        return packet_dict
                

    # This function is used to split the data into train, online and test
    # It should receive a dataframe and the percentages for each split
    # It should return three dataframes: train, online and test
    def get_all_data(self, cols=None, dataset=None, sample_size=None, seed=0):
        return self.db.get_data(collumns=cols, dataset=dataset, sample_size=sample_size, seed=seed)

    # This function is used to split the data into offline & online training
    def get_training_data(self,offp=60,onp=20,online=False,cols=None,dataset=None, labeled=True, seed=0):
        base_train = self.db.get_data(to_percent=offp,collumns=cols,seed=seed,dataset=dataset,labeled=labeled)
        if online:
            base_online = self.db.get_data(from_percent=offp,to_percent=offp+onp,collumns=cols,dataset=dataset,labeled=labeled,seed=seed)
            return base_train, base_online
        return base_train

    # This function is used to split the data into validation
    def get_validation_data(self,percentage=20,online=False,cols=None,dataset=None,seed=0):
        return self.db.get_data(from_percent=100-percentage,collumns=cols,dataset=dataset,labeled=True,seed=seed)

    