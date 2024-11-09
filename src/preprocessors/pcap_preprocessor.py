import pyshark
import pandas as pd
from collections import OrderedDict
from src.databases.mongo_connector import Database

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
        for data in datasets:
            (dataset, label) = data
            input_file = dataset
            if label.lower() not in ['allow', 'deny', 'unknown']:
                raise ValueError('The label must be either "allow", "deny" or "unknown"')
            
            # Open the PCAP file
            capture = pyshark.FileCapture(input_file)

            # convert the capture to a array of packets(dict) with proper data types instead of strings
            packets = []
            for packet in capture:
                fields = self.extract_fields(packet)
                fields['dataset'] = dataset
                fields['label'] = label.lower()
                packets.append(fields)
                if len(packets)%1000==0:
                    print(f'{len(packets)} packets processed')
            return packets
    
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
                

    def get_correct_column_type(self, base_data):
        for col in base_data.columns:
            
            try:
                converted_column = pd.to_numeric(base_data[col], errors='coerce')
                if converted_column.notna().sum() / len(converted_column) > 0.9:
                    base_data[col] = converted_column
                    continue
            except:
                pass
            
            try:
                converted_column = pd.to_datetime(base_data[col], errors='coerce')
                if converted_column.notna().sum() / len(converted_column) > 0.9:
                    base_data[col] = converted_column
                    continue
            except:
                pass

            try:
                values = base_data[col].unique()
                bool_values = [value for value in values if value.lower() in ['true', 'false']]
                if len(bool_values) / len(values) > 0.9:
                    base_data[col] = base_data[col].apply(lambda x: x.lower() == 'true')
                    continue
            except:
                pass

            try:
                unique_ratio = base_data[col].nunique() / len(base_data[col])
                if unique_ratio < 0.2:
                    base_data[col] = base_data[col].astype('category')
                    continue
            except:
                pass

            base_data[col] = base_data[col].astype('str')

        return base_data


    # This function is used to preprocess the data
    # It receives a dataframe and should return the preprocessed dataframe
    # This function should include the steps such as:
    # - Feature selection
    # - Feature engineering
    # - Data cleaning
    # - Data normalization
    # - Data transformation
    # - Outliar detection
    # It should return a dataframe with the preprocessed data
    def preprocess_dataframe(self, data):
        self.db.add_data(data)
        return True

    # This function is used to split the data into train, online and test
    # It should receive a dataframe and the percentages for each split
    # It should return three dataframes: train, online and test
    def get_all_data(self, cols=None, dataset=None):
        df = self.db.get_data(collumns=cols,dataset=dataset)
        return df

    def get_training_data(self,offp=60,onp=20,online=False,cols=None,dataset=None):
        base_train = self.db.get_data(to_percent=offp,collumns=cols)
        if online:
            base_online = self.db.get_data(from_percent=offp,to_percent=offp+onp,collumns=cols,dataset=dataset)
            return base_train, base_online
        return base_train

    def get_validation_data(self,percentage=20,online=False,cols=None,dataset=None):
        test = self.db.get_data(from_percent=100-percentage,collumns=cols,dataset=dataset)
        return test