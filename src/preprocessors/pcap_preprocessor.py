import pyshark
import pandas as pd
from collections import OrderedDict

class PcapPreprocessor():
    # This class is used to preprocess the data
    def __init__(self):
        pass

    # This function is used to load the datasets
    # It should allow to load the data from the formats bellow:
    # - PCAP
    # It should return a dataframes with the data
    def load_datasets(self, datasets=[]):
        input_file = datasets[0]
        # Open the PCAP file
        capture = pyshark.FileCapture(input_file)
        
        all_fields = set()
        packets_data = []
    
        # First pass: collect all possible fields
        for packet_number, packet in enumerate(capture, start=1):
            fields = self.extract_fields(packet)
            all_fields.update(fields.keys())
            packets_data.append(fields)
            
            if packet_number % 1000 == 0:
                print(f"Processed {packet_number} packets")
    
        # Sort the fields to ensure consistent column order
        fieldnames = sorted(list(all_fields))
    
        # Create DataFrame with the packets data
        df = pd.DataFrame([{field: packet_fields.get(field, '') for field in fieldnames} 
                           for packet_fields in packets_data])
    
        print("Conversion complete.")
        return df
        

    
    def extract_fields(self, packet):
        """Extract all fields from a packet."""
        fields = OrderedDict()

        # Extract layer names
        layer_names = [layer.layer_name for layer in packet.layers]
        fields['layers'] = ':'.join(layer_names)

        # Extract fields from each layer
        for layer in packet.layers:
            for field_name in layer.field_names:
                try:
                    field_value = getattr(layer, field_name)
                    fields[f"{layer.layer_name}.{field_name}"] = field_value
                except AttributeError:
                    # Skip fields that can't be accessed
                    pass
                
        return fields
    

    def get_correct_column_type(self, base_data):

        # Loop over each column to determine its appropriate type
        for col in base_data.columns:
            # 1. Try converting to numeric
            converted_column = pd.to_numeric(base_data[col], errors='coerce')

            # If it's mostly numeric, keep it as such
            if converted_column.notna().sum() / len(converted_column) > 0.9:
                base_data[col] = converted_column
                continue
            
            # 2. Try converting to datetime
            converted_column = pd.to_datetime(base_data[col], errors='coerce')
            if converted_column.notna().sum() / len(converted_column) > 0.9:
                base_data[col] = converted_column
                continue
            
            # 3. Convert to categorical if unique values are below a threshold (e.g., less than 50% unique values)
            unique_ratio = base_data[col].nunique() / len(base_data[col])
            if unique_ratio < 0.5:
                base_data[col] = base_data[col].astype('category')
            # Otherwise, leave it as an object if no specific type applies


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
        raise NotImplementedError

    # This function is used to split the data into train, online and test
    # It should receive a dataframe and the percentages for each split
    # It should return three dataframes: train, online and test
    def split_dataframe(self, data, train_percentage, online_percentage, test_percentage):
        raise NotImplementedError