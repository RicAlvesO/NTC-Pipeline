class PreProcessor():
    # This class is used to preprocess the data
    def __init__(self):
        pass

    # This function is used to load the datasets
    # It should allow to load the data from the formats bellow:
    # - PCAP
    # It should return a dataframes with the data
    def load_datasets(self, datasets=[]):
        raise NotImplementedError

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