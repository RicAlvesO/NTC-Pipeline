class Model():
    # This class is the template for all the models
    # It should indicate if the model is online or offline
    # It should have an id(name) to identify the model
    def __init__(self):
        self.online = False
        self.id = ""

    # This function is used to check if the model is online
    def is_online(self):
        return self.online

    # This function is used to train the model
    # It should receive a dataframe with the data
    # It should return True if the model was trained successfully
    def train(self, data):
        raise NotImplementedError

    # This function is used to classify the data in batch
    # It should receive a dataframe with the data
    # It should return the dataframe with a new column with the predictions
    def predict_batch(self, data):
        raise NotImplementedError

    # This function is used to classify the data points
    # It should receive a tuple with the data
    # It should return the prediction
    def predict(self, data):
        raise NotImplementedError

