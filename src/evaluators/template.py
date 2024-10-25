class Evaluator():
    def __init__(self):
        pass

    # This function is used to evaluate the performance of the models
    # It should receive a dataframe with expected and predicted values from one or more models
    # It should return a dataframe with the evaluation metrics and the results for each model
    def evaluate(self, data):
        raise NotImplementedError