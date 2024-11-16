import pandas as pd

class Evaluator():
    def __init__(self):
        pass

    # This function is used to evaluate the performance of the models
    # It should receive a dataframe with expected and predicted values from one or more models
    # It should return a dataframe with the evaluation metrics and the results for each model
    def evaluate(self, data):
        # calculate the accuracy for each collumn
        # first collumn is the expected values
        # the rest are all the models
        results = []
        for col in data.columns[1:]:
            accuracy = (data[col] == data[data.columns[0]]).mean()
            results.append(accuracy)
        return pd.DataFrame(results, index=data.columns[1:], columns=["accuracy"])