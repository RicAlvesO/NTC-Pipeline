import pymongo
import os
import pandas as pd
from dotenv import load_dotenv

class Database():

    # This class is used to interact with the database
    def __init__(self):
        load_dotenv()
        self.username = os.getenv("MONGO_USERNAME")
        self.password = os.getenv("MONGO_PASSWORD")
        self.host = os.getenv("MONGO_HOST")
        self.port = os.getenv("MONGO_PORT")
        self.db_name = os.getenv("MONGO_DB")
        self.collection_name = os.getenv("MONGO_COLLECTION")
        self.connect()

    def __del__(self):
        self.disconnect()

    # This function is used to connect to the database
    def connect(self):
        self.client = pymongo.MongoClient(f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/")
        # Get the database and collection (create them if they don't exist)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]


    # This function is used to disconnect from the database
    def disconnect(self):
        self.client.close()

    def add_data(self, data):
        self.collection.insert_many(data)

    def get_data(self, collumns=None, from_percent=0, to_percent=100, dataset=None):
        # Get all data in a dataframe
        if dataset:
            data = self.collection.find({"dataset": dataset})
        else:
            data = self.collection.find()
        # Sqaush the recursive json into a flat json
        data=pd.json_normalize(data)
        # Get the percentage of the data
        data = data.iloc[int(len(data)*from_percent/100):int(len(data)*to_percent/100)]
        df = pd.DataFrame(data)
        if collumns:
            df = df[collumns]
        return df
