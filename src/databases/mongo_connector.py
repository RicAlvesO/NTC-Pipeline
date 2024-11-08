import pymongo
import os
from dotenv import load_dotenv

class Database():

    # This class is used to interact with the database
    def __init__(self):
        load_dotenv()
        self.connect()

    def __del__(self):
        self.disconnect()

    # This function is used to connect to the database
    def connect(self):
        self.client = pymongo.MongoClient(os.getenv("MONGO_URI"))
        self.db = self.client[os.getenv("MONGO_DB")]
        self.collection = self.db[os.getenv("MONGO_COLLECTION")]

    # This function is used to disconnect from the database
    def disconnect(self):
        self.client.close()

    def add_data(self, data):
        self.collection.insert_many(data)

    def get_training_data(self):
        raise NotImplementedError
