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


    def get_data_random(self, collumns=None, dataset=None, sample_size=None):

        # Build the query based on the dataset if provided
        query = {}
        if dataset:
            query['dataset'] = dataset

        # Count total documents that match the query
        total_count = self.collection.count_documents(query)

        # If sample_size is provided, limit the number of documents retrieved
        if sample_size is not None:
            sample_size = min(sample_size, total_count)  # Ensure we don't exceed total count
            pipeline = [
                {"$match": query},
                {"$sample": {"size": sample_size}}  # Use the $sample stage for random sampling
            ]
        else:
            pipeline = [
                {"$match": query}
            ]

        # Execute the aggregation pipeline
        data = list(self.collection.aggregate(pipeline))

        # Normalize the JSON into a flat DataFrame
        data = pd.json_normalize(data)

        # Create a DataFrame from the sampled data
        df = pd.DataFrame(data)

        # Filter columns if specified
        if collumns:
            df = df[collumns]

        return df
    

    def count_by_dataset(self):
        """Counts the number of documents by dataset."""

        pipeline_dataset = [
            {
                "$group": {
                    "_id": "$dataset",  # Group by the 'dataset' field
                    "count": {"$sum": 1}  # Count the number of documents in each group
                }
            },
            {
                "$sort": {"count": -1}  # Sort by count in descending order
            }
        ]

        # Execute the aggregation
        dataset_counts = list(self.collection.aggregate(pipeline_dataset))

        # Convert the result into a DataFrame
        dataset_df = pd.DataFrame(dataset_counts)

        return dataset_df


    def count_by_label(self):
        """Counts the number of documents by label."""

        pipeline_label = [
            {
                "$group": {
                    "_id": "$label",  # Group by the 'label' field
                    "count": {"$sum": 1}  # Count the number of documents in each group
                }
            },
            {
                "$sort": {"count": -1}  # Sort by count in descending order
            }
        ]

        # Execute the aggregation
        label_counts = list(self.collection.aggregate(pipeline_label))

        # Convert the result into a DataFrame
        label_df = pd.DataFrame(label_counts)

        return label_df