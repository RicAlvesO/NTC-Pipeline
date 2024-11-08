class Database():

    # This class is used to interact with the database
    def __init__(self):
        self.connect()

    def __del__(self):
        self.disconnect()

    # This function is used to connect to the database
    def connect(self):
        raise NotImplementedError

    # This function is used to disconnect from the database
    def disconnect(self):
        raise NotImplementedError

    # This function is used to create a new entry in the database
    def create(self):
        raise NotImplementedError

    # This function is used to read the data from the database
    def read(self):
        raise NotImplementedError

    # This function is used to update the data in the database
    def update(self):
        raise NotImplementedError

    # This function is used to delete the data from the database
    def delete(self):
        raise NotImplementedError

    def get_training_data(self):
        raise NotImplementedError
