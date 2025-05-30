import flwr as fl
from model import create_model
from sklearn.model_selection import train_test_split


import pandas as pd

number=1
def load_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


class ECGClient(fl.client.NumPyClient):
    def __init__(self,x_train, x_test, y_train, y_test):
        self.x_train=x_train

        self.model = create_model(input_dim=self.x_train.shape[1])
        self.x_test=x_test
        print(self.x_train.shape[1])


        self.y_train=y_train
        self.y_test=y_test

    def get_parameters(self, config):
        print("config", config)
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history=self.model.fit(self.x_train, self.y_train, epochs=1, validation_data=(x_test, y_test), batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Eval acuracy: ", acc)
        return loss, len(self.x_test), {"accuracy": acc}


name=f"client{number}_data.csv"
x_train, x_test, y_train, y_test = load_data("client1_data.csv")

fl.client.start_numpy_client(
    server_address="localhost:8080",
    # server_address="localhost:8080",
    client=ECGClient(x_train, x_test, y_train, y_test),
    grpc_max_message_length = 1024*1024*1024 )
