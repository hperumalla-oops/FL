import flwr as fl
import numpy as np

class NumPyClient(fl.client.NumPyClient):
    def get_parameters(self, config): return [np.array([1.0, 2.0])]
    def fit(self, parameters, config): return parameters, 1, {}
    def evaluate(self, parameters, config): return 0.0, 0, {}

fl.client.start_numpy_client(server_address="localhost:8081", client=NumPyClient())
