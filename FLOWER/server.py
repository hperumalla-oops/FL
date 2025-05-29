import flwr as fl
import sys
import numpy as np
import pandas as pd

from model import create_model

def fit_config(rnd: int):
    return {"batch_size": 32, "local_epochs": 1}

def evaluate_config(rnd: int):
    return {"val_steps": 5}

def load_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def evaluate_fn(server_round, parameters, config):
    model = create_model(input_dim=187) ##need to change in the future
    model.set_weights(parameters)

    x_test, y_test = load_data(r"C:\Users\hperu\OneDrive\Desktop\fl\mitbih_test.csv")
    loss, acc = model.evaluate(x_test, y_test)
    return loss, {"accuracy": acc}





class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights


# strategy=SaveModelStrategy()

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=4,
    min_available_clients=4,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    evaluate_fn=evaluate_fn
)

# Start server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3) ,
    # grpc_max_message_length = 1024*1024*1024,
    strategy=strategy
)
