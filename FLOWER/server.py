import flwr as fl
import sys
import numpy as np
from model import create_model

def fit_config(rnd: int):
    return {"batch_size": 32, "local_epochs": 1}

def evaluate_config(rnd: int):
    return {"val_steps": 5}

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
    min_fit_clients=2,
    min_available_clients=2,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
)

# Start server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3) ,
    # grpc_max_message_length = 1024*1024*1024,
    strategy=strategy
)
