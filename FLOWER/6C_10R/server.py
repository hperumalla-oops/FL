import flwr as fl
import sys
import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
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

    x_test, y_test = load_data(r"C:\Users\hperu\OneDrive\Desktop\fl\FLOWER\1\mitbih_test.csv")  # You can also try with ptbdb files)
    loss, acc = model.evaluate(x_test, y_test)
    model.save("server_model.h5")
    y_pred = np.argmax(model.predict(x_test), axis=1)
    # print(classification_report(y_pred, y_test))

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

            if rnd == 20:  # Save the final global model at last round
                model = create_model(input_dim=187)  # Replace with actual input_dim if dynamic
                model.set_weights(aggregated_weights)
                model.save(f"Server_model.h5")
                print("Saved final global model to .h5")

        return aggregated_weights


# strategy=SaveModelStrategy()

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=6,
    min_available_clients=6,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    evaluate_fn=evaluate_fn
)

# Start server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=11) ,
    # grpc_max_message_length = 1024*1024*1024,
    strategy=strategy
)
