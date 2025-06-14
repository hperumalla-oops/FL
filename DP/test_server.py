import flwr as fl

# Define a simple strategy (optional)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
)

# Start the Flower server
fl.server.start_server(
    server_address="localhost:8081",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
