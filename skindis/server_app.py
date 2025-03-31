"""SkinCancer: A Flower / PyTorch app for skin lesion segmentation."""

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# Import UNet model and get_device for M4 Pro support
from skindis.task import UNet, get_weights, get_device


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Get proper device for Apple Silicon
    device = get_device()

    # Initialize UNet model with proper device support
    model = UNet(n_channels=3, n_classes=1)
    model.to(device)
    print(f"Server model initialized on {device}")

    # Get model parameters
    # In server_app.py
    ndarrays = get_weights(UNet(n_channels=3, n_classes=1, bilinear=False))
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy with timeout to prevent hanging
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=parameters,
    )


    # Add round timeout to prevent infinite waiting
    config = ServerConfig(
        num_rounds=num_rounds, round_timeout=600.0  # 10 minutes timeout
    )

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)



