"""SkinCancer: A Flower / PyTorch app for skin lesion segmentation."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

# Import from our updated task.py with M4 Pro support
from skindis.task import (
    UNet,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
    get_device,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

        # Use Apple Silicon MPS acceleration if available
        self.device = get_device()
        print(f"Client initialized with device: {self.device}")
        self.net.to(self.device)

    def fit(self, parameters, config):
        print(f"Starting client training with {self.local_epochs} epochs")
        set_weights(self.net, parameters)

        try:
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
            )
            print(f"Client training completed with loss: {train_loss:.4f}")
            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {"train_loss": train_loss},
            )
        except Exception as e:
            print(f"Error during client training: {e}")
            import traceback

            traceback.print_exc()
            # Return original weights if training fails
            return (
                parameters,
                0,
                {"train_loss": float("inf")},
            )

    def evaluate(self, parameters, config):
        print("Starting client evaluation")
        set_weights(self.net, parameters)

        try:
            # Get loss and dice_score from test function
            loss, dice_score = test(self.net, self.valloader, self.device)
            print(
                f"Client evaluation completed. Loss: {loss:.4f}, Dice: {dice_score:.4f}"
            )
            return loss, len(self.valloader.dataset), {"dice_score": dice_score}
        except Exception as e:
            print(f"Error during client evaluation: {e}")
            import traceback

            traceback.print_exc()
            # Return default values if evaluation fails
            return float("inf"), 0, {"dice_score": 0.0}


def client_fn(context: Context):
    print("Initializing Flower client")
    # Load UNet model
    net = UNet(n_channels=3, n_classes=1)

    # Get partition info
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    print(f"Client partition ID: {partition_id} of {num_partitions}")

    # Load data
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Get number of local epochs
    local_epochs = context.run_config["local-epochs"]
    print(f"Client will train for {local_epochs} local epochs")

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
