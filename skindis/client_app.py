"""SkinCancer: A Flower / PyTorch app for skin lesion segmentation."""

import torch
import torch.nn.functional as F
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


# IoU calculation function
def calculate_iou(pred, target, smooth=1e-6):
    """Calculate IoU (Intersection over Union)"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


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

            # Calculate IoU on training data
            train_iou = self._compute_iou(self.trainloader)

            print(
                f"Client training completed with loss: {train_loss:.4f}, IoU: {train_iou:.4f}"
            )
            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {"train_loss": train_loss, "train_iou": train_iou},
            )
        except Exception as e:
            print(f"Error during client training: {e}")
            import traceback

            traceback.print_exc()
            # Return original weights if training fails
            return (
                parameters,
                0,
                {"train_loss": float("inf"), "train_iou": 0.0},
            )

    def _compute_iou(self, dataloader):
        """Compute IoU on a dataloader"""
        self.net.eval()
        total_iou = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                outputs = self.net(images)
                iou = calculate_iou(outputs, masks)
                total_iou += iou.item()
                count += 1

                # Free memory
                del images, masks, outputs
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return total_iou / count if count > 0 else 0.0

    def evaluate(self, parameters, config):
        print("Starting client evaluation")
        set_weights(self.net, parameters)

        try:
            # Get loss and dice_score from test function
            loss, dice_score = test(self.net, self.valloader, self.device)

            # Calculate IoU separately since it's not part of the original test function
            test_iou = self._compute_iou(self.valloader)

            print(
                f"Client evaluation completed. Loss: {loss:.4f}, Dice: {dice_score:.4f}, IoU: {test_iou:.4f}"
            )
            return (
                loss,
                len(self.valloader.dataset),
                {"dice_score": dice_score, "iou_score": test_iou},
            )
        except Exception as e:
            print(f"Error during client evaluation: {e}")
            import traceback

            traceback.print_exc()
            # Return default values if evaluation fails
            return float("inf"), 0, {"dice_score": 0.0, "iou_score": 0.0}


def client_fn(context: Context):
    print("Initializing Flower client")

    # Load UNet model with additional parameters to improve performance
    net = UNet(
        n_channels=3, n_classes=1, bilinear=False
    )  # Use transposed conv instead of bilinear

    # Get partition info
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    print(f"Client partition ID: {partition_id} of {num_partitions}")

    # Load data
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Get number of local epochs
    local_epochs = context.run_config.get(
        "local-epochs", 1
    )  # Default to 1 if not specified
    print(f"Client will train for {local_epochs} local epochs")

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
