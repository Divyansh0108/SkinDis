import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from skindis.task import UNet, get_weights, load_data, set_weights, test, train


def generate_segmentation_masks(net, dataloader, device, output_dir, num_samples=9):
    """Generates and saves predicted segmentation masks along with original images and ground truth."""
    os.makedirs(output_dir, exist_ok=True)
    net.eval()
    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    masks = batch["mask"]
    images = images[:num_samples]
    masks = masks[:num_samples]

    with torch.no_grad():
        outputs = net(images)
        pred_masks = torch.sigmoid(outputs) > 0.5

    fig, axes = plt.subplots(
        min(num_samples, len(images)),
        3,
        figsize=(15, 5 * min(num_samples, len(images))),
    )

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].set_title("Original Image")
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 2].set_title("Predicted Mask")

    for i in range(min(num_samples, len(images))):
        img = images[i].cpu()
        true_mask = masks[i].cpu()
        pred_mask = pred_masks[i].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        axes[i, 0].imshow(img.permute(1, 2, 0))
        axes[i, 0].axis("off")
        axes[i, 1].imshow(true_mask.squeeze(), cmap="gray")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_mask.squeeze(), cmap="gray")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segmentation_comparison.png"))
    plt.close()
    return fig


class FlowerClient(NumPyClient):
    """Defines the Flower client for federated learning."""

    def __init__(
        self, net, trainloader, valloader, local_epochs, log_file, train_csv, test_csv
    ):
        self.net = net
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.net.to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.log_file = log_file
        self.train_csv = train_csv
        self.test_csv = test_csv

    def fit(self, parameters, config):
        """Trains the model locally and returns updated weights."""
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            self.log_file,
            self.train_csv,
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        masks_dir = os.path.join(script_dir, "predicted_masks", "training")
        generate_segmentation_masks(self.net, self.trainloader, self.device, masks_dir)
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": float(train_loss)},
        )

    def evaluate(self, parameters, config):
        """Evaluates the model locally and returns metrics."""
        set_weights(self.net, parameters)
        loss, dice = test(
            self.net, self.valloader, self.device, self.log_file, self.test_csv
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        masks_dir = os.path.join(script_dir, "predicted_masks", "evaluation")
        generate_segmentation_masks(self.net, self.valloader, self.device, masks_dir)
        return (
            float(loss),
            len(self.valloader.dataset),
            {"dice": float(dice)},
        )


def client_fn(context: Context):
    """Creates and returns a Flower client instance."""
    net = UNet(in_channels=3, out_channels=1)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "metrics.log")
    train_csv = os.path.join(script_dir, "training_metrics.csv")
    test_csv = os.path.join(script_dir, "testing_metrics.csv")
    return FlowerClient(
        net, trainloader, valloader, local_epochs, log_file, train_csv, test_csv
    ).to_client()


app = ClientApp(client_fn)
