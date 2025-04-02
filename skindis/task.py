"""fedapp: A Flower / PyTorch app."""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from PIL import Image
from collections import OrderedDict
import time
import matplotlib.pyplot as plt


def log_metrics(log_file, message):
    """Logs a message to a specified log file."""
    with open(log_file, "a") as f:
        f.write(message + "\n")


def log_metrics_to_csv(csv_file, counter, metrics):
    """Logs metrics to a CSV file with a counter as the x-axis."""
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write("Counter,Loss,Accuracy,IoU,DiceCoeff,DiceLoss\n")
    with open(csv_file, "a") as f:
        f.write(
            f"{counter},{metrics['loss']:.4f},{metrics['accuracy']:.4f},{metrics['iou']:.4f},{metrics['dice_coeff']:.4f},{metrics['dice_loss']:.4f}\n"
        )


def plot_metrics(csv_file, output_folder, title_prefix):
    """Generates line graphs for metrics from a CSV file."""
    data = pd.read_csv(csv_file)
    metrics = ["Loss", "Accuracy", "IoU", "DiceCoeff", "DiceLoss"]
    os.makedirs(output_folder, exist_ok=True)
    for metric in metrics:
        plt.figure()
        plt.plot(data.index, data[metric], marker="o", label=metric)
        plt.xlabel("Data Points")
        plt.ylabel(metric)
        plt.title(f"{title_prefix} {metric}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{metric.lower()}.png"))
        plt.close()


def dice_loss(pred, target):
    """Computes the Dice loss for segmentation tasks."""
    smooth = 1e-6
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def calculate_metrics(pred, target):
    """Calculates accuracy, IoU, and Dice coefficient for predictions."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    smooth = 1e-6
    accuracy = (pred == target).float().mean().item()
    iou = (intersection / (union + smooth)).item()
    dice_coeff = (2.0 * intersection / (pred.sum() + target.sum() + smooth)).item()
    return accuracy, iou, dice_coeff


class LocalDataset(torch.utils.data.Dataset):
    """Custom dataset class for loading images and corresponding masks."""

    def __init__(
        self,
        images_dir,
        masks_dir,
        ground_truth_df=None,
        transform=None,
        mask_transform=None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.ground_truth_df = ground_truth_df
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.mask_transform = mask_transform or transforms.Compose(
            [transforms.ToTensor()]
        )
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Fetches an image and its corresponding mask by index."""
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        img_id = img_filename.split(".")[0]
        mask_filename = f"{img_id}_segmentation.png"
        mask_path = os.path.join(self.masks_dir, mask_filename)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = self.transform(image)
        mask = self.mask_transform(mask)
        return {"image": image, "mask": mask}


class UNet(nn.Module):
    """UNet model implementation for image segmentation tasks."""

    def __init__(self, in_channels=3, out_channels=1, dropout=0.1):
        super(UNet, self).__init__()
        self.encoder1 = self._block(in_channels, 32, dropout)
        self.encoder2 = self._block(32, 64, dropout)
        self.encoder3 = self._block(64, 128, dropout)
        self.encoder4 = self._block(128, 256, dropout)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = self._block(256, 512, dropout)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder1 = self._block(512, 256, dropout)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self._block(256, 128, dropout)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self._block(128, 64, dropout)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder4 = self._block(64, 32, dropout)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels, dropout):
        """Creates a convolutional block with batch normalization and dropout."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Defines the forward pass of the UNet model."""
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec1 = self.decoder1(dec1)
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.decoder2(dec2)
        dec3 = self.upconv3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.decoder3(dec3)
        dec4 = self.upconv4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec4 = self.decoder4(dec4)
        return self.final_conv(dec4)


def load_data(partition_id: int, num_partitions: int):
    """Loads the dataset and creates train/test data loaders for a specific partition."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    ground_truth_path = os.path.join(data_dir, "GroundTruth.csv")

    try:
        ground_truth_df = pd.read_csv(ground_truth_path)
    except FileNotFoundError:
        ground_truth_df = None

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at {images_dir}")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory not found at {masks_dir}")

    img_transform = Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_transform = Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    full_dataset = LocalDataset(
        images_dir,
        masks_dir,
        ground_truth_df,
        transform=img_transform,
        mask_transform=mask_transform,
    )

    n_samples = len(full_dataset)
    samples_per_partition = n_samples // num_partitions
    partition_lengths = [samples_per_partition] * (num_partitions - 1)
    partition_lengths.append(n_samples - sum(partition_lengths))

    partitioned_datasets = torch.utils.data.random_split(
        full_dataset, partition_lengths, generator=torch.Generator().manual_seed(42)
    )

    current_partition = partitioned_datasets[partition_id]

    train_size = int(0.8 * len(current_partition))
    test_size = len(current_partition) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        current_partition,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=16, num_workers=2)

    return trainloader, testloader


def train(net, trainloader, epochs, device, log_file, csv_file):
    """Trains the model using the training dataset."""
    net.to(device)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    net.train()

    counter = 0
    for epoch in range(epochs):
        (
            running_loss,
            running_accuracy,
            running_iou,
            running_dice_coeff,
            running_dice_loss,
        ) = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        for batch_idx, batch in enumerate(trainloader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            accuracy, iou, dice_coeff = calculate_metrics(outputs, masks)
            dice_loss_value = dice_loss(outputs, masks).item()

            running_loss += loss.item()
            running_accuracy += accuracy
            running_iou += iou
            running_dice_coeff += dice_coeff
            running_dice_loss += dice_loss_value

        avg_loss = running_loss / len(trainloader)
        avg_accuracy = running_accuracy / len(trainloader)
        avg_iou = running_iou / len(trainloader)
        avg_dice_coeff = running_dice_coeff / len(trainloader)
        avg_dice_loss = running_dice_loss / len(trainloader)

        log_metrics(
            log_file,
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, IoU: {avg_iou:.4f}, Dice Coeff: {avg_dice_coeff:.4f}, Dice Loss: {avg_dice_loss:.4f}",
        )
        log_metrics_to_csv(
            csv_file,
            counter,
            {
                "loss": avg_loss,
                "accuracy": avg_accuracy,
                "iou": avg_iou,
                "dice_coeff": avg_dice_coeff,
                "dice_loss": avg_dice_loss,
            },
        )
        counter += 1

    return avg_loss


def test(net, testloader, device, log_file, csv_file):
    """Validates the model using the test dataset."""
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    net.eval()
    total_loss, total_accuracy, total_iou, total_dice_coeff, total_dice_loss = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    counter = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = net(images)

            loss = criterion(outputs, masks).item()
            dice_loss_value = dice_loss(outputs, masks).item()

            accuracy, iou, dice_coeff = calculate_metrics(outputs, masks)

            total_loss += loss
            total_dice_loss += dice_loss_value
            total_accuracy += accuracy
            total_iou += iou
            total_dice_coeff += dice_coeff

    avg_loss = total_loss / len(testloader)
    avg_dice_loss = total_dice_loss / len(testloader)
    avg_accuracy = total_accuracy / len(testloader)
    avg_iou = total_iou / len(testloader)
    avg_dice_coeff = total_dice_coeff / len(testloader)

    log_metrics(
        log_file,
        f"Testing - Loss: {avg_loss:.4f}, Dice Loss: {avg_dice_loss:.4f}, Accuracy: {avg_accuracy:.4f}, IoU: {avg_iou:.4f}, Dice Coeff: {avg_dice_coeff:.4f}",
    )

    log_metrics_to_csv(
        csv_file,
        counter,
        {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "iou": avg_iou,
            "dice_coeff": avg_dice_coeff,
            "dice_loss": avg_dice_loss,
        },
    )
    counter += 1

    return avg_loss, avg_dice_coeff


def get_weights(net):
    """Extracts model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Sets model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        net.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    """Main script for initializing and running the training/testing pipeline."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    log_file = os.path.join(script_dir, "metrics.log")
    train_csv = os.path.join(script_dir, "training_metrics.csv")
    test_csv = os.path.join(script_dir, "testing_metrics.csv")
    train_graph_folder = os.path.join(script_dir, "training_graphs")
    test_graph_folder = os.path.join(script_dir, "testing_graphs")

    with open(log_file, "w") as f:
        f.write("Metrics Log\n")
    with open(train_csv, "w") as f:
        f.write("Counter,Loss,Accuracy,IoU,DiceCoeff,DiceLoss\n")
    with open(test_csv, "w") as f:
        f.write("Counter,Loss,Accuracy,IoU,DiceCoeff,DiceLoss\n")

    img_transform = Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_transform = Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    dataset = LocalDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=img_transform,
        mask_transform=mask_transform,
    )

    plot_metrics(train_csv, train_graph_folder, "Training")
    plot_metrics(test_csv, test_graph_folder, "Testing")
