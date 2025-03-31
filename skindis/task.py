#!/usr/bin/env python3
"""SkinCancer: A Flower / PyTorch app for UNet segmentation on HAM10000 dataset."""

import os
from collections import OrderedDict
import pandas as pd
import numpy as np
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Custom Dataset for HAM10000
class HAM10000Dataset(Dataset):
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
        self.transform = transform
        self.mask_transform = mask_transform

        # Get list of image filenames
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        print(f"Found {len(self.image_filenames)} images in {images_dir}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get image filename
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_filename)

        # Extract image ID to find corresponding mask
        img_id = img_filename.split(".")[0]  # Remove extension
        mask_filename = f"{img_id}_segmentation.png"
        mask_path = os.path.join(self.masks_dir, mask_filename)

        # Load image and mask
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {img_path} or mask {mask_path}: {e}")
            # Return a placeholder if image loading fails
            image = Image.new("RGB", (224, 224), color="black")
            mask = Image.new("L", (224, 224), color=0)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        return {"image": image, "mask": mask, "image_id": img_id}


# UNet Model Components
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Padding if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Reduce feature map sizes to save memory on Apple Silicon
        base_features = 32  # Reduced from 64

        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_features * 8, base_features * 16 // factor)
        self.up1 = Up(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = Up(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = Up(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear)
        self.outc = OutConv(base_features, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# Dice loss for better segmentation results
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1 - dice


# Get the appropriate device for Mac M4 Pro
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# Function to load data
def load_data(partition_id: int, num_partitions: int):
    """Load partitioned HAM10000 data for federated learning."""
    # Define paths - use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    ground_truth_path = os.path.join(data_dir, "GroundTruth.csv")

    print(f"Loading data from: {data_dir}")
    print(f"Partition ID: {partition_id}, Total partitions: {num_partitions}")

    # Try to load ground truth
    try:
        ground_truth_df = pd.read_csv(ground_truth_path)
        print(f"Ground truth loaded with {len(ground_truth_df)} rows")
    except FileNotFoundError:
        print(
            f"Ground truth file not found at {ground_truth_path}. Continuing without it."
        )
        ground_truth_df = None

    # Check if data directories exist
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at {images_dir}")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory not found at {masks_dir}")

    # Define transforms
    img_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),  # Reduce image size for M4 Pro
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    mask_transform = transforms.Compose(
        [transforms.Resize((160, 160)), transforms.ToTensor()]  # Matching image size
    )

    # Create full dataset
    try:
        full_dataset = HAM10000Dataset(
            images_dir,
            masks_dir,
            ground_truth_df,
            transform=img_transform,
            mask_transform=mask_transform,
        )
        print(f"Dataset created with {len(full_dataset)} samples")
    except Exception as e:
        raise RuntimeError(f"Error creating dataset: {str(e)}")

    # Divide dataset into partitions for federated learning
    n_samples = len(full_dataset)
    samples_per_partition = n_samples // num_partitions
    partition_lengths = [samples_per_partition] * (num_partitions - 1)
    partition_lengths.append(
        n_samples - sum(partition_lengths)
    )  # Ensure all samples used

    print(
        f"Creating {num_partitions} partitions with {samples_per_partition} samples each"
    )

    # Split dataset into partitions
    partitioned_datasets = torch.utils.data.random_split(
        full_dataset, partition_lengths, generator=torch.Generator().manual_seed(42)
    )

    # Get current partition
    current_partition = partitioned_datasets[partition_id]
    print(f"Partition {partition_id} has {len(current_partition)} samples")

    # Split partition into train/test sets
    train_size = int(0.8 * len(current_partition))
    test_size = len(current_partition) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        current_partition,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create data loaders with smaller batch size for M4 Pro
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=4, num_workers=2)

    print(
        f"Created train loader with {len(train_dataset)} samples and test loader with {len(test_dataset)} samples"
    )
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the UNet model on the training set."""
    print(f"Training for {epochs} epochs on {device}")
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)  # Reduced learning rate

    net.train()
    running_loss = 0.0
    batch_count = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_count_epoch = 0

        for batch_idx, batch in enumerate(trainloader):
            try:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = net(images)

                # Calculate loss
                bce_loss = criterion(outputs, masks)
                dice_loss = dice_criterion(outputs, masks)
                loss = bce_loss + dice_loss

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                # Update statistics
                epoch_loss += loss.item()
                batch_count_epoch += 1

                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(
                        f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(trainloader)}, Loss: {loss.item():.4f}"
                    )

                # Explicitly clear variables to free memory
                del images, masks, outputs, loss
                if device.type == "mps":
                    # Special handling for MPS
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # End of epoch
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, Avg Loss: {epoch_loss/batch_count_epoch:.4f}"
        )
        running_loss += epoch_loss
        batch_count += batch_count_epoch

    if batch_count > 0:
        avg_trainloss = running_loss / batch_count
    else:
        avg_trainloss = 0
    print(f"Training completed. Average loss: {avg_trainloss:.4f}")
    return avg_trainloss


def test(net, testloader, device):
    """Validate the UNet model on the test set."""
    print(f"Testing on {device}")
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss()

    val_loss = 0.0
    dice_score = 0.0
    batch_count = 0

    net.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            try:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                outputs = net(images)

                # Calculate losses
                bce_loss = criterion(outputs, masks)
                dice_loss = dice_criterion(outputs, masks)
                loss = bce_loss + dice_loss

                val_loss += loss.item()

                # Calculate Dice coefficient (F1 score)
                pred = torch.sigmoid(outputs) > 0.5
                pred = pred.float()
                intersection = (pred * masks).sum()
                dice = (2.0 * intersection) / (pred.sum() + masks.sum() + 1e-8)
                dice_score += dice.item()

                batch_count += 1

                # Print progress
                if (batch_idx + 1) % 5 == 0:
                    print(f"  Testing batch {batch_idx+1}/{len(testloader)}")

                # Explicitly clear variables to free memory
                del images, masks, outputs, pred
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in test batch {batch_idx}: {e}")
                continue

    if batch_count > 0:
        avg_val_loss = val_loss / batch_count
        avg_dice = dice_score / batch_count
    else:
        avg_val_loss = 0
        avg_dice = 0

    print(
        f"Testing completed. Validation loss: {avg_val_loss:.4f}, Dice score: {avg_dice:.4f}"
    )
    return avg_val_loss, avg_dice


def get_weights(net):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def save_model(net, path):
    """Save model to disk."""
    torch.save(net.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, model=None):
    """Load model from disk."""
    if model is None:
        model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(path))
    return model


# Main function to test everything works locally before federated learning
def main():
    # Select device
    device = get_device()

    # Create UNet model (with reduced size)
    model = UNet(n_channels=3, n_classes=1)
    print(f"Model created: {model.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Load data for a single partition (for testing)
    try:
        trainloader, testloader = load_data(partition_id=0, num_partitions=5)
        print(f"Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Train the model
    try:
        train_loss = train(model, trainloader, epochs=1, device=device)
        print(f"Training completed. Final loss: {train_loss:.4f}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test the model
    try:
        val_loss, dice_score = test(model, testloader, device)
        print(
            f"Testing completed. Validation loss: {val_loss:.4f}, Dice score: {dice_score:.4f}"
        )
    except Exception as e:
        print(f"Error during testing: {e}")
        return

    # Save the model
    try:
        save_model(model, "unet_ham10000.pth")
    except Exception as e:
        print(f"Error saving model: {e}")


if __name__ == "__main__":
    main()
