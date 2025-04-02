import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth(values, weight=0.8):
    """Applies exponential smoothing to a list of values."""
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_value = last * weight + (1 - weight) * value
        smoothed.append(smoothed_value)
        last = smoothed_value
    return smoothed


def plot_tensorboard_logs(log_dir, output_dir, prefix):
    """Plots metrics from TensorBoard logs and saves them as images."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' does not exist.")
        return

    event_acc = EventAccumulator(log_dir)
    try:
        event_acc.Reload()
    except Exception as e:
        print(f"Error loading logs from '{log_dir}': {e}")
        return

    available_keys = event_acc.Tags().get("scalars", [])
    print(f"Available keys in '{log_dir}': {available_keys}")

    metrics = ["Loss", "Accuracy", "IoU", "DiceCoeff", "DiceLoss"]

    for metric in metrics:
        key = f"{prefix}/{metric}"
        if key not in available_keys:
            print(f"Metric '{metric}' not found in TensorBoard logs under key '{key}'.")
            continue

        try:
            scalar_events = event_acc.Scalars(key)
            values = [event.value for event in scalar_events]
            smoothed_values = smooth(values)
            steps = list(range(len(smoothed_values)))

            plt.figure()
            plt.plot(steps, smoothed_values, label=f"{prefix} {metric}")
            plt.xlabel("Rounds")
            plt.ylabel(metric)
            plt.title(f"{prefix} {metric} (Smoothed)")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(
                    output_dir, f"{prefix.lower()}_{metric.lower()}_smoothed.png"
                )
            )
            plt.close()
        except KeyError:
            print(f"Metric '{metric}' not found in TensorBoard logs.")


if __name__ == "__main__":
    """Main script for plotting training and testing metrics from TensorBoard logs."""
    training_log_dir = "runs/training"
    testing_log_dir = "runs/testing"
    output_dir = "plots"

    training_dir = os.path.join(output_dir, "training")
    testing_dir = os.path.join(output_dir, "testing")
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)

    print("Plotting training metrics...")
    plot_tensorboard_logs(training_log_dir, training_dir, prefix="Training")

    print("Plotting testing metrics...")
    plot_tensorboard_logs(testing_log_dir, testing_dir, prefix="Testing")

    print(f"Plots saved in '{output_dir}' directory.")
