import os
import matplotlib.pyplot as plt

rounds = [1, 2, 3]
loss = [0.32985, 0.23270, 0.18299]
dice_loss = [0.41274, 0.29212, 0.22528]
accuracy = [0.87799, 0.90855, 0.92268]
iou = [0.63275, 0.73982, 0.78926]
dice_coeff = [0.77209, 0.84998, 0.88114]

output_folder = "/Volumes/Projects/skindis/global_test_plots"
os.makedirs(output_folder, exist_ok=True)

metrics = {
    "Loss": loss,
    "Dice Loss": dice_loss,
    "Accuracy": accuracy,
    "IoU": iou,
    "Dice Coeff": dice_coeff,
}

for metric_name, metric_values in metrics.items():
    plt.figure()
    plt.plot(rounds, metric_values, marker="o", label=metric_name)
    plt.xlabel("Rounds")
    plt.ylabel(metric_name)
    plt.title(f"Global {metric_name} Over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_folder, f"{metric_name.lower().replace(' ', '_')}.png")
    )
    plt.close()

print(f"Plots saved in '{output_folder}'")
