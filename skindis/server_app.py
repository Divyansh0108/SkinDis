"""fedapp: A Flower / PyTorch app."""

import os
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from skindis.task import UNet, get_weights

script_dir = os.path.dirname(os.path.abspath(__file__))
global_log_file = os.path.join(script_dir, "global_metrics.log")


def server_fn(context: Context):
    """Defines the server-side logic for federated learning."""
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    ndarrays = get_weights(UNet())
    parameters = ndarrays_to_parameters(ndarrays)

    with open(global_log_file, "w") as f:
        f.write(
            "Round,Training Loss,Training Accuracy,Testing Loss,Testing Accuracy,Testing IoU,Testing DiceCoeff,Testing DiceLoss\n"
        )

    class LoggingFedAvg(FedAvg):
        """Custom FedAvg strategy with logging capabilities."""

        def aggregate_fit(self, rnd, results, failures):
            aggregated_metrics = super().aggregate_fit(rnd, results, failures)
            if aggregated_metrics:
                print(
                    f"Round {rnd} - Training Loss: {aggregated_metrics.get('loss', 'N/A')}, "
                    f"Training Accuracy: {aggregated_metrics.get('accuracy', 'N/A')}"
                )
            return aggregated_metrics

        def aggregate_evaluate(self, rnd, results, failures):
            aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
            if aggregated_metrics:
                testing_loss = aggregated_metrics.get("loss", "N/A")
                testing_accuracy = aggregated_metrics.get("accuracy", "N/A")
                testing_iou = aggregated_metrics.get("iou", "N/A")
                testing_dice_coeff = aggregated_metrics.get("dice_coeff", "N/A")
                testing_dice_loss = aggregated_metrics.get("dice_loss", "N/A")
                print(
                    f"Round {rnd} - Testing Loss: {testing_loss}, "
                    f"Testing Accuracy: {testing_accuracy}, "
                    f"Testing IoU: {testing_iou}, "
                    f"Testing DiceCoeff: {testing_dice_coeff}, "
                    f"Testing DiceLoss: {testing_dice_loss}"
                )
                with open(global_log_file, "a") as f:
                    f.write(
                        f"{rnd},{aggregated_metrics.get('loss', 'N/A')},{aggregated_metrics.get('accuracy', 'N/A')},"
                        f"{testing_loss},{testing_accuracy},{testing_iou},{testing_dice_coeff},{testing_dice_loss}\n"
                    )
            return aggregated_metrics

    strategy = LoggingFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
