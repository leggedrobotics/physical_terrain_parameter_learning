#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import yaml
import datetime
import matplotlib.pyplot as plt
from neptune.types import File
from pytorch_lightning.loggers.neptune import NeptuneLogger
from typing import Optional, Tuple, List, Dict, Any
from training_utils.run_config import ParamCollection


class BinAccuracy:
    def __init__(
        self,
        ord_num: int,
        output_type: str,
        discretization: str = "SID",
        shift_SID: float = 0.01,
    ):
        self.ord_num = ord_num
        self.output_type = output_type
        self.discretization = discretization
        self.shift = shift_SID
        if self.output_type == "fric":
            self.alpha, self.beta = 0, 1
        elif self.output_type == "stiff":
            self.alpha, self.beta = 1, 10
        else:
            raise ValueError("Unknown output type")

    def _get_bin(self, gt: torch.Tensor) -> torch.Tensor:
        if self.discretization == "SID":
            shift = torch.tensor(self.shift).to(gt.device)
            alpha = torch.tensor(self.alpha + self.shift).to(gt.device)
            beta = torch.tensor(self.beta + self.shift).to(gt.device)
            gt = gt + shift
            bin = torch.log(gt / alpha) / torch.log(beta / alpha) * self.ord_num
        elif self.discretization == "UD":
            bin = (gt - self.alpha) / (self.beta - self.alpha) * self.ord_num
        else:
            raise ValueError(f"Unknown discretization method: {self.discretization}")
        bin = bin.clamp(max=self.ord_num - 1, min=0)
        bin = bin.long()
        return bin

    def calculate_accuracy(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[List[float], torch.Tensor]:
        pred_bin = self._get_bin(pred)
        gt_bin = self._get_bin(gt)
        pred_bin = pred_bin.to(gt.device)

        accuracies = []
        # Initialize confusion matrix
        confusion_matrix = torch.zeros(self.ord_num, self.ord_num, dtype=torch.int)
        for i in range(self.ord_num):
            if self.discretization == "UD":
                bin_start = self.alpha + i * (self.beta - self.alpha) / self.ord_num
                bin_end = self.alpha + (i + 1) * (self.beta - self.alpha) / self.ord_num
            else:  # SID
                shift = torch.tensor(self.shift).to(gt.device)
                shifted_alpha = torch.tensor(self.alpha + self.shift).to(gt.device)
                shifted_beta = torch.tensor(self.beta + self.shift).to(gt.device)
                bin_start = (
                    torch.exp(
                        torch.log(shifted_alpha)
                        + torch.log(shifted_beta / shifted_alpha) * i / self.ord_num
                    )
                    - shift
                )
                bin_end = (
                    torch.exp(
                        torch.log(shifted_alpha)
                        + torch.log(shifted_beta / shifted_alpha)
                        * (i + 1)
                        / self.ord_num
                    )
                    - shift
                )

            correct = (pred_bin == i) & (gt_bin == i)
            total = gt_bin == i

            # Handle empty bins
            if total.float().sum() == 0:
                accuracies.append(float("nan"))
            else:
                accuracy = correct.float().sum() / total.float().sum()
                accuracies.append(accuracy.item())
            # Extract the predictions corresponding to the current gt bin
            mask = gt_bin == i
            binned_preds_for_current_gt_bin = pred_bin[mask]

            # Update the confusion matrix
            for j in range(self.ord_num):
                confusion_matrix[i, j] = (binned_preds_for_current_gt_bin == j).sum()
            print(
                f"Bin {i}: Range [{bin_start:.2f}, {bin_end:.2f}] - Accuracy: {accuracies[-1]*100:.2f}%"
            )
        # Display the confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix)
        return accuracies, confusion_matrix


def create_bin_accuracy_calculator(output_type: str) -> BinAccuracy:
    if output_type == "fric":
        discretization = "SID"
        ord_num = 5
        shift = 0.2
    else:
        discretization = "SID"
        ord_num = 5
        shift = 1.0
    bin_accuracy_cal = BinAccuracy(
        ord_num,
        output_type=output_type,
        discretization=discretization,
        shift_SID=shift,
    )
    return bin_accuracy_cal


def save_model(
    decoder: nn.Module,
    output_type: str,
    input_type: int,
    hyperparams: Dict[str, Any],
):
    """Save the model to a folder."""
    architecture_tag = decoder.config["mode"]
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    filename = f"model_{output_type}_{architecture_tag}_{input_type}_{current_time}.pth".replace(
        "__", "_"
    )

    folder_name = "models"
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist
    path_to_save = os.path.join(folder_name, filename)
    decoder.save(path_to_save)
    print(f"Model saved to {path_to_save}")

    if hyperparams:
        # Save hyperparameters to a .txt file
        hyperparams_filename = filename.replace(".pth", "_hyperparams.yaml")
        path_to_save_hyperparams = os.path.join(folder_name, hyperparams_filename)
        with open(path_to_save_hyperparams, "w") as f:
            yaml.dump(hyperparams, f)
        print(f"Hyperparameters saved to {path_to_save_hyperparams}")
    return current_time


def batch_eval(
    decoder: nn.Module,
    val_loader: DataLoader,
    logger: NeptuneLogger,
    timestamp: str,
    params: ParamCollection,
) -> None:
    output_type = params.model.output_type
    rnn_mode = params.model.mode
    input_type = params.model.input_type
    # Collect errors and plot histogram for model
    true_values, errors = collect_errors(decoder, val_loader)
    vis_data = VisData(
        true_values=true_values,
        errors=errors,
        interval=0.1 if output_type == "fric" else 1,
        min_range=0 if output_type == "fric" else 1,
        max_range=1 if output_type == "fric" else 10,
        output_type=output_type,
        input_type=input_type,
        timestamp=timestamp,
        rnn_mode=rnn_mode,
        logger=logger,
    )
    plot_error_histogram(vis_data)


def single_env_eval(
    decoder: nn.Module,
    single_env_val_loader: DataLoader,
    logger: NeptuneLogger,
    params: ParamCollection,
) -> None:
    output_type = params.model.output_type
    env_id = params.general.val_test_env_id

    true_values_list, predictions_list = [], []

    all_predictions = []
    all_true_values = []
    with torch.no_grad():
        for batch_inputs, batch_true_values in single_env_val_loader:
            batch_inputs = batch_inputs.to(params.model.device)
            predictions = decoder.predict(batch_inputs)

            batch_true_values = batch_true_values[:, -1, :].unsqueeze(1)
            predictions = predictions[:, -1, :].unsqueeze(1)

            all_predictions.append(predictions)
            all_true_values.append(batch_true_values)
            true_values_list.append(batch_true_values.squeeze(0).cpu().numpy())
            predictions_list.append(predictions.squeeze(0).cpu().numpy())

    # Convert list of arrays to single arrays
    true_values = np.concatenate(true_values_list, axis=0)
    predictions = np.concatenate(predictions_list, axis=0)

    #  Convert list of tensors to a single tensor
    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_values = torch.cat(all_true_values, dim=0)

    bin_accuracy_calculator = create_bin_accuracy_calculator(output_type)
    accuracies, confusion_matrix = bin_accuracy_calculator.calculate_accuracy(
        all_predictions, all_true_values
    )
    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average accuracy: {average_accuracy*100:.2f}%")
    metrics = {
        "Average accuracy": average_accuracy,
        "Confusion Matrix": str(confusion_matrix.numpy()),
        **{f"Accuracy for bin {idx}": acc for idx, acc in enumerate(accuracies)},
    }

    logger.log_metrics(metrics)

    # Plotting each dimension in separate subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    for i in range(4):
        axs[i].plot(true_values[:, i], label=f"True {output_type} values", color="blue")
        axs[i].plot(
            predictions[:, i],
            label=f"Predicted {output_type} values",
            color="red",
            linestyle="--",
        )
        axs[i].legend()
        axs[i].set_title(
            f"Dimension {i+1} - True vs Predicted {output_type} values for env_id {env_id}"
        )
    plt.tight_layout()
    plt.close()


def collect_errors(
    decoder: nn.Module, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    all_errors = []
    all_true_values = []
    test_last = True
    device = decoder.device
    for data in loader:
        input, label = data
        input = input.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = decoder.predict(input)

        if test_last:
            label = label[:, -1, :].unsqueeze(1)
            pred = pred[:, -1, :].unsqueeze(1)
        errors = np.abs(
            pred.cpu().numpy() - label.cpu().numpy()
        ).flatten()  # Flatten here
        all_errors.append(errors)
        all_true_values.append(label.cpu().numpy().flatten())  # Flatten here

    return np.hstack(all_true_values), np.hstack(all_errors)  # Stack horizontally


@dataclass
class VisData:
    true_values: np.ndarray
    errors: np.ndarray
    interval: float
    min_range: float
    max_range: float
    output_type: str
    input_type: int
    timestamp: str
    rnn_mode: str
    logger: Optional[NeptuneLogger] = None


def compute_mean_errors_per_bin(
    vis_data: VisData,
) -> Tuple[np.ndarray, List[float], List[float], List[List[float]]]:
    true_values = vis_data.true_values
    errors = vis_data.errors
    interval = vis_data.interval
    min_range = vis_data.min_range
    max_range = vis_data.max_range
    bins = np.arange(min_range, max_range + interval, interval)
    bin_means = []
    bin_stddevs = []
    errors_per_bin = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:  # If it's the last bin
            indices = np.logical_and(true_values >= bins[i], true_values <= bins[i + 1])
        else:
            indices = np.logical_and(true_values >= bins[i], true_values < bins[i + 1])

        bin_error = errors[indices].mean()
        bin_stddev = errors[indices].std()  # Compute standard deviation
        bin_means.append(bin_error)
        bin_stddevs.append(bin_stddev)
        errors_per_bin.append(errors[indices].tolist())

    return bins[:-1], bin_means, bin_stddevs, errors_per_bin


def plot_error_histogram(vis_data: VisData) -> None:
    interval = vis_data.interval
    output_type = vis_data.output_type
    input_type = vis_data.input_type
    timestamp = vis_data.timestamp
    logger = vis_data.logger
    rnn_mode = vis_data.rnn_mode
    errors = vis_data.errors

    bins, bin_means, bin_stddevs, bin_errors = compute_mean_errors_per_bin(vis_data)
    # Define colors for each output type
    colors = {"fric": "blue", "stiff": "green"}
    bars = plt.bar(
        bins,
        bin_means,
        width=interval,
        align="edge",
        color=colors[output_type],
        yerr=bin_stddevs,
        capsize=5,
    )

    # Annotating each bar with its height (value)
    for bar, stddev in zip(bars, bin_stddevs):
        yval = bar.get_height()
        rounded_stddev_str = "{:.2f}".format(stddev)
        plt.text(
            bar.get_x() + bar.get_width() / 2 - 0.2 * interval,
            yval,
            f"{round(yval, 2)} (+/- {rounded_stddev_str})",
            ha="center",
            va="bottom",
            rotation=90,
        )
    # Compute the total error mean and standard deviation and add to the plot
    total_error_mean = errors.mean()
    total_error_std = errors.std()
    plt.annotate(
        f"Total Error Mean: {total_error_mean:.3f}\nTotal Error Std: {total_error_std:.3f}",
        xy=(0.05, 0.9),
        xycoords="axes fraction",
    )

    plt.xlabel("True Value Range")
    plt.ylabel("Mean Prediction Error")
    plt.title(f"Mean Prediction Error for {output_type}")
    # Set y-axis range based on output_type
    if output_type == "stiff":
        plt.ylim([0, 10])
    elif output_type == "fric":
        plt.ylim([0, 1])
    # Save the plot
    current_time = timestamp
    folder_name = "models/histograms"
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist
    filename = (
        f"histogram_{output_type}_{rnn_mode}_{input_type}_{current_time}.png".replace(
            "__", "_"
        )
    )

    path_to_save = os.path.join(folder_name, filename)
    plt.savefig(path_to_save)
    if logger is not None:
        # Log to Neptune using filename as the channel name
        channel_name = f"histograms/{filename}"
        logger.experiment[channel_name].append(File(path_to_save))
    print(f"Histogram saved to {path_to_save}")
    plt.clf()
