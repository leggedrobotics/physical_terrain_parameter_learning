import torch
import torch.nn as nn
import os
from typing import Dict, Any, Tuple
from physical_decoder.decoder import ModelFactory
from physical_decoder.decoder_config import ModelParams
from physical_decoder.utils import (
    get_latest_file_in_directory,
    construct_search_pattern,
)


# During deployment, received input is per-step not a complete sequence as in training and batch size is 1.
class DeploymentWrapper:
    def __init__(self):
        self.fric_predictor, self.stiff_predictor, self.model_args = (
            load_fric_stiff_decoders_deploying()
        )
        self.input_seq_builder = RNNInputSequenceBuilder(self.model_args)

    def predict(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts the friction and stiffness parameters based on the input data (raw, unsliced).

        Args:
            data (torch.Tensor): shape (batch_size, feature_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - fric_pred (torch.Tensor): shape (batch_size, priv_size)
                - stiff_pred (torch.Tensor): shape (batch_size, priv_size)
        """
        input = self.input_seq_builder(data)
        fric_pred_seq = self.fric_predictor.predict(input)
        stiff_pred_seq = self.stiff_predictor.predict(input)
        # Take the last prediction (current step) in the sequence
        fric_pred = fric_pred_seq[:, -1, :]
        stiff_pred = stiff_pred_seq[:, -1, :]
        return fric_pred, stiff_pred


class RNNInputSequenceBuilder:
    def __init__(self, model_args: Dict[str, Any]):
        self.device = model_args["device"]
        self.seq_length = model_args["seq_length"]
        self.buffer = None

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Creates a sequence of inputs for the RNN by maintaining a buffer of the last `seq_length` inputs.

        Args:
            data (torch.Tensor): shape (batch_size, feature_dim)

        Returns:
            torch.Tensor: shape (batch_size, seq_length, feature_dim)
        """
        data = data.to(self.device)  # Ensure data is on the correct device
        self.add(data)
        return self.buffer

    def add(self, data: torch.Tensor) -> None:
        """
        Args:
            data (torch.Tensor): shape (batch_size, feature_dim)
        """
        batch_size, feature_dim = data.shape

        if self.buffer is None:
            self.buffer = torch.zeros(
                (batch_size, self.seq_length, feature_dim), device=self.device
            )  # Initialize buffer with the first data point, shape (batch_size, seq_length, feature_dim)

        # Shift buffer to the left by 1 timestep (drop the oldest)
        self.buffer[:, :-1, :] = self.buffer[:, 1:, :].clone()

        # Append new data at the last time step
        self.buffer[:, -1, :] = data


def load_decoder_deploying(model_args: Dict[str, Any], output_type: str) -> nn.Module:
    # Get the directory of the current file
    current_directory = os.path.dirname(__file__)
    models_path = os.path.join(current_directory, "models")
    # override output_type in model_args
    model_args["output_type"] = output_type
    search_pattern = construct_search_pattern(model_args)
    latest_model_file = get_latest_file_in_directory(
        models_path, pattern=search_pattern
    )
    if latest_model_file:
        model = ModelFactory.create_model(config=model_args)
        model.load(latest_model_file)
        model.eval()
        print(f"{output_type} decoder loaded from {latest_model_file}")
        return model
    else:
        raise ValueError(
            f"No '{output_type}' decoder found in the specified directory: {models_path}"
        )


def load_fric_stiff_decoders_deploying():
    """
    Load the models for the fric and stiff predictors
    """
    model_args = ModelParams().to_dict()
    model_args["is_training"] = False

    fric_predictor = load_decoder_deploying(model_args, "fric")
    stiff_predictor = load_decoder_deploying(model_args, "stiff")

    return fric_predictor, stiff_predictor, model_args


if __name__ == "__main__":
    fric_predictor, stiff_predictor, model_args = load_fric_stiff_decoders_deploying()
    print(model_args)
    print("done")
