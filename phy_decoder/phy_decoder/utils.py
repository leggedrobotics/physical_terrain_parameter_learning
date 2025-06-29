#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import os
import torch.nn as nn
import torch
import fnmatch
from datetime import datetime
from phy_decoder.decoder import ModelFactory
from phy_decoder.decoder_config import ModelParams

from typing import Dict, Any


def get_latest_file_in_directory(directory: str, pattern: str = "*.h5") -> str:
    """
    Returns the path to the latest (most recent) file
    of the joined path(s) or None if no file found.
    """
    try:
        # List all files in directory that match the given pattern
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and fnmatch.fnmatch(f, pattern)
        ]

        # # Get latest file
        # latest_file = max(files, key=os.path.getctime)
        # Parse the date from each filename and associate it with the file path
        date_file_pairs = []
        for file in files:
            # Extract the timestamp part of the filename
            basename = os.path.basename(file)
            timestamp_str = (
                basename.split("_")[-4]
                + basename.split("_")[-3]
                + basename.split("_")[-2]
                + basename.split("_")[-1][:-4]
            )  # YYYYMMDDHHMM
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
            date_file_pairs.append((timestamp, file))

        # Get the file with the latest date
        _, latest_file = max(date_file_pairs, key=lambda x: x[0])

        # Print the latest file name
        print(f"Latest file in {directory}: {os.path.basename(latest_file)}")

        return latest_file
    except ValueError as e:
        print(f"No files found in {directory} matching pattern {pattern}.")
        return None


class RNNInputBuffer:
    def __init__(self):
        self.capacity = CONFIG["seq_length"] - 1
        self.buffer = []

    def add(self, data):
        if len(self.buffer) >= self.capacity:
            if self.capacity > 0:
                self.buffer.pop(
                    0
                )  # remove the oldest data only if capacity is more than 0
        self.buffer.append(data)

    def get_last(self, n):
        return self.buffer[-n:]  # get the last n elements


def construct_search_pattern(rnn_mode: str, input_type: str, output_type: str) -> str:
    conditions = []

    # Check model type (RNN)
    conditions.append("RNN")
    conditions.append(rnn_mode)

    # Check input width
    input_width_mapping = {
        "pro+exte": "InputWidth341",
        "pro": "InputWidth133",
        "all": "InputWidth441",
    }
    conditions.append(input_width_mapping.get(input_type, ""))

    # Construct search pattern
    search_pattern = f"*{output_type}*"
    for condition in conditions:
        search_pattern += f"*{condition}*"
    search_pattern += ".pth"

    return search_pattern


def get_past_timesteps(buffer, n):
    return torch.cat(buffer.get_last(n), dim=0)


def prepare_padded_input(input_data, input_buffers, i, env_num):
    """
    Prepare the padded input for RNN decoders.

    Parameters:
    - input_data: The current timestep data
    - input_buffers: Buffers holding past timesteps data
    - i: Current iteration of the main loop
    - env_num: Number of environment instances

    Returns:
    - padded_inputs: A list containing padded input data for each environment instance
    """
    seq_length = CONFIG["seq_length"]
    if i == 0:
        # print error, i must start from 1
        raise ValueError("i must start from 1")
    padded_inputs = []
    for env_id in range(env_num):
        single_input_data = input_data[env_id].unsqueeze(0).to("cuda:0")

        # Calculate how many timesteps of past data we can fetch and how much padding is needed
        available_past_data = min(i - 1, seq_length - 1)
        required_padding = seq_length - 1 - available_past_data

        # Fetch the past data
        if available_past_data > 0:
            past_data = get_past_timesteps(
                input_buffers[env_id], available_past_data
            ).to("cuda:0")
        else:
            past_data = torch.empty(0, *single_input_data.shape[1:]).to("cuda:0")

        # Create the padding
        if required_padding > 0:
            padding = torch.zeros(required_padding, *single_input_data.shape[1:]).to(
                "cuda:0"
            )
            # print(padding.device,past_data.device,single_input_data.device)
            padded_data = torch.cat([padding, past_data, single_input_data], dim=0)

        else:
            padded_data = torch.cat([past_data, single_input_data], dim=0)

        padded_inputs.append(padded_data)

    return padded_inputs


def load_decoder(model_args: Dict[str, Any], output_type: str) -> nn.Module:
    # Get the directory of the current file
    current_directory = os.path.dirname(__file__)
    models_path = os.path.join(current_directory, "models")
    search_pattern = construct_search_pattern(
        model_args["mode"], model_args["input_type"], output_type
    )
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


def load_fric_stiff_decoders():
    """
    Load the models for the fric and stiff predictors
    """
    model_args = ModelParams().to_dict()
    model_args["is_training"] = False

    fric_predictor = load_decoder(model_args, "fric")
    stiff_predictor = load_decoder(model_args, "stiff")

    return fric_predictor, stiff_predictor, model_args


if __name__ == "__main__":
    fric_predictor, stiff_predictor, model_args = load_fric_stiff_decoders()
    print(model_args)
    print("done")
