#                                                                               
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#                                                                               
import sys
import os
import torch.nn as nn
import torch
import numpy as np
import re
import fnmatch
from datetime import datetime
from .decoder import BeliefDecoderLightning
from .decoder_config import config

def extract_timestamp_from_filename(filename):
    # Assuming the filename is in the format: YYYY_MM_DD_HHMM__dataset...h5
    pattern = re.compile(r"(\d{4}_\d{2}_\d{2}_\d{4})")
    match = pattern.search(filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract timestamp from filename: {filename}")

def create_subfolder(directory, timestamp):
    subfolder_path = os.path.join(directory, timestamp + "__analysis")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    return subfolder_path


def get_latest_file_in_directory(directory, pattern='*.h5', conditions=[]):
    """
    Returns the path to the latest (most recent) file 
    of the joined path(s) or None if no file found.
    """
    try:
        # Build the full pattern incorporating conditions
        for condition in conditions:
            pattern = f"*{condition}*{pattern}"

        # List all files in directory that match the given pattern
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                 if os.path.isfile(os.path.join(directory, f)) 
                 and fnmatch.fnmatch(f, pattern)]
        
        # # Get latest file
        # latest_file = max(files, key=os.path.getctime)
        # Parse the date from each filename and associate it with the file path
        date_file_pairs = []
        for file in files:
            # Extract the timestamp part of the filename
            basename = os.path.basename(file)
            timestamp_str = basename.split('_')[-4]+basename.split('_')[-3]+basename.split('_')[-2] + basename.split('_')[-1][:-4]  # YYYYMMDDHHMM
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


def compute_episode_statistics(data):
    # Assuming 'data' is a dictionary with keys as env_ids and values as nested dictionaries of sensor data

    episode_lengths_per_env = {}

    for env_id, sensors in data.items():
        counter_data = sensors['counter']
        total_length=len(counter_data)
        # Find the indices where the counter increments
        increment_indices = (counter_data[1:] - counter_data[:-1]).nonzero(as_tuple=True)[0] + 1
        # Compute episode lengths
        if increment_indices.numel() == 0:  # Handle the case when there are no increments
            episode_lengths = [len(counter_data)]
        else:
            episode_lengths = [increment_indices[0].item()] + (increment_indices[1:] - increment_indices[:-1]).tolist()
        episode_lengths_per_env[env_id] = episode_lengths

    # Flatten episode lengths across all envs to compute global statistics
    all_episode_lengths = [length for lengths in episode_lengths_per_env.values() for length in lengths]

    mean_length = np.mean(all_episode_lengths)
    std_length = np.std(all_episode_lengths)
    # Print the computed statistics
    print("\nGlobal Episode Statistics:")
    print(f"Total Simulation Length: {total_length}")
    print(f"Mean Episode Length: {mean_length}")
    print(f"Std Dev of Episode Length: {std_length}")
    
    print("\nEpisode Lengths Per Robot: (excluding the unfinished ones)")
    for env_id, lengths in episode_lengths_per_env.items():
        print(f"Robot {env_id}: {lengths}")
    return mean_length, std_length, episode_lengths_per_env

class NoDataFileError(Exception):
    pass

class RNNInputBuffer:
    def __init__(self):
        self.capacity = config['seq_length'] - 1
        self.buffer = []
    
    def add(self, data):
        if len(self.buffer) >= self.capacity:
            if self.capacity > 0:
                self.buffer.pop(0)  # remove the oldest data only if capacity is more than 0
        self.buffer.append(data)
    
    def get_last(self, n):
        return self.buffer[-n:]  # get the last n elements

def construct_search_pattern(use_rnn, rnn_mode, input_type, use_weight, model_name,paradigm=None):
    conditions = []

    # Check model type (RNN or MLP)
    if use_rnn:
        conditions.append("RNN")
        conditions.append(rnn_mode)
    else:
        conditions.append("MLP")

    # Check input width
    input_width_mapping = {
        "pro+exte": "InputWidth341",
        "hidden": "InputWidth100",
        "all": "InputWidth441"
    }
    conditions.append(input_width_mapping.get(input_type, ""))

    # Check if weighted
    if use_weight:
        conditions.append("weighted")
    if paradigm=="ordinal":
        conditions.append("ordinal")
    # Construct search pattern
    search_pattern = f"*{model_name}*"
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
    - seq_length: Desired sequence length
    - i: Current iteration of the main loop
    - env_num: Number of environment instances

    Returns:
    - padded_inputs: A list containing padded input data for each environment instance
    """
    subtract_mode = False
    seq_length=config['seq_length']
    if i==0:
        # print error, i must start from 1
        raise ValueError("i must start from 1")
    padded_inputs = []
    # print("shape of inputbuffer:",len(input_buffers[0].buffer))
    for env_id in range(env_num):
        single_input_data = input_data[env_id].unsqueeze(0).to('cuda:0')

        # Calculate how many timesteps of past data we can fetch and how much padding is needed
        available_past_data = min(i - 1, seq_length - 1)
        required_padding = seq_length - 1 - available_past_data

        # Fetch the past data
        if available_past_data > 0:
            past_data = get_past_timesteps(input_buffers[env_id], available_past_data).to('cuda:0')
        else:
            past_data = torch.empty(0, *single_input_data.shape[1:]).to('cuda:0')

        # Create the padding
        if required_padding > 0:
            padding = torch.zeros(required_padding, *single_input_data.shape[1:]).to('cuda:0')
            # print(padding.device,past_data.device,single_input_data.device)
            padded_data = torch.cat([padding, past_data, single_input_data], dim=0)
            
        else:
            padded_data = torch.cat([past_data, single_input_data], dim=0)

        if subtract_mode:
            # check if the last dim is 341 (pro+exte), raise error if not
            if padded_data.shape[-1] != 341:
                raise ValueError("The last dimension of the input data is not 341!")
            exte_dims = padded_data[0, -208:]
            padded_data[:,-208:]-=exte_dims

        padded_inputs.append(padded_data)

    return padded_inputs

def initialize_models():
        """ 
         Initialize the models for the fric and stiff predictors
           """
        # Get the directory of the current file
        current_directory = os.path.dirname(__file__)
        models_path = os.path.join(current_directory, "models")
        use_rnn = config["use_rnn"]
        seq_length = config["seq_length"]
        reset_hidden_each_epoch = config['reset_hidden_each_epoch']
        rnn_mode = config['rnn_mode']
        use_weight = config['use_weight']
        SEPARATE_DATASET_MODE = config['SEPARATE_DATASET_MODE']
        input_type = config['input_type']
        paradigm = config['paradigm']
        if use_rnn:
            sens_size = 341  # Extracted from the latent
            inner_state_size = 0  # Extracted from the latent
            model_args = {
                "sens_size": sens_size,
                "inner_state_size": inner_state_size,
                "priv_size": 4,
                "seq_length":seq_length,
                "hidden_to_pred_shape": [64] if paradigm=="ordinal" else [64, 32],
                "hidden_to_gate_shape": [64, 64],
                "gate_to_pred_shape": [64, 32],
                "hidden_size": 100,
                "num_rnn_layers": 1,
                "l1_alpha": 0.2,
                "activation_fn": nn.LeakyReLU,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "mode": rnn_mode,  # Or whatever mode you desire
                "paradigm":paradigm,
            }
        else:
            model_args = {
                "priv_size": 4,
                "priv_decoder_shape": [64, 32],
                "latent_size": 441,
                "l1_alpha": 0.2,
                "activation_fn": nn.LeakyReLU,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        # parallel mode and reset_hidden_each_epoch are not compatible
        if rnn_mode=="parallel" and reset_hidden_each_epoch is False:
            raise ValueError("parallel mode and using continued hidden states between sequences are not compatible")
        model_args["is_training"]=False
        search_pattern_fric = construct_search_pattern(use_rnn, rnn_mode, input_type, use_weight, "fric")
        search_pattern_stiff = construct_search_pattern(use_rnn, rnn_mode, input_type, use_weight, "stiff")
        latest_fric_model_file = get_latest_file_in_directory(models_path, pattern=search_pattern_fric)
        if latest_fric_model_file:
            model_fric = BeliefDecoderLightning(model_args, model_name="fric",use_rnn=use_rnn,reset_hidden_each_epoch=reset_hidden_each_epoch)
            model_fric.load_state_dict(torch.load(latest_fric_model_file))
            model_fric=model_fric.to('cuda:0')
            model_fric.eval()
            fric_predictor=model_fric.decoder
            print("fric_predictor loaded")
        latest_stiff_model_file = get_latest_file_in_directory(models_path, pattern=search_pattern_stiff)
        if latest_stiff_model_file:
            model_stiff = BeliefDecoderLightning(model_args, model_name="stiff",use_rnn=use_rnn,reset_hidden_each_epoch=reset_hidden_each_epoch)
            model_stiff.load_state_dict(torch.load(latest_stiff_model_file))
            model_stiff=model_stiff.to('cuda:0')
            model_stiff.eval()
            stiff_predictor=model_stiff.decoder
            print("stiff_predictor loaded")
        return fric_predictor,stiff_predictor,config

if __name__=="__main__":
    fric_predictor,stiff_predictor=initialize_models()
    print("done")