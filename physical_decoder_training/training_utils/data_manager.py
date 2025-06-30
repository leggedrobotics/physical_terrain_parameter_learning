#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
from __future__ import annotations
import deepdish as dd
import torch
import datetime
import os
import re
import gc
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional
from training_utils.run_config import ParamCollection


class DataManager:
    def __init__(self, param: ParamCollection):
        self.param = param
        self.train_loader = None
        self.label_weights = None
        self.label_weighting_bin_interval = None
        if self.param.general.mode == "train":
            self.train_loader = self._load_train_data()
            self.label_weights = self._compute_label_weights(self.train_loader.dataset)
        self.val_loader, self.single_env_val_loader = self._load_val_data()

    def _load_and_organize_data(
        self, directory: str
    ) -> Dict[int, List[Dict[str, torch.Tensor]]]:
        """
        Organize the data for training.
        Returns:
            {env_id: List[{combined_input/output_key: Tensor(sequence_length, feat_dim)}]}
        """
        data = combine_datasets_from_directory(directory, self.param.model.output_type)
        organized_data = organize_data(
            data,
            sequence_length=self.param.model.seq_length,
            overlapp=self.param.general.overlapp,
        )
        data = None  # Free up memory
        gc.collect()
        torch.cuda.empty_cache()
        return organized_data

    def _get_dataloader_for_env_id(
        self,
        organized_data: Dict[int, List[Dict[str, torch.Tensor]]],
    ) -> DataLoader:
        env_id = self.param.general.val_test_env_id
        # Extract data for the specific env_id
        env_id_data = {env_id: organized_data[env_id]}

        # Create a custom dataset using the RNNDataset class
        dataset = RNNDataset(
            env_id_data,
            param=self.param,
        )  # dataset.data: List[(InputTensor(sequence_length, feature_dim), LabelTensor(sequence_length, priv_size))]

        # Create a DataLoader for the custom dataset
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        return data_loader

    def _load_train_data(self) -> DataLoader:
        train_data_directory = os.path.expanduser(
            self.param.general.train_data_directory
        )
        organized_data = self._load_and_organize_data(train_data_directory)
        train_dataset = RNNDataset(
            organized_data,
            param=self.param,
        )
        organized_data = None  # Free up memory
        gc.collect()
        torch.cuda.empty_cache()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.param.general.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return train_loader

    def _compute_label_weights(self, train_dataset: RNNDataset) -> torch.Tensor:
        inputs, labels = zip(*train_dataset.data)
        if self.param.model.output_type == "fric":
            interval = 0.1
            max_range = 1
            min_range = 0
        elif self.param.model.output_type == "stiff":
            interval = 1
            max_range = 10
            min_range = 1
        else:
            raise ValueError("Unknown model type")
        # Convert tuple of tensors to a single numpy array
        labels = torch.cat(labels, dim=0).cpu().numpy()
        bins = np.arange(min_range, max_range + interval, interval)
        hist, _ = np.histogram(labels, bins=bins)
        self.label_weighting_bin_interval = interval
        # Compute weights based on the inverse of frequency
        if self.param.general.use_weight:
            weights = 1.0 / (hist + 1e-8)  # Adding epsilon to avoid division by zero
        else:
            weights = 0.0 * hist + 1.0
        weights /= weights.sum()
        return torch.tensor(weights, device=self.param.model.device)

    def _load_val_data(self) -> Tuple[DataLoader, DataLoader]:
        val_data_directory = os.path.expanduser(self.param.general.val_data_directory)
        organized_data = self._load_and_organize_data(val_data_directory)
        val_dataset = RNNDataset(
            organized_data,
            param=self.param,
        )
        single_env_val_loader = self._get_dataloader_for_env_id(organized_data)
        organized_data = None  # Free up memory
        gc.collect()
        torch.cuda.empty_cache()
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.param.general.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return val_loader, single_env_val_loader


class RNNDataset(Dataset):
    def __init__(
        self,
        data: Dict[int, List[Dict[str, torch.Tensor]]],
        param: ParamCollection,
    ):
        self.data = []
        self.input_slice = param.model.input_slice
        output_type = param.model.output_type
        add_noise = param.general.add_noise
        device = param.model.device

        for env_id, data_points in data.items():
            for data_point in data_points:
                input_data = data_point["input"].to(device)
                if add_noise:
                    # add noise to input
                    noise_std = (
                        torch.cat(
                            [
                                torch.zeros(6),
                                torch.ones(3) * 1e-2,
                                torch.ones(3) * 2e-2,
                                torch.ones(12) * 1e-3,
                                torch.ones(12) * 2e-2,
                                torch.ones(12) * 1e-3,
                                torch.zeros(12),
                                torch.ones(24) * 1e-3,
                                torch.ones(12) * 1e-3,
                                torch.ones(12) * 2e-2,
                                torch.zeros(12),
                                torch.zeros(12),
                                torch.zeros(1),
                                torch.ones(208) * 0.0,
                                torch.zeros(100),
                            ]
                        )
                        .to(device)
                        .detach()
                    )
                    self.noise_dist = torch.distributions.Normal(0, noise_std + 1e-10)
                    self.noise_exte = (2 * torch.rand_like(torch.ones(208)) - 1) * 0.5
                    cc = self.noise_dist.sample((input_data.shape[0],))
                    input_data += cc
                    input_data[:, 133:341] += self.noise_exte.to(device)

                if output_type == "fric":
                    output_data = data_point["output"][:, :4]
                elif output_type == "stiff":
                    output_data = data_point["output"][:, 4:]
                elif output_type == "all":
                    output_data = data_point["output"][:, :8]
                else:
                    raise ValueError("Invalid output_type. Choose 'fric' or 'stiff'")

                self.data.append(
                    (input_data.to(device), output_data.to(device))
                )  # List[(InputTensor(sequence_length, feature_dim), LabelTensor(sequence_length, priv_size))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data, output_data = self.data[idx]
        return input_data, output_data


def organize_data(
    data: Dict[int, Dict[str, torch.Tensor]],
    sequence_length: int = 10,
    overlapp: Optional[int] = None,
) -> Dict[int, List[Dict[str, torch.Tensor]]]:
    """
    Organize the data for training.
    Args:
        data: {env_id: {input/output_key: Tensor(complete_sequence, feat_dim)}
    Returns:
        {env_id: List[{combined_input/output_key: Tensor(sequence_length, feat_dim)}]}
    """
    organized_data = {}
    selected_cols = [i * 9 + 8 for i in range(4)]

    for env_id, sensors in data.items():
        # 'counter' is the sensor that determines episode boundaries
        counter_data = sensors["counter"]
        increment_indices = (counter_data[1:] - counter_data[:-1]).nonzero(
            as_tuple=True
        )[0] + 1
        increment_indices = increment_indices.tolist()
        boundaries = [0] + increment_indices + [len(counter_data)]
        data_points = []

        for i in range(len(boundaries) - 1):
            episode_start, episode_end = boundaries[i], boundaries[i + 1]

            interval = sequence_length
            if overlapp is not None:
                interval = overlapp
            for j in range(episode_start, episode_end, interval):
                if j + sequence_length > episode_end:
                    break

                policy_data = sensors["policy"][j : j + sequence_length]
                exte_data = sensors["priv_exte"][j : j + sequence_length]
                hidden_data = sensors["hidden"][j : j + sequence_length]
                fric_data = sensors["fric"][j : j + sequence_length]
                stiff_data = sensors["stiff"][j : j + sequence_length, selected_cols]

                data_points.append(
                    {
                        "input": torch.cat(
                            [policy_data, exte_data, hidden_data], dim=-1
                        ),
                        "output": torch.cat([fric_data, stiff_data], dim=-1),
                    }
                )

        organized_data[env_id] = data_points

    return organized_data


def generate_dest_filename(source_files: List[str]) -> str:
    # Extract parts from one of the source filenames
    sample_filename = os.path.basename(source_files[0])
    parts = sample_filename.split("__")

    # Get current timestamp
    current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")

    # Construct the destination filename
    dest_filename = f"{current_timestamp}__{parts[1]}"
    return dest_filename


def combine_datasets(
    source_files: List[str], dest_file: str
) -> Dict[str, Dict[str, torch.Tensor]]:
    data = {}
    last_max_counter = {}  # Dictionary to store the last counter for each env_id

    # Load each dataset and merge into the combined dataset
    for src_file in source_files:
        current_data = dd.io.load(src_file)
        # Print the number of rows in the current source file for an arbitrary env_id
        example_env_id = list(current_data.keys())[0]
        print(
            f"Row count for env_id {example_env_id} in source file {src_file}: {len(current_data[example_env_id]['counter'])}"
        )

        for env_id, sensors in current_data.items():
            if env_id not in data:
                data[env_id] = sensors
            else:
                for sensor_name, sensor_data in sensors.items():
                    if sensor_name == "counter":
                        # If we're processing the counter, adjust it based on the last maximum value
                        # from the previous dataset for the same env_id
                        if env_id in last_max_counter:
                            sensor_data += last_max_counter[env_id]
                    if sensor_name in data[env_id]:
                        data[env_id][sensor_name] = torch.cat(
                            [data[env_id][sensor_name], sensor_data]
                        )
                    else:
                        data[env_id][sensor_name] = sensor_data

                if "counter" in sensors:
                    last_max_counter[env_id] = (
                        sensors["counter"][-1].item() + 1
                    )  # update the last maximum counter value for the next iteration

    # Save the combined dataset
    # dd.io.save(dest_file, data)

    # Print the row number before and after for an arbitrary env_id (assuming the first env_id here)
    env_id_to_check = list(data.keys())[0]
    print(
        f"Row count for env_id {env_id_to_check} in combined data: {len(data[env_id_to_check]['counter'])}"
    )
    return data


def get_data_files_by_ratio(
    directory: str, min_ratio: float, max_ratio: float
) -> List[str]:
    """Get all data files in the given directory with a ratio value between min_ratio and max_ratio."""
    all_files = os.listdir(directory)
    filtered_files = []
    for f in all_files:
        match = re.search(r"rigidRatio=(\d+\.\d+)", f)
        if match:
            ratio = float(match.group(1))
            if min_ratio <= ratio <= max_ratio and f.endswith(".h5"):
                filtered_files.append(os.path.join(directory, f))
    return filtered_files


def combine_datasets_from_directory(
    dataset_directory: str, output_type: str
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Returns:
        {env_id: {input/output_key: Tensor(complete_sequence, feat_dim)}
    """

    if output_type == "fric":
        files = get_data_files_by_ratio(dataset_directory, 0.8, 1.0)
        dest_filename = generate_dest_filename(files)
        dest_file_path = f"{dataset_directory}/{dest_filename}"
        combined_data = combine_datasets(files, dest_file_path)
    elif output_type == "stiff":
        files = get_data_files_by_ratio(dataset_directory, 0.0, 0.2)
        dest_filename = generate_dest_filename(files)
        dest_file_path = f"{dataset_directory}/{dest_filename}"
        combined_data = combine_datasets(files, dest_file_path)
    else:
        raise ValueError(f"Unknown output_type: {output_type}")
    return combined_data


if __name__ == "__main__":
    dataset_directory = "dataset"
    combined_fric_data = combine_datasets_from_directory(dataset_directory, "fric")
    combined_stiff_data = combine_datasets_from_directory(dataset_directory, "stiff")
