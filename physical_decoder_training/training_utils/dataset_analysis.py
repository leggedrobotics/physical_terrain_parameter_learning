#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import sys
import os
import deepdish as dd
import matplotlib.pyplot as plt
import torch
import numpy as np
import re

from phy_decoder import get_latest_file_in_directory


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


def compute_episode_statistics(data):
    # Assuming 'data' is a dictionary with keys as env_ids and values as nested dictionaries of sensor data

    episode_lengths_per_env = {}

    for env_id, sensors in data.items():
        counter_data = sensors["counter"]
        total_length = len(counter_data)
        # Find the indices where the counter increments
        increment_indices = (counter_data[1:] - counter_data[:-1]).nonzero(
            as_tuple=True
        )[0] + 1
        # Compute episode lengths
        if (
            increment_indices.numel() == 0
        ):  # Handle the case when there are no increments
            episode_lengths = [len(counter_data)]
        else:
            episode_lengths = [increment_indices[0].item()] + (
                increment_indices[1:] - increment_indices[:-1]
            ).tolist()
        episode_lengths_per_env[env_id] = episode_lengths

    # Flatten episode lengths across all envs to compute global statistics
    all_episode_lengths = [
        length for lengths in episode_lengths_per_env.values() for length in lengths
    ]

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


if __name__ == "__main__":
    # Define the directory where you want to save the file
    output_dir = "statistics"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_stdout = sys.stdout  # Save the original stdout
    sensor_to_show = ["policy", "fric", "stiff"]
    sensor_to_vis = ["fric", "stiff"]
    feet_labels = ["FOOT_LF", "FOOT_RF", "FOOT_LH", "FOOT_RH"]
    selected_cols = [i * 9 + 8 for i in range(4)]
    # data = dd.io.load('data.h5')
    directory = "dataset"
    latest_data_file = get_latest_file_in_directory(directory)
    vis_env_id = 50
    if latest_data_file:
        data = dd.io.load(latest_data_file)

    else:
        raise NoDataFileError("No data file found in the specified directory.")

    timestamp = extract_timestamp_from_filename(latest_data_file)
    subfolder_path = create_subfolder(output_dir, timestamp)
    filename = os.path.join(
        subfolder_path, timestamp + "__dataset_analysis_Version0.txt"
    )
    base_plot_filename = os.path.join(
        subfolder_path, timestamp + "__dataset_analysis_Version0"
    )

    with open(filename, "w") as file:
        sys.stdout = file

        # Convert nested dictionary to DataFrame for easier manipulation
        df_list = []
        # Combine data for each sensor
        combined_data = {}
        timesteps_per_env = {}
        for env_id, sensors in data.items():
            for sensor, values in sensors.items():
                if sensor not in sensor_to_show:
                    continue
                if env_id not in timesteps_per_env:
                    timesteps_per_env[env_id] = values.shape[0]
                if sensor == "stiff":
                    values = [values[:, i].unsqueeze(1) for i in selected_cols]
                    values = torch.cat(values, dim=1)
                # temp_df = pd.DataFrame(values, columns=feet_labels)
                # temp_df['env_id'] = env_id
                # temp_df['sensor'] = sensor
                # df_list.append(temp_df)
                if sensor not in combined_data:
                    combined_data[sensor] = []
                combined_data[sensor].append(values)

        for sensor, values in combined_data.items():
            # print(values)

            combined_data[sensor] = torch.cat(values)

        # Compute statistics and visualize
        for sensor, values in combined_data.items():
            if sensor not in sensor_to_show:
                continue
            values_np = values.cpu().numpy()  # Convert to numpy for plotting
            print(f"\nSensor: {sensor}")
            print("Mean:", torch.mean(values, dim=0))
            print("Std Dev:", torch.std(values, dim=0))
            print("Min:", torch.min(values, dim=0).values)
            print("Max:", torch.max(values, dim=0).values)
            if sensor not in sensor_to_vis:
                continue
            # Plotting
            # When extracting data for a specific env_id, use the timesteps_per_env dictionary
            start_idx = sum(timesteps_per_env[i] for i in range(vis_env_id))
            end_idx = start_idx + timesteps_per_env[vis_env_id]
            values_np = values_np[start_idx:end_idx, :]

            # Assuming `counter_data` is the counter data for the current `env_id`
            counter_data = data[vis_env_id]["counter"].cpu().numpy()

            # Finding indices where the counter increments, marking the start of a new episode
            start_indices = np.where(counter_data[1:] - counter_data[:-1] > 0)[0]
            start_indices = np.concatenate(([0], start_indices, [len(counter_data)]))

            # Define a list of colors (extend this list if you have more episodes)
            colors = ["b", "g", "r", "c", "m", "y", "k"]

            # Plotting
            fig, axes = plt.subplots(
                nrows=2, ncols=2, figsize=(10, 10)
            )  # create a 2x2 grid of subplots
            axes = axes.ravel()  # flatten the 2x2 grid to easily loop through it

            for i, foot in enumerate(feet_labels):
                for j in range(len(start_indices) - 1):
                    # Select the color for this episode
                    color = colors[j % len(colors)]
                    axes[i].plot(
                        np.arange(start_indices[j], start_indices[j + 1]),
                        values_np[start_indices[j] : start_indices[j + 1], i],
                        label=f"Episode {j+1}" if foot == feet_labels[0] else "",
                        color=color,
                    )
                axes[i].set_title(foot)
                axes[i].set_xlabel("Timestep")
                axes[i].set_ylabel("Value")
                if foot == feet_labels[0]:
                    axes[i].legend()

            plt.suptitle(sensor)  # Set the main title for all subplots
            plt.tight_layout()
            # Saving the plot
            plot_filename = base_plot_filename + f"_{sensor}.png"
            plt.savefig(plot_filename)

            values = (
                combined_data[sensor].cpu().numpy()
            )  # Extract the values for the sensor from the combined data
            flattened_values = values.flatten()  # Flatten the values for histogram

            plt.figure(figsize=(10, 6))
            plt.hist(
                flattened_values, bins=10, edgecolor="black"
            )  # Adjust 'bins' as required
            plt.title(f"Histogram for {sensor}")
            plt.xlabel(sensor)
            plt.ylabel("Frequency")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()

            # Optionally save the histogram as well
            histogram_filename = base_plot_filename + f"_histogram_{sensor}.png"
            plt.savefig(histogram_filename)
            # plt.show()

        mean_length, std_length, episode_lengths_per_env = compute_episode_statistics(
            data
        )

    sys.stdout = original_stdout
    print("-----Process complete-----")
