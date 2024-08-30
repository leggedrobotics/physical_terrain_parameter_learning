import deepdish as dd
import torch
import datetime
import os
import re

def generate_dest_filename(source_files):
    # Extract parts from one of the source filenames
    sample_filename = os.path.basename(source_files[0])
    parts = sample_filename.split("__")

    # Get current timestamp
    current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")

    # Construct the destination filename
    dest_filename = f"{current_timestamp}__{parts[1]}"
    return dest_filename

def combine_datasets(source_files, dest_file):
    data = {}
    last_max_counter = {}  # Dictionary to store the last counter for each env_id

    # Load each dataset and merge into the combined dataset
    for src_file in source_files:
        current_data = dd.io.load(src_file)
        # Print the number of rows in the current source file for an arbitrary env_id
        example_env_id = list(current_data.keys())[0]
        print(f"Row count for env_id {example_env_id} in source file {src_file}: {len(current_data[example_env_id]['counter'])}")

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
                        data[env_id][sensor_name] = torch.cat([data[env_id][sensor_name], sensor_data])
                    else:
                        data[env_id][sensor_name] = sensor_data
                        
                if "counter" in sensors:
                    last_max_counter[env_id] = sensors["counter"][-1].item() + 1  # update the last maximum counter value for the next iteration

    # Save the combined dataset
    # dd.io.save(dest_file, data)

    # Print the row number before and after for an arbitrary env_id (assuming the first env_id here)
    env_id_to_check = list(data.keys())[0]
    print(f"Row count for env_id {env_id_to_check} in combined data: {len(data[env_id_to_check]['counter'])}")
    return data

def get_data_files_by_ratio(directory, min_ratio, max_ratio):
    """Get all data files in the given directory with a ratio value between min_ratio and max_ratio."""
    all_files = os.listdir(directory)
    filtered_files = []
    for f in all_files:
        match = re.search(r"rigidRatio=(\d+\.\d+)", f)
        if match:
            ratio = float(match.group(1))
            if min_ratio <= ratio <= max_ratio and f.endswith('.h5'):
                filtered_files.append(os.path.join(directory, f))
    return filtered_files

def combine_datasets_from_directory(dataset_directory, model_type,mode):
    # if os.path.exists(f"{dataset_directory}/{mode}_{model_type}_comb.pt"):
    #     print(f"Combined dataset for {model_type} already exists. Skipping combination.")
    #     return torch.load(f"{dataset_directory}/{mode}_{model_type}_comb.pt")

    if model_type == "fric":
        files = get_data_files_by_ratio(dataset_directory, 0.8, 1.0)
        dest_filename = generate_dest_filename(files)
        dest_file_path = f"{dataset_directory}/{dest_filename}"
        combined_data = combine_datasets(files, dest_file_path)
    elif model_type == "stiff":
        files = get_data_files_by_ratio(dataset_directory, 0.0, 0.2)
        dest_filename = generate_dest_filename(files)
        dest_file_path = f"{dataset_directory}/{dest_filename}"
        combined_data = combine_datasets(files, dest_file_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    # torch.save(combined_data, f"{dataset_directory}/{mode}_{model_type}_comb.pt")
    return combined_data

if __name__ == "__main__":
    dataset_directory = "dataset"
    combined_fric_data = combine_datasets_from_directory(dataset_directory, "fric")
    combined_stiff_data = combine_datasets_from_directory(dataset_directory, "stiff")
