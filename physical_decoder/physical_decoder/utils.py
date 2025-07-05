#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import os
import fnmatch
from datetime import datetime

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


def construct_search_pattern(model_args: Dict[str, Any]) -> str:
    conditions = []

    # Check model type
    conditions.append(model_args["mode"])

    conditions.append(model_args["input_type"])

    # Construct search pattern
    output_type = model_args["output_type"]
    search_pattern = f"*{output_type}*"
    for condition in conditions:
        search_pattern += f"*{condition}*"
    search_pattern += ".pth"

    return search_pattern
