#!/bin/bash
# play_rosbags.sh

# Navigate to the directory containing the bag files

cd /media/chenc/Chen/2024-01-25-white-board/2nd

# walk white
# cd /media/chen/Chen/20240211_Dodo_MPI/2024_02_11_Dodo_MPI_Vicon/2024-02-11-14-28-25/mission_data

rosbag play --clock --pause *.bag -r 0.5 