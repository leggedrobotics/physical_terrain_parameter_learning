#!/bin/bash

# Define the session name
SESSION_NAME="base_wvn"

# Define the Python interpreter path
PYTHON_PATH="/home/chenc/.conda/envs/vd/bin/python"

# Define the base path for your scripts
BASE_SCRIPT_PATH="/home/chenc/physical_terrain_parameter_learning/base_wvn/src/wild_visual_navigation_ros/scripts"

# ROS setup commands
ROS_SETUP="source ~/physical_terrain_parameter_learning/base_wvn/devel/setup.bash"

# Start a new tmux session in detached mode
tmux new-session -d -s $SESSION_NAME

# Run main_process_node.py in the first panel
tmux send-keys -t $SESSION_NAME "$ROS_SETUP && $PYTHON_PATH $BASE_SCRIPT_PATH/main_process_node.py" C-m

# Split the window vertically and run in the new panel
tmux split-window -h -t $SESSION_NAME.0
tmux send-keys -t $SESSION_NAME.1 "$ROS_SETUP && $PYTHON_PATH $BASE_SCRIPT_PATH/supervision_label_builder_node.py" C-m

# Split the first panel horizontally and run physical_decoder_node.py in the new panel
tmux split-window -v -t $SESSION_NAME.1
tmux send-keys -t $SESSION_NAME.2 "$ROS_SETUP && $PYTHON_PATH $BASE_SCRIPT_PATH/physical_decoder_node.py" C-m

# Optional: Attach to the session if you want to see the output immediately
tmux attach-session -t $SESSION_NAME

echo "base_wvn environment is set up in tmux. Use 'tmux attach -t $SESSION_NAME' to see it."
