#!/bin/bash

# Define the session name
SESSION_NAME="BaseWVN"

# Define the Python interpreter path
PYTHON_PATH="/home/chenc/.conda/envs/vd/bin/python"

# Define the base path for your scripts
BASE_SCRIPT_PATH="/home/chenc/physical_terrain_parameter_learning/BaseWVN/src/wild_visual_navigation_ros/scripts"

# ROS setup commands
ROS_SETUP="source ~/physical_terrain_parameter_learning/BaseWVN/devel/setup.bash"

# Start a new tmux session in detached mode
tmux new-session -d -s $SESSION_NAME

# Run Main_process_node.py in the first pane
tmux send-keys -t $SESSION_NAME "$ROS_SETUP && $PYTHON_PATH $BASE_SCRIPT_PATH/Main_process_node.py" C-m

# Split the window vertically and run Phy_decoder_node_old.py in the new pane
tmux split-window -h -t $SESSION_NAME.0
tmux send-keys -t $SESSION_NAME.1 "$ROS_SETUP && $PYTHON_PATH $BASE_SCRIPT_PATH/Phy_decoder_node.py" C-m

# Split the first pane horizontally and run Phy_publisher.py in the new pane
tmux split-window -v -t $SESSION_NAME.1
tmux send-keys -t $SESSION_NAME.2 "$ROS_SETUP && $PYTHON_PATH $BASE_SCRIPT_PATH/Phy_publisher.py" C-m

# Optional: Attach to the session if you want to see the output immediately
tmux attach-session -t $SESSION_NAME

echo "BaseWVN environment is set up in tmux. Use 'tmux attach -t $SESSION_NAME' to see it."
