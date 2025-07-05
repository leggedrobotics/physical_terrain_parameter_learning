#  Self-supervised Visual Decoder Learning
This code is for online training of the visual decoder to predict the physical terrain paramters estimated by the physical decoder from images. The pipeline is based on the previous repo from "[Fast Traversability Estimation for Wild Visual Navigation](https://github.com/leggedrobotics/wild_visual_navigation)".

## Installation
**Attention**: Please follow the installation order exactly as below. Otherwise, you may encounter some errors. Here we use mamba for virtual environment management with **Python 3.9**
### Install robostack ros first:
https://robostack.github.io/GettingStarted.html

then you can do `mamba install ros-noetic-plotjuggler-ros`

### Install pytorch next:

```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
### Install other dependencies:
```bash
pip install -r requirements.txt
```
If you encounter any errors, please follow the error message to install the missing dependencies.

**Attention**: As a prerequest you need to pip install the [Stand-alone pre-trained physical decoder](phy_decoder/README.md). Also install the package from https://github.com/utiasSTARS/liegroups.git and https://github.com/facebookresearch/segment-anything.git. Please also put [ros-comm](https://github.com/leggedrobotics/ros_comm) under src folder to avoid ros comm bugs during real-time running. You need to sudo apt install `tmux`.

Set you neptune API token, username, and project name in the system file `.bashrc`:
```bash
export NEPTUNE_API_TOKEN="your_neptune_api_token"
export NEPTUNE_USERNAME="your_neptune_username"
export NEPTUNE_PROJECT="your_neptune_username/your_neptune_project_name"
```

### Install this base_wvn package:
```bash
pip install -e .
```

### Build the base_wvn with ROS:
```bash
catkin build wild_visual_navigation_ros
source devel/setup.bash
```


## Vision pipeline - online training
All configs are set in [base_wvn/config/wvn_cfg.py](base_wvn/config/wvn_cfg.py), for all the training/testing, you should pay attention to path-related settings.

### Running
For different configs, please refer to the code and config file.
```bash
./src/wild_visual_navigation_ros/scripts/play_base_wvn.sh  # start all base_wvn nodes
```

### Tmux usage
`Ctrl+B` then press `arrow keys` to switch between panels in the cli and use `Ctrl+C` to stop. To kill the tmux session, press your prefix (e.g. `Ctrl+A` or `B`, which is the default) and then `:` and type `kill-session`, then hit `Enter`.

## Vision pipeline - offline training
Download segment_anything model checkpoint from [here](https://drive.google.com/file/d/1TU3asknvo1UKdhx0z50ghHDt1C_McKJu/view?usp=drive_link) and speicify the path in the config file.
### Offline Dataset
It is generated from the online rosbag playing. By setting `label_ext_mode: bool=True` you can record the dataset. The corresponding settings and paths are in config file.  Please change the paths in the `play_base_wvn.sh` file to your own paths.

You need to make sure the rosbag path is set to your own ones in `play_rosbag.sh` before executing the following.
```bash
roslaunch wild_visual_navigation_ros play.launch # start rosbag playing
./src/wild_visual_navigation_ros/scripts/play_base_wvn.sh  # start all base_wvn nodes

# Follow Tmux usage and use `Ctrl+C` to stop the recording.
```
### Tmux usage
`Ctrl+C` in the **left-most** panel will also finish the recording.

### Dataset structure
The default saving path is `~/base_wvn/results/manager` with the following files:

- `image_buffer.pt`: only store all the camera image tensors of the main nodes

- `train_data.pt`: only store the training data pairs, which are the same for online training

- `train_nodes.pt`:store all main nodes with all information

After running offline training for the first time, you will get additional files:

- `gt_masks_SAM.pt`: all the automatically generated GT masks from SAM
- `mask_img.pt`: the corresponding color image of the GT masks above
  
You can put the above files into seperate folders, like `~/base_wvn/results/manager/train/snow`

### Manual correction of GT masks
Beacause the automatically generated GT masks (from SAM) are not perfect, we need to manually correct them with segments.ai . 

You can use the `base_wvn/offline/seg_correction.py` to correct the masks. The detailed usage you can refer to the code.
### Running
For offline training/testing, you can switch the config and run the following command:
```bash
python base_wvn/offline/train_eval.py
```



## Acknowledgements

This pipeline includes code from the following open-source project(s) licensed under the MIT License:

- [wild_visual_navigation](https://github.com/leggedrobotics/wild_visual_navigation) by Matias Mattamala, Jonas Frey