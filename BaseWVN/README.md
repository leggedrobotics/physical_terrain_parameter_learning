# Vision pipeline
This repo is for the online vision pipeline that uses trained physical decoders to output dense prediction of the environments in the vision channel. The pipeline is based on the previous repo from "Fast Traversability Estimation for Wild Visual Navigation".

## Installation
**Attention**: Please follow the installation order exactly as below. Otherwise, you may encounter some errors. Here we use mamba for virtual environment management with **python 3.9**
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

**Attention**: You need to pip install the Physical decoders module. It is a package in folder (phy_decoder). The physical decoders compatible in the vision pipeline is trained using an old version of the physical decoder training repo (different from folder physical_decoder_training). So to avoid inconvenience we already provide trained decoders in package phy_decoder that will load its weights automatically upon running in the vision pipeline. Also install the package from https://github.com/utiasSTARS/liegroups.git and https://github.com/facebookresearch/segment-anything.git. Please also put [ros-comm](https://github.com/leggedrobotics/ros_comm) under src folder to avoid ros comm bugs during real-time running.

Set you neptune api token , username and project name in the system file `.bashrc`:
```bash
export NEPTUNE_API_TOKEN="your_neptune_api_token"
export NEPTUNE_USERNAME="your_neptune_username"
export NEPTUNE_PROJECT="your_neptune_username/your_neptune_project_name"
```

### Install this repo:
```bash
pip install -e .
```

### Build this repo with ROS:
```bash
catkin build
source devel/setup.bash
```

## Vision pipeline - offline training
All configs are set in `BaseWVN/config/wvn_config.py`, for all the training/testing, you should pay attention to path-related settings.

Download segment_anything model checkpoint from [here](https://drive.google.com/file/d/1TU3asknvo1UKdhx0z50ghHDt1C_McKJu/view?usp=drive_link) and speicify the path in the config file.
### Offline Dataset
It is generated from the online rosbag playing. By setting `label_ext_mode: bool=True` you can record the dataset. The corresponding settings and paths are in config file. You need to sudo apt install `tmux` first. Please change the paths in the `play_BaseWVN.sh` file to your own paths.

You need to make sure the rosbag path is set to your own ones in `play_rosbag.sh` before executing the following.
```bash
roslaunch wild_visual_navigation_ros play.launch # start rosbag playing
./src/wild_visual_navigation_ros/scripts/play_BaseWVN.sh  # start all BaseWVN nodes
```
`Ctrl+B` then press `arrow keys` to switch between panels in the cli and use `Ctrl+C` to stop/finish the recording. To kill the tmux session, press your prefix (e.g. `Ctrl+A` or `B`, which is the default) and then `:` and type `kill-session`, then hit `Enter`.

The default saving path is `~/BaseWVN/results/manager` with the following files:

- `image_buffer.pt`: only store all the camera image tensors of the main nodes

- `train_data.pt`: only store the training data pairs, which are the same for online training

- `train_nodes.pt`:store all main nodes with all information

After running offline training for the first time, you will get additional files:

- `gt_masks_SAM.pt`: all the automatically generated GT masks from SAM
- `mask_img.pt`: the corresponding color image of the GT masks above
  
You can put the above files into seperate folders, like `~/BaseWVN/results/manager/train/snow`

### Manual correction of GT masks
Beacause the automatically generated GT masks (from SAM or SEEM) are not perfect, we need to manually correct them with segments.ai . 

You can use the `BaseWVN/offline/seg_correction.py` to correct the masks. The detailed usage you can refer to the code.
### Running
For offline training/testing, you can switch the config and run the following command:
```bash
python BaseWVN/offline/offline_training_lightning.py
```

## Vision pipeline - online training

### Running
For different configs, please refer to the code and config file.
```bash
./src/wild_visual_navigation_ros/scripts/play_BaseWVN.sh  # start all BaseWVN nodes
```

## Acknowledgements

This pipeline includes code from the following open-source project(s) licensed under the MIT License:

- [wild_visual_navigation](https://github.com/leggedrobotics/wild_visual_navigation) by Matias Mattamala, Jonas Frey