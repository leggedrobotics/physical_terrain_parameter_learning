#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
"""
Main node to process ros messages, publish the relevant topics, train the model...
"""

from base_wvn import ParamCollection
from wild_visual_navigation_msgs.msg import AnymalState
from nav_msgs.msg import Odometry
import ros_converter as rc
import rospy
import seaborn as sns

import numpy as np
import signal
import sys


class RosNode:
    def __init__(self):
        # Read the parameters from the config file
        self.param = ParamCollection()
        self.import_params()
        self.color_palette = sns.color_palette(self.palette, as_cmap=False)
        # Dynamically set the class name for the timer

        # Register shotdown callbacks
        rospy.on_shutdown(self.shutdown_callback)
        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

    def import_params(self):
        """
        read cfg file and import the parameters
        """
        # TOPIC-REALTED PARAMETERS
        self.anymal_bag_name = self.param.roscfg.anymal_bag_name
        self.anymal_state_topic = self.param.roscfg.anymal_state_topic
        self.feet_list = self.param.roscfg.feet_list
        self.camera_bag_name = self.param.roscfg.camera_bag_name
        self.camera_topic = self.param.roscfg.camera_topic
        self.phy_decoder_input_topic = self.param.roscfg.phy_decoder_input_topic
        self.camera_info_topic = self.param.roscfg.camera_info_topic
        self.visual_odom_topic = self.param.roscfg.visual_odom_topic
        self.phy_temp_topic = self.param.roscfg.phy_decoder_temp_topic
        # Frames
        self.world_frame = self.param.roscfg.world_frame
        self.base_frame = self.param.roscfg.base_frame

        # Robot dimensions
        self.robot_length = self.param.roscfg.robot_length
        self.robot_height = self.param.roscfg.robot_height
        self.robot_width = self.param.roscfg.robot_width
        self.robot_max_velocity = self.param.roscfg.robot_max_velocity
        self.foot_radius = self.param.roscfg.foot_radius

        # THREAD PARAMETERS
        self.camera_callback_rate = self.param.thread.camera_callback_rate
        self.supervision_signal_callback_rate = (
            self.param.thread.supervision_signal_callback_rate
        )
        self.learning_thread_rate = self.param.thread.learning_rate
        self.logging_thread_rate = self.param.thread.logging_rate

        # RUN PARAMETERS
        self.device = self.param.run.device
        self.mode = self.param.run.mode
        self.palette = self.param.run.palette
        self.print_time = self.param.run.print_time

        # Feature extractor parameters
        self.feature_type = self.param.feat.feature_type
        self.input_size = self.param.feat.input_size
        self.interp = self.param.feat.interp
        self.center_crop = self.param.feat.center_crop

        # Loss parameters
        self.confidence_std_factor = self.param.loss.confidence_std_factor
        self.method = self.param.loss.method
        self.log_enabled = self.param.loss.log_enabled
        self.log_folder = self.param.loss.log_folder
        self.verbose = self.param.loss.verbose

        if "v4l2" in self.camera_topic:
            self.camera_in_base = self.param.roscfg.rear_hdr_camera_in_base
        elif "wide_angle" in self.camera_topic and "rear" in self.camera_topic:
            self.camera_in_base = self.param.roscfg.rear_wide_angle_camera_in_base
        elif "wide_angle" in self.camera_topic and "front" in self.camera_topic:
            self.camera_in_base = self.param.roscfg.front_wide_angle_camera_in_base
        elif "hdr" in self.camera_topic:
            self.camera_in_base = self.param.roscfg.front_hdr_camera_in_base
        self.lidar_in_base = self.param.roscfg.lidar_in_base

        self.use_vo = self.param.roscfg.use_vo
        self.pub_pred_as_layer = self.param.general.pub_pred_as_layer
        if self.pub_pred_as_layer == "single":
            self.layer_num = 1
        elif self.pub_pred_as_layer == "RGB":
            self.layer_num = 3
        else:
            raise ValueError("pub_pred_as_layer must be either 'single' or 'RGB'")

    def get_pose_base_in_world(
        self, state_msg: AnymalState, visual_odom_msg: Odometry
    ) -> np.ndarray:
        """Get the pose of the base in world frame"""
        if self.use_vo:
            return self.get_pose_base_in_world_from_slam(visual_odom_msg)
        else:
            return self.get_pose_base_in_world_from_state_estimator(state_msg)

    def get_pose_base_in_world_from_state_estimator(
        self, state_msg: AnymalState
    ) -> np.ndarray:
        """From proprioceptive state estimator"""
        transform = state_msg.pose
        pose_base_in_world = rc.msg_to_se3(transform)
        return pose_base_in_world.astype(np.float32)

    def get_pose_base_in_world_from_slam(self, visual_odom_msg: Odometry) -> np.ndarray:
        """From SLAM system, visual_odom_msg: from visual odom frame to lidar frame"""
        pose_lidar_in_world = rc.msg_to_se3(visual_odom_msg)
        pose_base_in_lidar = np.linalg.inv(self.lidar_in_base)
        pose_base_in_world = pose_lidar_in_world @ pose_base_in_lidar
        return pose_base_in_world.astype(np.float32)

    def shutdown_callback(self, *args, **kwargs):
        print(f"Node killed {args}")
        rospy.signal_shutdown(f"Node killed {args}")
        sys.exit(0)
