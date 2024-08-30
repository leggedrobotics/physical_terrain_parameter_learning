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

import rospy
import seaborn as sns

import numpy as np
from typing import Optional
import signal
import sys


class NodeForROS:
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
        self.fixed_frame = self.param.roscfg.fixed_frame
        self.base_frame = self.param.roscfg.base_frame
        self.footprint_frame = self.param.roscfg.footprint_frame

        # Robot dimensions
        self.robot_length = self.param.roscfg.robot_length
        self.robot_height = self.param.roscfg.robot_height
        self.robot_width = self.param.roscfg.robot_width
        self.robot_max_velocity = self.param.roscfg.robot_max_velocity
        self.foot_radius = self.param.roscfg.foot_radius

        # THREAD PARAMETERS
        self.image_callback_rate = self.param.thread.image_callback_rate
        self.proprio_callback_rate = self.param.thread.proprio_callback_rate
        self.learning_thread_rate = self.param.thread.learning_rate
        self.logging_thread_rate = self.param.thread.logging_rate

        # RUN PARAMETERS
        self.device = self.param.run.device
        self.mode = self.param.run.mode
        self.palette = self.param.run.palette
        self.print_time = self.param.run.print_time

        # Feature extractor parameters
        self.segmentation_type = self.param.feat.segmentation_type
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

    def query_tf(
        self,
        parent_frame: str,
        child_frame: str,
        stamp: Optional[rospy.Time] = None,
        from_message: Optional[AnymalState] = None,
    ):
        """Helper function to query TFs

        Args:
            parent_frame (str): Frame of the parent TF
            child_frame (str): Frame of the child
            from_message (AnymalState, optional): AnymalState message containing the TFs
        """
        if from_message is None:
            # error, must have from_message
            raise ValueError("Must provide from_message")
        else:
            # New behavior: extract the TF from the AnymalState message
            if child_frame not in self.feet_list:
                for frame_transform in from_message.frame_transforms:
                    if (
                        frame_transform.header.frame_id == parent_frame
                        and frame_transform.child_frame_id == child_frame
                    ):
                        trans = (
                            frame_transform.transform.translation.x,
                            frame_transform.transform.translation.y,
                            frame_transform.transform.translation.z,
                        )
                        rot = np.array(
                            [
                                frame_transform.transform.rotation.x,
                                frame_transform.transform.rotation.y,
                                frame_transform.transform.rotation.z,
                                frame_transform.transform.rotation.w,
                            ]
                        )
                        rot /= np.linalg.norm(rot)
                        return (trans, tuple(rot))
            else:
                for foot_transform in from_message.contacts:
                    if (
                        foot_transform.name == child_frame
                        and foot_transform.header.frame_id == parent_frame
                    ):
                        trans = (
                            foot_transform.position.x,
                            foot_transform.position.y,
                            foot_transform.position.z,
                        )
                        # FOOT is merely a point, no rotation
                        rot = np.array([0, 0, 0, 1])
                        return (trans, tuple(rot))

            # If the requested TF is not found in the message
            # rospy.logwarn(f"Couldn't find tf between {parent_frame} and {child_frame} in the provided message")
            raise ValueError(
                f"Couldn't find tf between {parent_frame} and {child_frame} in the provided message"
            )

    def shutdown_callback(self, *args, **kwargs):
        print(f"Node killed {args}")
        rospy.signal_shutdown(f"Node killed {args}")
        sys.exit(0)
