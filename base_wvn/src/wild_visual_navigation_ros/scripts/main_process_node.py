#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
"""
Main node to process ros messages, publish the relevant topics, train the model...
"""

from base_wvn.utils import (
    FeatureExtractor,
    ImageProjector,
    MaskedPredictionData,
    plot_pred_w_overlay,
    plot_overlay_image,
    add_color_bar_and_save,
)
from base_wvn.graph_manager import Manager, MainNode, SubNode
import ros_converter as rc
import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Header
from nav_msgs.msg import Odometry
from wild_visual_navigation_msgs.msg import AnymalState
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from wild_visual_navigation_msgs.msg import PhyDecoderOutput, ChannelInfo
from ros_node import RosNode
import os
import rospy
from cv_bridge import CvBridge
import torch
import numpy as np
import traceback
import sys
import cv2
from threading import Thread, Event, Lock
from prettytable import PrettyTable
from termcolor import colored
import PIL.Image
import tf2_ros
from pytictac import ClassTimer, accumulate_time
import datetime
from copy import deepcopy
from typing import Optional, Union


class MainProcess(RosNode):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.sub_step = 0
        # Timers to control the rate of the callbacks
        self.start_time = rospy.get_time()
        self.last_image_ts = self.start_time
        self.last_supervision_ts = self.start_time
        self.last_learning_ts = self.start_time
        self.last_logging_ts = self.start_time
        self.last_image_saved_ts = self.start_time
        self.today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.feat_extractor = FeatureExtractor(self.param)

        # Init graph manager
        self.manager = Manager(param=self.param)

        # Init Camera handler
        self.camera_handler = {}
        self.system_events = {}
        self.timer = ClassTimer(
            objects=[
                self,
                self.manager,
            ],
            names=["Main_process", "Manager"],
            enabled=self.param.general.timestamp,
        )

        self.log_data = {}
        self.log_data["Lock"] = Lock()
        if self.verbose:
            self.logging_thread_stop_event = Event()
            self.logging_thread = Thread(
                target=self.logging_thread_loop, name="logging"
            )
            self.run_logging_thread = True
            self.logging_thread.start()

        # Initialize ROS nodes
        self.ros_init()

        if "label" not in self.mode:
            # Init learning thread
            self.learning_thread_stop_event = Event()
            self.learning_thread = Thread(
                target=self.learning_thread_loop, name="learning"
            )
            self.learning_thread.start()

        # for paper video
        self.overlay_img_frames = {}

        print("Main process node initialized!")

    def shutdown_callback(self, *args, **kwargs) -> None:
        if self.verbose:
            self.run_logging_thread = False
            self.logging_thread_stop_event.set()
            self.logging_thread.join()
        self.visualize_self_supervision_label_image_overlay()
        if self.param.general.timestamp:
            print(self.timer)

        # self.manager.save("results/manager","try1")

        if "label" not in self.mode:
            self.learning_thread_stop_event.set()
            self.learning_thread.join()
        if self.manager._label_ext_mode:
            self.manager.save(self.param.offline.data_folder, "train")
        rospy.signal_shutdown(f"FeatExtractor Node killed {args}")

        print("Storing learned checkpoint...", end="")
        self.manager.save_ckpt(self.param.general.model_path, "last_checkpoint.pt")
        saving = deepcopy(self.overlay_img_frames)
        torch.save(
            saving,
            os.path.join(
                self.param.offline.ckpt_parent_folder, "online", "overlay_img_frames.pt"
            ),
        )
        print("done")

        sys.exit(0)

    def logging_thread_loop(self) -> None:
        # Log loop
        while self.run_logging_thread:
            ts = rospy.get_time()
            self.logging_thread_stop_event.wait(timeout=0.01)
            if self.logging_thread_stop_event.is_set():
                rospy.logwarn("Stopped logging thread")
                break
            if self.is_bad_rate_with_log(
                ts,
                self.last_logging_ts,
                self.logging_thread_rate,
                "logging_thread",
            ):
                continue
            self.last_logging_ts = ts

            x = PrettyTable()
            x.field_names = ["Key", "Value"]
            with self.log_data["Lock"]:  # Acquire the lock
                for k, v in self.log_data.items():
                    if "Lock" in k:
                        continue
                    if "time" in k:
                        d = ts - v
                        if d < 0:
                            c = "red"
                        if d < 0.2:
                            c = "green"
                        elif d < 1.0:
                            c = "yellow"
                        else:
                            c = "red"
                        x.add_row([k, colored(round(d, 2), c)])
                    else:
                        x.add_row([k, v])

            print(x)

        self.logging_thread_stop_event.clear()

    def ros_init(self) -> None:
        """
        start ros subscribers and publishers and filter/process topics
        """
        self.log("time_last_model", -1)
        self.log("num_model_updates", -1)
        self.log("num_images", 0)
        self.log("time_last_image", -1)
        self.log("camera_callback", "N/A")
        self.log("prediction_done", 0)

        print("Waiting for Camera topic being published!")
        # Camera info
        camera_info_msg: CameraInfo = rospy.wait_for_message(
            self.camera_info_topic, CameraInfo
        )
        new_camera_info_msg = deepcopy(camera_info_msg)
        K, H, W = rc.ros_cam_info_to_tensors(camera_info_msg, device=self.device)

        self.camera_handler["camera_info"] = camera_info_msg
        self.camera_handler["K"] = K
        self.camera_handler["H"] = H
        self.camera_handler["W"] = W
        self.camera_handler["distortion_model"] = camera_info_msg.distortion_model

        self.feat_extractor.init_transform(original_width=W, original_height=H)
        ratio_x, ratio_y = self.feat_extractor.scaling_ratio
        offset_x, offset_y = self.feat_extractor.crop_offset
        # scale the intrinsic matrix
        K_scaled = rc.scale_intrinsic(K, ratio_x, ratio_y, offset_x, offset_y)
        W_scaled, H_scaled = self.feat_extractor.new_size
        # update the camera info
        self.camera_handler["K_scaled"] = K_scaled
        self.camera_handler["H_scaled"] = H_scaled
        self.camera_handler["W_scaled"] = W_scaled

        fx = K_scaled[0, 0, 0]
        fy = K_scaled[0, 1, 1]
        cx = K_scaled[0, 0, 2]
        cy = K_scaled[0, 1, 2]

        camera_info_msg = new_camera_info_msg
        # Update the camera_info message
        camera_info_msg.height = H_scaled
        camera_info_msg.width = W_scaled

        # K is a 3x3 matrix flattened into an array, in row-major order
        camera_info_msg.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

        # If P is affected, update it accordingly. Assuming no change in depth/translation:
        camera_info_msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_handler["new_camera_info"] = camera_info_msg

        # Camera and anymal state subscriber
        camera_sub = message_filters.Subscriber(
            self.camera_topic, CompressedImage, queue_size=1
        )
        anymal_state_sub = message_filters.Subscriber(
            self.anymal_state_topic, AnymalState
        )
        if self.use_vo:
            visual_odom_sub = message_filters.Subscriber(
                self.visual_odom_topic, Odometry
            )
            self.world_frame = self.param.roscfg.vo_world_frame
            sync = message_filters.ApproximateTimeSynchronizer(
                [camera_sub, anymal_state_sub, visual_odom_sub],
                queue_size=200,
                slop=0.2,
            )
            sync.registerCallback(self.camera_callback, self.camera_topic)
        else:
            sync = message_filters.ApproximateTimeSynchronizer(
                [camera_sub, anymal_state_sub], queue_size=200, slop=0.2
            )
            sync.registerCallback(self.camera_callback_no_vo, self.camera_topic)

        # Phy-decoder info subscriber
        phy_sub = rospy.Subscriber(
            self.param.roscfg.phy_decoder_output_topic,
            PhyDecoderOutput,
            self.supervision_signal_callback,
            queue_size=1,
        )

        # Fill in handler
        self.camera_handler["name"] = self.camera_topic
        self.camera_handler["img_sub"] = camera_sub

        # Results publisher
        fric_pub = rospy.Publisher(
            self.param.roscfg.masked_dense_friction_prediction_image_topic,
            Image,
            queue_size=10,
        )
        stiff_pub = rospy.Publisher(
            self.param.roscfg.masked_dense_stiffness_prediction_image_topic,
            Image,
            queue_size=10,
        )
        info_pub = rospy.Publisher(
            self.param.roscfg.prediction_image_camera_info_topic,
            CameraInfo,
            queue_size=10,
        )
        channel_pub = rospy.Publisher(
            self.param.roscfg.prediction_image_channel_topic, ChannelInfo, queue_size=10
        )

        latest_main_node = rospy.Publisher(
            self.param.roscfg.main_node_visualization_topic, Marker, queue_size=10
        )
        latest_sub_node = rospy.Publisher(
            self.param.roscfg.sub_node_visualization_topic, Marker, queue_size=10
        )

        # Fill in handler
        self.camera_handler["fric_pub"] = fric_pub
        self.camera_handler["stiff_pub"] = stiff_pub
        self.camera_handler["info_pub"] = info_pub
        self.camera_handler["channel_pub"] = channel_pub
        self.camera_handler["latest_main_node"] = latest_main_node
        self.camera_handler["latest_sub_node"] = latest_sub_node

    def log(self, entry: str, msg: str) -> None:
        """Grab the lock and log the message in verbose mode."""
        with self.log_data["Lock"]:
            self.log_data[entry] = msg

    def is_bad_rate_with_log(
        self,
        current_timestamp_sec: float,
        last_timestamp_sec: float,
        rate_upper_limit: float,
        log_entry: str,
    ) -> bool:
        """Here we output signal to outer loop if the current iteration above the rate upper limit:
        True - too fast or jump back in time, skip this iteration
        False - ok
        """
        current_interval_sec = current_timestamp_sec - last_timestamp_sec
        current_rate = (
            1.0 / current_interval_sec if current_interval_sec != 0 else float("inf")
        )

        if current_rate <= 0.0:
            self.log(log_entry, "skipping: jump back in time")
            return True

        elif current_rate > rate_upper_limit:
            self.log(log_entry, "skipping: rate too high")
            return True
        else:
            self.log(log_entry, "processing: rate ok")
            return False

    @accumulate_time
    def camera_callback(
        self,
        img_msg: CompressedImage,
        state_msg: AnymalState,
        visual_odom_msg: Optional[Odometry],
        cam: str,
    ) -> None:
        """
        callback function for camera img, visual odometry msg can be None if self.use_vo is False
        """
        self.system_events["camera_callback_received"] = {
            "time": rospy.get_time(),
            "value": "message received",
        }
        try:
            ts = img_msg.header.stamp.to_sec()
            if self.is_bad_rate_with_log(
                ts, self.last_image_ts, self.camera_callback_rate, "camera_callback"
            ):
                self.system_events["camera_callback_cancelled"] = {
                    "time": rospy.get_time(),
                    "value": "cancelled due to rate",
                }
                return

            self.log("ros_time_now", rospy.get_time())

            # prepare tf from base to camera
            pose_base_in_world = self.get_pose_base_in_world(state_msg, visual_odom_msg)

            # transform the camera pose from base to world
            pose_cam_in_world = np.matmul(pose_base_in_world, self.camera_in_base)
            self.camera_handler["pose_cam_in_world"] = pose_cam_in_world

            # prepare image
            img_torch = rc.ros_image_to_torch(
                img_msg, device=self.device, desired_encoding="rgb8"
            )
            img_torch = img_torch[None]
            transformed_img, compressed_feats = self.feat_extractor.extract(img_torch)

            image_projector = ImageProjector(
                K=self.camera_handler["K_scaled"],
                h=self.camera_handler["H_scaled"],
                w=self.camera_handler["W_scaled"],
            )

            if self.manager._label_ext_mode:
                if abs(ts - self.last_image_saved_ts) > 10.0:
                    self.last_image_saved_ts = ts
                    self.log("time_last_image", rospy.get_time())
                    self.log("num_images", self.log_data["num_images"] + 1)

            main_node = MainNode(
                timestamp=img_msg.header.stamp.to_sec(),
                pose_base_in_world=torch.from_numpy(pose_base_in_world).to(self.device),
                pose_cam_in_world=torch.from_numpy(pose_cam_in_world).to(self.device),
                image=transformed_img,
                features=compressed_feats,
                feature_type=self.feature_type,
                image_projector=image_projector,
                camera_name=cam,
                phy_dim=self.param.feat.physical_dim,
            )

            # add to main graph
            added_new_node = self.manager.add_main_node(main_node, logger=self.log_data)

            if (
                self.manager.distance_between_last_main_node_and_last_sub_node
                is not None
            ):
                self.log(
                    "head dist of main/sub graph",
                    "{:.2f}".format(
                        self.manager.distance_between_last_main_node_and_last_sub_node.item()
                    ),
                )

            # ----- visualization -----
            # 1. publish a green cube marker where the latest main node is
            if self.param.logger.vis_new_node and added_new_node:
                self.visualize_new_node(
                    pose_base_in_world,
                    ts,
                    self.camera_handler["latest_main_node"],
                    "latest_main_node",
                )

            # 2. publish the label image overlay on vis_node, very slow so commented
            # self.visualize_self_supervision_label_image_overlay()

            # 3. publish the masked dense prediction, using the img received in the callback
            header = Header()
            header.stamp = rospy.Time.from_sec(ts)
            self.pub_node_prediction(main_node, header=header, ts=ts)

            self.system_events["camera_callback_state"] = {
                "time": rospy.get_time(),
                "value": "executed successfully",
            }
            self.last_image_ts = ts
        except Exception as e:
            traceback.print_exc()
            print("error camera callback", e)
            self.system_events["camera_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }

    @accumulate_time
    def camera_callback_no_vo(
        self, img_msg: CompressedImage, state_msg: AnymalState, cam: str
    ) -> None:
        self.camera_callback(img_msg, state_msg, None, cam)

    @accumulate_time
    def supervision_signal_callback(self, phy_output: PhyDecoderOutput) -> None:
        self.system_events["supervision_signal_callback_received"] = {
            "time": rospy.get_time(),
            "value": "message received",
        }
        try:
            ts = phy_output.header.stamp.to_sec()
            if self.is_bad_rate_with_log(
                ts,
                self.last_supervision_ts,
                self.supervision_signal_callback_rate,
                "supervision_signal_callback",
            ):
                self.system_events["supervision_signal_callback_cancelled"] = {
                    "time": rospy.get_time(),
                    "value": "cancelled due to rate",
                }
                return
            self.last_supervision_ts = ts

            pose_base_in_world = rc.pose_msg_to_se3_torch(
                phy_output.base_pose, device=self.device
            )

            fric = torch.tensor(phy_output.prediction[:4]).to(self.device)
            stiff = torch.tensor(phy_output.prediction[4:]).to(self.device)
            phy_label = torch.stack([fric, stiff], dim=0)
            feet_planes = []
            for i, plane in enumerate(phy_output.feet_planes):
                if self.feet_list[i] != plane.name:
                    raise ValueError(
                        "The order of the feet planes not match feet list!"
                    )
                edge = rc.plane_edge_to_torch(plane, device=self.device)
                feet_planes.append(edge)
            feet_planes = torch.stack(feet_planes, dim=0)
            feet_contact = torch.tensor(phy_output.feet_contact).to(self.device)
            sub_node = SubNode(
                timestamp=ts,
                pose_base_in_world=pose_base_in_world,
                feet_planes=feet_planes,
                feet_contact=feet_contact,
                phy_pred=phy_label,
            )

            # add subnode
            added_sub_node = self.manager.add_sub_node(sub_node, logger=self.log_data)

            # ----- visualization -----
            # 1. publish a yellow sphere marker where the latest sub node is
            if self.param.logger.vis_new_node and added_sub_node:
                self.visualize_new_node(
                    pose_base_in_world,
                    ts,
                    self.camera_handler["latest_sub_node"],
                    "latest_sub_node",
                )

            self.sub_step += 1

            self.system_events["supervision_signal_callback_state"] = {
                "time": rospy.get_time(),
                "value": "executed successfully",
            }

        except Exception as e:
            traceback.print_exc()
            print("error supervision_signal_callback", e)
            self.system_events["supervision_signal_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }

            raise Exception("Error in supervision_signal_callback")

    @accumulate_time
    def learning_thread_loop(self) -> None:
        i = 0
        # Learning loop
        while True:
            self.system_events["learning_thread_loop"] = {
                "time": rospy.get_time(),
                "value": "running",
            }
            self.learning_thread_stop_event.wait(timeout=0.01)
            if self.learning_thread_stop_event.is_set():
                rospy.logwarn("Stopped learning thread")
                break
            ts = rospy.get_time()
            if self.is_bad_rate_with_log(
                ts,
                self.last_learning_ts,
                self.learning_thread_rate,
                "learning_thread",
            ):
                self.system_events["learning_thread_iteration_cancelled"] = {
                    "time": rospy.get_time(),
                    "value": "cancelled due to rate",
                }
                continue
            self.last_learning_ts = ts

            self.log("learning_thread_step", i)

            if not self.param.general.online_training:
                self.manager.pause_learning = True
            self.manager.train()

            self.log("training_step", self.manager.step)

            i += 1

        self.system_events["learning_thread_loop"] = {
            "time": rospy.get_time(),
            "value": "finished",
        }
        self.learning_thread_stop_event.clear()

    @accumulate_time
    def pub_node_prediction(
        self,
        node: MainNode,
        ts: Optional[float],
        header: Optional[Header],
    ) -> None:
        if not hasattr(node, "image") or node.image is None:
            return
        # the image feats in node is on cpu
        trans_img = node.image.to(self.device)
        if isinstance(node.features, dict):
            feats_input = {}
            for key, tensor in node.features.items():
                feats_input[key] = tensor.to(self.device)
        else:
            raise ValueError(
                "The features in node is not a dict, only support dict now!"
            )
        res: MaskedPredictionData = (
            self.manager.get_confidence_masked_prediction_from_img(
                trans_img=trans_img,
                compressed_feats=feats_input,
            )
        )
        fric_vis_imgs, stiff_vis_imgs = plot_pred_w_overlay(
            data=res,
            time="online",
            image_name=str(node.timestamp),
            step=self.manager.step,
            param=self.param,
            foothold_label_mask=node._supervision_mask,
            save_local=False,
        )

        if self.param.general.pub_which_pred == "fric":
            overlay_imgs = fric_vis_imgs
            masked_phy_pub = res.masked_output_phy[0]
        elif self.param.general.pub_which_pred == "stiff":
            overlay_imgs = stiff_vis_imgs
            masked_phy_pub = res.masked_output_phy[1]
        else:
            raise ValueError(
                "pub_which_pred should be either 'fric' or 'stiff', got {}".format(
                    self.param.general.pub_which_pred
                )
            )

        if ts:
            self.overlay_img_frames[ts] = overlay_imgs
        else:
            self.overlay_img_frames[node.timestamp] = overlay_imgs
        bridge = CvBridge()
        if self.layer_num == 1:
            masked_phy_pub_np = masked_phy_pub.cpu().numpy()
            # Ensure the numpy array is float32
            if masked_phy_pub_np.dtype != np.float32:
                masked_phy_pub_np = masked_phy_pub_np.astype(np.float32)
            # can adjust to stiff if needed
            img_msg = bridge.cv2_to_imgmsg(masked_phy_pub_np)
        elif self.layer_num == 3:
            last_image_pil = overlay_imgs[-1]  # This is masked prediction image

            #    Convert PIL.Image to a numpy array
            last_image_np = np.array(last_image_pil)

            # Convert the numpy array to a ROS CompressedImage message
            last_image_np = cv2.cvtColor(last_image_np, cv2.COLOR_RGB2BGR)
            img_msg = bridge.cv2_to_imgmsg(last_image_np, "bgr8")

        # calculate phy loss
        self.log("prediction_done", self.log_data["prediction_done"] + 1)
        img_msg.header = header

        # Publish the message
        if self.param.general.pub_which_pred == "fric":
            self.camera_handler["fric_pub"].publish(img_msg)
        elif self.param.general.pub_which_pred == "stiff":
            self.camera_handler["stiff_pub"].publish(img_msg)
        timestamp = rospy.Time.from_sec(ts)
        self.camera_handler["new_camera_info"].header.stamp = timestamp
        self.camera_handler["info_pub"].publish(self.camera_handler["new_camera_info"])

        channel_info_msg = ChannelInfo()
        channel_info_msg.header = deepcopy(
            self.camera_handler["new_camera_info"].header
        )
        channel_str = self.param.general.pub_which_pred
        if self.layer_num == 1:
            channel_info_msg.channels = [channel_str]
        elif self.layer_num == 3:
            channel_info_msg.channels = [channel_str + "_rgb"]
        self.camera_handler["channel_pub"].publish(channel_info_msg)

    def visualize_new_node(
        self,
        pose: Union[np.ndarray, torch.Tensor],
        stamp_sec: float,
        handle: rospy.Publisher,
        name: str,
    ) -> None:
        # publish the last sub node
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)
        msg = Marker()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = rospy.Time.from_sec(stamp_sec)
        msg.pose = rc.se3_to_pose_msg(pose)
        msg.action = Marker.ADD

        msg.id = self.sub_step
        # Set the color of the marker
        if name == "latest_main_node":
            msg.type = Marker.CUBE
            msg.scale.x = 0.2  # size in meters
            msg.scale.y = 0.2
            msg.scale.z = 0.2
            msg.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)
        elif name == "latest_sub_node":
            msg.type = Marker.SPHERE
            msg.scale.x = 0.1  # size in meters
            msg.scale.y = 0.1
            msg.scale.z = 0.1
            msg.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)
        handle.publish(msg)

    def visualize_self_supervision_label_image_overlay(self) -> None:
        """Visualizes the self-supervision label overlaid on the camera image of the vis_node,
        it published as a camera topic and can save locally as well.
        """
        save_local = True
        # Ensure the results/reprojection directory exists
        output_dir = "results/reprojection"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        vis_node: MainNode = self.manager.get_main_node_for_visualization()
        if vis_node is not None:
            torch_image = vis_node._image
            torch_mask = vis_node._supervision_mask
            for i in range(torch_mask.shape[0]):
                out = plot_overlay_image(
                    torch_image, overlay_mask=torch_mask, channel=i, alpha=0.9
                )
                if not save_local:
                    img_msg = rc.numpy_to_ros_image(out)
                    if i == 0:
                        self.camera_handler["fric_overlay_pub"].publish(img_msg)
                    elif i == 1:
                        self.camera_handler["stiff_overlay_pub"].publish(img_msg)
                else:
                    # Convert the numpy array to an image
                    out_image = PIL.Image.fromarray(out)
                    if "v4l2" in self.param.roscfg.camera_topic:
                        out_image = out_image.rotate(180)
                    # Construct a filename
                    filename = f"channel_{i}.jpg"
                    file_path = os.path.join(output_dir, filename)
                    add_color_bar_and_save(out_image, i, file_path)


if __name__ == "__main__":
    node_name = "main_process_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)

    phy_node = MainProcess()

    rospy.spin()
