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
    NodeForROS,
    FeatureExtractor,
    ImageProjector,
    plot_overlay_image,
    compute_phy_mask,
    add_color_bar_and_save,
)
from base_wvn.graph_manager import Manager, MainNode, SubNode
import ros_converter as rc
import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Float32
from nav_msgs.msg import Path, Odometry
from wild_visual_navigation_msgs.msg import AnymalState
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from wild_visual_navigation_msgs.msg import SystemState, PhyDecoderOutput, ChannelInfo
import os
import rospy
import time
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
from msg_to_transmatrix import msg_to_se3
import datetime
from copy import deepcopy


class MainProcess(NodeForROS):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.sub_step = 0
        # Timers to control the rate of the callbacks
        self.start_time = rospy.get_time()
        self.last_image_ts = self.start_time
        self.last_node_position = None
        self.last_supervision_ts = self.start_time
        self.image_buffer = {}
        self.last_image_saved = self.start_time
        self.today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Init feature extractor
        self.feat_extractor = FeatureExtractor(
            device=self.device,
            segmentation_type=self.segmentation_type,
            feature_type=self.feature_type,
            input_size=self.input_size,
            interp=self.interp,
            center_crop=self.center_crop,
        )

        # Init graph manager
        self.manager = Manager(
            device=self.device,
            param=self.param,
        )

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

        if self.verbose:
            self.log_data = {}
            self.status_thread_stop_event = Event()
            self.status_thread = Thread(target=self.status_thread_loop, name="status")
            self.log_data["Lock"] = Lock()
            self.run_status_thread = True
            self.status_thread.start()

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

    def shutdown_callback(self, *args, **kwargs):
        self.run_status_thread = False
        self.status_thread_stop_event.set()
        self.status_thread.join()
        self.visualize_image_overlay()
        if self.param.general.timestamp:
            print(self.timer)

        # self.manager.save("results/manager","try1")

        if "label" not in self.mode:
            self.learning_thread_stop_event.set()
            self.learning_thread.join()
        if self.manager._label_ext_mode:
            self.manager.save(self.param.offline.data_folder, "train")
            image_buffer_path = os.path.join(
                self.param.offline.data_folder, self.param.offline.image_file
            )
            torch.save(self.image_buffer, image_buffer_path)
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

    def clear_screen(self):
        # For Windows
        if os.name == "nt":
            _ = os.system("cls")
        # For Linux and Mac
        else:
            _ = os.system("clear")

    def status_thread_loop(self):
        # Log loop
        # TODO: make the table prettier, into columns...
        while self.run_status_thread:
            self.status_thread_stop_event.wait(timeout=0.01)
            if self.status_thread_stop_event.is_set():
                rospy.logwarn("Stopped learning thread")
                break

            t = rospy.get_time()
            x = PrettyTable()
            x.field_names = ["Key", "Value"]
            with self.log_data["Lock"]:  # Acquire the lock
                for k, v in self.log_data.items():
                    if "Lock" in k:
                        continue
                    if "time" in k:
                        d = t - v
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
            # self.clear_screen()
            print(x)
            time.sleep(0.1)
            # try:
            #    rate.sleep()
            # except Exception as e:
            #    rate = rospy.Rate(self.ros_params.status_thread_rate)
            #    print("Ignored jump pack in time!")
        self.status_thread_stop_event.clear()

    def ros_init(self):
        """
        start ros subscribers and publishers and filter/process topics
        """

        if self.verbose:
            # DEBUG Logging
            with self.log_data["Lock"]:
                self.log_data["time_last_model"] = -1
                self.log_data["num_model_updates"] = -1
                self.log_data["num_images"] = 0
                self.log_data["time_last_image"] = -1
                self.log_data["image_callback"] = "N/A"
                self.log_data["prediction_done"] = 0

        print("Start waiting for Camera topic being published!")
        # Camera info
        camera_info_msg = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
        new_camera_info_msg = deepcopy(camera_info_msg)
        K, H, W = rc.ros_cam_info_to_tensors(camera_info_msg, device=self.device)

        self.camera_handler["camera_info"] = camera_info_msg
        self.camera_handler["K"] = K
        self.camera_handler["H"] = H
        self.camera_handler["W"] = W
        self.camera_handler["distortion_model"] = camera_info_msg.distortion_model

        # update size info in the feature extractor
        self.feat_extractor.set_original_size(original_height=H, original_width=W)
        ratio_x, ratio_y = self.feat_extractor.resize_ratio
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
            self.fixed_frame = "map_o3d"
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
            sync.registerCallback(self.camera_callback_ori, self.camera_topic)

        # Phy-decoder info subscriber
        phy_sub = rospy.Subscriber(
            self.param.roscfg.phy_decoder_output_topic,
            PhyDecoderOutput,
            self.phy_decoder_callback,
            queue_size=1,
        )

        # Fill in handler
        self.camera_handler["name"] = self.camera_topic
        self.camera_handler["img_sub"] = camera_sub

        # camera tf broadcaster
        self.camera_br = tf2_ros.TransformBroadcaster()

        # Results publisher
        input_pub = rospy.Publisher("/vd_pipeline/image_input", Image, queue_size=10)
        fric_pub = rospy.Publisher("/vd_pipeline/friction", Image, queue_size=10)
        stiff_pub = rospy.Publisher(
            "/vd_pipeline/stiffness", CompressedImage, queue_size=10
        )
        conf_pub = rospy.Publisher("/vd_pipeline/confidence", Image, queue_size=10)
        info_pub = rospy.Publisher(
            "/vd_pipeline/camera_info", CameraInfo, queue_size=10
        )
        channel_pub = rospy.Publisher(
            "/vd_pipeline/channel_info", ChannelInfo, queue_size=10
        )
        freq_pub = rospy.Publisher("/test", Float32, queue_size=10)
        main_graph_pub = rospy.Publisher("/vd_pipeline/main_graph", Path, queue_size=10)
        nodes_pub = rospy.Publisher("/vd_pipeline/nodes", Path, queue_size=10)
        sub_node_pub = rospy.Publisher("/vd_pipeline/sub_node", Marker, queue_size=10)
        latest_img_pub = rospy.Publisher(
            "/vd_pipeline/latest_img", Marker, queue_size=10
        )
        latest_phy_pub = rospy.Publisher(
            "/vd_pipeline/latest_phy", Marker, queue_size=10
        )
        system_state_pub = rospy.Publisher(
            "/vd_pipeline/system_state", SystemState, queue_size=10
        )

        # Fill in handler
        self.camera_handler["input_pub"] = input_pub
        self.camera_handler["fric_pub"] = fric_pub
        self.camera_handler["stiff_pub"] = stiff_pub
        self.camera_handler["conf_pub"] = conf_pub
        self.camera_handler["info_pub"] = info_pub
        self.camera_handler["channel_pub"] = channel_pub
        self.camera_handler["freq_pub"] = freq_pub
        self.camera_handler["main_graph_pub"] = main_graph_pub
        self.camera_handler["sub_node_pub"] = sub_node_pub
        self.camera_handler["system_state_pub"] = system_state_pub
        self.camera_handler["latest_img_pub"] = latest_img_pub
        self.camera_handler["latest_phy_pub"] = latest_phy_pub
        self.camera_handler["nodes_pub"] = nodes_pub
        pass

    @accumulate_time
    def camera_callback(
        self,
        img_msg: CompressedImage,
        state_msg: AnymalState,
        visual_odom_msg: Odometry,
        cam: str,
    ):
        """
        callback function for the anymal state subscriber
        """
        self.system_events["camera_callback_received"] = {
            "time": rospy.get_time(),
            "value": "message received",
        }
        try:
            # Run the callback so as to match the desired rate
            ts = img_msg.header.stamp.to_sec()
            if abs(ts - self.last_image_ts) < 1.0 / self.image_callback_rate:
                if self.verbose:
                    with self.log_data["Lock"]:
                        self.log_data["image_callback"] = "skipping"
                return
            else:
                if self.verbose:
                    with self.log_data["Lock"]:
                        self.log_data["image_callback"] = "processing"

            self.log_data["ros_time_now"] = rospy.get_time()
            if "debug" in self.mode:
                # pub for testing frequency
                freq_pub = self.camera_handler["freq_pub"]
                msg = Float32()
                msg.data = 1.0
                freq_pub.publish(msg)

            # load MLP and confidence generator params if possible,
            # don't need it, just use the training model to predict, no need for save and reload
            # self.load_model()

            # prepare tf from base to camera
            transform = state_msg.pose
            trans = (
                transform.pose.position.x,
                transform.pose.position.y,
                transform.pose.position.z,
            )
            rot = (
                transform.pose.orientation.x,
                transform.pose.orientation.y,
                transform.pose.orientation.z,
                transform.pose.orientation.w,
            )
            suc, pose_base_in_world = rc.ros_tf_to_numpy((trans, rot))

            # calculate the world_in_map tf
            world_in_map = (
                msg_to_se3(visual_odom_msg)
                @ np.linalg.inv(self.lidar_in_base)
                @ np.linalg.inv(pose_base_in_world)
            )
            # switch to o3d_map from odom--base
            pose_base_in_world = world_in_map @ pose_base_in_world
            pose_base_in_world = pose_base_in_world.astype(np.float32)

            # if self.last_node_position is not None:
            #     dist=np.linalg.norm(pose_base_in_world[:3,3]-self.last_node_position)
            #     if dist<self.param.graph.edge_dist_thr_main_graph:
            #         with self.log_data["Lock"]:
            #             self.log_data[f"image_callback"] = "skipping, mnode too close"
            #         return
            self.last_node_position = pose_base_in_world[:3, 3]
            if self.param.logger.vis_callback:
                self.visualize_callback(
                    pose_base_in_world,
                    ts,
                    self.camera_handler["latest_img_pub"],
                    "latest_img",
                )

            if not suc:
                self.system_events["camera_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to pose_base_in_world",
                }
                return

            # transform the camera pose from base to world
            pose_cam_in_base = self.camera_in_base
            pose_cam_in_world = np.matmul(pose_base_in_world, pose_cam_in_base)
            self.camera_handler["pose_cam_in_world"] = pose_cam_in_world

            # send tf , vis in rviz
            # self.broadcast_tf_from_matrix(pose_cam_in_world,self.fixed_frame,"hdr_rear_camera")

            # prepare image
            img_torch = rc.ros_image_to_torch(img_msg, device=self.device)
            img_torch = img_torch[None]
            features, seg, transformed_img, compressed_feats = (
                self.feat_extractor.extract(img_torch)
            )

            image_projector = ImageProjector(
                K=self.camera_handler["K_scaled"],
                h=self.camera_handler["H_scaled"],
                w=self.camera_handler["W_scaled"],
            )
            # tolist is expensive
            # msg=FeatExtractorOutput()
            # msg.header=img_msg.header
            # # msg.features=features.reshape(-1).cpu().numpy()
            # msg.segments=seg.cpu().numpy().flatten().tolist()
            # msg.resized_image=transformed_img.cpu().numpy().flatten().tolist()
            # msg.ori_camera_info=self.camera_handler["camera_info"]
            # msg.resized_K=self.camera_handler["K_scaled"].cpu().numpy().flatten().tolist()
            # msg.resized_height=self.camera_handler["H_scaled"]
            # msg.resized_width=self.camera_handler["W_scaled"]

            if self.manager._label_ext_mode:
                if abs(ts - self.last_image_saved) > 10.0:
                    self.image_buffer[ts] = transformed_img
                    self.last_image_saved = ts
                    if self.verbose:
                        with self.log_data["Lock"]:
                            self.log_data["num_images"] += 1
                            self.log_data["time_last_image"] = rospy.get_time()
            main_node = MainNode(
                timestamp=img_msg.header.stamp.to_sec(),
                pose_base_in_world=torch.from_numpy(pose_base_in_world).to(self.device),
                pose_cam_in_world=torch.from_numpy(pose_cam_in_world).to(self.device),
                image=transformed_img,
                features=features
                if self.segmentation_type != "pixel"
                else compressed_feats,
                feature_type=self.feature_type,
                segments=seg,
                image_projector=image_projector,
                camera_name=cam,
                use_for_training=self.param.graph.use_for_training,
                phy_dim=self.param.feat.physical_dim,
            )

            # add to main graph
            added_new_node = self.manager.add_main_node(
                main_node, verbose=self.verbose, logger=self.log_data
            )
            # if added_new_node:
            # add for paper video

            if self.param.logger.vis_mgraph:
                # publish the main graph
                self.visualize_main_graph()
            if added_new_node:
                self.manager.update_visualization_node()
            if self.manager.subnodes_update is not None:
                self.visualize_nodes(self.manager.subnodes_update)
            with self.log_data["Lock"]:
                if self.manager._graph_distance is not None:
                    self.log_data["head dist of main/sub graph"] = "{:.2f}".format(
                        self.manager._graph_distance.item()
                    )
            # self.visualize_image_overlay()
            self.system_events["image_callback_state"] = {
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
        pass

    @accumulate_time
    def camera_callback_ori(
        self, img_msg: CompressedImage, state_msg: AnymalState, cam: str
    ):
        """
        callback function for the anymal state subscriber
        """
        self.system_events["camera_callback_received"] = {
            "time": rospy.get_time(),
            "value": "message received",
        }
        try:
            # Run the callback so as to match the desired rate
            ts = img_msg.header.stamp.to_sec()
            if abs(ts - self.last_image_ts) < 1.0 / self.image_callback_rate:
                if self.verbose:
                    with self.log_data["Lock"]:
                        self.log_data["image_callback"] = "skipping"
                return
            else:
                if self.verbose:
                    with self.log_data["Lock"]:
                        self.log_data["image_callback"] = "processing"

            self.log_data["ros_time_now"] = rospy.get_time()
            if "debug" in self.mode:
                # pub for testing frequency
                freq_pub = self.camera_handler["freq_pub"]
                msg = Float32()
                msg.data = 1.0
                freq_pub.publish(msg)

            # load MLP and confidence generator params if possible,
            # don't need it, just use the training model to predict, no need for save and reload
            # self.load_model()

            # prepare tf from base to camera
            transform = state_msg.pose
            trans = (
                transform.pose.position.x,
                transform.pose.position.y,
                transform.pose.position.z,
            )
            rot = (
                transform.pose.orientation.x,
                transform.pose.orientation.y,
                transform.pose.orientation.z,
                transform.pose.orientation.w,
            )
            suc, pose_base_in_world = rc.ros_tf_to_numpy((trans, rot))

            pose_base_in_world = pose_base_in_world.astype(np.float32)
            if self.param.logger.vis_callback:
                self.visualize_callback(
                    pose_base_in_world,
                    ts,
                    self.camera_handler["latest_img_pub"],
                    "latest_img",
                )

            if not suc:
                self.system_events["camera_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to pose_base_in_world",
                }
                return

            # transform the camera pose from base to world
            pose_cam_in_base = self.camera_in_base
            pose_cam_in_world = np.matmul(pose_base_in_world, pose_cam_in_base)
            self.camera_handler["pose_cam_in_world"] = pose_cam_in_world

            # send tf , vis in rviz
            # self.broadcast_tf_from_matrix(pose_cam_in_world,self.fixed_frame,"hdr_rear_camera")

            # prepare image
            img_torch = rc.ros_image_to_torch(img_msg, device=self.device)
            img_torch = img_torch[None]
            features, seg, transformed_img, compressed_feats = (
                self.feat_extractor.extract(img_torch)
            )

            image_projector = ImageProjector(
                K=self.camera_handler["K_scaled"],
                h=self.camera_handler["H_scaled"],
                w=self.camera_handler["W_scaled"],
            )
            # tolist is expensive
            # msg=FeatExtractorOutput()
            # msg.header=img_msg.header
            # # msg.features=features.reshape(-1).cpu().numpy()
            # msg.segments=seg.cpu().numpy().flatten().tolist()
            # msg.resized_image=transformed_img.cpu().numpy().flatten().tolist()
            # msg.ori_camera_info=self.camera_handler["camera_info"]
            # msg.resized_K=self.camera_handler["K_scaled"].cpu().numpy().flatten().tolist()
            # msg.resized_height=self.camera_handler["H_scaled"]
            # msg.resized_width=self.camera_handler["W_scaled"]

            if self.manager._label_ext_mode:
                if abs(ts - self.last_image_saved) > 10.0:
                    self.image_buffer[ts] = transformed_img
                    self.last_image_saved = ts
                    if self.verbose:
                        with self.log_data["Lock"]:
                            self.log_data["num_images"] += 1
                            self.log_data["time_last_image"] = rospy.get_time()
            main_node = MainNode(
                timestamp=img_msg.header.stamp.to_sec(),
                pose_base_in_world=torch.from_numpy(pose_base_in_world).to(self.device),
                pose_cam_in_world=torch.from_numpy(pose_cam_in_world).to(self.device),
                image=transformed_img,
                features=features
                if self.segmentation_type != "pixel"
                else compressed_feats,
                feature_type=self.feature_type,
                segments=seg,
                image_projector=image_projector,
                camera_name=cam,
                use_for_training=self.param.graph.use_for_training,
                phy_dim=self.param.feat.physical_dim,
            )

            # add to main graph
            added_new_node = self.manager.add_main_node(
                main_node, verbose=self.verbose, logger=self.log_data
            )
            # if added_new_node:
            #     # add for paper video
            self.update_prediction(main_node, ts, img_msg.header)
            if self.param.logger.vis_mgraph:
                # publish the main graph
                self.visualize_main_graph()
            if added_new_node:
                self.manager.update_visualization_node()
            if self.manager.subnodes_update is not None:
                self.visualize_nodes(self.manager.subnodes_update)
            with self.log_data["Lock"]:
                if self.manager._graph_distance is not None:
                    self.log_data["head dist of main/sub graph"] = "{:.2f}".format(
                        self.manager._graph_distance.item()
                    )

            self.system_events["image_callback_state"] = {
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
        pass

    @accumulate_time
    def phy_decoder_callback(self, phy_output: PhyDecoderOutput):
        self.system_events["phy_decoder_callback_received"] = {
            "time": rospy.get_time(),
            "value": "message received",
        }
        try:
            ts = phy_output.header.stamp.to_sec()
            if abs(ts - self.last_supervision_ts) < 1.0 / self.proprio_callback_rate:
                self.system_events["phy_decoder_callback_cancled"] = {
                    "time": rospy.get_time(),
                    "value": "cancled due to rate",
                }
                if self.verbose:
                    with self.log_data["Lock"]:
                        self.log_data["phy_decoder_callback"] = "skipping"
                return
            else:
                if self.verbose:
                    with self.log_data["Lock"]:
                        self.log_data["phy_decoder_callback"] = "processing"
            self.last_supervision_ts = ts

            pose_base_in_world = rc.ros_pose_to_torch(
                phy_output.base_pose, device=self.device
            )

            if self.param.logger.vis_callback:
                self.visualize_callback(
                    pose_base_in_world,
                    ts,
                    self.camera_handler["latest_phy_pub"],
                    "latest_phy",
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
            self.manager.add_sub_node(sub_node, logger=self.log_data)

            self.sub_step += 1

            if self.param.logger.vis_snodes:
                self.visualize_sub_node()
            self.system_events["phy_decoder_callback_state"] = {
                "time": rospy.get_time(),
                "value": "executed successfully",
            }

        except Exception as e:
            traceback.print_exc()
            print("error phy_decoder_callback", e)
            self.system_events["phy_decoder_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }

            raise Exception("Error in phy_decoder_callback")

        pass

    @accumulate_time
    def learning_thread_loop(self):
        # rate = rospy.Rate(self.param.thread.learning_rate)
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
            with self.log_data["Lock"]:
                self.log_data["learning_thread_step"] = i

            if not self.param.general.online_training:
                self.manager.pause_learning = True
            res = self.manager.train()

            with self.log_data["Lock"]:
                self.log_data["training_step"] = self.manager.step
            step_time = rospy.get_time()
            step = self.manager.step
            # Publish loss
            system_state = SystemState()
            for k in res.keys():
                if hasattr(system_state, k):
                    setattr(system_state, k, res[k])
            system_state.pause_learning = self.manager.pause_learning
            system_state.mode = self.mode
            system_state.step = step
            system_state.header.stamp = rospy.Time.from_sec(step_time)
            self.camera_handler["system_state_pub"].publish(system_state)
            # rate.sleep()
            # save model every 10 steps, comment for paper video
            # if i % 10== 0:
            #     if self.param.general.online_training:
            #         self.manager.save_ckpt(self.param.general.model_path,f"checkpoint_{step}.pt")
            #     # update real-time pred once
            #     # self.update_prediction(self.manager._vis_main_node)
            #     linodes=self.manager._main_graph.get_valid_nodes()
            #     if len(linodes)>0:
            #         self.update_prediction(linodes[0])
            # linodes=self.manager._main_graph.get_valid_nodes()
            # if len(linodes)>0:
            #     self.update_prediction(linodes[0],step_time)
            # or plot the latest node
            # self.update_prediction(main_node)
            i += 1
            time.sleep(1 / self.param.thread.learning_rate)

        self.system_events["learning_thread_loop"] = {
            "time": rospy.get_time(),
            "value": "finished",
        }
        self.learning_thread_stop_event.clear()

    @accumulate_time
    def update_prediction(self, node: MainNode, ts=None, header=None):
        if not hasattr(node, "image") or node.image is None:
            return
        img = node.image.to(self.device)
        B, C, H, W = img.shape
        self.feat_extractor.set_original_size(W, H)
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
        res_dict = compute_phy_mask(
            img,
            self.feat_extractor,
            self.manager._model,
            self.manager._phy_loss,
            self.param.loss.confidence_threshold,
            self.param.loss.confidence_mode,
            self.param.general.plot_overlay_online,
            self.manager.step,
            image_name=str(node.timestamp),
            trans_img=trans_img,
            compressed_feats=feats_input,
            param=self.param,
            label_mask=node._supervision_mask,
        )
        # conf_mask=res_dict['conf_mask']
        # pred_phy_mask=res_dict['output_phy']
        # ori_phy_mask= node._supervision_mask.to(self.param.run.device)
        overlay_imgs = res_dict["overlay_imgs"]
        if ts:
            self.overlay_img_frames[ts] = overlay_imgs
        else:
            self.overlay_img_frames[node.timestamp] = overlay_imgs
        bridge = CvBridge()
        if self.layer_num == 1:
            masked_phy = res_dict["output_phy"]
            fric_masked_phy = masked_phy[0]
            if fric_masked_phy.is_cuda:
                fric_masked_phy_np = fric_masked_phy.cpu().numpy()
            else:
                fric_masked_phy_np = fric_masked_phy.numpy()

            # Ensure the numpy array is float32
            if fric_masked_phy_np.dtype != np.float32:
                fric_masked_phy_np = fric_masked_phy_np.astype(np.float32)
            img_msg = bridge.cv2_to_imgmsg(fric_masked_phy_np)
        elif self.layer_num == 3:
            last_image_pil = overlay_imgs[-1]  # This is your PIL.Image object

            #    Convert PIL.Image to a numpy array
            last_image_np = np.array(last_image_pil)

            # Convert the numpy array to a ROS CompressedImage message

            last_image_np = cv2.cvtColor(last_image_np, cv2.COLOR_RGB2BGR)
            img_msg = bridge.cv2_to_imgmsg(last_image_np, "bgr8")

        # calculate phy loss
        # phy_loss_dict=compute_pred_phy_loss(img,conf_mask,pred_phy_mask=pred_phy_mask,ori_phy_mask=ori_phy_mask)
        with self.log_data["Lock"]:
            self.log_data["prediction_done"] += 1

        img_msg.header = header

        # Publish the message
        self.camera_handler["fric_pub"].publish(img_msg)
        timestamp = rospy.Time.from_sec(ts)
        self.camera_handler["new_camera_info"].header.stamp = timestamp
        self.camera_handler["info_pub"].publish(self.camera_handler["new_camera_info"])

        channel_info_msg = ChannelInfo()
        channel_info_msg.header = deepcopy(
            self.camera_handler["new_camera_info"].header
        )
        if self.layer_num == 1:
            channel_info_msg.channels = ["friction"]
        elif self.layer_num == 3:
            channel_info_msg.channels = ["friction_rgb"]
        self.camera_handler["channel_pub"].publish(channel_info_msg)

    @accumulate_time
    def broadcast_tf_from_matrix(self, matrix, parent_frame, child_frame):
        br = self.camera_br
        t = TransformStamped()

        if isinstance(matrix, torch.Tensor):
            pass
        elif isinstance(matrix, np.ndarray):
            matrix = torch.from_numpy(matrix)
        pose = rc.torch_to_ros_pose(matrix)
        # Extract translation
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # Copy position
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z

        # Copy orientation
        t.transform.rotation.x = pose.orientation.x
        t.transform.rotation.y = pose.orientation.y
        t.transform.rotation.z = pose.orientation.z
        t.transform.rotation.w = pose.orientation.w

        br.sendTransform(t)

    def load_model(self):
        """
        load the model from the checkpoint
        """
        try:
            self.step += 1
            if self.step % 10 == 0:
                print(f"Loading model from checkpoint {self.step}")

                pass
        except Exception as e:
            if self.verbose:
                print(f"Model Loading Failed: {e}")

    def visualize_sub_node(self):
        # publish the last sub node
        msg = Marker()
        msg.header.frame_id = self.fixed_frame
        msg.header.stamp = rospy.Time.from_sec(self.manager.last_sub_node.timestamp)
        msg.pose = rc.torch_to_ros_pose(self.manager.last_sub_node.pose_base_in_world)
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD
        msg.scale.x = 0.2  # size in meters
        msg.scale.y = 0.2
        msg.scale.z = 0.2
        msg.id = self.sub_step
        # Set the color of the marker
        msg.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)
        self.camera_handler["sub_node_pub"].publish(msg)

    def visualize_callback(self, pose, stamp, handle, name):
        # publish the last sub node
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)
        msg = Marker()
        msg.header.frame_id = self.fixed_frame
        msg.header.stamp = rospy.Time.from_sec(stamp)
        msg.pose = rc.torch_to_ros_pose(pose)
        msg.type = Marker.CUBE
        msg.action = Marker.ADD

        msg.id = self.sub_step
        # Set the color of the marker
        if name == "latest_img":
            msg.scale.x = 0.2  # size in meters
            msg.scale.y = 0.2
            msg.scale.z = 0.2
            msg.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)
        elif name == "latest_phy":
            msg.scale.x = 0.1  # size in meters
            msg.scale.y = 0.1
            msg.scale.z = 0.1
            msg.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)
        handle.publish(msg)

    def visualize_nodes(self, nodes):
        """Publishes all the visualizations related to the nodes"""
        now = rospy.Time.from_sec(self.last_image_ts)

        # publish main graph
        nodes_msg = Path()
        nodes_msg.header.frame_id = self.fixed_frame
        nodes_msg.header.stamp = now

        for node in nodes:
            pose = PoseStamped()
            pose.header.frame_id = self.fixed_frame
            pose.header.stamp = rospy.Time.from_sec(node.timestamp)
            pose.pose = rc.torch_to_ros_pose(node.pose_base_in_world)
            nodes_msg.poses.append(pose)
        self.camera_handler["nodes_pub"].publish(nodes_msg)

    @accumulate_time
    def visualize_main_graph(self):
        """Publishes all the visualizations related to the mission graph"""
        now = rospy.Time.from_sec(self.last_image_ts)

        # publish main graph
        main_graph_msg = Path()
        main_graph_msg.header.frame_id = self.fixed_frame
        main_graph_msg.header.stamp = now

        for node in self.manager.get_main_nodes():
            pose = PoseStamped()
            pose.header.frame_id = self.fixed_frame
            pose.header.stamp = rospy.Time.from_sec(node.timestamp)
            pose.pose = rc.torch_to_ros_pose(node.pose_cam_in_world)
            main_graph_msg.poses.append(pose)
        self.camera_handler["main_graph_pub"].publish(main_graph_msg)

    def visualize_image_overlay(self):
        """Publishes all the debugging, slow visualizations"""
        save_local = True
        # Ensure the results/reprojection directory exists
        output_dir = "results/reprojection"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        vis_node: MainNode = self.manager.get_main_node_for_visualization()
        if vis_node is not None:
            cam = vis_node.camera_name
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
                    # Save the image
                    # out_image.save(file_path)


if __name__ == "__main__":
    node_name = "main_process_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)

    phy_node = MainProcess()

    rospy.spin()
