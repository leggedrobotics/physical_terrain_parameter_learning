#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
"""
Pub node to relay friction and stiffness prediction msgs from phy_publisher and additionally add foothold and other visulization info.
Attention: this use an old stable version of the PHY-decoder.
"""

from base_wvn.utils import make_ellipsoid, FootFilter, NodeForROS

import ros_converter as rc
from wild_visual_navigation_msgs.msg import AnymalState
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from wild_visual_navigation_msgs.msg import PhyDecoderOutput, PlaneEdge
import message_filters
import rospy
import numpy as np
import traceback
from typing import Optional, Dict, Union


class SupervisionLabelBuilder(NodeForROS):
    def __init__(self):
        super().__init__()

        # Init for PHYSICS DECODERs
        self.step = 0

        # Init contact filter
        self.foot_filters = {foot: FootFilter(foot) for foot in self.feet_list}

        self.system_events = {}

        # Initialize ROS nodes
        self.ros_init()

    def ros_init(self) -> None:
        """
        start ros subscribers and publishers and filter/process topics
        """

        # Anymal state subscriber
        print("Start waiting for AnymalState topic being published!")
        rospy.wait_for_message(self.anymal_state_topic, AnymalState)
        anymal_state_sub = message_filters.Subscriber(
            self.anymal_state_topic, AnymalState
        )

        print("Start waiting for Phy_temp_topic topic being published!")
        rospy.wait_for_message(self.phy_temp_topic, PhyDecoderOutput)
        incomplete_label_sub = message_filters.Subscriber(
            self.phy_temp_topic, PhyDecoderOutput
        )

        if self.use_vo:
            print("Start waiting for visual odom topic being published!")
            rospy.wait_for_message(self.visual_odom_topic, Odometry)
            visual_odom_sub = message_filters.Subscriber(
                self.visual_odom_topic, Odometry
            )
            self.world_frame = "map_o3d"
            
            self.sub = message_filters.ApproximateTimeSynchronizer(
                [anymal_state_sub, incomplete_label_sub, visual_odom_sub],
                queue_size=100,
                slop=0.1,
                allow_headerless=False,
            )
            print("Current ros time is: ", rospy.get_time())

            self.sub.registerCallback(self.callback)
        else:
            self.sub = message_filters.ApproximateTimeSynchronizer(
                [anymal_state_sub, incomplete_label_sub],
                queue_size=100,
                slop=0.1,
                allow_headerless=False,
            )

            print("Current ros time is: ", rospy.get_time())

            self.sub.registerCallback(self.callback_no_vo)

        # Results publisher
        self.phy_decoder_pub = rospy.Publisher(
            "/vd_pipeline/phy_decoder_out", PhyDecoderOutput, queue_size=10
        )
        self.marker_array_pub = rospy.Publisher(
            "/vd_pipeline/visualization_planes", MarkerArray, queue_size=10
        )

    def callback_no_vo(self, anymal_state_msg: AnymalState, phy_tmp_msg: PhyDecoderOutput) -> None:
        self.callback(anymal_state_msg, phy_tmp_msg, None)

    def callback(
        self,
        anymal_state_msg: AnymalState,
        phy_tmp_msg: PhyDecoderOutput,
        visual_odom_msg: Optional[Odometry],
    ) -> None:
        """
        callback function for the anymal state subscriber
        """

        self.step += 1
        self.system_events["callback_received"] = {
            "time": rospy.get_time(),
            "value": "message received",
        }
        msg = phy_tmp_msg
        try:
            pose_base_in_world = self.get_pose_base_in_world(anymal_state_msg, visual_odom_msg)
            msg.base_pose = rc.se3_to_pose_msg(pose_base_in_world)
            msg.header.frame_id = self.world_frame

            # Query 4 feet transforms from AnymalState message
            foot_poses = []
            foot_contacts = []
            foot_planes = []

            feet_info = self.get_feet_info_from_msg(anymal_state_msg, visual_odom_msg)

            for foot in self.feet_list:
                suc, pose_foot_in_world = rc.pq_to_se3((feet_info[foot]["position"], feet_info[foot]["rotation"]))
                if not suc:
                    self.system_events["callback_cancelled"] = {
                        "time": rospy.get_time(),
                        "value": f"cancelled due to pose_{foot}_in_world",
                    }
                    return
                foot_pose = rc.se3_to_pose_msg(pose_foot_in_world)
                foot_poses.append(foot_pose)
                # Make feet circle planes
                d = 2 * self.foot_radius
                foot_plane_points = make_ellipsoid(
                    d, d, 0, pose_foot_in_world, grid_size=24
                )
                foot_plane_points = rc.np_to_geometry_msgs_PointArray(foot_plane_points)
                foot_plane = PlaneEdge()
                foot_plane.edge_points = foot_plane_points
                foot_plane.name = foot
                foot_planes.append(foot_plane)

                foot_contacts.append(feet_info[foot]["contact"])
                        
            msg.feet_poses = foot_poses
            msg.feet_planes = foot_planes
            msg.feet_contact = foot_contacts

            # Publish results
            self.phy_decoder_pub.publish(msg)
            if "debug" in self.mode:
                self.visualize_plane(msg)

        except Exception as e:
            traceback.print_exc()
            print("error callback", e)
            self.system_events["callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }


    def switch_to_feet_pose_in_vo_if_provided(self, state_msg: AnymalState, visual_odom_msg: Optional[Odometry], feet_info: Dict[str, Dict[str, Union[np.ndarray, str, bool]]]) -> None:
        if visual_odom_msg is None:
            return
        pose_base_in_world = rc.msg_to_se3(state_msg.pose.pose)
        # calculate the world_in_map tf
        world_in_map = (
            rc.msg_to_se3(visual_odom_msg)
            @ np.linalg.inv(self.lidar_in_base)
            @ np.linalg.inv(pose_base_in_world)
        )
        for _, info in feet_info.items():
            _, pose_foot_in_world = rc.pq_to_se3((info["position"], info["rotation"]))
            # switch to o3d_map from odom--feet
            pose_foot_in_world = world_in_map @ pose_foot_in_world
            pose_foot_in_world = pose_foot_in_world.astype(np.float32)
            info["position"], info["rotation"] = rc.se3_to_pq(pose_foot_in_world)
    
    def apply_foot_contact_filter(self, feet_info: Dict[str, Dict[str, Union[np.ndarray, str, bool]]],anymal_state_msg: AnymalState) -> None:
        for foot, info in feet_info.items():
            input_pose = np.concatenate([feet_info[foot]["position"], feet_info[foot]["rotation"]])
            filtered_contact = int(
                self.foot_filters[foot].filter(
                    input_pose,
                    feet_info[foot]["contact"],
                    anymal_state_msg.header.stamp.to_sec(),
                )
            )
            feet_info[foot]["contact"] = filtered_contact

    def get_feet_info_from_msg(
        self,
        state_msg: AnymalState,
        visual_odom_msg: Optional[Odometry]
    ) -> Dict[str, Dict[str, Union[np.ndarray, str, bool]]]:
        """Helper function to query TFs

        Args:

            msg (AnymalState, optional): AnymalState message containing the TFs
        """
        res={}

        for transform in state_msg.contacts:
            if "FOOT" in transform.name:
                foot_name = transform.name
                res[foot_name] = {
                    "position": np.array([
                        transform.position.x,
                        transform.position.y,
                        transform.position.z,
                    ]),
                    "rotation": np.array([0, 0, 0, 1]),
                    "frame_id": transform.header.frame_id,
                    "contact": transform.state,
                }
        self.apply_foot_contact_filter(res, state_msg) # must be done before switch to feet pose in VO, because filter operate on the state_estimator data (higher frequency than slam)
        self.switch_to_feet_pose_in_vo_if_provided(state_msg, visual_odom_msg, res)
        return res

    def visualize_plane(self, msg: PhyDecoderOutput) -> None:
        # color.a will be set to 1.0 for foot plane if its contact is true
        header = msg.header
        planes = [x.edge_points for x in msg.feet_planes]
        names = [x.name for x in msg.feet_planes]
        marker_array = MarkerArray()
        for i, plane in enumerate(planes):
            marker = Marker()
            marker.header = header
            marker.ns = names[i]
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            # uncomment this line if you want to see the plane history
            marker.id = self.step * len(planes) + i
            # marker.lifetime=rospy.Duration(10)
            rgb_color = self.color_palette[i % len(self.color_palette)]
            rgba_color = (
                rgb_color[0],
                rgb_color[1],
                rgb_color[2],
                1.0,
            )  # Add alpha value

            marker.pose.orientation.w = 1.0
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0

            # Set the color of the marker
            marker.color.r = rgba_color[0]
            marker.color.g = rgba_color[1]
            marker.color.b = rgba_color[2]
            marker.color.a = rgba_color[3]
            if names[i] in self.feet_list and msg.feet_contact[i] == 0:
                marker.color.a = 0.1
            else:
                marker.points = plane

                marker_array.markers.append(marker)
        self.marker_array_pub.publish(marker_array)


if __name__ == "__main__":
    node_name = "supervision_label_builder_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)
    phy_node = SupervisionLabelBuilder()
    rospy.spin()
