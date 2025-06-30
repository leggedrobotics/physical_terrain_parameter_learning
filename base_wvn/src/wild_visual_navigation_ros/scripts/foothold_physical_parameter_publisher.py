#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
"""
Pub node to process ros messages, publish the friction and stiffness prediction.
Attention: this use an old stable version of the PHY-decoder.
"""

from base_wvn.utils import NodeForROS
from physical_decoder import DeploymentWrapper

from wild_visual_navigation_msgs.msg import AnymalState
from std_msgs.msg import Float32, Float32MultiArray
from wild_visual_navigation_msgs.msg import PhyDecoderOutput
import message_filters
import rospy
import torch
import traceback


class FootholdPhysicalParameterPublisher(NodeForROS):
    """
    Process /debug_info and publish friction and stiffness prediction. It will be passed on to supervision_label_builder_node to add foothold and other info.
    """

    def __init__(self):
        super().__init__()

        self.physical_decoder = DeploymentWrapper()

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

        print("Start waiting for Phy_decoder_input topic being published!")
        rospy.wait_for_message(self.phy_decoder_input_topic, Float32MultiArray)
        phy_decoder_input_sub = message_filters.Subscriber(
            self.phy_decoder_input_topic, Float32MultiArray
        )

        # Here we only use anymal_state to get timestamp for headerless phy_decoder_input
        self.sub = message_filters.ApproximateTimeSynchronizer(
            [anymal_state_sub, phy_decoder_input_sub],
            queue_size=100,
            slop=0.1,
            allow_headerless=True,
        )

        print("Current ros time is: ", rospy.get_time())

        self.sub.registerCallback(self.callback)

        # Results publisher
        self.phy_decoder_pub = rospy.Publisher(
            self.phy_temp_topic, PhyDecoderOutput, queue_size=10
        )

    def callback(
        self, anymal_state_msg: AnymalState, phy_decoder_input_msg: Float32MultiArray
    ):
        """
        callback function for the anymal state subscriber
        """

        self.system_events["callback_received"] = {
            "time": rospy.get_time(),
            "value": "message received",
        }
        msg = PhyDecoderOutput()
        msg.header.stamp = anymal_state_msg.header.stamp
        msg.header.seq = anymal_state_msg.header.seq
        # msg.header.frame_id shall be determined later in builder

        try:
            """ 
            Fric/Stiff-Decoder input topics & prediction
            """
            phy_decoder_input = torch.tensor(
                phy_decoder_input_msg.data, device=self.device
            ).unsqueeze(0)
            fric_pred, stiff_pred = self.physical_decoder.predict(phy_decoder_input)
            # pub fric and stiff together
            fric_pred = torch.clamp(fric_pred, min=0, max=1)
            stiff_pred = torch.clamp(stiff_pred, min=1, max=10)
            new_priv = torch.cat([fric_pred, stiff_pred], dim=-1)
            new_priv = new_priv.squeeze(0).cpu().numpy()
            msg.prediction = new_priv

            # Publish results
            self.phy_decoder_pub.publish(msg)

        except Exception as e:
            traceback.print_exc()
            print("error callback", e)
            self.system_events["callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
        pass


if __name__ == "__main__":
    node_name = "foothold_physical_parameter_publisher_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)
    phy_node = FootholdPhysicalParameterPublisher()
    rospy.spin()
