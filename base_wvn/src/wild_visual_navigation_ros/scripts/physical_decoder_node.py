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

from physical_decoder import DeploymentWrapper
from ros_node import RosNode

from signal_logger_msgs.msg import Float32MultiArrayStamped
from wild_visual_navigation_msgs.msg import PhyDecoderOutput
import rospy
import torch
import traceback


class PhysicalDecoderNode(RosNode):
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

        print("Waiting for Phy_decoder_input topic being published!")
        rospy.wait_for_message(self.phy_decoder_input_topic, Float32MultiArrayStamped)

        sub = rospy.Subscriber(
            self.phy_decoder_input_topic,
            Float32MultiArrayStamped,
            self.callback,
            queue_size=1,
        )

        print(
            "OK: Physical decoders begin to publish, and current ros time is: ",
            rospy.get_time(),
        )

        # Results publisher
        self.phy_decoder_pub = rospy.Publisher(
            self.phy_temp_topic, PhyDecoderOutput, queue_size=1
        )

    def callback(self, phy_decoder_input_msg: Float32MultiArrayStamped):
        self.system_events["callback_received"] = {
            "time": rospy.get_time(),
            "value": "message received",
        }
        msg = PhyDecoderOutput()
        msg.header.stamp = phy_decoder_input_msg.header.stamp
        msg.header.seq = phy_decoder_input_msg.header.seq
        # msg.header.frame_id shall be determined later in builder

        try:
            """ 
            Fric/Stiff-Decoder input topics & prediction
            """
            phy_decoder_input = torch.tensor(
                phy_decoder_input_msg.matrix.data, device=self.device
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
    node_name = "physical_decoder_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)
    phy_node = PhysicalDecoderNode()
    rospy.spin()
