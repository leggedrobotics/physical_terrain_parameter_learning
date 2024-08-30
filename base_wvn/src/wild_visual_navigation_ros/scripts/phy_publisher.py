#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
"""
Main node to process ros messages, publish the relevant topics, train the model...
Attention: this use an old stable version of the PHY-decoder.
"""

from base_wvn.utils import *
from phy_decoder import initialize_models, prepare_padded_input, RNNInputBuffer

from wild_visual_navigation_msgs.msg import AnymalState
from std_msgs.msg import Float32, Float32MultiArray
from wild_visual_navigation_msgs.msg import PhyDecoderOutput
import message_filters
import rospy
import torch
import traceback


class PhyPublsiher(NodeForROS):
    """
    Process /debug_info and publish friction and stiffness prediction. It will be passed on to phy_decoder_node to add foothold and other info.
    """

    def __init__(self):
        super().__init__()

        # Init for PHYSICS DECODERs
        self.step = 0
        self.env_num = 1
        self.fric_predictor, self.stiff_predictor, self.predictor_cfg = (
            initialize_models()
        )
        self.fric_hidden = self.fric_predictor.init_hidden(self.env_num)
        self.stiff_hidden = self.stiff_predictor.init_hidden(self.env_num)
        self.input_buffers = {0: RNNInputBuffer()}

        # Init for storing last footprint pose
        self.last_footprint_pose = None
        self.current_footprint_pose = None

        # Init Decoder handler
        self.decoder_handler = {}
        self.system_events = {}

        # Init contact filter
        self.foot_filters = {foot: FootFilter(foot) for foot in self.feet_list}

        # Initialize ROS nodes
        self.ros_init()

    def ros_init(self):
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
        self.state_sub = message_filters.ApproximateTimeSynchronizer(
            [anymal_state_sub, phy_decoder_input_sub],
            queue_size=100,
            slop=0.1,
            allow_headerless=True,
        )

        print("Current ros time is: ", rospy.get_time())

        self.state_sub.registerCallback(self.state_callback_ori)

        # Results publisher
        phy_decoder_pub = rospy.Publisher(
            self.phy_temp_topic, PhyDecoderOutput, queue_size=10
        )
        test_pub = rospy.Publisher("/vd_pipeline/test", Float32, queue_size=10)
        # stamped_debug_info_pub=rospy.Publisher('/stamped_debug_info', StampedFloat32MultiArray, queue_size=10)
        # Fill in handler
        self.decoder_handler["phy_decoder_pub"] = phy_decoder_pub
        self.decoder_handler["test_pub"] = test_pub

    def state_callback_ori(
        self, anymal_state_msg: AnymalState, phy_decoder_input_msg: Float32MultiArray
    ):
        """
        callback function for the anymal state subscriber
        """

        self.step += 1
        self.system_events["state_callback_received"] = {
            "time": rospy.get_time(),
            "value": "message received",
        }
        msg = PhyDecoderOutput()
        msg.header = anymal_state_msg.header

        # print((rospy.Time.now()-anymal_state_msg.header.stamp)*1e-9)
        try:
            """ 
            Fric/Stiff-Decoder input topics & prediction
            """
            phy_decoder_input = torch.tensor(
                phy_decoder_input_msg.data, device=self.device
            ).unsqueeze(0)
            obs, hidden = torch.split(phy_decoder_input, [341, 100], dim=1)
            input_data = obs[:, :341]
            # debug
            # if not foot_contacts[0]:
            #     A=1
            # foot_scan=obs[:,133:341]
            # mean_foot_scan=torch.mean(foot_scan)
            # mean_lf_scan=torch.mean(foot_scan[:,:52])
            # testmsg=Float32()
            # testmsg.data=mean_lf_scan
            # self.decoder_handler['test_pub'].publish(testmsg)
            padded_inputs = prepare_padded_input(
                input_data, self.input_buffers, self.step, self.env_num
            )
            padded_input = torch.stack(padded_inputs, dim=0)
            if self.predictor_cfg["reset_hidden_each_epoch"]:
                self.fric_hidden = self.fric_predictor.init_hidden(self.env_num)
                self.stiff_hidden = self.stiff_predictor.init_hidden(self.env_num)
            with torch.no_grad():
                # Predict using the friction predictor
                fric_pred, self.fric_hidden = (
                    self.fric_predictor.get_unnormalized_recon(
                        padded_input, self.fric_hidden
                    )
                )

                # Predict using the stiffness predictor
                stiff_pred, self.stiff_hidden = (
                    self.stiff_predictor.get_unnormalized_recon(
                        padded_input, self.stiff_hidden
                    )
                )

            self.input_buffers[0].add(input_data[0].unsqueeze(0))
            # pub fric and stiff together
            if isinstance(fric_pred, torch.Tensor):
                fric_pred = torch.clamp(fric_pred, min=0, max=1)
                stiff_pred = torch.clamp(stiff_pred, min=1, max=10)
                new_priv = torch.cat([fric_pred, stiff_pred], dim=-1)
                new_priv = new_priv[:, -1, :].squeeze(0).cpu().numpy()
                msg.prediction = new_priv
            else:
                fric_recon_loss = fric_pred[2]
                fric_pred_var = fric_pred[1]
                fric_pred_mean = fric_pred[0]
                fric_pred_mean = torch.clamp(fric_pred_mean, min=0, max=1)
                stiff_recon_loss = stiff_pred[2]
                stiff_pred_var = stiff_pred[1]
                stiff_pred_mean = stiff_pred[0]
                stiff_pred_mean = torch.clamp(stiff_pred_mean, min=1, max=10)
                new_priv = torch.cat([fric_pred_mean, stiff_pred_mean], dim=-1)
                new_priv = new_priv[:, -1, :].squeeze(0).cpu().numpy()
                msg.prediction = new_priv

                new_priv_var = torch.cat([fric_pred_var, stiff_pred_var], dim=-1)
                new_priv_var = new_priv_var[:, -1, :].squeeze(0).cpu().numpy()
                msg.prediction_var = new_priv_var
                if fric_recon_loss.shape[0] > 1:
                    recon_loss = torch.cat([fric_recon_loss, stiff_recon_loss], dim=-1)
                else:
                    recon_loss = torch.Tensor([fric_recon_loss, stiff_recon_loss])
                recon_loss = recon_loss.cpu().numpy()
                msg.recon_loss = recon_loss

            # Publish results
            self.decoder_handler["phy_decoder_pub"].publish(msg)
            # if "debug" in self.mode:
            #     self.visualize_plane(msg)

        except Exception as e:
            traceback.print_exc()
            print("error state callback", e)
            self.system_events["state_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
        pass


if __name__ == "__main__":
    node_name = "phy_publisher_node"
    rospy.set_param("/use_sim_time", True)
    rospy.init_node(node_name)
    phy_node = PhyPublsiher()
    rospy.spin()
