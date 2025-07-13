#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
"""
This file contains the all configurations for the Wild-Visual-Navigation project.
"""

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Tuple, List, Optional
from simple_parsing.helpers import Serializable
import numpy as np
import yaml


@dataclass
class ParamCollection(Serializable):
    """A collection of parameters."""

    @dataclass
    class GeneralParams:
        """General parameters for the experiment."""

        name: str = "debug"
        timestamp: bool = True
        online_training: bool = True
        resume_training: bool = False
        resume_training_path: str = "model/last_checkpoint.pt"
        model_path: str = "model"
        pub_pred_as_layer: str = "RGB"  # single or RGB
        pub_which_pred: str = "fric"  # fric or stiff, publish dense prediction under topic name /vd_pipeline/friction or /vd_pipeline/stiffness
        verbose: bool = True

    general: GeneralParams = GeneralParams()

    @dataclass
    class RosParams:
        """Parameters for ROS."""

        feet_list: List[str] = field(
            default_factory=lambda: ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        )

        anymal_state_topic: str = "/anymal/state_estimator/anymal_state"
        phy_decoder_input_topic: str = "/anymal/locomotion_policy/debug_info"
        phy_decoder_temp_topic: str = "/phy_decoder/temp_out"
        phy_decoder_output_topic: str = "/vd_pipeline/phy_decoder_out"
        supervision_label_visualization_topic: str = "/vd_pipeline/visualization_planes"
        masked_dense_friction_prediction_image_topic: str = "/vd_pipeline/friction"
        masked_dense_stiffness_prediction_image_topic: str = "/vd_pipeline/stiffness"
        prediction_image_camera_info_topic: str = "/vd_pipeline/camera_info"
        prediction_image_channel_topic: str = "/vd_pipeline/channel_info"
        main_node_visualization_topic: str = "/vd_pipeline/latest_main_node"
        sub_node_visualization_topic: str = "/vd_pipeline/latest_sub_node"

        camera_topic: str = "/wide_angle_camera_front/image_color_rect/compressed"
        camera_info_topic: str = "/wide_angle_camera_front/camera_info"

        use_vo: bool = False
        vo_world_frame: str = "map_o3d"
        visual_odom_topic: str = "/open3d_slam/scan2map_odometry"

        world_frame: str = "odom"
        base_frame: str = "base"

        foot_radius: float = 0.03269

        # {child}_in_{parent}
        front_hdr_camera_in_base = np.array(
            [
                [-3.63509055e-06, 1.43680318e-01, 9.89624154e-01, 3.53700000e-01],
                [1.00000000e00, -1.34923184e-11, 3.67320510e-06, 0.00000000e00],
                [5.27780629e-07, 9.89624154e-01, -1.43680318e-01, 1.63400000e-01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )

        rear_hdr_camera_in_base = np.array(
            [
                [3.63509049e-06, -1.43680305e-01, -9.89624138e-01, -3.53700000e-01],
                [-9.99999820e-01, 1.34923159e-11, -3.67320444e-06, 0.00000000e00],
                [5.27780582e-07, 9.89623958e-01, -1.43680305e-01, 1.63400000e-01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        rear_wide_angle_camera_in_base = np.array(
            [
                [1.22464680e-16, -1.11022302e-16, -1.00000000e00, -4.04490000e-01],
                [1.00000000e00, 2.22044605e-16, 1.14423775e-17, -5.15576302e-19],
                [0.00000000e00, -1.00000000e00, 0.00000000e00, 2.05000000e-02],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        front_wide_angle_camera_in_base = np.array(
            [
                [0.00000000e00, 1.11022302e-16, 1.00000000e00, 4.04490000e-01],
                [-1.00000000e00, -2.22044605e-16, 1.11022302e-16, 0.00000000e00],
                [0.00000000e00, -1.00000000e00, 0.00000000e00, 2.05000000e-02],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )

        lidar_in_base = np.array(
            [
                [-1.0, 0.0, 0.0, -0.31],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1585],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    roscfg: RosParams = RosParams()

    @dataclass
    class ThreadParams:
        """Parameters for the threads."""

        camera_callback_rate: float = 1.0
        supervision_signal_callback_rate: float = 4.0
        learning_rate: float = 4.0
        logging_rate: float = 1.0

    thread: ThreadParams = ThreadParams()

    @dataclass
    class RunParams:
        """Parameters for the run."""

        device: str = "cuda"
        mode: str = "debug"
        palette: str = "husl"
        seed: int = 42

    run: RunParams = RunParams()

    @dataclass
    class LoggerParams:
        name: str = "neptune"
        neptune_project_name: str = "RSL/WVN"
        vis_new_node: bool = True

    logger: LoggerParams = LoggerParams()

    @dataclass
    class OptimizerParams:
        name: str = "ADAM"
        lr: float = 0.001  # 0.0001 ,0.001
        weight_decay: float = 0.001  # 0.001 ,0

    optimizer: OptimizerParams = OptimizerParams()

    @dataclass
    class FeatParams:
        """Parameters for the feature extractor."""

        feature_type: str = "dinov2"
        resize: bool = True
        input_size: int = 1078  # 1260 for dinov2 , 1078 for dinov2-wideangle camera
        interp: str = "bilinear"
        center_crop: Tuple[bool, int, int] = (True, 910, 910)  # crop or not, H,W
        physical_dim: int = 2

    feat: FeatParams = FeatParams()

    @dataclass
    class LossParams:
        w_pred: float = 0.1  # 0.1
        w_reco: float = 0.9  # 0.9
        reco_loss_type: str = "mse"  # mse or cosine

        confidence_mode: str = "gmm_1d_history"  # gmm_1d, gmm_1d_history, or fixed

    loss: LossParams = LossParams()

    @dataclass
    class GraphParams:
        """Parameters for the graph."""

        max_node_number: int = 40
        update_range_main_graph: float = 5  # 50
        edge_dist_thr_main_graph: float = 0.2  # 0.2

        use_sub_graph: bool = False
        edge_dist_thr_sub_graph: float = 0.05
        max_distance_sub_graph: float = 5
        update_range_sub_graph: float = 5

        min_samples_for_training: int = 1
        random_sample_num: int = 100

        vis_node_index_from_last: int = 1
        label_ext_mode: bool = False  # turn on if want to record the online collected data (nodes, training batches)

    graph: GraphParams = GraphParams()

    @dataclass
    class ModelParams:
        name: str = "SeperateMLP"

        @dataclass
        class SimpleMlpCfgParams:
            input_size: int = 384  # 384 is embedding dim from dinov2-small
            # hidden_sizes: List[int] = field(default_factory=lambda: [32, 8, 32, 2])  # S
            hidden_sizes: List[int] = field(
                default_factory=lambda: [128, 32, 128, 2]
            )  # M
            # hidden_sizes: List[int] = field(default_factory=lambda: [512, 128, 512, 2])
            # hidden_sizes: List[int] = field(
            #     default_factory=lambda: [1024, 512, 256, 512, 1024, 2]
            # )  # L
            # hidden_sizes: List[int] = field(default_factory=lambda: [4096,2048,1024, 512, 1024,2048,4096, 2]) # XL
            reconstruction: bool = True

            def to_dict(self):
                return vars(self)

        simple_mlp_cfg: SimpleMlpCfgParams = SimpleMlpCfgParams()

        @dataclass
        class SeperateMLPCfgParams:
            input_size: int = 384
            hidden_sizes: List[int] = field(
                default_factory=lambda: [128, 32, 128, 2]
            )  # M

            def to_dict(self):
                return vars(self)

        seperate_mlp_cfg: SeperateMLPCfgParams = SeperateMLPCfgParams()

    model: ModelParams = ModelParams()

    @dataclass
    class OfflineParams:
        mode: str = "train"  # or 'eval'
        max_epochs: int = 1
        env: str = "white"  # your own defined, please put the data files into this folder first
        reload_model: bool = False
        use_online_ckpt: bool = False
        replicate_online_training: bool = True
        ckpt_parent_folder: str = "results/overlay"
        data_folder: str = "results/manager"
        train_datafile: str = "train_data.pt"
        nodes_datafile: str = "train_nodes.pt"

        test_on_images: bool = False  # test and output vis for user-provided images
        test_on_nodes: bool = False  # test and output vis for node data
        test_video: bool = False  # output dense pred video from image bag
        img_bag_path: str = "/media/chenc/Chen/2024-01-25-white-board/1st/2024-01-25-19-36-11_anymal-d020-npc_0.bag"  # for video prediction
        process_option: str = "all"  # 'all' or 'first_half' or 'first_100

        upload_error_stats_in_training: bool = True
        gt_model: str = "SAM"
        SAM_type: str = "vit_h"
        SAM_ckpt: str = "/media/chenc/Chen/sam_vit_h_4b8939.pth"

        # vis options, will siginificantly slow down the training since it plots each training step
        # recommend to turn off in training
        plot_hist: bool = False
        plot_nodes: bool = False
        plot_masks_compare: bool = False

    offline: OfflineParams = OfflineParams()


def dataclass_to_dict(obj):
    if is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    return obj


def save_to_yaml(dataclass_instance, filename):
    data_dict = dataclass_to_dict(dataclass_instance)
    with open(filename, "w") as file:
        yaml.dump(data_dict, file)


if __name__ == "__main__":
    params = ParamCollection()
    save_to_yaml(params, "test.yaml")
    pass
