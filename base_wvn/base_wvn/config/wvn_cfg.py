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

        name: str = "debug/debug"
        timestamp: bool = True
        online_training: bool = True
        resume_training: bool = False
        resume_training_path: str = "model/last_checkpoint.pt"
        plot_overlay_online: bool = True
        model_path: str = "model"
        pub_pred_as_layer: str = "RGB"  # single or RGB
        pub_which_pred: str = "fric"  # fric or stiff
        # ... [rest of the attributes]

    general: GeneralParams = GeneralParams()

    @dataclass
    class RosParams:
        """Parameters for ROS."""

        anymal_bag_name: str = "lpc"
        anymal_state_topic: str = "/state_estimator/anymal_state"
        feet_list: List[str] = field(
            default_factory=lambda: ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        )
        phy_decoder_input_topic: str = "/debug_info"
        phy_decoder_temp_topic: str = "/phy_decoder/temp_out"
        phy_decoder_output_topic: str = "/vd_pipeline/phy_decoder_out"

        camera_bag_name: str = "jetson"
        # camera_topic: str='/hdr_camera/image_raw/compressed'
        # camera_info_topic: str='/hdr_camera/camera_info'
        # camera_topic: str='/v4l2_camera/image_raw_throttle/compressed'
        # camera_info_topic: str='/v4l2_camera/camera_info_throttle'
        camera_topic: str = "/wide_angle_camera_rear/image_color_rect/compressed"
        camera_info_topic: str = "/wide_angle_camera_rear/camera_info"
        # camera_topic: str = "/wide_angle_camera_front/image_color_rect/compressed"
        # camera_info_topic: str = "/wide_angle_camera_front/camera_info"

        use_vo: bool = False
        visual_odom_topic: str = "/open3d_slam/scan2map_odometry"

        fixed_frame: str = "odom"
        base_frame: str = "base"
        footprint_frame: str = "footprint"

        robot_length: float = 0.930
        robot_height: float = 0.890
        robot_width: float = 0.530
        robot_max_velocity: float = 1.2
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

        pass

    roscfg: RosParams = RosParams()

    @dataclass
    class ThreadParams:
        """Parameters for the threads."""

        image_callback_rate: float = 1.0
        proprio_callback_rate: float = 4.0
        learning_rate: float = 2
        logging_rate: float = 0.5

    thread: ThreadParams = ThreadParams()

    @dataclass
    class RunParams:
        """Parameters for the run."""

        device: str = "cuda"
        mode: str = "debug"
        palette: str = "husl"
        print_time: bool = True
        pass

    run: RunParams = RunParams()

    @dataclass
    class LoggerParams:
        name: str = "neptune"
        neptune_project_name: str = "RSL/WVN"
        vis_callback: bool = True
        vis_mgraph: bool = True
        vis_snodes: bool = True

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

        segmentation_type: str = "pixel"
        feature_type: str = "dinov2"  # dinov2, focal
        resize: bool = True
        input_size: int = (
            1078  # 1260 for dinov2 , 1280 for focal,1078 for dinov2-wideangle camera
        )
        interp: str = "bilinear"
        center_crop: Tuple[bool, int, int] = (True, 910, 910)  # crop or not, H,W
        physical_dim: int = 2

    feat: FeatParams = FeatParams()

    @dataclass
    class LossParams:
        w_pred: float = 0.1  # 0.1
        w_reco: float = 0.9  # 0.9
        method: str = "running_mean"
        confidence_std_factor: float = 1.0
        confidence_threshold: float = 0.5
        confidence_mode: str = "gmm_1d"  # gmm_1d,gmm_all,fixed
        init_precision: List[float] = field(default_factory=lambda: [0.1, 10.0])
        init_mean: List[float] = field(default_factory=lambda: [0.0, 2.0])
        log_enabled: bool = False
        log_folder: str = "/tmp"
        verbose: bool = True

        reco_loss_type: str = "mse"  # mse or cosine

    loss: LossParams = LossParams()

    @dataclass
    class GraphParams:
        """Parameters for the graph."""

        update_range_main_graph: float = 5  # 50
        cut_threshold: float = 5.0
        edge_dist_thr_main_graph: float = 1.0  # 0.2

        use_sub_graph: bool = False  # only use when the robot is walking reversely
        edge_dist_thr_sub_graph: float = 0.05
        max_distance_sub_graph: float = 5
        update_range_sub_graph: float = 5

        min_samples_for_training: int = 1
        random_sample_num: int = 100

        vis_node_index: int = 10
        label_ext_mode: bool = False
        extraction_store_folder: str = "LabelExtraction"
        use_for_training: bool = True

    graph: GraphParams = GraphParams()

    @dataclass
    class ModelParams:
        name: str = "SimpleMLP"  #  SimpleMLP, SeperateMLP,RndMLP,SeprndMLP
        load_ckpt: Optional[str] = None

        @dataclass
        class SimpleMlpCfgParams:
            input_size: int = (
                384  # 2880 for focal, 384 is embedding dim from dinov2-small
            )
            # hidden_sizes: List[int] = field(default_factory=lambda: [32, 8, 32, 2])  # S
            # hidden_sizes: List[int] = field(default_factory=lambda: [128, 32, 128, 2])   # M
            # hidden_sizes: List[int] = field(default_factory=lambda: [512, 128, 512, 2])
            hidden_sizes: List[int] = field(
                default_factory=lambda: [1024, 512, 256, 512, 1024, 2]
            )  # L
            # hidden_sizes: List[int] = field(default_factory=lambda: [4096,2048,1024, 512, 1024,2048,4096, 2]) # XL
            reconstruction: bool = True

            def to_dict(self):
                return vars(self)

        simple_mlp_cfg: SimpleMlpCfgParams = SimpleMlpCfgParams()

        @dataclass
        class SeperateMLPCfgParams:
            input_size: int = 384
            hidden_sizes: List[int] = field(default_factory=lambda: [256, 64, 256, 2])

            def to_dict(self):
                return vars(self)

        seperate_mlp_cfg: SeperateMLPCfgParams = SeperateMLPCfgParams()

        @dataclass
        class RndMLPCfgParams:
            input_size: int = 384
            hidden_sizes_target: List[int] = field(default_factory=lambda: [128, 32])
            hidden_sizes_pred: List[int] = field(default_factory=lambda: [128, 32])
            pred_head: int = 2

            def to_dict(self):
                return vars(self)

        rnd_mlp_cfg: RndMLPCfgParams = RndMLPCfgParams()

        @dataclass
        class SeprndMLPCfgParams:
            input_size: int = 384
            hidden_sizes_target: List[int] = field(default_factory=lambda: [256, 64])
            hidden_sizes_pred: List[int] = field(default_factory=lambda: [256, 64])
            pred_head: int = 2

            def to_dict(self):
                return vars(self)

        seprnd_mlp_cfg: SeprndMLPCfgParams = SeprndMLPCfgParams()

    model: ModelParams = ModelParams()

    @dataclass
    class OfflineParams:
        mode: str = "train"
        env: str = "vowhite_both"  # vowhite_both
        reload_model: bool = False
        use_online_ckpt: bool = False
        ckpt_parent_folder: str = "results/overlay"
        data_folder: str = "results/manager"
        train_datafile: str = "train_data.pt"
        nodes_datafile: str = "train_nodes.pt"
        image_file: str = "image_buffer.pt"
        # img_bag_path:str='/media/chen/UDisk1/vis_rosbag/snow/2022-12-10-15-40-10_anymal-d020-npc_mission_0.bag'
        # img_bag_path:str='/media/chen/Chen/20240211_Dodo_MPI/2024_02_11_Dodo_MPI_Vicon/2024-02-11-14-28-25/mission_data/2024-02-11-14-28-25_npc_wide_angle_camera_0.bag'
        img_bag_path: str = "/media/chenc/Chen/2024-01-25-white-board/2nd/2024-01-25-19-38-19_anymal-d020-npc_0.bag"
        # img_bag_path:str='/media/chen/Chen/2024-01-25-white-board/1st/2024-01-25-19-36-11_anymal-d020-npc_0.bag'
        # img_bag_path:str='/media/chen/Chen/rosbag_white/2nd/2024-01-16-21-45-48_anymal-d020-npc_0-003.bag'
        # img_bag_path:str='/media/chen/Chen/rosbag_lee/2023-12-03-11-57-12_anymal-d020-npc_1-004.bag'
        traindata_option: str = (
            "each_full"  # 'each_full' or 'each_partial' or 'all_full' or 'all_partial'
        )

        test_images: bool = False  # output vis for image_buffer
        test_nodes: bool = False  # output vis for node data
        test_video: bool = False  # output dense pred video
        process_option: str = "all"  # 'all' or 'first_half' or 'first_100

        random_datasample: Tuple[bool, int] = (False, 40)
        upload_error_stats_in_training: bool = True
        gt_model: str = "SAM"  # 'SEEM' or 'SAM'
        SAM_type: str = "vit_h"
        SAM_ckpt: str = "/media/chenc/Chen/sam_vit_h_4b8939.pth"
        # SAM_ckpt='/media/chen/UDisk1/sam_hq_vit_h.pth'

        # vis options
        plot_hist: bool = False
        hist_colormap: str = "plasma"
        colored_mask_alpha: float = 0.7

        plot_tsne: bool = False
        plot_overlay: bool = False
        plot_nodes: bool = False
        plot_masks_compare: bool = False

        fake_phy: bool = False
        augment: bool = True

        analyze_path: str = "results/analyze"
        white_board_gt_masks: str = "white_masks.pt"
        ground_gt_masks: str = "ground_masks.pt"

        white_gt_val = (0.0, 0.2)
        ground_gt_val = (0.3, 1.0)

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
