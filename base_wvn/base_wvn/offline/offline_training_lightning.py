#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
import numpy as np
import os
import rosbag
import cv2
import matplotlib.pyplot as plt
import datetime
import time
from base_wvn import WVN_ROOT_DIR
from base_wvn.offline.model_helper import *
from base_wvn.offline.dataset_cls import *
from base_wvn.utils import (
    PhyLoss,
    FeatureExtractor,
    plot_overlay_image,
    compute_phy_mask,
)
from base_wvn.model import get_model
from base_wvn.config.wvn_cfg import ParamCollection
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import Trainer


class DecoderLightning(pl.LightningModule):
    def __init__(self, model, params: ParamCollection):
        super().__init__()
        self.model = model
        self.params = params
        loss_params = self.params.loss
        self.step = 0

        try:
            # Attempt to load the image from the 'val' directory
            self.test_img = load_one_test_image(
                os.path.join(
                    WVN_ROOT_DIR,
                    params.offline.data_folder,
                    "val",
                    params.offline.env,
                    params.offline.image_file,
                )
            )
        except FileNotFoundError:
            # If the image is not found in the 'val' directory, try the 'train' directory
            self.test_img = load_one_test_image(
                os.path.join(
                    WVN_ROOT_DIR,
                    params.offline.data_folder,
                    "train",
                    params.offline.env,
                    params.offline.image_file,
                )
            )
        B, C, H, W = self.test_img.shape
        self.feat_extractor = FeatureExtractor(self.params)
        self.loss_fn = PhyLoss(
            w_pred=loss_params.w_pred,
            w_reco=loss_params.w_reco,
            method=loss_params.method,
            reco_loss_type=loss_params.reco_loss_type,
        )
        self.validator = Validator(params)
        self.val_loss = 0.0
        self.time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        xs = xs.squeeze(0)
        ys = ys.squeeze(0)
        # chech if xs and ys both have shape of 2
        if len(xs.shape) != 2 or len(ys.shape) != 2:
            raise ValueError("xs and ys must have shape of 2")
        res = self.model(xs)
        loss, confidence, loss_dict = self.loss_fn((xs, ys), res)
        self.log("train_loss", loss)

        if self.params.offline.upload_error_stats_in_training:
            stats_dict = self.validator.go(self, self.feat_extractor)

            # upload the error stats calculated by the validator
            # for all recorded nodes of the current model
            self.log("fric_error_mean", stats_dict["fric_mean"])
            self.log("fric_error_std", stats_dict["fric_std"])
            self.log("stiff_error_mean", stats_dict["stiffness_mean"])
            self.log("stiff_error_std", stats_dict["stiffness_std"])
            self.log("iou_mean", stats_dict["iou_mean"])
            self.log("iou_std", stats_dict["iou_std"])

        if self.params.offline.plot_hist or self.step == 19:
            res_dict = compute_phy_mask(
                self.test_img,
                self.feat_extractor,
                self.model,
                self.loss_fn,
                self.params.loss.confidence_threshold,
                self.params.loss.confidence_mode,
                self.params.offline.plot_overlay,
                self.step,
                time=self.time,
                param=self.params,
            )
            conf_mask = res_dict["conf_mask"]
            loss_reco = res_dict["loss_reco"]

            colored_loss_mask = calculate_uncertainty_plot(
                loss_reco,
                conf_mask,
                all_reproj_masks=None,
                save_path=os.path.join(
                    WVN_ROOT_DIR,
                    self.params.offline.ckpt_parent_folder,
                    self.time,
                    "hist",
                    f"trainstep_{self.step}_uncertainty_histogram.png",
                ),
                colormap=self.params.offline.hist_colormap,
            )
            alpha = self.params.offline.colored_mask_alpha
            overlay_img = (
                self.test_img.squeeze(0).permute(1, 2, 0).cpu().numpy() * (1 - alpha)
                + colored_loss_mask * alpha
            )
            plt.imsave(
                os.path.join(
                    WVN_ROOT_DIR,
                    self.params.offline.ckpt_parent_folder,
                    self.time,
                    "hist",
                    f"trainstep_{self.step}_overlay.png",
                ),
                overlay_img,
            )
        self.step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        xs = xs.squeeze(0)
        ys = ys.squeeze(0)
        # chech if xs and ys both have shape of 2
        if len(xs.shape) != 2 or len(ys.shape) != 2:
            raise ValueError("xs and ys must have shape of 2")
        res = self.model(xs)
        loss, confidence, loss_dict = self.loss_fn(
            (xs, ys), res, step=self.step, update_generator=False
        )
        # if batch_idx==0 and self.step%10==0:
        if batch_idx == 0:
            res_dict = compute_phy_mask(
                self.test_img,
                self.feat_extractor,
                self.model,
                self.loss_fn,
                self.params.loss.confidence_threshold,
                self.params.loss.confidence_mode,
                self.params.offline.plot_overlay,
                self.step,
                time=self.time,
                param=self.params,
            )
            conf_mask = res_dict["conf_mask"]
            loss_reco = res_dict["loss_reco"]

            colored_loss_mask = calculate_uncertainty_plot(
                loss_reco,
                conf_mask,
                all_reproj_masks=None,
                save_path=os.path.join(
                    WVN_ROOT_DIR,
                    self.params.offline.ckpt_parent_folder,
                    self.time,
                    "hist",
                    f"step_{self.step}_uncertainty_histogram.png",
                ),
                colormap=self.params.offline.hist_colormap,
            )
            alpha = self.params.offline.colored_mask_alpha
            overlay_img = (
                self.test_img.squeeze(0).permute(1, 2, 0).cpu().numpy() * (1 - alpha)
                + colored_loss_mask * alpha
            )
            plt.imsave(
                os.path.join(
                    WVN_ROOT_DIR,
                    self.params.offline.ckpt_parent_folder,
                    self.time,
                    "hist",
                    f"trainstep_{self.step}_overlay.png",
                ),
                overlay_img,
            )
        self.log("val_loss", loss)
        self.log("reco_loss", loss_dict["loss_reco"])
        self.log("phy_loss", loss_dict["loss_pred"])
        self.val_loss = loss

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.params.optimizer.lr,
            weight_decay=self.params.optimizer.weight_decay,
        )
        return optimizer


def train_and_evaluate(param: ParamCollection):
    """Train and evaluate the model."""

    mode = param.offline.mode
    ckpt_parent_folder = os.path.join(WVN_ROOT_DIR, param.offline.ckpt_parent_folder)

    m = get_model(param.model).to(param.run.device)
    model = DecoderLightning(m, param)
    if mode == "train":
        if param.offline.reload_model:
            if not param.offline.use_online_ckpt:
                checkpoint_path = find_latest_checkpoint(ckpt_parent_folder)
            else:
                checkpoint_path = os.path.join(
                    WVN_ROOT_DIR, param.general.resume_training_path
                )
            if checkpoint_path:
                print(f"Latest checkpoint path: {checkpoint_path}")
            else:
                print("No checkpoint found.")
                return None
            checkpoint = torch.load(checkpoint_path)
            model.model.load_state_dict(checkpoint["model_state_dict"])
            model.loss_fn.load_state_dict(checkpoint["phy_loss_state_dict"])
            model.step = checkpoint["step"]
            model.time = (
                checkpoint["time"] if not param.offline.use_online_ckpt else "online"
            )
            model.val_loss = checkpoint["loss"]
            model.model.train()
            print("Reloaded model from {}".format(checkpoint_path))
        # Initialize the Neptune logger
        neptune_logger = NeptuneLogger(
            api_key=os.getenv("NEPTUNE_API_TOKEN", "default_api_token_if_not_set"),
            project=os.getenv("NEPTUNE_PROJECT", "default_project_if_not_set"),
            tags=["offline", param.offline.env, param.general.name],
        )

        max_epochs = 5  # 8 ,3 for 2nd , 5 for 1st, TODO, move into config file

        # load train and val data from nodes_datafile (should include all pixels of supervision masks)
        train_data_raw = load_data(
            os.path.join(
                param.offline.data_folder,
                "train",
                param.offline.env,
                param.offline.nodes_datafile,
            )
        )
        try:
            val_data_raw = load_data(
                os.path.join(
                    param.offline.data_folder,
                    "val",
                    param.offline.env,
                    param.offline.nodes_datafile,
                )
            )
        except FileNotFoundError:
            val_data_raw = load_data(
                os.path.join(
                    param.offline.data_folder,
                    "train",
                    param.offline.env,
                    param.offline.nodes_datafile,
                )
            )
        train_data = create_dataset_from_nodes(
            param,
            train_data_raw,
            model.feat_extractor,
        )
        train_dataset = EntireDataset(train_data)
        val_data = create_dataset_from_nodes(
            param,
            val_data_raw,
            model.feat_extractor,
        )
        val_dataset = EntireDataset(val_data)
        batch_size = 100
        shuffle = True

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # batch in loader is a tuple of (xs, ys)
        # xs:(1, 100, feat_dim), ys:(1, 100, 2)
        sample_input, sample_output = next(iter(train_loader))
        device = sample_input.device
        feat_dim = sample_input.shape[-1]
        label_dim = sample_output.shape[-1]

        trainer = Trainer(
            accelerator="gpu",
            devices=[0],
            logger=neptune_logger,
            max_epochs=max_epochs,
            max_steps=80,
            log_every_n_steps=1,
            limit_train_batches=0.5,
            limit_val_batches=0.5,
        )
        trainer.fit(model, train_loader, val_loader)
        torch.save(
            {
                "time": model.time,
                "step": model.step,
                "model_state_dict": model.model.state_dict(),
                "phy_loss_state_dict": model.loss_fn.state_dict(),
                "loss": model.val_loss.item(),
            },
            os.path.join(ckpt_parent_folder, model.time, "last_checkpoint.pt"),
        )
        torch.cuda.empty_cache()
        return None
    else:
        if not param.offline.use_online_ckpt:
            checkpoint_path = find_latest_checkpoint(ckpt_parent_folder)
        else:
            checkpoint_path = os.path.join(
                WVN_ROOT_DIR, param.general.resume_training_path
            )
        if checkpoint_path:
            print(f"Latest checkpoint path: {checkpoint_path}")
        else:
            print("No checkpoint found.")
            return None
        checkpoint = torch.load(checkpoint_path)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.loss_fn.load_state_dict(checkpoint["phy_loss_state_dict"])
        model.step = checkpoint["step"]
        model.time = (
            checkpoint["time"] if not param.offline.use_online_ckpt else "online"
        )
        model.val_loss = checkpoint["loss"]
        model.model.eval()
        feat_extractor = FeatureExtractor(param)
        """ 
        plot phy_masks (two channels) on a set of test images
        """
        if param.offline.test_images:
            test_imgs = load_all_test_images(
                os.path.join(param.offline.data_folder, "val", param.offline.env)
            )
            for name, img in test_imgs.items():
                compute_phy_mask(
                    img,
                    feat_extractor,
                    model.model,
                    model.loss_fn,
                    param.loss.confidence_threshold,
                    param.loss.confidence_mode,
                    True,
                    -1,
                    time=model.time,
                    image_name=name,
                    param=param,
                    use_conf_mask=False,
                )

        """ 
        test on the recorded main nodes
        """
        if param.offline.test_nodes:
            # READ nodes datafile and gt_masks datafile
            validator = Validator(param)
            stats_dict = validator.go(model, feat_extractor)
            return stats_dict

        """ 
        output prediction video from rosbag
        """
        if param.offline.test_video:
            import src.wild_visual_navigation_ros.scripts.ros_converter as rc
            from tqdm import tqdm

            i = 0  # channel index
            use_conf_mask = False
            process_option = param.offline.process_option
            with rosbag.Bag(param.offline.img_bag_path, "r") as bag:
                total_messages = bag.get_message_count(
                    topic_filters=[param.roscfg.camera_topic]
                )
                # Determine the number of messages to process based on the selected option
                if process_option == "all":
                    messages_to_process = total_messages
                elif process_option == "first_half":
                    messages_to_process = total_messages // 2
                elif process_option == "first_1000":
                    messages_to_process = min(1000, total_messages)
                else:
                    raise ValueError("Invalid process option")
                progress_bar = tqdm(
                    total=messages_to_process, desc="Processing ROS Bag"
                )
                # Set up video writer
                first_frame = True
                for _, msg, _ in bag.read_messages(topics=[param.roscfg.camera_topic]):
                    if progress_bar.n >= messages_to_process:
                        break
                    # Convert ROS Image message to OpenCV image
                    img_torch = rc.ros_image_to_torch(msg, device=param.run.device)
                    img = img_torch[None]

                    out_dict = compute_phy_mask(
                        img,
                        feat_extractor,
                        model.model,
                        model.loss_fn,
                        param.loss.confidence_threshold,
                        param.loss.confidence_mode,
                        False,
                        -1,
                        time=model.time,
                        param=param,
                        use_conf_mask=False,
                    )
                    trans_img = out_dict["trans_img"].detach()
                    output_phy = out_dict["output_phy"].detach()
                    overlay_img = plot_overlay_image(
                        trans_img, overlay_mask=output_phy, channel=i, alpha=1.0
                    )
                    ori_img = plot_overlay_image(trans_img)

                    out_dict = compute_phy_mask(
                        img,
                        feat_extractor,
                        model.model,
                        model.loss_fn,
                        param.loss.confidence_threshold,
                        param.loss.confidence_mode,
                        False,
                        -1,
                        time=model.time,
                        param=param,
                        use_conf_mask=True,
                    )
                    trans_img = out_dict["trans_img"].detach()
                    output_phy = out_dict["output_phy"].detach()
                    max_val, mean_val = calculate_mask_values(output_phy[i])
                    overlay_img_wm = plot_overlay_image(
                        trans_img, overlay_mask=output_phy, channel=i, alpha=1.0
                    )

                    # Convert back to OpenCV image and store
                    frame = np.concatenate(
                        (ori_img, overlay_img, overlay_img_wm), axis=1
                    )
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Initialize video writer with the first frame's size
                    if first_frame:
                        height, width, layers = frame.shape
                        size = (width, height)
                        output_video_path = get_output_video_path(
                            param,
                            os.path.dirname(param.offline.img_bag_path),
                            i,
                            use_conf_mask,
                        )  # assuming get_output_video_path is a function you define
                        out = cv2.VideoWriter(
                            output_video_path, cv2.VideoWriter_fourcc(*"DIVX"), 15, size
                        )
                        first_frame = False
                    # Define headers for each section
                    headers = ["Original Image", "Dense Pred Raw", "Dense Pred w. Mask"]

                    # Calculate the width of each section (assuming all sections are equal width)
                    section_width = frame.shape[1] // len(headers)

                    # Add headers to the concatenated frame
                    # frame = add_headers_to_frame(frame, headers, section_width)
                    # start_x_right_section = 2 * section_width

                    # # Overlay these values on the right section of the frame
                    # frame = overlay_values_on_section(frame, max_val, mean_val, start_x_right_section)
                    out.write(frame)
                    progress_bar.update(1)
                progress_bar.close()

            out.release()

        return None


class Validator:
    def __init__(self, param: ParamCollection) -> None:
        mode = (
            "train"  # use train dataset for validation or test dataset for validation
        )
        nodes = torch.load(
            os.path.join(
                WVN_ROOT_DIR,
                param.offline.data_folder,
                mode,
                param.offline.env,
                param.offline.nodes_datafile,
            )
        )
        self.nodes = nodes
        self.param = param
        self.ckpt_parent_folder = os.path.join(
            WVN_ROOT_DIR, param.offline.ckpt_parent_folder
        )
        output_dir = os.path.join(
            WVN_ROOT_DIR, param.offline.data_folder, mode, param.offline.env
        )

        # Construct the path for gt_masks.pt
        if param.offline.gt_model == "SAM":
            gt_masks_path = os.path.join(output_dir, "gt_masks_SAM.pt")
        else:
            raise ValueError(
                f"Unsupported gt_model: {param.offline.gt_model}. Supported models: SAM."
            )
        img_path = os.path.join(output_dir, "mask_img.pt")
        # gt_masks_path = os.path.join(output_dir, 'gt_masks.pt')

        if os.path.exists(gt_masks_path):
            # Load the existing gt_masks
            gt_masks = torch.load(gt_masks_path)
        else:
            # Generate gt_masks
            if param.offline.gt_model == "SAM":
                gt_masks, cur_imags = SAM_label_mask_generate(param, nodes)
            else:
                raise ValueError(
                    f"Unsupported gt_model: {param.offline.gt_model}. Supported models: SAM."
                )
            torch.save(cur_imags, img_path)
            torch.save(gt_masks, gt_masks_path)
        self.gt_masks = gt_masks
        print("gt_masks shape:{}".format(gt_masks.shape))

    def go(self, model: pl.LightningModule, feat_extractor: FeatureExtractor):
        output_dict = conf_mask_generate(
            self.param, self.nodes, feat_extractor, model, self.gt_masks
        )
        return_dict = {}

        conf_masks = output_dict["all_conf_masks"]
        ori_imgs = output_dict["ori_imgs"]
        fric_mean, fric_std = output_dict["loss_fric_mean+std"]
        stiffness_mean, stiffness_std = output_dict["loss_stiff_mean+std"]
        conf_masks = conf_masks.to(self.param.run.device)
        ori_imgs = ori_imgs.to(self.param.run.device)
        print("conf_masks shape:{}".format(conf_masks.shape))

        stats_outputdict = masks_iou_stats(
            self.gt_masks,
            conf_masks,
            os.path.join(self.ckpt_parent_folder, model.time, "masks_stats.txt"),
            self.param.general.name,
        )

        if self.param.offline.plot_masks_compare:
            plot_masks_compare(
                self.gt_masks,
                conf_masks,
                ori_imgs,
                os.path.join(
                    self.ckpt_parent_folder, model.time, self.param.offline.gt_model
                ),
                layout_type="grid",
                param=self.param,
            )
        return_dict.update(
            {
                "fric_mean": fric_mean,
                "fric_std": fric_std,
                "stiffness_mean": stiffness_mean,
                "stiffness_std": stiffness_std,
                "iou_mean": stats_outputdict["iou_mean"],
                "iou_std": stats_outputdict["iou_std"],
            }
        )
        return return_dict


if __name__ == "__main__":
    param = ParamCollection()
    train_and_evaluate(param)
