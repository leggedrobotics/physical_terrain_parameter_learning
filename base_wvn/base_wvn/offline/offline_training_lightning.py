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


class AverageTrainingTimeCallback(pl.Callback):
    def __init__(self, warmup_steps=10, mode="gpu"):
        assert mode in ["cpu", "gpu"], "mode must be either 'cpu' or 'gpu'"
        self.warmup_steps = warmup_steps
        self.mode = mode
        self.total_time = 0
        self.num_steps = 0
        self.step_count = 0
        self.memory_usage = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.mode == "gpu":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.time()

    def on_after_backward(self, trainer, pl_module):
        if self.mode == "gpu":
            self.end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded
            elapsed_time = self.start_event.elapsed_time(
                self.end_event
            )  # Elapsed time in milliseconds
        else:
            self.end_time = time.time()
            elapsed_time = (
                self.end_time - self.start_time
            ) * 1000  # Convert to milliseconds

        self.step_count += 1
        if self.step_count > self.warmup_steps:
            self.total_time += elapsed_time
            self.num_steps += 1
            if self.mode == "gpu":
                mem_allocated = torch.cuda.memory_allocated()
                self.memory_usage.append(mem_allocated)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.num_steps > 0:
            avg_time_per_step = self.total_time / self.num_steps
            avg_memory_usage = (
                sum(self.memory_usage) / len(self.memory_usage)
                if self.memory_usage
                else 0
            )
            pl_module.log(
                "avg_training_time_per_step", avg_time_per_step, on_epoch=True
            )
            pl_module.log("avg_memory_usage", avg_memory_usage, on_epoch=True)
            print(
                f"Average training time per step (after warmup): {avg_time_per_step:.2f} ms"
            )
            print(
                f"Average memory usage per step: {avg_memory_usage / (1024 ** 2):.2f} MB"
            )
        else:
            print("Warmup period, average training time per step not recorded yet.")
            print("Warmup period, average memory usage per step not recorded yet.")


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
        self.feat_extractor = FeatureExtractor(
            device=self.params.run.device,
            input_size=self.params.feat.input_size,
            feature_type=self.params.feat.feature_type,
            interp=self.params.feat.interp,
            center_crop=self.params.feat.center_crop,
            original_width=W,
            original_height=H,
        )
        self.loss_fn = PhyLoss(
            w_pred=loss_params.w_pred,
            w_reco=loss_params.w_reco,
            method=loss_params.method,
            confidence_std_factor=loss_params.confidence_std_factor,
            log_enabled=loss_params.log_enabled,
            log_folder=loss_params.log_folder,
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
        loss, confidence, loss_dict = self.loss_fn((xs, ys), res, step=self.step)
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
            self.log(
                "digital_twin_fric_error_white_mean",
                stats_dict["digital_twin_fric_error_white_mean"],
            )
            self.log(
                "digital_twin_fric_error_white_std",
                stats_dict["digital_twin_fric_error_white_std"],
            )
            self.log(
                "digital_twin_fric_error_ground_mean",
                stats_dict["digital_twin_fric_error_ground_mean"],
            )
            self.log(
                "digital_twin_fric_error_ground_std",
                stats_dict["digital_twin_fric_error_ground_std"],
            )
            self.log(
                "digital_twin_fric_error_both_mean",
                stats_dict["digital_twin_fric_error_both_mean"],
            )
            self.log(
                "digital_twin_fric_error_both_std",
                stats_dict["digital_twin_fric_error_both_std"],
            )
            # self.log('over_conf_mean',stats_dict['over_conf_mean'])
            # self.log('over_conf_std',stats_dict['over_conf_std'])
            # self.log('under_conf_mean',stats_dict['under_conf_mean'])
            # self.log('under_conf_std',stats_dict['under_conf_std'])
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
            loss_reco_raw = res_dict["loss_reco_raw"]
            conf_mask_raw = res_dict["conf_mask_raw"]

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
        # if batch_idx==0:
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
            loss_reco_raw = res_dict["loss_reco_raw"]
            conf_mask_raw = res_dict["conf_mask_raw"]

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
            plot_tsne(
                conf_mask_raw,
                loss_reco_raw,
                title=f"step_{self.step}_t-SNE with Confidence Highlighting",
                path=os.path.join(
                    WVN_ROOT_DIR,
                    self.params.offline.ckpt_parent_folder,
                    self.time,
                    "tsne",
                ),
            )
            pass
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

        max_epochs = 5  # 8 ,3 for 2nd , 5 for 1st
        if "partial" in param.offline.traindata_option:
            if "each" in param.offline.traindata_option:
                # load train and val data from online collected dataset (each batch is 100 samples from six nodes)
                train_data = load_data(
                    os.path.join(
                        param.offline.data_folder,
                        "train",
                        param.offline.env,
                        param.offline.train_datafile,
                    )
                )

                try:
                    val_data = load_data(
                        os.path.join(
                            param.offline.data_folder,
                            "val",
                            param.offline.env,
                            param.offline.train_datafile,
                        )
                    )
                except FileNotFoundError:
                    val_data = load_data(
                        os.path.join(
                            param.offline.data_folder,
                            "train",
                            param.offline.env,
                            param.offline.train_datafile,
                        )
                    )
            elif "all" in param.offline.traindata_option:
                train_data_hiking = load_data(
                    os.path.join(
                        param.offline.data_folder,
                        "train",
                        "hiking",
                        param.offline.train_datafile,
                    )
                )
                train_data_snow = load_data(
                    os.path.join(
                        param.offline.data_folder,
                        "train",
                        "snow",
                        param.offline.train_datafile,
                    )
                )
                val_data = load_data(
                    os.path.join(
                        param.offline.data_folder,
                        "val",
                        param.offline.env,
                        param.offline.train_datafile,
                    )
                )
                train_data = train_data_hiking + train_data_snow

            else:
                raise ValueError("Invalid traindata_option")
            if param.offline.random_datasample[0]:
                print(
                    "Randomly sample {} data from the dataset".format(
                        param.offline.random_datasample[1]
                    )
                )
                train_dataset = BigDataset(
                    train_data, param.offline.random_datasample[1]
                )
            else:
                train_dataset = BigDataset(train_data)

            if param.offline.random_datasample[0]:
                print(
                    "Randomly sample {} data from the dataset".format(
                        param.offline.random_datasample[1]
                    )
                )
                val_dataset = BigDataset(val_data, param.offline.random_datasample[1])
            else:
                val_dataset = BigDataset(val_data)

            # mimic online training fashion
            batch_size = 1
            shuffle = True
        elif "full" in param.offline.traindata_option:
            if "each" in param.offline.traindata_option:
                # load train and val data from nodes_datafile (should include all pixels of supervision masks)
                train_data_raw = load_data(
                    os.path.join(
                        param.offline.data_folder,
                        "train",
                        param.offline.env,
                        param.offline.nodes_datafile,
                    )
                )
                if param.offline.env == "vowhite_2nd":
                    additional_data = load_data(
                        os.path.join(
                            param.offline.data_folder,
                            "train",
                            "vowhite_1st",
                            param.offline.nodes_datafile,
                        )
                    )
                    train_data_raw += additional_data
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
            elif "all" in param.offline.traindata_option:
                train_data_hiking = load_data(
                    os.path.join(
                        param.offline.data_folder,
                        "train",
                        "hiking",
                        param.offline.nodes_datafile,
                    )
                )
                train_data_snow = load_data(
                    os.path.join(
                        param.offline.data_folder,
                        "train",
                        "snow",
                        param.offline.nodes_datafile,
                    )
                )
                val_data_raw = load_data(
                    os.path.join(
                        param.offline.data_folder,
                        "val",
                        param.offline.env,
                        param.offline.nodes_datafile,
                    )
                )
                train_data_raw = train_data_hiking + train_data_snow
            else:
                raise ValueError("Invalid traindata_option")
            train_data = create_dataset_from_nodes(
                param,
                train_data_raw,
                model.feat_extractor,
                fake_phy=param.offline.fake_phy,
            )
            train_dataset = EntireDataset(train_data)
            val_data = create_dataset_from_nodes(
                param,
                val_data_raw,
                model.feat_extractor,
                fake_phy=param.offline.fake_phy,
            )
            val_dataset = EntireDataset(val_data)
            batch_size = 100
            shuffle = True
        else:
            raise ValueError("Invalid traindata_option")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # batch in loader is a tuple of (xs, ys)
        # xs:(1, 100, feat_dim), ys:(1, 100, 2)
        sample_input, sample_output = next(iter(train_loader))
        device = sample_input.device
        feat_dim = sample_input.shape[-1]
        label_dim = sample_output.shape[-1]

        # timer callback
        avg_time_callback = AverageTrainingTimeCallback()
        trainer = Trainer(
            accelerator="gpu",
            devices=[0],
            logger=neptune_logger,
            max_epochs=max_epochs,
            max_steps=80,
            log_every_n_steps=1,
            limit_train_batches=0.5,
            limit_val_batches=0.5,
            callbacks=[avg_time_callback],
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
        feat_extractor = FeatureExtractor(
            device=param.run.device,
            input_size=param.feat.input_size,
            feature_type=param.feat.feature_type,
            interp=param.feat.interp,
            center_crop=param.feat.center_crop,
        )
        """ 
        plot phy_masks (two channels) on a set of test images
        """
        if param.offline.test_images:
            test_imgs = load_all_test_images(
                os.path.join(param.offline.data_folder, "val", param.offline.env)
            )
            for name, img in test_imgs.items():
                B, C, H, W = img.shape
                feat_extractor.set_original_size(W, H)
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

                    B, C, H, W = img.shape
                    feat_extractor.set_original_size(W, H)
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
        if "vowhite" in self.param.offline.env:
            digital_twin_error_dict = phy_mask_total_accuracy(
                self.param, self.nodes, feat_extractor, model
            )
            return_dict.update(digital_twin_error_dict)
        conf_masks = output_dict["all_conf_masks"]
        ori_imgs = output_dict["ori_imgs"]
        fric_mean, fric_std = output_dict["loss_fric_mean+std"]
        stiffness_mean, stiffness_std = output_dict["loss_stiff_mean+std"]
        conf_masks = conf_masks.to(self.param.run.device)
        ori_imgs = ori_imgs.to(self.param.run.device)
        print("conf_masks shape:{}".format(conf_masks.shape))

        # stats_outputdict=masks_stats(self.gt_masks,conf_masks,os.path.join(self.ckpt_parent_folder,model.time,"masks_stats.txt"),self.param.general.name)
        # over_conf_mean=stats_outputdict['over_conf_mean']
        # over_conf_std=stats_outputdict['over_conf_std']
        # under_conf_mean=stats_outputdict['under_conf_mean']
        # under_conf_std=stats_outputdict['under_conf_std']
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
