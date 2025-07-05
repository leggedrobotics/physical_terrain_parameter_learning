#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
import os
import datetime
from base_wvn import WVN_ROOT_DIR
from base_wvn.offline.helper import (
    nodes_mask_generation_and_phy_pred_error_computation,
    SAM_label_mask_generate,
    plot_masks_compare,
    masks_iou_stats,
)
from base_wvn.utils import (
    PhyLoss,
    FeatureExtractor,
    ConfidenceMaskGeneratorFactory,
)
from base_wvn.config.wvn_cfg import ParamCollection
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from typing import Dict


class DecoderLightning(pl.LightningModule):
    def __init__(self, model, params: ParamCollection):
        super().__init__()
        self.model = model
        self.params = params
        loss_params = self.params.loss
        self.step = 0
        seed_everything(params.run.seed)

        self.feat_extractor = FeatureExtractor(self.params)
        self.conf_mask_generator = ConfidenceMaskGeneratorFactory.create(
            mode=params.loss.confidence_mode, device=params.run.device
        )
        self.loss_fn = PhyLoss(
            w_pred=loss_params.w_pred,
            w_reco=loss_params.w_reco,
            reco_loss_type=loss_params.reco_loss_type,
        )
        self.err_computer = MaskedPredErrorComputer(params)
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
        loss, loss_dict = self.loss_fn((xs, ys), res)
        self.log("train_loss", loss)
        self.log("train_reco_loss", loss_dict["loss_reco"])
        self.log("train_phy_loss", loss_dict["loss_pred"])

        if self.params.offline.upload_error_stats_in_training:
            # upload the error stats calculated by the err_computer
            # for all recorded nodes of the current model
            self.err_computer.compte_and_log(self, log=True)

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
        loss, loss_dict = self.loss_fn((xs, ys), res)

        self.log("val_loss", loss)
        self.log("val_reco_loss", loss_dict["loss_reco"])
        self.log("val_phy_loss", loss_dict["loss_pred"])
        self.val_loss = loss

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.params.optimizer.lr,
            weight_decay=self.params.optimizer.weight_decay,
        )
        return optimizer


class MaskedPredErrorComputer:
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

    def compte_and_log(self, model: DecoderLightning, log: bool) -> Dict[str, float]:
        output_dict = nodes_mask_generation_and_phy_pred_error_computation(
            self.param, self.nodes, model, self.gt_masks
        )

        conf_masks = output_dict["all_conf_masks"]
        ori_imgs = output_dict["ori_imgs"]
        fric_mean, fric_std = output_dict["loss_fric_mean+std"]
        stiffness_mean, stiffness_std = output_dict["loss_stiff_mean+std"]
        conf_masks = conf_masks.to(self.param.run.device).unsqueeze(1)
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

        if log:
            model.log("fric_error_mean", fric_mean)
            model.log("fric_error_std", fric_std)
            model.log("stiff_error_mean", stiffness_mean)
            model.log("stiff_error_std", stiffness_std)
            model.log("iou_mean", stats_outputdict["iou_mean"])
            model.log("iou_std", stats_outputdict["iou_std"])

        return {
            "fric_error_mean": fric_mean,
            "fric_error_std": fric_std,
            "stiff_error_mean": stiffness_mean,
            "stiff_error_std": stiffness_std,
            "iou_mean": stats_outputdict["iou_mean"],
            "iou_std": stats_outputdict["iou_std"],
        }
