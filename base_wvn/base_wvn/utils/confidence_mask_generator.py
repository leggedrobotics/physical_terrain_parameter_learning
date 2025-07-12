#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

from abc import ABC, abstractmethod
from sklearn.mixture import GaussianMixture
from dataclasses import dataclass
from collections import deque
from .loss import PhyLoss
from ..config import save_to_yaml, ParamCollection
import PIL.Image
from base_wvn import WVN_ROOT_DIR
import os
from .visualizer import plot_overlay_image, plot_image, add_color_bar_and_save


class ConfidenceMaskGeneratorFactory:
    @staticmethod
    def create(mode: str, device: str):
        if mode == "gmm_1d":
            return GMM1D(
                num_components=2,
                deivice=device,
            )
        elif mode == "fixed":
            return FixedThreshold(
                # adjustable parameters if needed
                std_factor=1.0,
                threshold=0.5,
                method="running_mean",
                device=device,
            )
        elif mode == "gmm_1d_history":
            return GMM1DHistory(
                num_components=2,
                deivice=device,
                max_history_length=10,  # adjustable
            )
        else:
            raise ValueError(f"Confidence mask generator mode {mode} not implemented")


@dataclass
class MaskedPredictionData:
    masked_output_phy: torch.Tensor
    raw_output_phy: torch.Tensor
    trans_img: torch.Tensor
    conf_mask: torch.Tensor
    loss_reco: torch.Tensor


class MaskGenerator(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def get_confidence_mask_from_recon_loss(self):
        """Returns a boolean mask based on the confidence scores"""
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_confidence_masked_prediction_from_img(
        self,
        trans_img: torch.Tensor,
        compressed_feats: Dict[Tuple[float, float], torch.Tensor],
        model: nn.Module,
        loss_fn: PhyLoss,
    ) -> MaskedPredictionData:
        """process the original_img and return the phy_mask in resized img shape(non-confident--> nan)
        Shape of phy_mask: (2,H,W) H,W is the size of resized img
        Shape of conf_mask/loss_reco (H,W) H,W is the size of resized img
        Shape of output_phy: (2,H,W) H,W is the size of resized img

        """
        feat_input, H, W = concat_feat_dict(compressed_feats)
        feat_input = feat_input.squeeze(0)
        output = model(feat_input)
        loss_reco = loss_fn.compute_reconstruction_loss(
            output,
            feat_input,  # output - shape (H*W, C), feat_input - shape (H*W, C), loss_reco - shape (H*W,)
        )

        phy_dim = output.shape[1] - feat_input.shape[1]
        output_phy = output[:, -phy_dim:].reshape(H, W, 2).permute(2, 0, 1)  # (2,H,W)

        conf_mask = self.get_confidence_mask_from_recon_loss(loss_reco).reshape(H, W)
        loss_reco = loss_reco.reshape(H, W)
        unconf_mask = ~conf_mask
        mask = unconf_mask.unsqueeze(0).repeat(output_phy.shape[0], 1, 1)  # (2,H,W)
        output_phy_ori = output_phy.clone()
        output_phy[mask] = torch.nan
        if (
            output_phy.shape[-2] != trans_img.shape[-2]
            or output_phy.shape[-1] != trans_img.shape[-1]
        ):
            # upsample the output
            output_phy = F.interpolate(
                output_phy.unsqueeze(0).type(torch.float32), size=trans_img.shape[-2:]
            ).squeeze(0)
            output_phy_ori = F.interpolate(
                output_phy_ori.unsqueeze(0).type(torch.float32),
                size=trans_img.shape[-2:],
            ).squeeze(0)
        conf_mask_resized = (
            (
                F.interpolate(
                    conf_mask.type(torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0),  # (1, 1, H, W)
                    size=trans_img.shape[-2:],
                )
                > 0
            )
            .squeeze(0)
            .squeeze(0)
        )  # (H, W)
        loss_reco_resized = (
            F.interpolate(
                loss_reco.unsqueeze(0).unsqueeze(0), size=trans_img.shape[-2:]
            )
            .squeeze(0)
            .squeeze(0)
        )  # (H, W)
        # loss_threshold=thresholds[0]
        torch.cuda.empty_cache()
        return MaskedPredictionData(
            masked_output_phy=output_phy.detach(),
            raw_output_phy=output_phy_ori.detach(),
            trans_img=trans_img,
            conf_mask=conf_mask_resized,
            loss_reco=loss_reco_resized,
        )


class FixedThreshold(MaskGenerator):
    def __init__(
        self,
        std_factor: float,
        threshold: float,
        method: str,
        device: str,
    ):
        """Returns a confidence value for each number"""
        super(FixedThreshold, self).__init__()
        self.device = device
        self.std_factor = std_factor
        self.threshold = threshold

        mean = torch.zeros(1, dtype=torch.float32)
        var = torch.ones(1, dtype=torch.float32)
        std = torch.ones(1, dtype=torch.float32)
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.var = torch.nn.Parameter(var, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)

        if method == "running_mean":
            running_n = torch.zeros(1, dtype=torch.float64)
            running_sum = torch.zeros(1, dtype=torch.float64)
            running_sum_of_squares = torch.zeros(1, dtype=torch.float64)

            self.running_n = running_n.to(self.device)
            self.running_sum = running_sum.to(self.device)
            self.running_sum_of_squares = running_sum_of_squares.to(self.device)

            self._update = self.update_running_mean
            self._reset = self.reset_running_mean
        else:
            raise ValueError(f"Method {method} not implemented")

    def update_running_mean(self, x: torch.Tensor) -> torch.Tensor:
        # We assume the positive samples' loss follows a Gaussian distribution
        # We estimate the parameters empirically
        if x.device != self.device:
            x = x.to(self.device)
        self.running_n += x.numel()
        self.running_sum += x.sum()
        self.running_sum_of_squares += (x**2).sum()

        self.mean[0] = self.running_sum[0] / self.running_n
        self.var[0] = self.running_sum_of_squares / self.running_n - self.mean**2
        self.std[0] = torch.sqrt(self.var)

        return self.inference_without_update(x)

    def reset(self):
        self._reset()

    def reset_running_mean(self):
        self.running_n[0] = 0
        self.running_sum[0] = 0
        self.running_sum_of_squares[0] = 0

    def get_dict(self):
        return {"mean": self.mean, "var": self.var, "std": self.std}

    def update(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Input a tensor with multiple error predictions.
        Returns the estimated confidence score within 2 standard deviations based on the running mean and variance.

        Args:
            x (torch.tensor): (H*W,)
        Returns:
            (torch.tensor): (H*W,)
        """
        output = self._update(x)
        return output

    def inference_without_update(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.mean.device:
            x = x.to(self.mean.device)

        confidence = torch.exp(
            -(((x - self.mean) / (self.std * self.std_factor)) ** 2) * 0.5
        )
        confidence[x < self.mean] = 1.0  # debug, I think it fit the paper
        return confidence.type(torch.float32)

    def get_confidence_mask_from_recon_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loss (torch.Tensor): shape (H*W,)

        Returns:
            torch.Tensor: shape (H*W,), boolean mask, need to be reshaped later
        """
        assert loss.dim() == 1, "Loss tensor should be 1D"
        confidence = self.inference_without_update(loss)
        conf_mask = confidence >= self.threshold
        return conf_mask


class GMM1D(MaskGenerator):
    def __init__(self, num_components: int, deivice: str):
        super(GMM1D, self).__init__()
        self.num_components = num_components
        self.device = deivice
        self.gmm_1d = GaussianMixture(
            n_components=self.num_components,
            random_state=42,
        )

    def update(self, x: torch.Tensor):
        # GMM does not need train-time update
        return

    def get_confidence_mask_from_recon_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loss (torch.Tensor): shape (H*W,)

        Returns:
            torch.Tensor: shape (H*W,), boolean mask, need to be reshaped later
        """
        assert loss.dim() == 1, "Loss tensor should be 1D"
        loss = loss.detach().cpu().numpy().reshape(-1, 1)  # (H*W, 1)
        self.gmm_1d.fit(loss)
        labels = self.gmm_1d.predict(loss)
        confident_cluster = self.gmm_1d.means_.argmin()
        confidence_mask = labels == confident_cluster
        return torch.tensor(confidence_mask, dtype=torch.bool, device=self.device)


class GMM1DHistory(MaskGenerator):
    def __init__(self, num_components: int, deivice: str, max_history_length: int):
        super(GMM1DHistory, self).__init__()
        self.num_components = num_components
        self.device = deivice
        self.loss_history_raw = deque(
            maxlen=max_history_length
        )  # keep the last few loss Tensors
        self.gmm_1d = GaussianMixture(
            n_components=self.num_components,
            random_state=42,
        )

    def update(self, x: torch.Tensor) -> None:
        self.loss_history_raw.append(x.clone().detach())

    @property
    def loss_history(self) -> torch.Tensor:
        return torch.cat(list(self.loss_history_raw), dim=0)

    def get_confidence_mask_from_recon_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loss (torch.Tensor): shape (H*W,)

        Returns:
            torch.Tensor: shape (H*W,), boolean mask, need to be reshaped later
        """
        assert loss.dim() == 1, "Loss tensor should be 1D"
        loss = loss.detach().cpu()
        loss_history = self.loss_history.detach().cpu()
        loss_to_fit = (
            torch.cat([loss, loss_history], dim=0).numpy().reshape(-1, 1)
        )  # (H*W+history, 1)
        self.gmm_1d.fit(loss_to_fit)
        labels = self.gmm_1d.predict(loss.numpy().reshape(-1, 1))
        confident_cluster = self.gmm_1d.means_.argmin()
        confidence_mask = labels == confident_cluster
        return torch.tensor(confidence_mask, dtype=torch.bool, device=self.device)


def plot_pred_w_overlay(
    data: MaskedPredictionData,
    time: str,
    image_name: str,
    step: int,
    param: ParamCollection,
    foothold_label_mask: torch.Tensor = None,
    save_local: bool = False,
) -> Tuple[List[PIL.Image.Image], List[PIL.Image.Image]]:
    output_dir = os.path.join(WVN_ROOT_DIR, param.offline.ckpt_parent_folder, time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_phy = data.masked_output_phy
    output_phy_ori = data.raw_output_phy
    trans_img = data.trans_img
    channel_num = output_phy.shape[0]
    # process trans_img for plotting
    trans_img_uint = plot_image(trans_img.squeeze(0))
    trans_img_pil = PIL.Image.fromarray(trans_img_uint)
    trans_img_pil = rot_or_not(trans_img_pil, param)

    fric_vis_imgs = []
    stiff_vis_imgs = []

    for i in range(channel_num):
        output_phy = output_phy.detach()
        output_phy_ori = output_phy_ori.detach()
        overlay_img = plot_overlay_image(
            trans_img, overlay_mask=output_phy, channel=i, alpha=1.0
        )
        overlay_img_raw = plot_overlay_image(
            trans_img, overlay_mask=output_phy_ori, channel=i, alpha=1.0
        )
        # ---debug for paper plotting, conf mask section---
        # overlay_img=plot_overlay_image_binary(trans_img, overlay_mask=output_phy, channel=i,alpha=0.7)
        out_image = PIL.Image.fromarray(overlay_img)  # masked prediction
        out_image_raw = PIL.Image.fromarray(overlay_img_raw)  # raw/unmasked prediction
        rotated_image = rot_or_not(out_image, param)
        rotated_image_raw = rot_or_not(out_image_raw, param)

        vis_imgs = [trans_img_pil]

        # process possible foothold_label_mask for plotting if given
        if foothold_label_mask is not None:
            label = foothold_label_mask.detach()
            overlay_label = plot_overlay_image(
                trans_img, overlay_mask=label, channel=i, alpha=0.9
            )
            overlay_label_img = PIL.Image.fromarray(overlay_label)
            overlay_label_img = rot_or_not(overlay_label_img, param)
            vis_imgs.append(overlay_label_img)

        vis_imgs.append(rotated_image_raw)
        vis_imgs.append(rotated_image)
        # add colorbar to overlay image and then save
        # for paper video, uncomment
        if i == 0:
            fric_vis_imgs = vis_imgs
        elif i == 1:
            stiff_vis_imgs = vis_imgs

        if save_local:
            # Construct a filename
            if i == 0:
                filename = f"{image_name}_fric_den_pred_step_{step}_{param.loss.confidence_mode}.jpg"
            elif i == 1:
                filename = f"{image_name}_stiff_den_pred_step_{step}_{param.loss.confidence_mode}.jpg"
            file_path = os.path.join(output_dir, filename)
            # Save the image
            add_color_bar_and_save(vis_imgs, i, file_path)

    # return output_phy,trans_img,confidence,conf_mask_resized
    torch.cuda.empty_cache()
    return fric_vis_imgs, stiff_vis_imgs


def concat_feat_dict(
    feat_dict: Dict[Tuple[float, float], torch.Tensor],
) -> Tuple[torch.Tensor, int, int]:
    """Concatenate features from different scales, all upsamples to the first scale (expected to be the highest resolution)"""
    """ Return: features (B,H*W,C)
        feat_height: H
        feat_width: W
    """
    first_shape = list(feat_dict.values())[0].shape
    scales_h = [first_shape[2] / feat.shape[2] for feat in feat_dict.values()]
    scales_w = [first_shape[3] / feat.shape[3] for feat in feat_dict.values()]
    # upsampling the feat of each scale
    resized_feats = [
        F.interpolate(feat.type(torch.float32), scale_factor=(scale_h, scale_w))
        for scale_h, scale_w, feat in zip(scales_h, scales_w, feat_dict.values())
    ]
    resized_feats = torch.cat(resized_feats, dim=1)
    resized_feats = resized_feats.permute(0, 2, 3, 1)
    features = resized_feats.reshape(
        resized_feats.shape[0], resized_feats.shape[1] * resized_feats.shape[2], -1
    )
    return features, first_shape[2], first_shape[3]


def rot_or_not(img, param: ParamCollection) -> PIL.Image.Image:
    if param is not None:
        if isinstance(img, PIL.Image.Image) and "v4l2" in param.roscfg.camera_topic:
            # Rotate the image by 180 degrees
            img = img.rotate(180)
    return img


if __name__ == "__main__":
    cg = FixedThreshold()
    for i in range(100000):
        inp = (
            torch.rand(
                10,
            )
            * 10
        )
        res = cg.update(inp, inp, i)
        print("inp ", inp, " res ", res, "std", cg.std)
