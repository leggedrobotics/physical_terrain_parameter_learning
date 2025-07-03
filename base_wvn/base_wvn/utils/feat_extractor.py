#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch

from .dinov2_interface import Dinov2Interface

from ..config import ParamCollection

from torchvision import transforms as T
from typing import Dict, Tuple
from base_wvn import WVN_ROOT_DIR


class FeatureExtractor:
    def __init__(self, params: ParamCollection):
        """Feature extraction from image"""
        self._device = params.run.device
        self._feature_type = params.feat.feature_type
        self._input_size = params.feat.input_size
        self._input_interp = params.feat.interp
        self.center_crop = params.feat.center_crop

        # extract crop info
        self.crop_size = self.center_crop[1:]
        self.center_crop = self.center_crop[0]

        if self.center_crop:
            self.target_height = self.crop_size[0]
        else:
            self.target_height = self._input_size

        # feature extractor
        if self._feature_type == "dinov2":
            self.patch_size = 14
            self.extractor = Dinov2Interface(self._device)
        else:
            raise ValueError(f"Extractor[{self._feature_type}] not supported!")

        assert (
            self._input_size % self.patch_size == 0
        ), "Input size must be a multiple of patch_size"
        assert (
            self.crop_size[0] % self.patch_size == 0
            and self.crop_size[1] % self.patch_size == 0
        ), "Crop size must be a multiple of patch_size"

        # Interpolation type
        if self._input_interp == "bilinear":
            self.interp = T.InterpolationMode.BILINEAR
        elif self._input_interp == "nearest":
            self.interp = T.InterpolationMode.NEAREST
        elif self._input_interp == "bicubic":
            self.interp = T.InterpolationMode.BICUBIC

        # to be initialized later during first call of extract
        self.original_height, self.original_width = None, None
        self.new_height, self.new_width = None, None
        self.transform = None

    def extract(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[Tuple[float, float], torch.Tensor]]:
        """Extract features from image

        Args:
            img (torch.tensor): Image tensor (B,C,H,W)

        Returns:
            transformed_img (torch.tensor, shape:(B,C,H,W)): Transformed image
            compressed_feats (Dict): DINOv2 outputs feat in patches {(scale_h,scale_w):feat-->(B,C,H,W))}
        """
        B, C, H, W = img.shape
        img = img.to(self._device)
        if H != self.target_height:  # we scale based on the height
            self.init_transform(original_height=H, original_width=W)
            transformed_img = self.transform(img)
        else:
            transformed_img = img
        # Compute features
        compressed_feats = self.compute_features(transformed_img)
        torch.cuda.empty_cache()
        return transformed_img, compressed_feats

    def init_transform(self, original_width: int, original_height: int) -> None:
        if self.transform is not None:
            return
        self.original_height = original_height
        self.original_width = original_width

        if self.original_height < self._input_size:
            raise ValueError(
                f"Original image height {self.original_height} is smaller than desired input size {self._input_size}!"
            )

        self._create_transform()

    @property
    def feature_type(self) -> str:
        return self._feature_type

    @property
    def scaling_ratio(self) -> Tuple[float, float]:
        return self.scaling_ratio_x, self.scaling_ratio_y

    @property
    def crop_offset(self) -> Tuple[float, float]:
        return self.crop_offset_x, self.crop_offset_y

    @property
    def new_size(self) -> Tuple[int, int]:
        return self.new_width, self.new_height

    def _create_transform(self) -> None:
        # Calculate aspect ratio preserving size
        aspect_ratio = self.original_width / self.original_height

        # we scale based on the height
        new_height = self._input_size
        new_width = int(new_height * aspect_ratio)
        # check if new_width is a multiple of self.patch_size
        if new_width % self.patch_size != 0:
            new_width = new_width - new_width % self.patch_size

        # Resize and then center crop to the expected input size
        self.transform = T.Compose(
            [
                T.Resize((new_height, new_width), self.interp, antialias=None),
                T.CenterCrop(self.crop_size)
                if self.center_crop
                else T.CenterCrop((new_height, new_width)),
                T.ConvertImageDtype(torch.float),
            ]
        )

        # actual resize ratio along x and y of resize step
        self.scaling_ratio_x = float(new_width) / float(self.original_width)
        self.scaling_ratio_y = float(new_height) / float(self.original_height)
        if not self.center_crop:
            self.new_height = new_height
            self.new_width = new_width
            self.crop_offset_x = 0
            self.crop_offset_y = 0
        else:
            self.new_height = self.crop_size[0]
            self.new_width = self.crop_size[1]
            self.crop_offset_x = (new_width - self.crop_size[1]) / 2
            self.crop_offset_y = (new_height - self.crop_size[0]) / 2

    def change_device(self, device: str) -> None:
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._device = device
        self.extractor.change_device(device)

    def compute_features(
        self, img: torch.Tensor
    ) -> Dict[Tuple[float, float], torch.Tensor]:
        img_internal = img.clone()
        B, C, H, W = img_internal.shape

        if self._feature_type == "dinov2":
            feat_dict = {}
            feat = self.extractor.inference(img_internal)
            # compute ratio of original image size to feature map size
            ratio_h = H / feat.shape[-2]
            ratio_w = W / feat.shape[-1]
            feat_dict[(ratio_h, ratio_w)] = feat
        else:
            raise ValueError(f"Extractor[{self._feature_type}] not supported!")

        return feat_dict


def compute_pred_phy_loss(
    ori_phy_mask: torch.Tensor,
    pred_phy_mask: torch.Tensor,
):
    """
    To calculate the mean error of predicted physical params value in confident area of a image
    phy_mask (2,H,W) H,W is the size of resized img
    """
    # check dim of phy_masks first
    if (
        ori_phy_mask.shape[-2] != pred_phy_mask.shape[-2]
        or ori_phy_mask.shape[-1] != pred_phy_mask.shape[-1]
    ):
        raise ValueError("ori_phy_mask and pred_phy_mask should have the same shape!")
    compare_regions = ~torch.isnan(ori_phy_mask)
    regions_in_pred = pred_phy_mask * compare_regions
    regions_in_pred = torch.where(regions_in_pred == 0, torch.nan, regions_in_pred)
    delta = torch.abs(regions_in_pred - ori_phy_mask)
    delta_mask = ~torch.isnan(delta[0])

    fric_dvalues = delta[0][delta_mask]
    fric_mean_deviation = torch.mean(fric_dvalues)
    fric_std_deviation = torch.std(fric_dvalues)

    stiff_dvalues = delta[1][delta_mask]
    stiff_mean_deviation = torch.mean(stiff_dvalues)
    stiff_std_deviation = torch.std(stiff_dvalues)

    return {
        "fric_mean_deviat": fric_mean_deviation,
        "fric_std_deviation": fric_std_deviation,
        "fric_loss_raw": fric_dvalues,
        "stiff_mean_deviat": stiff_mean_deviation,
        "stiff_std_deviation": stiff_std_deviation,
        "stiff_loss_raw": stiff_dvalues,
    }


def test_extractor():
    import cv2
    import os
    import time
    from base_wvn import ParamCollection

    param = ParamCollection()
    image_relative_path = (
        "image/forest_clean.png"  # Update to the relative path of your image
    )
    feat_relative_path = "image/forest_clean_feat.png"
    # Use os.path.join to get the full path of the image
    image_path = os.path.join(WVN_ROOT_DIR, image_relative_path)
    np_img = cv2.imread(image_path)
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(param.run.device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
    extractor = FeatureExtractor(param)
    start_time = time.time()

    trans_img, compressed_feat = extractor.extract(img)
    features = next(iter(compressed_feat.values()))
    # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time
    inference_time = end_time - start_time

    # Print the resulting feature tensor and inference time
    print("Feature shape:", features.shape)
    print("Extract time: {:.3f} seconds".format(inference_time))

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    B, C, H, W = features.shape
    features = features.permute(0, 2, 3, 1)
    features = features.reshape(B, H * W, C)
    features = features[0].cpu().numpy()
    n = 3
    pca = PCA(n_components=n)
    pca.fit(features)

    pca_features = pca.transform(features)
    for i in range(n):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (
            pca_features[:, i].max() - pca_features[:, i].min()
        )
    # pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    plt.imshow(pca_features.reshape(H, W, n).astype(np.uint8))
    image_path = os.path.join(WVN_ROOT_DIR, feat_relative_path)
    plt.savefig(image_path)


if __name__ == "__main__":
    test_extractor()
