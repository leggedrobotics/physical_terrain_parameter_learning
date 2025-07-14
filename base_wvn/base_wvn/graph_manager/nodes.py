#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
from liegroups.torch import SE3
import torch
import torch.nn.functional as F
from typing import Union
from base_wvn.utils import ImageProjector


class BaseNode:
    """Base node data structure"""

    _name = "base_node"

    def __init__(
        self, timestamp: float = 0.0, pose_base_in_world: torch.Tensor = torch.eye(4)
    ):
        assert isinstance(pose_base_in_world, torch.Tensor)

        self._timestamp = timestamp
        self._pose_base_in_world = pose_base_in_world

    def __str__(self):
        return f"{self._name}_{self._timestamp}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if other is None:
            return False
        return (
            self._name == other.name
            and self._timestamp == other.timestamp
            and torch.equal(self._pose_base_in_world, other.pose_base_in_world)
        )

    def __lt__(self, other):
        return self._timestamp < other.timestamp

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._pose_base_in_world = self._pose_base_in_world.to(device)

    @classmethod
    def from_node(cls, instance):
        return cls(
            timestamp=instance.timestamp, pose_base_in_world=instance.pose_base_in_world
        )

    def is_valid(self):
        return True

    def pose_between(self, other):
        """Computes pose difference (SE(3)) between this state and other

        Args:
            other (BaseNode): Other state

        Returns:
            tensor (torch.tensor): Pose difference expressed in this' frame
        """
        return other.pose_base_in_world.inverse() @ self.pose_base_in_world

    def distance_to(self, other):
        """Computes the relative distance between states

        Args:
            other (BaseNode): Other state

        Returns:
            distance (float): absolute distance between the states
        """
        # Compute pose difference, then log() to get a vector, then extract position coordinates, finally get norm
        return (
            SE3.from_matrix(
                self.pose_base_in_world.inverse() @ other.pose_base_in_world,
                normalize=True,
            )
            .log()[:3]
            .norm()
        )

    @property
    def name(self):
        return self._name

    @property
    def pose_base_in_world(self):
        return self._pose_base_in_world

    @property
    def timestamp(self):
        return self._timestamp

    @pose_base_in_world.setter
    def pose_base_in_world(self, pose_base_in_world: torch.Tensor):
        self._pose_base_in_world = pose_base_in_world

    @timestamp.setter
    def timestamp(self, timestamp: float):
        self._timestamp = timestamp


class MainNode(BaseNode):
    """Main node stores the minimum information required for visual decoder.
    All the information is stored on the image plane

    image shape (B,C,H,W):transformed image
    feat shape (B,num_segs or H*W,C): Sparse features tensor
    seg (H,W): Segmentation map
    """

    _name = "main_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        pose_base_in_world: torch.Tensor = torch.eye(4),
        pose_cam_in_world: torch.Tensor = None,
        image: torch.Tensor = None,
        image_projector: ImageProjector = None,
        features: Union[torch.tensor, dict] = None,
        feature_type: str = None,
        camera_name="cam",
        phy_dim: int = 2,
    ):
        super().__init__(timestamp=timestamp, pose_base_in_world=pose_base_in_world)
        # Initialize members
        self._pose_cam_in_world = pose_cam_in_world
        self._pose_base_in_world = pose_base_in_world
        self._image = image
        self._image_projector = image_projector
        self._camera_name = camera_name
        self._features = features
        self._feature_type = feature_type
        self._phy_dim = phy_dim
        H, W = self._image.shape[-2], self._image.shape[-1]
        self._supervision_mask = (
            torch.ones(
                (self._phy_dim, H, W), dtype=torch.float32, device=self._image.device
            )
            * torch.nan
        )
        """ 
        Warning: to save GPU memory, move img and features to cpu
        """
        self._image = self._image.cpu()
        for key, tensor in self._features.items():
            self._features[key] = tensor.cpu()

        # Uninitialized members
        self._supervision_signal_valid = None

    @property
    def camera_name(self):
        return self._camera_name

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def image(self):
        return self._image

    @property
    def features(self):
        return self._features

    @property
    def image_projector(self):
        return self._image_projector

    @property
    def pose_cam_in_world(self):
        return self._pose_cam_in_world

    @property
    def supervision_signal_valid(self):
        return self._supervision_signal_valid

    @property
    def supervision_mask(self):
        return self._supervision_mask

    @camera_name.setter
    def camera_name(self, camera_name):
        self._camera_name = camera_name

    @features.setter
    def features(self, features):
        self._features = features

    @feature_type.setter
    def feature_type(self, feature_type):
        self._feature_type = feature_type

    @image_projector.setter
    def image_projector(self, image_projector):
        self._image_projector = image_projector

    @supervision_signal_valid.setter
    def supervision_signal_valid(self, _supervision_signal_valid):
        self._supervision_signal_valid = _supervision_signal_valid

    @supervision_mask.setter
    def supervision_mask(self, supervision_mask):
        self._supervision_mask = supervision_mask

    def clear_debug_data(self):
        """Removes all data not required for training"""
        try:
            del self._image
            del self._supervision_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            pass  # Image already removed

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._image_projector.change_device(device)
        self._image = self._image.to(device)
        self._pose_base_in_world = self._pose_base_in_world.to(device)
        self._pose_cam_in_world = self._pose_cam_in_world.to(device)

        # TODO: also move features dict to device
        if self._supervision_mask is not None:
            self._supervision_mask = self._supervision_mask.to(device)
        if self._supervision_signal_valid is not None:
            self._supervision_signal_valid = self._supervision_signal_valid.to(device)

    def is_valid(self):
        valid_members = (
            isinstance(self._features, torch.Tensor) or isinstance(self._features, dict)
        ) and isinstance(self._supervision_signal_valid, torch.Tensor)
        valid_signals = self._supervision_signal_valid.any() if valid_members else False

        return valid_members and valid_signals

    def update_supervision_signal(self):
        self._supervision_signal_valid = ~torch.isnan(self._supervision_mask)

    def recover_feat(self):
        """Recover the feature from compression if needed"""
        # check it must be dict
        if not isinstance(self._features, dict):
            raise TypeError("The feature must be dict")
        # feat in dict: (ratio_h,ratio_w):(B, C, H, W)
        # recover to tensor (B, H*W,C)
        recover_feat = []
        for key, value in self._features:
            scale_h, scale_w = key
            resized_feats = F.interpolate(
                value.type(torch.float32), scale_factor=(scale_h, scale_w)
            )
            recover_feat.append(resized_feats)
        recover_feat = torch.cat(recover_feat, dim=1)
        recover_feat = recover_feat.permute(0, 2, 3, 1)
        res_features = recover_feat.reshape(
            recover_feat.shape[0], recover_feat.shape[1] * recover_feat.shape[2], -1
        )
        return res_features

    def query_valid_batch(self):
        """Since the feat is compressed with ratio info in dict, we need to query the valid feat
        given supervision_signal_valid ---> get the dataset for model training

        Return:
            valid_feats: (num_valid, C)
            valid_masks: (num_valid, C (=2))
        """
        # transfer the features to the same device as supervision_mask
        temp_feat = {}
        for key, value in self._features.items():
            temp_feat[key] = value.to(self._supervision_mask.device)

        _, H_s, W_s = self._supervision_signal_valid.shape
        valid_indices = torch.where(self._supervision_signal_valid[0] == 1)
        h_indices, w_indices = valid_indices
        scale_layer = 0
        all_selected_feats = []
        for key, value in temp_feat.items():
            B, C, H, W = value.shape
            if H_s / H != key[0] or W_s / W != key[1]:
                raise ValueError(
                    "The ratio in feats dict is not equal to the real ratio"
                )
            ratio_h, ratio_w = key
            # Vectorize the computation of patch indices
            patch_h_indices = h_indices // ratio_h
            patch_w_indices = w_indices // ratio_w
            patch_h_indices = patch_h_indices.type(torch.int64)
            patch_w_indices = patch_w_indices.type(torch.int64)

            selected_feats = value[:, :, patch_h_indices, patch_w_indices]
            selected_feats = selected_feats.squeeze(0).permute(1, 0)
            all_selected_feats.append(selected_feats)
            if scale_layer == 0:
                selected_masks = self._supervision_mask[
                    :, h_indices, w_indices
                ].permute(1, 0)
            scale_layer += 1
        return torch.cat(all_selected_feats, dim=1), selected_masks


class SubNode(BaseNode):
    """Local node stores all the information required for traversability estimation and debugging
    All the information matches a real frame that must be respected to keep consistency"""

    _name = "sub_node"

    def __init__(
        self,
        timestamp: float = 0,
        pose_base_in_world: torch.Tensor = torch.eye(4),
        phy_pred: torch.Tensor = None,
        feet_planes: torch.Tensor = None,
        feet_contact: torch.Tensor = None,
    ):
        assert isinstance(pose_base_in_world, torch.Tensor)
        assert len(feet_contact.shape) == 1
        super().__init__(timestamp, pose_base_in_world)

        self._pose_base_in_world = pose_base_in_world
        self._phy_pred = phy_pred
        self._feet_planes = feet_planes
        self._feet_contact = feet_contact

    def change_device(self, device):
        super().change_device(device)
        self._pose_base_in_world = self._pose_base_in_world.to(device)
        self._phy_pred = self._phy_pred.to(device)
        self._feet_planes = self._feet_planes.to(device)
        self._feet_contact = self._feet_contact.to(device)

    def is_valid(self):
        return isinstance(self._phy_pred, torch.Tensor) and isinstance(
            self._feet_planes, torch.Tensor
        )

    @property
    def phy_pred(self):
        return self._phy_pred

    @property
    def feet_planes(self):
        return self._feet_planes

    @property
    def feet_contact(self):
        return self._feet_contact

    @phy_pred.setter
    def phy_pred(self, phy_pred):
        self._phy_pred = phy_pred

    @feet_planes.setter
    def feet_planes(self, feet_planes):
        self._feet_planes = feet_planes

    @feet_contact.setter
    def feet_contact(self, feet_contact):
        self._feet_contact = feet_contact


if __name__ == "__main__":
    H = 4
    W = 6
    first_tensor = torch.rand((1, H, W))
    second_tensor = (
        torch.rand((1, H, W)) * 9 + 1
    )  # Scaling and shifting to get values from 1 to 10
    supervision_tensor = torch.cat((first_tensor, second_tensor), dim=0)
    super_copy = supervision_tensor.clone() * torch.nan

    super_copy[:, 1:3, 2:5] = supervision_tensor[:, 1:3, 2:5]

    # Generate a (3, H, W) random tensor with values from 0 to 1
    img_tensor = torch.rand((3, H, W))

    # Generate a (6, 4) tensor with values from 0 to 24
    mask_tensor = torch.arange(24).reshape(H, W)

    features = {
        (2, 2): torch.rand(1, 10, H // 2, W // 2)  # Example feature tensor
    }

    main_node = MainNode(features=features, segments=mask_tensor, image=img_tensor)
    main_node.supervision_mask = super_copy
    main_node.update_supervision_signal()
    valid_feats, valid_masks = main_node.query_valid_batch()
    print(valid_feats.shape)

    pass
