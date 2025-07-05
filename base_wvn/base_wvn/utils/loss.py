#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch.nn.functional as F
import torch
from torch import nn

from typing import Tuple, Dict, Union

from ..model import VD_dataset


class PhyLoss(nn.Module):
    def __init__(
        self,
        w_pred: float,
        w_reco: float,
        reco_loss_type: str,
    ):
        super(PhyLoss, self).__init__()
        self._w_pred = w_pred
        self._w_recon = w_reco

        self.loss_type = reco_loss_type

    def forward(
        self,
        dataset: Union[VD_dataset, tuple],
        res: torch.Tensor,
        batch_idx: int = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(dataset, tuple):
            x_label, y_label = dataset
        elif isinstance(dataset, VD_dataset):
            x_label = dataset.get_x(batch_idx)
            y_label = dataset.get_y(batch_idx)
        else:
            raise ValueError("dataset must be a tuple or a VD_dataset")

        loss_reco = self.compute_reconstruction_loss(res, x_label)
        # need to normalize the last two dim of res seperately since their range is different
        nr_channel_reco = x_label.shape[1]
        normalized_y = self.normalize_tensor(y_label)
        normalized_res = self.normalize_tensor(res[:, nr_channel_reco:])
        loss_pred = F.mse_loss(normalized_res, normalized_y, reduction="none").mean(
            dim=1
        )

        loss_final = self._w_pred * loss_pred.mean() + self._w_recon * loss_reco.mean()

        return (
            loss_final,
            {"loss_pred": loss_pred.mean(), "loss_reco": loss_reco.mean()},
        )

    def compute_reconstruction_loss(
        self, res: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            res (torch.Tensor): output from the model, consists of input reconstruction and physical parameters.
            input (torch.Tensor): the input tensor to be reconstructed. shape is (H*W, C).

        Returns:
            torch.Tensor: The reconstruction loss, shape is (H*W,).
        """
        nr_channel_reco = input.shape[1]
        if self.loss_type == "mse":
            loss_reco_raw = F.mse_loss(
                res[:, :nr_channel_reco], input, reduction="none"
            )
            loss_reco = loss_reco_raw.mean(
                dim=1
            )  # loss_reco shape is (H*W,), _raw shape is (H*W, C)
        elif self.loss_type == "cosine":
            # Cosine Similarity Loss
            cosine_sim = F.cosine_similarity(res[:, :nr_channel_reco], input, dim=1)
            loss_reco = 1 - cosine_sim
        else:
            raise ValueError("Invalid loss type specified. Choose 'mse' or 'cosine'")
        return loss_reco

    def normalize_tensor(self, tensor):
        # Assuming tensor shape is [batch_size, 2]

        # Extract each dimension
        dim0 = tensor[:, 0]  # First dimension (already 0-1)--friction
        dim1 = tensor[:, 1]  # Second dimension (1-10)--stiffness

        # Normalize the second dimension
        dim1_normalized = dim1 / 10.0
        # Combine the dimensions back into a tensor
        normalized_tensor = torch.stack((dim0, dim1_normalized), dim=1)

        return normalized_tensor
