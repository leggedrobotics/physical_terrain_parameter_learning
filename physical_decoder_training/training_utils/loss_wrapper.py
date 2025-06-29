#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch.nn as nn
import torch
from phy_decoder import ModelFactory

from typing import Dict, Any, Optional


class LossWrapperFactory:
    @staticmethod
    def create_loss_wrapper(config: Dict[str, Any]) -> nn.Module:
        return Base(config)


class Base(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.output_type = config["output_type"]
        self.l1_alpha = config.get("l1_alpha", 0.2)
        self.device = config.get("device", "cpu")

        self.label_weights = config["label_weights"]
        self.interval = config["label_weighting_bin_interval"]

        self.decoder = ModelFactory.create_model(config)
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pred = self.decoder(input)
        return pred

    def reset_parameters(self) -> None:
        kaimin_normal = torch.nn.init.kaiming_normal_
        xavier_normal = torch.nn.init.xavier_normal_
        gain = nn.init.calculate_gain("relu")
        orthogonal = torch.nn.init.orthogonal_

        def weights_init():
            def fn(m):
                classname = m.__class__.__name__
                if classname.find("Conv") != -1 or classname.find("Linear") != -1:
                    kaimin_normal(m.weight.data)
                    try:
                        m.bias.data.fill_(0)
                    except:
                        pass

                if classname.find("PFGRU") != -1:
                    xavier_normal(m.fc_z.weight.data)
                    xavier_normal(m.fc_r.weight.data)
                    kaimin_normal(m.fc_n.weight.data, nonlinearity="relu")
                    orthogonal(m.fc_obs.weight.data, gain=gain)
                    m.fc_z.bias.data.fill_(0)
                    m.fc_r.bias.data.fill_(0)
                    m.fc_n.bias.data.fill_(0)
                    m.fc_obs.bias.data.fill_(0)

            return fn

        self.apply(weights_init())

    def forward_and_get_loss(
        self, input: torch.Tensor, label: torch.Tensor, is_training: bool
    ) -> torch.Tensor:
        output = self.forward(input)

        if output.shape[1] != label.shape[1]:
            raise ValueError(
                "The output and label should have the same sequence length."
            )
        normalized_label = self.decoder.normalize(label, is_training)
        loss_w = None
        if self.label_weights is not None and self.interval is not None:
            indices = (label / self.interval).long()
            indices = torch.clamp(
                indices, 0, len(self.label_weights) - 1
            )  # ensure indices are within range
            loss_w = self.label_weights[indices]
        return self.loss(output, normalized_label, loss_w)

    def loss(
        self,
        output: torch.Tensor,
        true_state: torch.Tensor,
        loss_w_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = true_state.shape[0]

        if loss_w_in is not None:
            # weights based on data balance
            loss_w = loss_w_in
        else:
            loss_w = torch.ones(
                batch_size, true_state.shape[1], true_state.shape[2], device=self.device
            )
        loss_l2 = self.weighted_mse(output, true_state, loss_w)
        loss_l1 = self.weighted_l1(output, true_state, loss_w)
        loss = loss_l2 + self.l1_alpha * loss_l1
        return loss

    def weighted_mse(
        self, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        # w = w.reshape(a.shape[0], 1)
        return (w * (a - b) ** 2).mean()

    def weighted_l1(
        self, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        # w = w.reshape(a.shape[0], 1)
        return (w * torch.abs(a - b)).mean()

    def save(self, path: str) -> None:
        """
        Save the model state to a file.
        """
        self.decoder.save(path)

    def load(self, path: str) -> None:
        """
        Load the model state from a file.
        """
        self.decoder.load(path)
