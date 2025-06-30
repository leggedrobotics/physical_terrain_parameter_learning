#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
from torch import nn

import pytorch_lightning as pl
from training_utils.loss_wrapper import LossWrapperFactory

from typing import Dict, Any, Tuple


class LightningWrapper:
    @staticmethod
    def create_lightning(args: Dict[str, Any]) -> pl.LightningModule:
        return DefaultLightning(args)


class DefaultLightning(pl.LightningModule):
    def __init__(self, args: Dict[str, Any]):
        super().__init__()

        self.wrapped_decoder = LossWrapperFactory.create_loss_wrapper(args)

        self.output_type = args["output_type"]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Forward method is not implemented in the LightningWrapper. Use the wrapped decoder's method instead."
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Args:
            batch (torch.Tensor):input and label tensors, shape (batch_size, sequence_length, feature_dim/priv_size)

        Returns:
            torch.Tensor: shape ()
        """
        input, label = batch
        loss = self.wrapped_decoder.forward_and_get_loss(input, label, is_training=True)
        self.log(f"{self.output_type}_train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input, label = batch
        loss = self.wrapped_decoder.forward_and_get_loss(
            input, label, is_training=False
        )
        self.log(f"{self.output_type}_val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        lr = 0.001
        weight_decay = 0.00001
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer

    def save(self, path: str) -> None:
        self.wrapped_decoder.save(path)

    def load(self, path: str) -> None:
        self.wrapped_decoder.load(path)

    def get_unwrapped_decoder(self) -> nn.Module:
        """Returns the underlying decoder."""
        return self.wrapped_decoder.decoder
