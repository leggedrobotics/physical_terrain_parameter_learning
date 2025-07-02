#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
from typing import List


class SimpleMLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        hidden_sizes: List[int] = [255],
        reconstruction: bool = False,
    ):
        super(SimpleMLP, self).__init__()
        layers = []
        hidden_sizes = hidden_sizes.copy()
        self.nr_sigmoid_layers = hidden_sizes[-1]

        if reconstruction:
            hidden_sizes[-1] = hidden_sizes[-1] + input_size

        for hs in hidden_sizes[:-1]:
            layers.append(torch.nn.Linear(input_size, hs))
            layers.append(torch.nn.ReLU())
            input_size = hs
        layers.append(torch.nn.Linear(input_size, hidden_sizes[-1]))

        self.layers = torch.nn.Sequential(*layers)
        self.output_features = hidden_sizes[-1]

    def forward(self, x) -> torch.Tensor:
        # Checked data is correctly memory aligned and can be reshaped
        # If you change something in the dataloader make sure this is still working
        x = self.layers(x)
        # scale the stiffness values by 10x, to make it easier for the network to learn
        x[:, -1] = x[:, -1] * 10
        return x
