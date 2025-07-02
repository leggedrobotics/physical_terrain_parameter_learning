#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
from ..model import VD_dataset


class EntireDataset(torch.utils.data.Dataset):
    def __init__(self, data: VD_dataset):
        if len(data.batches) > 1:
            self.x = torch.cat(
                [data.batches[i][0] for i in range(len(data.batches))], dim=0
            )
            self.y = torch.cat(
                [data.batches[i][1] for i in range(len(data.batches))], dim=0
            )
        else:
            self.x, self.y = data.batches[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
