#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
import os
from torch.utils.data import DataLoader
from typing import Tuple
from .helper import load_data, create_dataset_from_nodes
from ..model import VD_dataset
from ..config.wvn_cfg import ParamCollection
from .lightening_module import DecoderLightning


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


def get_train_eval_dataloaders(
    param: ParamCollection, model: DecoderLightning
) -> Tuple[DataLoader, DataLoader]:
    # load train and val data from nodes_datafile (should include all pixels of supervision masks)
    train_data_raw = load_data(
        os.path.join(
            param.offline.data_folder,
            "train",
            param.offline.env,
            param.offline.nodes_datafile,
        )
    )
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
    train_data = create_dataset_from_nodes(
        param,
        train_data_raw,
        model.feat_extractor,
    )
    train_dataset = EntireDataset(
        train_data
    )  # input contain all datapoints as one batch, so need the EntireDataset wrapper to split the batch
    val_data = create_dataset_from_nodes(
        param,
        val_data_raw,
        model.feat_extractor,
    )
    val_dataset = EntireDataset(val_data)
    batch_size = param.graph.random_sample_num

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # batch in loader is a tuple of (xs, ys)
    # xs:(1, 100, feat_dim), ys:(1, 100, 2)
    return train_loader, val_loader
