#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
from torch.utils.data import Dataset
import torch
from typing import List, Tuple


class VD_dataset(Dataset):
    def __init__(
        self, list_of_batches: List[Tuple[torch.Tensor, torch.Tensor]], random_num=100
    ):
        # list_of_batches is a list of tuples: [(x1, y1), (x2, y2), ...], x shapes (N,C), y shapes (N,2)
        # Dimension check
        for x, y in list_of_batches:
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    "Mismatch in batch size: x and y should have the same first dimension."
                )

        # Combine all batches into one large batch
        xs, ys = zip(*list_of_batches)
        self.xs = torch.cat(xs, dim=0)
        self.ys = torch.cat(ys, dim=0)

        # randomly pick random_num samples from the dataset as the new batches
        # Check if random_num is greater than dataset size
        if random_num > self.xs.size(0) or random_num == -1:
            self.batches = [(self.xs, self.ys)]
        else:
            # Randomly pick random_num samples from the dataset
            indices = torch.randperm(self.xs.size(0))[:random_num]
            self.xs = self.xs[indices]
            self.ys = self.ys[indices]
            # Storing the randomly picked samples as the new batch
            self.batches = [(self.xs, self.ys)]

    def __len__(self):
        return sum(len(batch[0]) for batch in self.batches)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Find the right batch and index within that batch
        for x_batch, y_batch in self.batches:
            if index < len(x_batch):
                return x_batch[index], y_batch[index]
            index -= len(x_batch)
        raise IndexError("Index out of range")

    def get_batch_num(self):
        return len(self.batches)

    def get_x(self, batch_idx: int = None) -> torch.Tensor:
        if batch_idx is None:
            raise ValueError(
                "batch_idx must be specified when batches are not combined."
            )
        return self.batches[batch_idx][0]

    def get_y(self, batch_idx: int = None) -> torch.Tensor:
        if batch_idx is None:
            raise ValueError(
                "batch_idx must be specified when batches are not combined."
            )
        return self.batches[batch_idx][1]

    def add_batches(
        self, new_list_of_batches: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        # Perform dimension check as before
        for x, y in new_list_of_batches:
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    "Mismatch in batch size: x and y should have the same first dimension."
                )

        # If batches are combined, just concatenate the new data
        new_xs, new_ys = zip(*new_list_of_batches)
        xs = self.batches[0][0]
        ys = self.batches[0][1]
        self.xs = torch.cat([xs] + list(new_xs), dim=0)
        self.ys = torch.cat([ys] + list(new_ys), dim=0)
        self.batches = [(self.xs, self.ys)]

    @staticmethod
    def merge_VD_datasets(
        vd_datasets: List["VD_dataset"], kept_num: int
    ) -> "VD_dataset":
        """
        Merges multiple VD_dataset instances into a single VD_dataset.
        """
        all_batches = []
        for dataset in vd_datasets:
            all_batches.extend(dataset.batches)
        return VD_dataset(all_batches, random_num=kept_num)

    def save_dataset(dataset: "VD_dataset", file_path: str) -> None:
        torch.save(dataset, file_path)

    def load_dataset(file_path: str) -> "VD_dataset":
        return torch.load(file_path)


if __name__ == "__main__":
    # Example usage
    # list_of_batches = [(torch.randn(batch_size1, x_dim), torch.randn(batch_size1, y_dim)), ...]
    batch_size1 = 10
    batch_size2 = 25
    x_dim = 5
    y_dim = 2
    list_of_batches = [
        (torch.randn(batch_size1, x_dim), torch.randn(batch_size1, y_dim)),
        (torch.randn(batch_size2, x_dim), torch.randn(batch_size2, y_dim)),
    ]
    dataset = VD_dataset(list_of_batches)
    ss = dataset[1]
    batch_size1 = 20
    batch_size2 = 35
    x_dim = 5
    y_dim = 2
    list_of_batches = [
        (torch.randn(batch_size1, x_dim), torch.randn(batch_size1, y_dim)),
        (torch.randn(batch_size2, x_dim), torch.randn(batch_size2, y_dim)),
    ]
    dataset.add_batches(list_of_batches)
    pass
