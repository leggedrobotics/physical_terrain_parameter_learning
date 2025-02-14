#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
from torch.utils.data import Dataset
import torch


class VD_dataset(Dataset):
    def __init__(self, list_of_batches, combine_batches=True, random_num=100):
        # list_of_batches is a list of tuples: [(x1, y1), (x2, y2), ...]
        self.combine_batches = combine_batches
        # Dimension check
        for x, y in list_of_batches:
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    "Mismatch in batch size: x and y should have the same first dimension."
                )

        if combine_batches:
            # Combine all batches into one large batch
            xs, ys = zip(*list_of_batches)
            self.xs = torch.cat(xs, dim=0)
            self.ys = torch.cat(ys, dim=0)

            # randomly pick random_num samples from the dataset as the new batches
            # Check if random_num is greater than dataset size
            if random_num > self.xs.size(0):
                self.batches = [(self.xs, self.ys)]
            else:
                # Randomly pick random_num samples from the dataset
                indices = torch.randperm(self.xs.size(0))[:random_num]
                self.xs = self.xs[indices]
                self.ys = self.ys[indices]
                # Storing the randomly picked samples as the new batch
                self.batches = [(self.xs, self.ys)]
        else:
            # Keep batches separate
            self.batches = list_of_batches
            # Keep batches separate and pick random_num from each batch
            self.batches = []
            for x, y in list_of_batches:
                if random_num > x.size(0):
                    self.batches.append((x, y))
                else:
                    indices = torch.randperm(x.size(0))[:random_num]
                    batch_x = x[indices]
                    batch_y = y[indices]
                    self.batches.append((batch_x, batch_y))

    def __len__(self):
        return sum(len(batch[0]) for batch in self.batches)

    def __getitem__(self, index):
        # Find the right batch and index within that batch
        for x_batch, y_batch in self.batches:
            if index < len(x_batch):
                return x_batch[index], y_batch[index]
            index -= len(x_batch)
        raise IndexError("Index out of range")

    def get_batch_num(self):
        return len(self.batches)

    def get_x(self, batch_idx=None):
        if batch_idx is None:
            raise ValueError(
                "batch_idx must be specified when batches are not combined."
            )
        return self.batches[batch_idx][0]

    def get_y(self, batch_idx=None):
        if batch_idx is None:
            raise ValueError(
                "batch_idx must be specified when batches are not combined."
            )
        return self.batches[batch_idx][1]

    def add_batches(self, new_list_of_batches):
        # Perform dimension check as before
        for x, y in new_list_of_batches:
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    "Mismatch in batch size: x and y should have the same first dimension."
                )

        if self.combine_batches:
            # If batches are combined, just concatenate the new data
            new_xs, new_ys = zip(*new_list_of_batches)
            xs = self.batches[0][0]
            ys = self.batches[0][1]
            self.xs = torch.cat([xs] + list(new_xs), dim=0)
            self.ys = torch.cat([ys] + list(new_ys), dim=0)
            self.batches = [(self.xs, self.ys)]
        else:
            # If batches are separate, extend the list
            self.batches.extend(new_list_of_batches)

    def save_dataset(dataset, file_path):
        torch.save(dataset, file_path)

    def load_dataset(file_path):
        return torch.load(file_path)


if __name__ == "__main__":
    # Example usage
    # list_of_batches = [(torch.randn(batch_size1, x_dim), torch.randn(batch_size1, y_dim)), ...]
    # dataset = VD_dataset(list_of_batches, combine_batches=True or False)
    batch_size1 = 10
    batch_size2 = 25
    x_dim = 5
    y_dim = 2
    list_of_batches = [
        (torch.randn(batch_size1, x_dim), torch.randn(batch_size1, y_dim)),
        (torch.randn(batch_size2, x_dim), torch.randn(batch_size2, y_dim)),
    ]
    dataset = VD_dataset(list_of_batches, combine_batches=True)
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
