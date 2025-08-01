#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
# MIT License
#
# Copyright (c) 2020 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values.
    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        batch_axis (int): Batch axis.
        eps (float): Small value for stability.
        dtype (dtype): Dtype of input values.
        until (int or None): If this arg is specified, the link learns input
            values until the sum of batch sizes exceeds it.
    """

    def __init__(
        self,
        shape,
        batch_axis=0,
        eps=1e-2,
        dtype=np.float32,
        until=None,
        clip_threshold=None,
        is_training=True,
    ):
        super(EmpiricalNormalization, self).__init__()
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = eps
        self.until = until
        self.clip_threshold = clip_threshold
        # print(f"Shape: {shape}, Batch Axis: {batch_axis}, dtype: {dtype}")
        self.register_buffer(
            "_mean",
            torch.tensor(np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)),
        )
        self.register_buffer(
            "_var",
            torch.tensor(np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)),
        )
        self.register_buffer("count", torch.tensor(0))

        # cache
        self._cached_std_inverse = torch.tensor(
            np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)
        )
        self._is_std_cached = False
        self._is_training = is_training

    @property
    def mean(self):
        return torch.squeeze(self._mean, self.batch_axis).clone()

    @property
    def std(self):
        return torch.sqrt(torch.squeeze(self._var, self.batch_axis)).clone()

    @property
    def _std_inverse(self):
        if self._is_std_cached is False:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5
            self._is_std_cached = True
        return self._cached_std_inverse

    def experience(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return
        if isinstance(self.batch_axis, tuple):
            count_x = x.shape[self.batch_axis[0]]
        else:
            count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        self.count += count_x
        rate = count_x / self.count.float()
        assert rate > 0
        assert rate <= 1

        var_x = torch.var(x, dim=self.batch_axis, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=self.batch_axis, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        if torch.isnan(self._var).any():
            print("var contains nan value")
            x_max = torch.max(x)
            x_min = torch.min(x)
            ss = torch.any(torch.isnan(x))
        # clear cache
        self._is_std_cached = False

    def forward(self, x):
        """Normalize mean and variance of values based on emprical values.
        Args:
            x (ndarray or Variable): Input values
        Returns:
            ndarray or Variable: Normalized output values
        """

        if self._is_training:
            self.experience(x)

        if not x.is_cuda:
            self._is_std_cached = False
        normalized = (x - self._mean) * self._std_inverse

        if torch.isnan(x).any():
            print("x contains nan value")
        if torch.isnan(self._mean).any():
            print("mean contains nan value")
        if torch.isnan(self._var).any():
            print("var contains nan value")
        # if torch.any(self._var == 0):
        #     print('var contains zero value')
        if torch.isnan(self._std_inverse).any():
            print("std inverse contains nan value")
        if torch.isnan(normalized).any():
            print("normalized contains nan value")

        if self.clip_threshold is not None:
            normalized = torch.clamp(
                normalized, -self.clip_threshold, self.clip_threshold
            )
        if not x.is_cuda:
            self._is_std_cached = False
        return normalized

    def inverse(self, y):
        std = torch.sqrt(self._var + self.eps)
        return y * std + self._mean

    def load_numpy(self, mean, var, count, device="cpu"):
        self._mean = torch.from_numpy(np.expand_dims(mean, self.batch_axis)).to(device)
        self._var = torch.from_numpy(np.expand_dims(var, self.batch_axis)).to(device)
        self.count = torch.tensor(count).to(device)

    def set_training_mode(self):
        self._is_training = True

    def set_validation_mode(self):
        self._is_training = False
