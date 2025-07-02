#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch

from abc import ABC, abstractmethod
from sklearn.mixture import GaussianMixture


class ConfidenceMaskGeneratorFactory:
    @staticmethod
    def create(mode: str, device: str):
        if mode == "gmm_1d":
            return GMM1D(
                num_components=2,
                deivice=device,
            )
        elif mode == "fixed":
            return FixedThreshold(
                # adjustable parameters if needed
                std_factor=1.0,
                threshold=0.8,
                method="running_mean",
                device=device,
            )
        else:
            raise ValueError(f"Confidence mask generator mode {mode} not implemented")


class MaskGenerator(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def get_confidence_mask(self):
        """Returns a boolean mask based on the confidence scores"""
        raise NotImplementedError("This method should be overridden by subclasses")


class FixedThreshold(MaskGenerator):
    def __init__(
        self,
        std_factor: float,
        threshold: float,
        method: str,
        device: str,
    ):
        """Returns a confidence value for each number"""
        super(FixedThreshold, self).__init__()
        self.device = device
        self.std_factor = std_factor
        self.threshold = threshold

        mean = torch.zeros(1, dtype=torch.float32)
        var = torch.ones(1, dtype=torch.float32)
        std = torch.ones(1, dtype=torch.float32)
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.var = torch.nn.Parameter(var, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)

        if method == "running_mean":
            running_n = torch.zeros(1, dtype=torch.float64)
            running_sum = torch.zeros(1, dtype=torch.float64)
            running_sum_of_squares = torch.zeros(1, dtype=torch.float64)

            self.running_n = running_n.to(self.device)
            self.running_sum = running_sum.to(self.device)
            self.running_sum_of_squares = running_sum_of_squares.to(self.device)

            self._update = self.update_running_mean
            self._reset = self.reset_running_mean
        else:
            raise ValueError(f"Method {method} not implemented")

    def update_running_mean(self, x: torch.Tensor):
        # We assume the positive samples' loss follows a Gaussian distribution
        # We estimate the parameters empirically
        if x.device != self.device:
            x = x.to(self.device)
        self.running_n += x.numel()
        self.running_sum += x.sum()
        self.running_sum_of_squares += (x**2).sum()

        self.mean[0] = self.running_sum[0] / self.running_n
        self.var[0] = self.running_sum_of_squares / self.running_n - self.mean**2
        self.std[0] = torch.sqrt(self.var)

        return self.inference_without_update(x)

    def reset(self):
        self._reset()

    def reset_running_mean(self):
        self.running_n[0] = 0
        self.running_sum[0] = 0
        self.running_sum_of_squares[0] = 0

    def get_dict(self):
        return {"mean": self.mean, "var": self.var, "std": self.std}

    def update(
        self,
        x: torch.Tensor,
    ):
        """Input a tensor with multiple error predictions.
        Returns the estimated confidence score within 2 standard deviations based on the running mean and variance.

        Args:
            x (torch.tensor): (H*W,)
        Returns:
            (torch.tensor): (H*W,)
        """
        output = self._update(x)
        return output

    def inference_without_update(self, x: torch.Tensor):
        if x.device != self.mean.device:
            x = x.to(self.mean.device)

        confidence = torch.exp(
            -(((x - self.mean) / (self.std * self.std_factor)) ** 2) * 0.5
        )
        confidence[x < self.mean] = 1.0  # debug, I think it fit the paper
        return confidence.type(torch.float32)

    def get_confidence_mask(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loss (torch.Tensor): shape (H*W,)

        Returns:
            torch.Tensor: shape (H*W,), boolean mask, need to be reshaped later
        """
        assert loss.dim() == 1, "Loss tensor should be 1D"
        confidence = self.inference_without_update(loss)
        conf_mask = confidence >= self.threshold
        return conf_mask


class GMM1D(MaskGenerator):
    def __init__(self, num_components: int, deivice: str):
        super(GMM1D, self).__init__()
        self.num_components = num_components
        self.device = deivice
        self.gmm_1d = GaussianMixture(
            n_components=self.num_components,
            random_state=42,
        )

    def update(self, x: torch.Tensor):
        # GMM does not need train-time update
        return

    def get_confidence_mask(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loss (torch.Tensor): shape (H*W,)

        Returns:
            torch.Tensor: shape (H*W,), boolean mask, need to be reshaped later
        """
        assert loss.dim() == 1, "Loss tensor should be 1D"
        loss = loss.detach().cpu().numpy().reshape(-1, 1)  # (H*W, 1)
        self.gmm_1d.fit(loss)
        labels = self.gmm_1d.predict(loss)
        confident_cluster = self.gmm_1d.means_.argmin()
        confidence_mask = labels == confident_cluster
        return torch.tensor(confidence_mask, dtype=torch.bool, device=self.device)


if __name__ == "__main__":
    cg = FixedThreshold()
    for i in range(100000):
        inp = (
            torch.rand(
                10,
            )
            * 10
        )
        res = cg.update(inp, inp, i)
        print("inp ", inp, " res ", res, "std", cg.std)
