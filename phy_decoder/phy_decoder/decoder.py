#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from . import modules


class ModelFactory:
    @staticmethod
    def create_model(config: Dict[str, Any]) -> nn.Module:
        model_type = config["mode"]
        if model_type == "base":
            return Base(config)
        elif model_type == "parallel":
            return Parallel(config)
        elif model_type == "mlp":
            return Mlp(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# temporal-wise attention
class SelfAttention(nn.Module):
    def __init__(self, input_dim: int):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim**0.5)
        if mask is not None:
            scores += mask
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class Model(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(Model, self).__init__()
        if not isinstance(config, dict):
            raise ValueError("Config must be a dict.")
        self.config = config
        # Set default values and override with any values provided in config
        self.input_size = config["input_size"]
        self.priv_size = config["priv_size"]
        self.hidden_to_pred_shape = config["hidden_to_pred_shape"]
        self.hidden_to_gate_shape = config["hidden_to_gate_shape"]
        self.gate_to_pred_shape = config["gate_to_pred_shape"]
        self.hidden_size = config["hidden_size"]
        self.num_rnn_layers = config["num_rnn_layers"]
        self.seq_length = config["seq_length"]
        self.input_slice = config["input_slice"]

        self.reset_hidden = config.get("reset_hidden_each_epoch", True)
        self.activation_fn = config.get("activation_fn", "LeakyReLU")
        if self.activation_fn == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU
        self.device = config.get("device", "cpu")
        self.norm = config.get("norm", False)
        if self.norm:
            self.priv_normalizer = modules.EmpiricalNormalization(
                [self.seq_length, self.priv_size],
                until=1e8,
                clip_threshold=10,
            )
        else:
            self.priv_normalizer = None

    def forward(self, input: torch.Tensor):
        raise NotImplementedError

    def predict(self, input: torch.Tensor):
        raise NotImplementedError

    def normalize(self, value: torch.Tensor, is_training: bool) -> torch.Tensor:
        """
        Normalize the label / prediction.
        """
        if self.priv_normalizer is not None:
            if is_training:
                self.priv_normalizer.set_training_mode()
            else:
                self.priv_normalizer.set_validation_mode()
            return self.priv_normalizer(value)
        else:
            return value

    def denormalize(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Inverse normalization.
        """
        if self.priv_normalizer is not None:
            return self.priv_normalizer.inverse(pred)
        else:
            return pred

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Return new latent state.
        batch_size is the number of parallel environments being used.
        """
        h0 = torch.zeros(
            self.num_rnn_layers, batch_size, self.hidden_size, device=self.device
        )
        return h0

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)  # Ensure the model is on the correct device after loading


class Base(Model):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hidden_to_pred = modules.MLP(
            self.hidden_to_pred_shape,
            self.activation_fn,
            self.hidden_size,
            self.priv_size,
            init_scale=1.0 / np.sqrt(2),
        )
        self.rnn = nn.GRU(
            self.input_size,
            self.hidden_size,
            self.num_rnn_layers,
            bidirectional=False,
        )
        self.attention = SelfAttention(self.hidden_size)
        self.mask = None
        self.hidden = None

    def forward_rnn(self, x: torch.Tensor) -> torch.Tensor:
        if self.reset_hidden or self.hidden is None:
            self.hidden = self.init_hidden(x.shape[1])
        self.hidden = self.hidden.detach()
        output, self.hidden = self.rnn(x, self.hidden)
        output = output.permute(1, 0, 2)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized prediction (used during training).
        """
        x = input_selection(x, self.input_slice)  # Select input based on input type
        x = x.permute(1, 0, 2)
        output = self.forward_rnn(x)
        output_weighted = self.attention(output)
        pred = self.hidden_to_pred(output_weighted)
        return pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns de-normalized output (used during inference).
        """
        with torch.no_grad():
            norm_pred = self.forward(x)
            real_pred = self.denormalize(norm_pred)
            return real_pred


class Parallel(Base):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.alpha = None
        self.pro_size = self.input_slice["pro"][1] - self.input_slice["pro"][0]
        self.rnn_pro = nn.GRU(
            self.pro_size, self.hidden_size, self.num_rnn_layers, bidirectional=False
        )
        self.hidden_to_pred = modules.MLP(
            self.hidden_to_pred_shape,
            self.activation_fn,
            self.hidden_size,
            self.priv_size,
            init_scale=1.0 / np.sqrt(2),
        )
        self.hidden_to_gate = modules.MLP(
            self.hidden_to_gate_shape,
            self.activation_fn,
            self.hidden_size,
            self.priv_size,
            init_scale=1.0 / np.sqrt(2),
        )
        self.hidden_to_pred_pro = modules.MLP(
            self.hidden_to_pred_shape,
            self.activation_fn,
            self.hidden_size,
            self.priv_size,
            init_scale=1.0 / np.sqrt(2),
        )
        self.attention_pro = SelfAttention(self.hidden_size)
        self.mask = None
        self.hidden_pro = None

    def forward_rnn(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Path 1 rnn
        if self.reset_hidden or self.hidden is None:
            self.hidden = self.init_hidden(x.shape[1])
        self.hidden = self.hidden.detach()
        output, self.hidden = self.rnn(x, self.hidden)
        output = output.permute(1, 0, 2)

        # Path 2 rnn
        if self.reset_hidden or self.hidden_pro is None:
            self.hidden_pro = self.init_hidden(x.shape[1])
        self.hidden_pro = self.hidden_pro.detach()
        pro_input = x[:, :, : self.pro_size]
        pro_output, self.hidden_pro = self.rnn_pro(pro_input, self.hidden_pro)
        pro_output = pro_output.permute(1, 0, 2)
        return output, pro_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized prediction (used during training).
        """
        x = input_selection(x, self.input_slice)  # Select input based on input type
        x = x.permute(1, 0, 2)  # convert for rnn input
        output, pro_output = self.forward_rnn(x)
        output_weighted = self.attention(output, self.mask)
        pro_output_weighted = self.attention_pro(pro_output, self.mask)
        pro_pred = self.hidden_to_pred_pro(pro_output_weighted)
        self.alpha = F.sigmoid(self.hidden_to_gate(output_weighted))
        filtered_pro_output = self.alpha * pro_pred
        pred = filtered_pro_output + self.hidden_to_pred(output_weighted)
        return pred


class Mlp(Model):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.in_to_pred = modules.MLP(
            self.hidden_to_pred_shape,
            self.activation_fn,
            self.input_size,
            self.priv_size,
            init_scale=1.0 / np.sqrt(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized prediction (used during training).
        """
        x = input_selection(x, self.input_slice)  # Select input based on input type
        pred = self.in_to_pred(x)
        return pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns de-normalized output (used during inference).
        """
        with torch.no_grad():
            norm_pred = self.forward(x)
            real_pred = self.denormalize(norm_pred)
            return real_pred


def input_selection(
    input_data: torch.Tensor, input_slice: Dict[str, Tuple[int, int]]
) -> torch.Tensor:
    sliced_data = []
    for key, (start, end) in input_slice.items():
        slice = input_data[..., start:end]
        sliced_data.append(slice)
    sliced_data = torch.cat(sliced_data, dim=-1)
    return sliced_data
