#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple


@dataclass
class RawInputParams:
    component_dims: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "pro": (0, 133),  # first 133 dims
            "exte": (133, 341),  # next 208 dims
            "hidden": (341, 441),  # last 100 dims
            # one by one, not overlapping
        }
    )

    # input_type composition mapping
    input_type_map: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "pro": ["pro"],
            "pro+exte": ["pro", "exte"],
            "all": ["pro", "exte", "hidden"],
            "all-e": ["pro", "hidden"],
        }
    )

    def input_components(self, desired_input_type) -> List[str]:
        return self.input_type_map[desired_input_type]

    def input_size(self, desired_input_type) -> int:
        return sum(
            self.component_dims[comp][1] - self.component_dims[comp][0]
            for comp in self.input_components(desired_input_type)
        )

    def input_slice(self, desired_input_type) -> Dict[str, Tuple[int, int]]:
        components = self.input_components(desired_input_type)
        sorted_components = sorted(
            components, key=lambda comp: self.component_dims[comp][0]
        )
        return {comp: self.component_dims[comp] for comp in sorted_components}


@dataclass
class ModelParams:
    seq_length: int = 50
    reset_hidden_each_epoch: bool = True
    input_type: str = "pro+exte"  # Choose 'pro','pro+exte', or 'all'or 'all-e'
    output_type: str = "fric"  # or 'stiff'
    mode: str = (
        "parallel"  # Choose  'base'  or 'mlp' or 'parallel' ... please check Modes.py
    )

    norm: bool = True
    priv_size: int = 4

    hidden_to_pred_shape: List[int] = field(default_factory=lambda: [64, 32])
    hidden_to_gate_shape: List[int] = field(default_factory=lambda: [64, 64])
    gate_to_pred_shape: List[int] = field(default_factory=lambda: [64, 32])

    hidden_size: int = 100  # 50 for S, 100 for M, 200 for L
    num_rnn_layers: int = 1  # 1 for S M, 2 for L
    l1_alpha: float = 0.2
    activation_fn: str = "LeakyReLU"
    device: str = "cuda"  # or "cpu"

    # Derived values
    input_size: int = field(init=False)
    input_slice: Dict[str, Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        raw = RawInputParams()
        self.input_size = raw.input_size(self.input_type)
        self.input_slice = raw.input_slice(self.input_type)

    def to_dict(self):
        d = asdict(self)  # Convert all fields to a dict
        return d
