#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from simple_parsing.helpers import Serializable
import yaml
import os


@dataclass
class ParamCollection(Serializable):
    """A collection of parameters."""

    @dataclass
    class GeneralParams:
        mode: str = "train"
        max_epochs: int = 100
        num_workers: int = 8
        batch_size: int = 64
        SEPERATE_DATASET_MODE: bool = True
        continue_training: bool = False
        overlapp: int = 25
        add_noise: bool = False
        use_weight: bool = True
        model_types: List[str] = field(default_factory=lambda: ["fric"])  # or 'stiff'

        model_directory: str = "models"
        train_data_directory: str = "~/dataset"
        val_data_directory: str = "~/dataset/validation"

        # train_data_directory:str = "~/dataset/try"
        # val_data_directory:str = "~/dataset/try/val"

        api_key: str = os.getenv("NEPTUNE_API_TOKEN", "default_api_token_if_not_set")
        project: str = os.getenv("NEPTUNE_PROJECT", "default_project_if_not_set")
        tags: List[str] = field(default_factory=lambda: ["tag"])

        def to_dict(self):
            d = asdict(self)  # Convert all fields to a dict
            return d

    general: GeneralParams = GeneralParams()

    @dataclass
    class ModelDefaultParams:
        seq_length: int = 50
        use_rnn: bool = True  # fixed, don't change
        reset_hidden_each_epoch: bool = True
        input_type: str = "pro+exte"  # Choose 'hidden', 'pro','pro+exte', or 'all'or 'all-e' or 'nosubstep'
        mode: str = "parallel"  # Choose  'base'  or 'mlp' or 'parallel' or 'triparallel'... please check Modes.py
        paradigm: Optional[str] = None  #  should be set to None

        backbone: Optional[str] = None
        norm: bool = True
        middle_only: bool = False
        priv_size: int = 4

        hidden_to_pred_shape: List[int] = field(default_factory=lambda: [64, 32])
        hidden_to_gate_shape: List[int] = field(default_factory=lambda: [64, 64])
        gate_to_pred_shape: List[int] = field(default_factory=lambda: [64, 32])

        hidden_size: int = 100  # 50 for S, 100 for M, 200 for L
        num_rnn_layers: int = 1  # 1 for S M, 2 for L
        l1_alpha: float = 0.2
        activation_fn: str = "LeakyReLU"
        device: str = "cuda"

        @property
        def sens_size(self):
            if self.input_type == "pro":
                return 133
            elif self.input_type == "all":
                return 441
            elif self.input_type == "hidden":
                return 100
            elif self.input_type == "pro+exte":
                return 341
            elif self.input_type == "nosubstep":
                return 268
            elif self.input_type == "justsubstep":
                return 281
            elif self.input_type == "nosubstep+cpg":
                return 268 - 24
            else:
                raise ValueError(f"Unknown input type: {self.input_type}")

        def to_dict(self):
            d = asdict(self)  # Convert all fields to a dict
            d["sens_size"] = self.sens_size  # Manually add property value
            return d

    model: ModelDefaultParams = ModelDefaultParams()

    def create_model_params(self, mode):
        cfgs = self.model.to_dict()
        return cfgs

    def all_to_dict(self, cfgs: dict):
        general_dict = self.general.to_dict()
        general_dict.update(cfgs)  # Updates general_dict with key-value pairs from cfgs
        return general_dict


if __name__ == "__main__":
    param = ParamCollection()
    model_args = param.create_model_params(None)
    print(model_args)
    x = param.all_to_dict(model_args)
    print(asdict(param))
    print("ok")
    param_dict = asdict(param)

    # Save to a YAML file
    with open("param_config.yaml", "w") as file:
        yaml.dump(param_dict, file)

    with open("param_config.yaml", "r") as file:
        loaded_dict = yaml.safe_load(file)

    # Assuming Serializable or a similar mechanism properly implements from_dict()
    param_loaded = ParamCollection.from_dict(loaded_dict)
    print(param_loaded)
    print(param_loaded.model.paradigm is None)
