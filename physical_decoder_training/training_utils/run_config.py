#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from simple_parsing.helpers import Serializable
import yaml
import os

from physical_decoder import ModelParams


@dataclass
class ParamCollection(Serializable):
    """A collection of parameters."""

    @dataclass
    class GeneralParams:
        mode: str = "train"
        max_epochs: int = 100
        num_workers: int = 8
        batch_size: int = 64
        continue_training: bool = False
        overlapp: int = 25
        add_noise: bool = False
        use_weight: bool = True

        model_directory: str = "models"
        train_data_directory: str = "~/dataset"
        val_data_directory: str = "~/dataset/validation"
        val_test_env_id: int = 0

        # train_data_directory: str = "~/dataset/try"
        # val_data_directory: str = "~/dataset/try/val"

        api_key: str = os.getenv("NEPTUNE_API_TOKEN", "default_api_token_if_not_set")
        project: str = os.getenv("NEPTUNE_PROJECT", "default_project_if_not_set")
        tags: List[str] = field(default_factory=lambda: ["tag"])

        def to_dict(self) -> Dict[str, Any]:
            d = asdict(self)  # Convert all fields to a dict
            return d

    general: GeneralParams = GeneralParams()

    model: ModelParams = ModelParams()

    def model_params_to_dict(self) -> Dict[str, Any]:
        cfgs = self.model.to_dict()
        return cfgs

    def all_to_dict(self) -> Dict[str, Any]:
        cfgs = {}

        # Prefix general config keys
        general_dict = {f"general.{k}": v for k, v in self.general.to_dict().items()}
        cfgs.update(general_dict)

        # Prefix model config keys
        model_dict = {f"model.{k}": v for k, v in self.model_params_to_dict().items()}
        cfgs.update(model_dict)

        return cfgs


if __name__ == "__main__":
    param = ParamCollection()
    # model_args = param.model_params_to_dict()
    # print(model_args)
    x = param.all_to_dict()
    print(x)
    print("ok")

    # # Save to a YAML file
    # with open("param_config.yaml", "w") as file:
    #     yaml.dump(param_dict, file)

    # with open("param_config.yaml", "r") as file:
    #     loaded_dict = yaml.safe_load(file)

    # # Assuming Serializable or a similar mechanism properly implements from_dict()
    # param_loaded = ParamCollection.from_dict(loaded_dict)
    # print(param_loaded)
