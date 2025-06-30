#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
from pytorch_lightning import Trainer

from pytorch_lightning.loggers.neptune import NeptuneLogger

from training_utils.dataset_analysis import (
    extract_timestamp_from_filename,
)
from training_utils.lightning_wrapper import LightningWrapper
from training_utils.run_config import ParamCollection
from training_utils.data_manager import DataManager
from training_utils.utils import (
    save_model,
    single_env_eval,
    batch_eval,
)

from typing import Optional

from physical_decoder import (
    ModelFactory,
    get_latest_file_in_directory,
    construct_search_pattern,
)


def train(param: ParamCollection, data_manager: DataManager) -> NeptuneLogger:
    # Initialize the Neptune logger
    neptune_logger = NeptuneLogger(
        api_key=param.general.api_key,
        project=param.general.project,
        tags=param.general.tags,
    )
    model_args = param.model_params_to_dict()
    # Extract configuration
    max_epochs = param.general.max_epochs
    continue_training = param.general.continue_training
    model_directory = param.general.model_directory

    input_type = model_args["input_type"]
    output_type = model_args["output_type"]

    train_loader, val_loader = data_manager.train_loader, data_manager.val_loader

    # here only used for logging
    hyperparameters = param.all_to_dict()

    model_args["is_training"] = True
    model_args["label_weights"] = data_manager.label_weights
    model_args["label_weighting_bin_interval"] = (
        data_manager.label_weighting_bin_interval
    )
    # Train the model
    if continue_training:
        search_pattern = construct_search_pattern(model_args)
        latest_decoder_file = get_latest_file_in_directory(
            model_directory, pattern=search_pattern
        )
        if not latest_decoder_file:
            raise ValueError(
                f"No '{output_type}' model found in the specified directory for continuation."
            )

        model = LightningWrapper.create_lightning(model_args)
        model.load(latest_decoder_file)
    else:
        model = LightningWrapper.create_lightning(model_args)

    neptune_logger.log_hyperparams(hyperparameters)
    trainer = Trainer(
        logger=neptune_logger,
        max_epochs=max_epochs,
        accelerator="gpu" if "cuda" in model_args["device"] else "cpu",
    )
    trainer.fit(model, train_loader, val_loader)
    decoder = model.get_unwrapped_decoder()
    save_model(decoder, output_type, input_type, hyperparameters)

    return neptune_logger


def evaluate(
    param: ParamCollection,
    data_manager: DataManager,
    neptune_logger: Optional[NeptuneLogger] = None,
) -> None:
    if neptune_logger is None:
        neptune_logger = NeptuneLogger(
            api_key=param.general.api_key,
            project=param.general.project,
            tags=param.general.tags,
        )
    model_args = param.model_params_to_dict()

    model_directory = param.general.model_directory
    output_type = model_args["output_type"]

    val_loader, single_env_val_loader = (
        data_manager.val_loader,
        data_manager.single_env_val_loader,
    )

    search_pattern = construct_search_pattern(model_args)
    latest_decoder_file = get_latest_file_in_directory(
        model_directory, pattern=search_pattern
    )
    timestamp = extract_timestamp_from_filename(latest_decoder_file)

    model_args["is_training"] = False

    if latest_decoder_file:
        decoder = ModelFactory.create_model(config=model_args)
        decoder.load(latest_decoder_file)
        decoder.eval()
        single_env_eval(
            decoder,
            single_env_val_loader,
            logger=neptune_logger,
            params=param,
        )
        batch_eval(
            decoder,
            val_loader,
            logger=neptune_logger,
            timestamp=timestamp,
            params=param,
        )
    else:
        raise ValueError(
            f"No '{output_type}' decoder found in the specified directory."
        )


if __name__ == "__main__":
    param = ParamCollection()
    logger = None
    data_manager = DataManager(param)
    if param.general.mode == "train":
        logger = train(param, data_manager)
    evaluate(param, data_manager, logger)
