from dataclasses import asdict
import deepdish as dd
import torch
from torch.utils.data import  DataLoader
from pytorch_lightning import Trainer
import os

from pytorch_lightning.loggers.neptune import NeptuneLogger
import gc
# Imports from the Decoder_training directory

from training_utils.dataset_analysis import get_latest_file_in_directory, NoDataFileError, extract_timestamp_from_filename
from training_utils.decoder import *
from training_utils.decoder_config import ParamCollection
from training_utils.combine_dataset import combine_datasets_from_directory
from training_utils.utils import *

""" If the model use middle_only mode, the output will be sequence_length=1. Evaluation function like evaluate_on_data will not work. They need to be modified to fit for the middle-only option. """

def organize_data(data, sequence_length=10, use_rnn=False,overlapp=None):
    """
    Organize the data for training. If use_rnn is True, the data will be organized 
    for RNN training, otherwise for MLP training. 
    Returns a dictionary with keys as env_ids and values as lists of sequences (for RNN) 
    or single data points (for MLP).

    Also, if subtraction mode is on, subtract the exte and policy data from its previous 1 time step.
    """
    organized_data = {}
    selected_cols = [i * 9+8 for i in range(4)]

    for env_id, sensors in data.items():
        # 'counter' is the sensor that determines episode boundaries
        counter_data = sensors['counter']
        increment_indices = (counter_data[1:] - counter_data[:-1]).nonzero(as_tuple=True)[0] + 1
        increment_indices = increment_indices.tolist()
        boundaries = [0] + increment_indices + [len(counter_data)]
        data_points = []
        
        for i in range(len(boundaries) - 1):
            episode_start, episode_end = boundaries[i], boundaries[i+1]

            if use_rnn:
                interval=sequence_length
                if overlapp is not None:
                    interval=overlapp
                for j in range(episode_start, episode_end, interval):
                    if j + sequence_length > episode_end:
                        break

                    policy_data = sensors['policy'][j:j+sequence_length]
                    exte_data = sensors['priv_exte'][j:j+sequence_length]
                    hidden_data = sensors['hidden'][j:j+sequence_length]
                    fric_data = sensors['fric'][j:j+sequence_length]
                    stiff_data = sensors['stiff'][j:j+sequence_length, selected_cols]
                   
                    data_points.append({
                        "input": torch.cat([policy_data,exte_data, hidden_data], dim=-1),
                        "output": torch.cat([fric_data, stiff_data], dim=-1)
                    })

            else:
                policy_data = sensors['policy'][episode_start:episode_end]
                exte_data = sensors['priv_exte'][episode_start:episode_end]
                hidden_data = sensors['hidden'][episode_start:episode_end]
                fric_data = sensors['fric'][episode_start:episode_end]
                stiff_data = sensors['stiff'][episode_start:episode_end, selected_cols]

                for j in range(0, episode_end-episode_start):
                    data_points.append({
                        "input": torch.cat([policy_data[j].unsqueeze(0),exte_data[j].unsqueeze(0), hidden_data[j].unsqueeze(0)], dim=-1),
                        "output": torch.cat([fric_data[j].unsqueeze(0), stiff_data[j].unsqueeze(0)], dim=-1)
                    })

        organized_data[env_id] = data_points

    return organized_data



def train_and_evaluate(param:ParamCollection):
    
    # Initialize the Neptune logger

    neptune_logger = NeptuneLogger(
        api_key=param.general.api_key,
        project=param.general.project,
        tags=param.general.tags,
        
    )
    model_args=param.create_model_params(param.model.mode)
    # Extract configuration
    mode = param.general.mode
    max_epochs = param.general.max_epochs
    num_workers = param.general.num_workers
    batch_size = param.general.batch_size
    use_weight = param.general.use_weight
    SEPARATE_DATASET_MODE=param.general.SEPERATE_DATASET_MODE
    overlapp=param.general.overlapp
    continue_training=param.general.continue_training
    model_types=param.general.model_types
    model_directory = param.general.model_directory
    add_noise=param.general.add_noise
    
    train_data_directory=os.path.expanduser(param.general.train_data_directory) 
    val_data_directory=os.path.expanduser(param.general.val_data_directory)
    
    seq_length = model_args['seq_length']
    use_rnn = model_args['use_rnn']
    reset_hidden_each_epoch = model_args['reset_hidden_each_epoch']
    rnn_mode = model_args['mode']
    input_type = model_args['input_type']
    paradigm = model_args['paradigm']

    for model_type in model_types:
        # Load training data
        if SEPARATE_DATASET_MODE:
            train_data = combine_datasets_from_directory(train_data_directory, model_type,'train')
            
            organized_train_data = organize_data(train_data, seq_length, use_rnn, overlapp)
        else:
            latest_train_data_file = get_latest_file_in_directory(train_data_directory)
            if latest_train_data_file:
                train_data = dd.io.load(latest_train_data_file)
                organized_train_data = organize_data(train_data, seq_length, use_rnn, overlapp)
            else:
                raise NoDataFileError("No training data file found in the specified directory.")
        train_dataset = RNNDataset(organized_train_data, use_rnn, output_type=model_type, input_type=input_type,add_noise=add_noise)
        train_data = None
        # Load validation data
        if SEPARATE_DATASET_MODE:
            val_data = combine_datasets_from_directory(val_data_directory, model_type,'val')
            organized_val_data = organize_data(val_data, seq_length, use_rnn, overlapp)
        else:
            latest_val_data_file = get_latest_file_in_directory(val_data_directory)
            if latest_val_data_file:
                val_data = dd.io.load(latest_val_data_file)
                organized_val_data = organize_data(val_data, seq_length, use_rnn, overlapp)
            else:
                raise NoDataFileError("No validation data file found in the specified directory.")
        val_dataset = RNNDataset(organized_val_data, use_rnn, output_type=model_type, input_type=input_type,add_noise=add_noise)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        val_data = None
        # Once the data loaders are created, the original datasets can be set to None.
        # This is to free up memory.
        # train_dataset = None
        val_dataset = None
        gc.collect()
        torch.cuda.empty_cache()
        # Get a single batch from the data loader
        sample_input, sample_output = next(iter(train_loader))

        # Determine the width/dimension of the input data
        input_width = sample_input.shape[-1]

        # here only used for logging, asdict might ignore transformer cfg in that mode
        hyperparameters = asdict(param)
    
        if mode == "train":
            model_args["is_training"]=True
            # Train the model
            if continue_training:
                search_pattern = construct_search_pattern(use_rnn, rnn_mode, input_type, use_weight, model_type,paradigm=paradigm)
                latest_model_file = get_latest_file_in_directory(model_directory, pattern=search_pattern)
                if not latest_model_file:
                    raise ValueError(f"No '{model_type}' model found in the specified directory for continuation.")

                model = LightningWrapper.create_lightning(model_args, model_name=model_type)
                model.load_state_dict(torch.load(latest_model_file))
            else:
                model = LightningWrapper.create_lightning(model_args, model_name=model_type)
            data_inputs, data_outputs = zip(*train_dataset.data)
            train_dataset = None
            gc.collect()
            torch.cuda.empty_cache()
            model.decoder.set_range_weight(data_outputs, use_weight)
            neptune_logger.log_hyperparams(hyperparameters)
            trainer = Trainer(accelerator="gpu", devices=[0], logger=neptune_logger, max_epochs=max_epochs)
            trainer.fit(model, train_loader, val_loader)
            save_model(model, model_type, use_rnn, input_width, use_weight, hyperparameters)

        # Testing mode after training
        
        search_pattern = construct_search_pattern(use_rnn, rnn_mode, input_type, use_weight, model_type,paradigm=paradigm)
        latest_model_file = get_latest_file_in_directory(model_directory, pattern=search_pattern)
        timestamp = extract_timestamp_from_filename(latest_model_file)
    
        model_args["is_training"]=False

        if latest_model_file:
            model = LightningWrapper.create_lightning(model_args, model_name=model_type)
            model.load_state_dict(torch.load(latest_model_file))
            model=model.to('cuda:0')
            if paradigm!='dropout':
                model.train()
            evaluate_on_data(
                model, 
                organized_val_data, 
                env_id=0, 
                model_name=model_type, 
                use_rnn=use_rnn, 
                reset_hidden_each_epoch=reset_hidden_each_epoch, 
                output_type=model_type, 
                input_type=input_type,
                overlapp=overlapp,
                bin_accuracy_calculator=model.decoder.bin_accuracy_cal,
                logger=neptune_logger
            )
            if not model.decoder.prob:
                # Collect errors and plot histogram for fric model
                true_values, errors = collect_errors(model, val_loader,use_rnn,reset_hidden_each_epoch)
                if SEPARATE_DATASET_MODE:
                    plot_error_histogram(true_values, errors, 0.1 if model_type == "fric" else 1, 0 if model_type == "fric" else 1, 1 if model_type == "fric" else 10, model_type, use_rnn, input_width, timestamp, ratio=1.0 if model_type == "fric" else 0.1, use_weight=use_weight, rnn_mode=rnn_mode, logger=neptune_logger)
                else:
                    plot_error_histogram(true_values, errors, 0.1 if model_type == "fric" else 1, 0 if model_type == "fric" else 1, 1 if model_type == "fric" else 10, model_type, use_rnn, input_width, timestamp, use_weight=use_weight, rnn_mode=rnn_mode, logger=neptune_logger)
            else:
                true_values, errors,stds=collect_errors_prob(model, val_loader,use_rnn,reset_hidden_each_epoch)
                plot_error_histogram_prob(true_values, errors,stds, 0.1 if model_type == "fric" else 1, 0 if model_type == "fric" else 1, 1 if model_type == "fric" else 10, model_type, use_rnn, input_width, timestamp,ratio=1.0 if model_type == "fric" else 0.1, use_weight=use_weight, rnn_mode=rnn_mode, logger=neptune_logger)
        else:
            print(f"No '{model_type}' model found in the specified directory.")

if __name__ == "__main__":

    param=ParamCollection()
    # Call the function with the configuration
    train_and_evaluate(param)
    