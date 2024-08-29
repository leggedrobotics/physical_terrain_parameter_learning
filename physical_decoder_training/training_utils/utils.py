import torch
from torch.utils.data import Dataset
import numpy as np
import os
import yaml
import datetime
import matplotlib.pyplot as plt
from neptune.types import File
from .dataset_analysis import get_latest_file_in_directory

class RNNDataset(Dataset):
    def __init__(self, data, use_rnn=False, output_type="fric", input_type="hidden",noise_level=0.1,add_noise=False):
        self.data = []
        self.use_rnn = use_rnn
        self.output_type = output_type
        self.input_type = input_type

        for env_id, data_points in data.items():
            for data_point in data_points:
                input_data = data_point['input'].to('cuda:0')
                if add_noise:
                    # add noise to input
                    noise_std = (
                        torch.cat(
                            [
                                torch.zeros(6),
                                torch.ones(3) * 1e-2,
                                torch.ones(3) * 2e-2,
                                torch.ones(12) * 1e-3,
                                torch.ones(12) * 2e-2,
                                torch.ones(12) * 1e-3,
                                torch.zeros(12),
                                torch.ones(24) * 1e-3,
                                torch.ones(12) * 1e-3,
                                torch.ones(12) * 2e-2,
                                torch.zeros(12),
                                torch.zeros(12),
                                torch.zeros(1),
                                torch.ones(208)*0.0,
                                torch.zeros(100),
                            ]
                        )
                        .to('cuda:0')
                        .detach()
                    )
                    self.noise_dist = torch.distributions.Normal(0, noise_std + 1e-10)
                    self.noise_exte= (2 * torch.rand_like(torch.ones(208)) - 1) * 0.5
                    cc=self.noise_dist.sample((input_data.shape[0],))
                    # input_data=input_data+torch.randn_like(input_data)*noise_level
                    input_data+=cc
                    input_data[:,133:341]+=self.noise_exte.to('cuda:0')
                
                if output_type == "fric":
                    output_data = data_point['output'][:, :4]
                elif output_type == "stiff":
                    output_data = data_point['output'][:, 4:]
                elif output_type =="all":
                    output_data = data_point['output'][:, :8]
                else:
                    raise ValueError("Invalid output_type. Choose 'fric' or 'stiff'")
                
                self.data.append((input_data, output_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data, output_data = self.data[idx]
        
        if not self.use_rnn:
            input_data =self.input_selection(self,input_data,self.input_type)
            input_data = input_data.squeeze(0)
            output_data = output_data.squeeze(0)
        else:
            input_data =self.input_selection(self,input_data,self.input_type)
        return input_data, output_data

    @staticmethod
    def input_selection(self, input_data, input_type):
        if input_type == "hidden":
            input_data = input_data[:, -100:]  # Extract the last 100 columns
        elif input_type == "pro+exte":
            input_data = input_data[:, :341]  # Extract the first 341 columns
        elif input_type == "pro":
            input_data = input_data[:, :133]  # Extract the first 133 columns
        elif input_type == "all-e":
            first_part = input_data[:, :133]
            last_part = input_data[:, -100:]
            input_data = torch.cat((first_part, last_part), dim=-1)
        elif input_type == "all":
            pass  # Use all columns
        elif input_type == 'nosubstep':
            first_part = input_data[:, :60]
            last_part = input_data[:, 133:341]
            input_data = torch.cat((first_part, last_part), dim=-1)
        elif input_type == 'justsubstep':
            input_data = input_data[:, 60:341]
        elif input_type == 'nosubstep+cpg':
            first_part = input_data[:, :36]
            last_part = input_data[:, 133:341]
            input_data = torch.cat((first_part, last_part), dim=-1)
        else:
            raise ValueError("Invalid input_type. Choose 'hidden', 'pro+exte', or 'all'")
        return input_data
    
def save_model(model, model_name, use_rnn, input_width=None, use_weight=False,hyperparams=None):
    """Save the model to a folder."""
    architecture_tag = "RNN" if use_rnn else "MLP"
    rnn_mode = model.decoder.mode if hasattr(model,'decoder') and (hasattr(model.decoder, "mode") and use_rnn) else ""  # Add mode only for RNN
    weight_tag = "weighted" if use_weight else ""
    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')
    paradigm=model.decoder.paradigm if hasattr(model,'decoder') and hasattr(model.decoder, "paradigm") else ""
    if input_width:
        filename = f"model_{model_name}_{architecture_tag}_{rnn_mode}_InputWidth{input_width}_{weight_tag}_{paradigm}_{current_time}.pth".replace("__", "_")
    else:
        filename = f"model_{model_name}_{architecture_tag}_{rnn_mode}_{weight_tag}_{current_time}.pth".replace("__", "_")

    folder_name = "models"
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist
    path_to_save = os.path.join(folder_name, filename)
    torch.save(model.state_dict(), path_to_save)
    print(f"Model saved to {path_to_save}")

    if hyperparams:
        # Save hyperparameters to a .txt file
        hyperparams_filename = filename.replace(".pth", "_hyperparams.yaml")
        path_to_save_hyperparams = os.path.join(folder_name, hyperparams_filename)
        with open(path_to_save_hyperparams, 'w') as f:
            yaml.dump(hyperparams, f)
        print(f"Hyperparameters saved to {path_to_save_hyperparams}")
    return current_time

def get_dataloader_for_env_id(organized_data, env_id, batch_size, use_rnn=False, output_type="fric", input_type="hidden"):
    # Extract data for the specific env_id
    env_id_data = {env_id: organized_data[env_id]}
    
    # Create a custom dataset using the RNNDataset class
    dataset = RNNDataset(env_id_data, use_rnn=use_rnn, output_type=output_type, input_type=input_type)

    # Create a DataLoader for the custom dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader


def evaluate_on_data(model, organized_data, env_id, model_name, use_rnn, reset_hidden_each_epoch, output_type, input_type,overlapp=None,bin_accuracy_calculator=None,logger=None):
    """ Also need to add a option for the middle-only mode. 
     In middle_only mode:
     1. The first timesteps will still need padded inputs but the output will be the middle timestep.So the padded length is not same as the current one.
     2. The output will be a single timestep.
     3. no overlapping and running average of the overlapped predictions
      
        """
    # if model.decoder.middle_only or model.decoder.mode=="cnn" or model.decoder.mode=="transformer":
    #     overlapp=None
    # if overlapp is not None:
    #     # Get DataLoader for the specific env_id
    #     data_loader = get_dataloader_for_env_id(organized_data, env_id, 1, use_rnn=use_rnn, output_type=output_type, input_type=input_type)

    #     # Initialize accumulators for averaging
    #     true_values_accum = np.zeros((len(organized_data[env_id])*overlapp+organized_data[env_id][0]['input'].shape[0]-overlapp, 4))
    #     predictions_accum = np.zeros_like(true_values_accum)
    #     counts_accum = np.zeros_like(true_values_accum)

    #     model = model.to('cuda:0')

    #     # Initialize hidden state if using RNN
    #     hidden = model.decoder.model.init_hidden(1) if use_rnn else None
    #     all_predictions = []
    #     all_probs = []
    #     all_true_values = []
    #     with torch.no_grad():
    #         for i, (batch_inputs, batch_true_values) in enumerate(data_loader):
    #             batch_inputs = batch_inputs.to('cuda:0')
    #             batch_true_values=batch_true_values.to('cuda:0')
    #             if use_rnn:
    #                 predictions, hidden = model.decoder.get_unnormalized_recon(batch_inputs, hidden)
    #                 if model.decoder.paradigm=="ordinal":
    #                     probs=model.decoder.prob
    #                 elif model.decoder.prob:
    #                     predictions=predictions[0]
    #                 # Reset hidden state if required
    #                 if reset_hidden_each_epoch:
    #                     hidden = model.decoder.model.init_hidden(1)
    #             else:
    #                 predictions = model(batch_inputs)

    #             all_predictions.append(predictions)
    #             all_true_values.append(batch_true_values)
    #             if model.decoder.paradigm=="ordinal":
    #                 all_probs.append(probs)

    #             # Update accumulators
    #             start_index = i*overlapp 
    #             end_index = start_index + batch_true_values.shape[1]
    #             true_values_accum[start_index:end_index] += batch_true_values.squeeze(0).cpu().numpy()
    #             predictions_accum[start_index:end_index] += predictions.squeeze(0).cpu().numpy()
    #             counts_accum[start_index:end_index] += 1

    #       # Convert list of tensors to a single tensor
    #     all_predictions = torch.cat(all_predictions, dim=0)
    #     all_true_values = torch.cat(all_true_values, dim=0)
    #     if model.decoder.paradigm=="ordinal":
    #         all_probs = torch.cat(all_probs, dim=0)
    #     else:
    #         all_probs=None
    #     if bin_accuracy_calculator:  # Ensure that the BinAccuracy instance is provided
    #         accuracies,confusion_matrix = bin_accuracy_calculator.calculate_accuracy(all_predictions, all_true_values,all_probs)
    #         average_accuracy = sum(accuracies) / len(accuracies)
    #         print(f"Average accuracy: {average_accuracy*100:.2f}%")
    #         metrics={"Average accuracy":average_accuracy,
    #                  "Confusion Matrix":str(confusion_matrix.numpy()),
    #                  **{f"Accuracy for bin {idx}": acc for idx, acc in enumerate(accuracies)},
    #                  }
            
    #         logger.log_metrics(metrics)

    #     # Averaging
    #     true_values = true_values_accum / counts_accum
    #     predictions = predictions_accum / counts_accum

    #     # Plotting each dimension in separate subplots
    #     fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    #     for i in range(4):
    #         axs[i].plot(true_values[:, i], label=f"True {model_name} values", color="blue")
    #         axs[i].plot(predictions[:, i], label=f"Predicted {model_name} values", color="red", linestyle="--")
    #         axs[i].legend()
    #         axs[i].set_title(f"Dimension {i+1} - True vs Predicted {model_name} values for env_id {env_id}")
    #     plt.tight_layout()
    #     plt.show()
    #     plt.savefig("compare.png")
    # else:
    # also add ordinal mode and bin accuracy in the none overlapp mode just like the overlapp mode
    data_loader = get_dataloader_for_env_id(organized_data, env_id, 1, use_rnn=use_rnn, output_type=output_type, input_type=input_type)

    true_values_list, predictions_list = [], []
    model = model.to('cuda:0')

    # Initialize hidden state if using RNN
    hidden = model.decoder.model.init_hidden(1) if use_rnn else None
    all_predictions = []
    all_probs = []
    all_true_values = []
    with torch.no_grad():
        for batch_inputs, batch_true_values in data_loader:
            batch_inputs = batch_inputs.to('cuda:0')
            if use_rnn:
                predictions, hidden = model.decoder.get_unnormalized_recon(batch_inputs, hidden)
                if model.decoder.paradigm=="ordinal":
                    probs=model.decoder.prob
                elif model.decoder.prob:
                    predictions=predictions[0]
                # Reset hidden state if required
                if reset_hidden_each_epoch:
                    hidden = model.decoder.model.init_hidden(1)
            else:
                predictions = model(batch_inputs)
            if model.decoder.paradigm=="ordinal":
                all_probs.append(probs)
                # extract the middle timestep if middle_only mode is used
            if model.decoder.middle_only:
                batch_true_values=batch_true_values[:,model.decoder.exact_middle_step,:].unsqueeze(1)
            # if model.decoder.mode=="cnn" or model.decoder.mode=="transformer":
            batch_true_values=batch_true_values[:,-1,:].unsqueeze(1)
            predictions=predictions[:,-1,:].unsqueeze(1)
                
            all_predictions.append(predictions)
            all_true_values.append(batch_true_values)
            true_values_list.append(batch_true_values.squeeze(0).numpy())
            predictions_list.append(predictions.squeeze(0).cpu().numpy())

    # Convert list of arrays to single arrays
    true_values = np.concatenate(true_values_list, axis=0)
    predictions = np.concatenate(predictions_list, axis=0)

    #  Convert list of tensors to a single tensor
    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_values = torch.cat(all_true_values, dim=0)
    if model.decoder.paradigm=="ordinal":
        all_probs = torch.cat(all_probs, dim=0)
    else:
        all_probs=None
    if bin_accuracy_calculator:  # Ensure that the BinAccuracy instance is provided
        accuracies,confusion_matrix = bin_accuracy_calculator.calculate_accuracy(all_predictions, all_true_values,all_probs)
        average_accuracy = sum(accuracies) / len(accuracies)
        print(f"Average accuracy: {average_accuracy*100:.2f}%")
        metrics={"Average accuracy":average_accuracy,
                    "Confusion Matrix":str(confusion_matrix.numpy()),
                    **{f"Accuracy for bin {idx}": acc for idx, acc in enumerate(accuracies)},
                    }
        
        logger.log_metrics(metrics)

    # Plotting each dimension in separate subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    for i in range(4):
        axs[i].plot(true_values[:, i], label=f"True {model_name} values", color="blue")
        axs[i].plot(predictions[:, i], label=f"Predicted {model_name} values", color="red", linestyle="--")
        axs[i].legend()
        axs[i].set_title(f"Dimension {i+1} - True vs Predicted {model_name} values for env_id {env_id}")
    plt.tight_layout()
    plt.close()




def collect_errors(model, loader,use_rnn,reset_hidden_each_epoch):
    """ Here needs to add a option for the middle-only mode. """
    all_errors = []
    all_true_values = []
    test_last=True
    if use_rnn:
        # Initialize hidden state for RNN
        batch_size = loader.batch_size
        hidden = model.decoder.model.init_hidden(batch_size)
    for data in loader:
        latent, priv = data
        latent = latent.to('cuda:0')
        priv = priv.to('cuda:0')
        
        if reset_hidden_each_epoch and use_rnn:
            hidden = model.decoder.model.init_hidden(batch_size)
        with torch.no_grad():
            if use_rnn:
                recon,hidden = model.decoder.get_unnormalized_recon(latent,hidden)
            else:
                recon = model.decoder.get_unnormalized_recon(latent).cpu().numpy()
        if model.decoder.middle_only:
            priv=priv[:,model.decoder.exact_middle_step,:].unsqueeze(1)
        elif test_last:
            priv=priv[:,-1,:].unsqueeze(1)
            recon=recon[:,-1,:].unsqueeze(1)
        errors = np.abs(recon.cpu().numpy() - priv.cpu().numpy()).flatten()  # Flatten here
        all_errors.append(errors)
        all_true_values.append(priv.cpu().numpy().flatten())  # Flatten here

    return np.hstack(all_true_values), np.hstack(all_errors)  # Stack horizontally


def collect_errors_prob(model, loader,use_rnn,reset_hidden_each_epoch):
    """ Here needs to add a option for the middle-only mode. """
    all_errors = []
    all_true_values = []
    all_stds=[]
    test_last=True
    if use_rnn:
        # Initialize hidden state for RNN
        batch_size = loader.batch_size
        hidden = model.decoder.model.init_hidden(batch_size)
    for data in loader:
        latent, priv = data
        latent = latent.to('cuda:0')
        priv = priv.to('cuda:0')
            
        if reset_hidden_each_epoch and use_rnn:
            hidden = model.decoder.model.init_hidden(batch_size)
        with torch.no_grad():
            out,hidden = model.decoder.get_unnormalized_recon(latent,hidden)
            recon=out[0]
            std=out[1]
        if model.decoder.middle_only:
            priv=priv[:,model.decoder.exact_middle_step,:].unsqueeze(1)
        elif test_last:
            priv=priv[:,-1,:].unsqueeze(1)
            recon=recon[:,-1,:].unsqueeze(1)
            std=std[:,-1,:].unsqueeze(1)
        errors = np.abs(recon.cpu().numpy() - priv.cpu().numpy()).flatten()  # Flatten here
        all_errors.append(errors)
        all_true_values.append(priv.cpu().numpy().flatten())  # Flatten here
        all_stds.append(std.cpu().numpy().flatten())

    return np.hstack(all_true_values), np.hstack(all_errors),np.hstack(all_stds)  # Stack horizontally

def compute_mean_errors_per_bin(true_values, errors, interval,min_range,max_range):
    bins = np.arange(min_range, max_range + interval, interval)
    bin_means = []
    bin_stddevs = []
    errors_per_bin=[]
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:  # If it's the last bin
            indices = np.logical_and(true_values >= bins[i], true_values <= bins[i+1])
        else:
            indices = np.logical_and(true_values >= bins[i], true_values < bins[i+1])
        
        bin_error = errors[indices].mean()
        bin_stddev = errors[indices].std()  # Compute standard deviation
        bin_means.append(bin_error)
        bin_stddevs.append(bin_stddev)
        errors_per_bin.append(errors[indices].tolist())


    return bins[:-1], bin_means, bin_stddevs,errors_per_bin

def compute_mean_errors_per_bin_prob(true_values, errors,stds, interval,min_range,max_range):
    bins = np.arange(min_range, max_range + interval, interval)
    bin_means = []
    bin_stddevs = []
    bin_meanstds=[]
    bin_stdstds=[]
    errors_per_bin=[]
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:  # If it's the last bin
            indices = np.logical_and(true_values >= bins[i], true_values <= bins[i+1])
        else:
            indices = np.logical_and(true_values >= bins[i], true_values < bins[i+1])
        
        bin_error = errors[indices].mean()
        bin_stddev = errors[indices].std()  # Compute standard deviation
        bin_meanstd = stds[indices].mean()  
        bin_stdstd = stds[indices].std()  # Compute standard deviation
        bin_means.append(bin_error)
        bin_stddevs.append(bin_stddev)
        bin_meanstds.append(bin_meanstd)
        bin_stdstds.append(bin_stdstd)
        errors_per_bin.append(errors[indices].tolist())


    return bins[:-1], bin_means, bin_stddevs,bin_meanstds,bin_stdstds,errors_per_bin

def plot_error_histogram(true_values, errors, interval, min_range, max_range, model_name,use_rnn,input_width,timestamp,ratio,use_weight,rnn_mode,logger=None):
    bins, bin_means, bin_stddevs,bin_errors= compute_mean_errors_per_bin(true_values, errors, interval, min_range, max_range)
    # Define colors for each model type
    colors = {
        "fric": "blue",
        "stiff": "green"
    }
    bars = plt.bar(bins, bin_means, width=interval, align='edge',color=colors[model_name], yerr=bin_stddevs, capsize=5)

    # Annotating each bar with its height (value)
    for bar, stddev in zip(bars, bin_stddevs):
        yval = bar.get_height()
        rounded_stddev_str = "{:.2f}".format(stddev)
        plt.text(bar.get_x() + bar.get_width()/2-0.2*interval, yval, f"{round(yval, 2)} (+/- {rounded_stddev_str})", ha='center', va='bottom', rotation=90)
    # Compute the total error mean and standard deviation and add to the plot
    total_error_mean = errors.mean()
    total_error_std = errors.std()
    plt.annotate(f'Total Error Mean: {total_error_mean:.3f}\nTotal Error Std: {total_error_std:.3f}', xy=(0.05, 0.9), xycoords='axes fraction')
    
    plt.xlabel('True Value Range')
    plt.ylabel('Mean Prediction Error')
    plt.title(f'Mean Prediction Error for {model_name}')
    # Set y-axis range based on model_name
    if model_name == "stiff":
        plt.ylim([0, 10])
    elif model_name == "fric":
        plt.ylim([0, 1])
    # Save the plot
    current_time = timestamp
    rnn_or_mlp = "RNN" + "_" + rnn_mode if use_rnn else "MLP" # Modify this line to include the RNN mode
    weight_tag = "weighted" if use_weight else ""
    folder_name = "models/histograms"
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist
    if ratio is not None:
        filename = f"histogram_{model_name}_{rnn_or_mlp}_{input_width}_ratio{ratio}_{weight_tag}_{current_time}.png".replace("__", "_")
    else:
        filename = f"histogram_{model_name}_{rnn_or_mlp}_{input_width}_{weight_tag}_{current_time}.png".replace("__", "_")

    path_to_save = os.path.join(folder_name, filename)
    plt.savefig(path_to_save)
    if logger:
        # Log to Neptune using filename as the channel name
        channel_name = f"histograms/{filename}"
        logger.experiment[channel_name].append(File(path_to_save))
    print(f"Histogram saved to {path_to_save}")
    plt.clf()

def plot_error_histogram_prob(true_values, errors, stds,interval, min_range, max_range, model_name,use_rnn,input_width,timestamp,ratio,use_weight,rnn_mode,logger=None):
    bins, bin_means, bin_stddevs,bin_meanstds,bin_stdstds,bin_errors= compute_mean_errors_per_bin_prob(true_values, errors, stds,interval, min_range, max_range)
    # Define colors for each model type
    colors = {
        "fric": "blue",
        "stiff": "green"
    }
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the histogram on the first subplot
    bars = ax1.bar(bins, bin_means, width=interval, align='edge', color=colors[model_name], yerr=bin_stddevs, capsize=5)
    # Annotating each bar with its height (value)
    for bar, stddev in zip(bars, bin_stddevs):
        yval = bar.get_height()
        rounded_stddev_str = "{:.2f}".format(stddev)
        ax1.text(bar.get_x() + bar.get_width()/2-0.2*interval, yval, f"{round(yval, 2)} (+/- {rounded_stddev_str})", ha='center', va='bottom', rotation=90)
    # Compute the total error mean and standard deviation and add to the plot
    total_error_mean = errors.mean()
    total_error_std = errors.std()
    ax1.annotate(f'Total Error Mean: {total_error_mean:.3f}\nTotal Error Std: {total_error_std:.3f}', xy=(0.05, 0.9), xycoords='axes fraction')
    
    ax1.set_xlabel('True Value Range')
    ax1.set_ylabel('Mean Prediction Error')
    ax1.set_title(f'Mean Prediction Error for {model_name}')
    
    # Plot the mean variance with standard deviation lines on the second subplot
    ax2.errorbar(bins, bin_meanstds, yerr=bin_stdstds, fmt='o', ecolor='r', linestyle='None', marker='^', capsize=5)
    ax2.set_xlabel('Bins')
    ax2.set_ylabel('Mean Variance and Std Dev of Variance')
    ax2.set_title('Mean Variance and Std Dev per Bin')
    
    plt.tight_layout()
    
    # Set y-axis range based on model_name
    if model_name == "stiff":
        ax1.set_ylim([0, 10])
    elif model_name == "fric":
        ax1.set_ylim([0, 1])
    # Save the plot
    current_time = timestamp
    rnn_or_mlp = "RNN" + "_" + rnn_mode if use_rnn else "MLP" # Modify this line to include the RNN mode
    weight_tag = "weighted" if use_weight else ""
    folder_name = "models/histograms"
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist
    if ratio is not None:
        filename = f"histogram_{model_name}_{rnn_or_mlp}_{input_width}_ratio{ratio}_{weight_tag}_{current_time}.png".replace("__", "_")
    else:
        filename = f"histogram_{model_name}_{rnn_or_mlp}_{input_width}_{weight_tag}_{current_time}.png".replace("__", "_")

    path_to_save = os.path.join(folder_name, filename)
    plt.savefig(path_to_save)
    if logger:
        # Log to Neptune using filename as the channel name
        channel_name = f"histograms/{filename}"
        logger.experiment[channel_name].append(File(path_to_save))
    print(f"Histogram saved to {path_to_save}")
    plt.clf()

def check_normalization_changes(model, data_loader):
    """
    Check if normalization parameters of a model change during data feeding.
    
    Args:
    - model: The model to check.
    - data_loader: The DataLoader to use for feeding data.
    
    Returns:
    - A boolean indicating whether the normalization parameters changed.
    """
    # Store initial normalization parameters
    initial_mean = model.decoder.priv_normalizer._mean.clone().detach()
    

    for data in data_loader:
        latent, priv = data
        with torch.no_grad():
            # recon = model.decoder.get_unnormalized_recon(latent).cpu().numpy()
            model.decoder.priv_normalizer(priv)
    
    # Check for changes in normalization parameters
    final_mean = model.decoder.priv_normalizer._mean
    has_changed = not torch.allclose(initial_mean, final_mean,atol=1e-10,rtol=1e-10)
    
    return has_changed

def get_data_file_by_ratio(directory, ratio):
    pattern = f"*rigidRatio={ratio}*.h5"
    return get_latest_file_in_directory(directory, pattern)

def construct_search_pattern(use_rnn, rnn_mode, input_type, use_weight, model_name,paradigm=None):
    conditions = []

    # Check model type (RNN or MLP)
    if use_rnn:
        conditions.append("RNN")
        conditions.append(rnn_mode)
    else:
        conditions.append("MLP")

    # Check input width
    input_width_mapping = {
        "pro+exte": "InputWidth341",
        "hidden": "InputWidth100",
        "all": "InputWidth441"
    }
    conditions.append(input_width_mapping.get(input_type, ""))

    # Check if weighted
    if use_weight:
        conditions.append("weighted")
    if paradigm=="ordinal":
        conditions.append("ordinal")
    # Construct search pattern
    search_pattern = f"*{model_name}*"
    for condition in conditions:
        search_pattern += f"*{condition}*"
    search_pattern += ".pth"
    
    return search_pattern


