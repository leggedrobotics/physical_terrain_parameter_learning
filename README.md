# Physical Terrain Parameters Learning

[View the Framework Overview (PDF)](.docs/header_figure.pdf)

This repository accompanies the paper "Identifying Terrain Physical Parameters from Vision - Towards Physical-Parameter-Aware Locomotion and Navigation". Check project website [here](https://bit.ly/3Xo5AA8)

The repo mainly contains three parts: 
1. Stand-alone pre-trained physical decoder
2. Physical decoder training
3. Self-supervised visual decoder learning


**Maintainer**: Jiaqi Chen 
**Affiliation**: ETH Zurich  
**Contact**: chenjiaq@student.ethz.ch 

If this code supports your research, please consider citing the following work. We also welcome feedback or collaboration opportunities:
```
@INPROCEEDINGS{Chen24physical, 
        AUTHOR    = {Jiaqi Chen AND Jonas Frey AND Ruyi Zhou AND Takahiro Miki AND Georg Martius AND Marco Hutter}, 
        TITLE     = {Identifying Terrain Physical Parameters from Vision - Towards Physical-Parameter-Aware Locomotion and Navigation}, 
        BOOKTITLE = {accepted for IEEE Robotics and Automation Letters (RA-L)}, 
        YEAR      = {2024}
      }
```

## Stand-alone Pre-trained Physical Decoder (Folder: phy_decoder)
You can try out our pre-trained physical decoder as follows:

### Installing
First, clone this repository to your local machine and install the dependencies.
```shell
cd phy_decoder/

# Install the dependencies, the listed versions are for compatibility with the visual decoder environment
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Explanation & Usage
This two decoders use sequence data as input and output a physical parameters sequence (friction or stiffness). 
The main architecture is GRU+Self-Attention with a parallel structure.
The model_pth is automatically loaded from the package folder. 
Here we showcase how to use the decoders in legged_gym. 
If you want to deploy it on real robots with ros, please check the `base_wvn` folder for ros usage.

**Attention**: Only env_num=1 is tested now. If you want to use env_num>1, you may need to modify the code a little bit.

```python
from phy_decoder import initialize_models, prepare_padded_input, RNNInputBuffer
# Initializing
step = 0
env_num=1

fric_predictor, stiff_predictor, predictor_cfg = initialize_models()

fric_hidden = fric_predictor.init_hidden(env_num)
stiff_hidden = stiff_predictor.init_hidden(env_num)

input_buffers = {0: RNNInputBuffer()}

# Main loop
while True:
    step += 1 # Pre-increment to 1 for first step not 0
    input_data=obs[:,:341] # Input is batch_size (env_num)x341dim(prop. + ext.)
    padded_inputs = prepare_padded_input(input_data, input_buffers, step, env_num)
    padded_input = torch.stack(padded_inputs, dim=0) # reshape to env_num x num_timesteps (sequence_len) x 341
    
    if predictor_cfg['reset_hidden_each_epoch']:
        fric_hidden = fric_predictor.init_hidden(env_num)
        stiff_hidden = stiff_predictor.init_hidden(env_num)
    
    with torch.no_grad():
        # Predict using the friction predictor
        fric_pred, fric_hidden = fric_predictor.get_unnormalized_recon(padded_input, fric_hidden)           
        # Predict using the stiffness predictor
        stiff_pred, stiff_hidden = stiff_predictor.get_unnormalized_recon(padded_input, stiff_hidden)

        # fric_pred, stiff_pred are dimension: env_num x num_timesteps x 4 (number of feet)

    input_buffers[0].add(input_data[0].unsqueeze(0))

# You may use the last step of output sequence as the prediction
fric_pred=fric_pred[:,-1,:]
stiff_pred=stiff_pred[:,-1,:]
```

## Physical Decoder Training (Folder: physical_decoder_training)

Install the required packages:
```bash
cd physical_decoder_training
pip install -r requirements.txt
```

Set your Neptune API token, username and project name in the system file `.bashrc`:
```bash
export NEPTUNE_API_TOKEN="your_neptune_api_token"
export NEPTUNE_USERNAME="your_neptune_username"
export NEPTUNE_PROJECT="your_neptune_username/your_neptune_project_name"
```

Configure the decoder model settings and other parameters in `physical_decoder_training/training_utils/decoder_config.py`

The main training loop happens in `physical_decoder_training/train.py`

Be advised that the datasets are seperated for friction and stiffness prediction, and the training is also seperated. Change the `model_types` in the config for different decoders training. For detailed information, please refer to code.

You may use our pre-collected dataset for training. Download the `dataset` folder from [this link](https://drive.google.com/drive/folders/1GiX66anCw4DuOGTlS3FzBez0hATTrJbL?usp=drive_link). Specify the paths for training and validation data in the configuration file.


Here is the training command:
```bash
python physical_decoder_training/train.py
```
## Self-supervised Visual Decoder Learning (Folder: base_wvn)

Please check the Readme in `base_wvn` folder for detailed instructions.
