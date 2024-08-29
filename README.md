# Terrain physical parmaeter learning framework

This repository accompanies the paper "Identifying Terrain Physical Parameters from Vision - Towards Physical-Parameter-Aware Locomotion and Navigation". 

The repo mainly contains two parts: 
1. Physical decoder training
2. Online vision pipeline learning


**Maintainer**: Jiaqi Chen 
**Affiliation**: ETH Zurich  
**Contact**: chenjiaq@student.ethz.ch 

### Physical decoder training

Set your neptune api token , username and project name in the system file `.bashrc`:
```bash
export NEPTUNE_API_TOKEN="your_neptune_api_token"
export NEPTUNE_USERNAME="your_neptune_username"
export NEPTUNE_PROJECT="your_neptune_username/your_neptune_project_name"
```

Configure the decoder model settings and other parameters in `physical_decoder_training/training_utils/decoder_config.py`

The main training loop happens in `physical_decoder_training/train.py`

Be advised that the datasets are seperated for friction and stiffness prediction, and the training is also seperated. Change the `model_types` in the config for different decoders training. For detailed information, please refer to code.

You can use our collected dataset for training, download the `dataset` folder from [here](https://drive.google.com/drive/folders/1GiX66anCw4DuOGTlS3FzBez0hATTrJbL?usp=drive_link). And specify the train data and val data path in the config file.

Here is the training command:
```bash
python physical_decoder_training/train.py
```
