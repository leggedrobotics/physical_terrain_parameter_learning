# Physical Terrain Parameters Learning

[View the Framework Overview (PDF)](.docs/header_figure.pdf)

This repository accompanies the paper "Identifying Terrain Physical Parameters from Vision - Towards Physical-Parameter-Aware Locomotion and Navigation". Check project website [here](https://bit.ly/3Xo5AA8)

The repo mainly contains three parts: 
1. Physical decoder training
2. Self-supervised visual decoder learning
3. Stand-alone pre-trained physical decoder

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

## Stand-alone Pre-trained Physical Decoder (Folder: phy_decoder)

Please check the Readme in `phy_decoder` folder for detailed instructions.