# Terrain physical parmaeter learning framework

This repository accompanies the paper "Identifying Terrain Physical Parameters from Vision - Towards Physical-Parameter-Aware Locomotion and Navigation". Check project website [here](https://bit.ly/3Xo5AA8)

The repo mainly contains three parts: 
1. Physical decoder training
2. Online vision pipeline learning
3. A seperate phyiscal decoder module that can directly be used for inference in vision pipeline.


**Maintainer**: Jiaqi Chen 
**Affiliation**: ETH Zurich  
**Contact**: chenjiaq@student.ethz.ch 

If you find this code useful in your research, please consider citing:
```
@INPROCEEDINGS{Chen24physical, 
        AUTHOR    = {Jiaqi Chen AND Jonas Frey AND Ruyi Zhou AND Takahiro Miki AND Georg Martius AND Marco Hutter}, 
        TITLE     = {Identifying Terrain Physical Parameters from Vision - Towards Physical-Parameter-Aware Locomotion and Navigation}, 
        BOOKTITLE = {accepted for IEEE Robotics and Automation Letters (RA-L)}, 
        YEAR      = {2024}
      }
```

## Physical decoder training (Folder: physical_decoder_training)

Install the required packages:
```bash
cd physical_decoder_training
pip install -r requirements.txt
```

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
## Online vision pipeline learning (Folder: BaseWVN)

Please check the Readme in `BaseWVN` folder for detailed instructions.

## Physical decoder module (Folder: Phy_Decoder)

Please check the Readme in `Phy_Decoder` folder for detailed instructions.