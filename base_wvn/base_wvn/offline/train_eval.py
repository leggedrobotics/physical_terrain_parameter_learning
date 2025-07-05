#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
import os

from base_wvn import WVN_ROOT_DIR
from base_wvn.offline.helper import (
    find_latest_checkpoint,
    load_all_test_images,
)
from base_wvn.offline.dataset_cls import get_train_eval_dataloaders
from base_wvn.offline.lightening_module import DecoderLightning, MaskedPredErrorComputer
from base_wvn.offline.pred_video_producer import RosBagPredictionVideoProducer
from base_wvn.utils import (
    plot_pred_w_overlay,
)
from base_wvn.model import get_model
from base_wvn.config.wvn_cfg import ParamCollection
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import Trainer

from typing import Dict, Optional


def train(param: ParamCollection) -> None:
    """Train and evaluate the model."""
    ckpt_parent_folder = os.path.join(WVN_ROOT_DIR, param.offline.ckpt_parent_folder)

    m = get_model(param.model).to(param.run.device)
    model = DecoderLightning(m, param)
    if param.offline.reload_model:
        if not param.offline.use_online_ckpt:
            checkpoint_path = find_latest_checkpoint(ckpt_parent_folder)
        else:
            checkpoint_path = os.path.join(
                WVN_ROOT_DIR, param.general.resume_training_path
            )
        if checkpoint_path:
            print(f"Latest checkpoint path: {checkpoint_path}")
        else:
            print("No checkpoint found.")
            return None
        checkpoint = torch.load(checkpoint_path)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.loss_fn.load_state_dict(checkpoint["phy_loss_state_dict"])
        model.step = checkpoint["step"]
        model.time = (
            checkpoint["time"] if not param.offline.use_online_ckpt else "online"
        )
        model.val_loss = checkpoint["loss"]
        model.model.train()
        print("Reloaded model from {}".format(checkpoint_path))
    # Initialize the Neptune logger
    neptune_logger = NeptuneLogger(
        api_key=os.getenv("NEPTUNE_API_TOKEN", "default_api_token_if_not_set"),
        project=os.getenv("NEPTUNE_PROJECT", "default_project_if_not_set"),
        tags=["offline", param.offline.env, param.general.name],
    )
    train_loader, val_loader = get_train_eval_dataloaders(param=param, model=model)
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=neptune_logger,
        max_epochs=param.offline.max_epochs,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)
    torch.save(
        {
            "time": model.time,
            "step": model.step,
            "model_state_dict": model.model.state_dict(),
            "phy_loss_state_dict": model.loss_fn.state_dict(),
            "loss": model.val_loss.item(),
        },
        os.path.join(ckpt_parent_folder, model.time, "last_checkpoint.pt"),
    )
    torch.cuda.empty_cache()
    return None


def evaluate(param: ParamCollection) -> Optional[Dict[str, float]]:
    ckpt_parent_folder = os.path.join(WVN_ROOT_DIR, param.offline.ckpt_parent_folder)

    m = get_model(param.model).to(param.run.device)
    model = DecoderLightning(m, param)
    if not param.offline.use_online_ckpt:
        checkpoint_path = find_latest_checkpoint(ckpt_parent_folder)
    else:
        checkpoint_path = os.path.join(WVN_ROOT_DIR, param.general.resume_training_path)
    if checkpoint_path:
        print(f"Latest checkpoint path: {checkpoint_path}")
    else:
        print("No checkpoint found.")
        return None
    checkpoint = torch.load(checkpoint_path)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.loss_fn.load_state_dict(checkpoint["phy_loss_state_dict"])
    model.step = checkpoint["step"]
    model.time = checkpoint["time"] if not param.offline.use_online_ckpt else "online"
    model.val_loss = checkpoint["loss"]
    model.model.eval()
    """ 
    plot phy_masks (two channels) on a set of test images
    """
    if param.offline.test_on_images:
        test_imgs = load_all_test_images(
            os.path.join(param.offline.data_folder, "val", param.offline.env)
        )
        for name, img in test_imgs.items():
            trans_img, compressed_feat = model.feat_extractor.extract(img)
            res = model.conf_mask_generator.get_confidence_masked_prediction_from_img(
                trans_img=trans_img,
                compressed_feats=compressed_feat,
                model=model.model,
                loss_fn=model.loss_fn,
            )
            plot_pred_w_overlay(
                data=res,
                time=model.time,
                step=-1,
                image_name=name,
                param=param,
                save_local=True,
            )

    """ 
    test on the recorded main nodes
    """
    if param.offline.test_on_nodes:
        # READ nodes datafile and gt_masks datafile
        err_computer = MaskedPredErrorComputer(param)
        return err_computer.compte_and_log(model, log=False)

    """ 
    output prediction video from rosbag
    """
    if param.offline.test_video:
        RosBagPredictionVideoProducer(
            model=model,
            param=param,
        ).produce()

    return None


if __name__ == "__main__":
    param = ParamCollection()
    if param.offline.mode == "train":
        train(param)
    evaluate(param)
