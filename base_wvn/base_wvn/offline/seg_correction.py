#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import os
import torchvision.transforms.functional as F
import torch
from segments import SegmentsDataset
import re
from base_wvn import WVN_ROOT_DIR
from base_wvn.config.wvn_cfg import ParamCollection


# ----Step2. download and automatically replace the flaw masks (uncomment the following lines)----
# use your own API key
# api_key = 'xxx'
# client = SegmentsClient(api_key)
# dataset_identifier = "xxx"
# name = "v0.1"
# client.add_release(dataset_identifier, name)
def save_imperfect_images(image_batch, mask_batch, imperfect_indexes, save_folder):
    """
    Save images with imperfect masks to a folder.

    :param image_batch: Batch of images (B, C, H, W)
    :param mask_batch: Corresponding batch of masks (B, 1, H, W)
    :param imperfect_indexes: List of indexes of imperfect masks
    :param save_folder: Folder to save the images
    """
    os.makedirs(save_folder, exist_ok=True)

    for idx in imperfect_indexes:
        image = image_batch[idx]
        # Convert from tensor to PIL image
        image_pil = F.to_pil_image(image)
        image_pil.save(os.path.join(save_folder, f"imperfect_image_{idx}.png"))


def download_and_replace_masks(mask_batch, corrected_mask_paths):
    """
    Replace the original imperfect masks with the corrected ones.

    :param mask_batch: The original batch of masks (B, 1, H, W)
    :param corrected_mask_paths: List of file paths to the corrected mask images
    :param imperfect_indexes: List of indexes of the masks to replace
    """

    release = client.get_release(dataset_identifier, name)
    dataset = SegmentsDataset(release, labelset="ground-truth")
    for sample in dataset:
        # Extract index from sample name (assuming format 'imperfect_image_INDEX.png')
        match = re.search(r"imperfect_image_(\d+)\.png", sample["name"])
        if not match:
            print(f"Could not extract index from sample name: {sample['name']}")
            continue
        idx = int(match.group(1))
        # Show the instance segmentation label
        corrected_mask_tensor = F.to_tensor(sample["segmentation_bitmap"]).type(
            torch.bool
        )  # Add channel dimension
        # Replace the corresponding mask in mask_batch
        if idx < len(mask_batch):
            mask_batch[idx] = corrected_mask_tensor
        else:
            print(f"Index {idx} is out of bounds for the mask batch.")
    output_path = os.path.join(corrected_mask_paths, "Ground_comp.pt")
    torch.save(mask_batch, output_path)
    print(f"Saved corrected masks to {output_path}")


def seg_union_correction(
    mask_batch, additional_batch, union_indices, corrected_mask_paths
):
    # complement the mask_batch with additional_batch
    for ind in union_indices:
        mask_batch[ind] = torch.logical_or(mask_batch[ind], additional_batch[ind])
    output_path = os.path.join(corrected_mask_paths, "gt_masks_SAM_union.pt")
    torch.save(mask_batch, output_path)
    print(f"Saved corrected masks (union) to {output_path}")


def seg_subtract_correction(
    mask_batch, additional_batch, subtract_indices, corrected_mask_paths
):
    # subtract the area of additional_batch from mask_batch
    for ind in subtract_indices:
        mask_batch[ind] = torch.logical_and(mask_batch[ind], ~additional_batch[ind])
    output_path = os.path.join(corrected_mask_paths, "gt_masks_SAM_subtract.pt")
    torch.save(mask_batch, output_path)
    print(f"Saved corrected masks (subtract) to {output_path}")


if __name__ == "__main__":
    param = ParamCollection()
    output_dir = os.path.join(
        WVN_ROOT_DIR, param.offline.data_folder, "train", param.offline.env
    )

    gt_masks_path = os.path.join(output_dir, "gt_masks_SAM.pt")
    img_path = os.path.join(output_dir, "mask_img.pt")

    gt_masks = torch.load(gt_masks_path)
    mask_imgs = torch.load(img_path)
    # ----Step0. list the indexes of imperfect masks ----
    imperfect_indexes = torch.arange(len(gt_masks))
    # imperfect_indexes = [78,79]

    # ----Step1. save for manual correction (uncomment the following line)----
    # save_imperfect_images(mask_imgs, gt_masks, imperfect_indexes, os.path.join(output_dir, 'imperfect_images'))
    # ----Step2. download and automatically replace the flaw masks (uncomment the following lines)----
    # download_and_replace_masks(gt_masks,os.path.join(output_dir, 'imperfect_images'))

    # ----Step2 (optional). combine the mask with a complementary mask (uncomment the following lines)----
    # move the 'gt_masks_SAM.pt' from imperfect image folder to the train/your_env folder, load the complementary mask
    # comp_masks=torch.load(os.path.join(output_dir, 'white_masks.pt'))
    # seg_union_correction(gt_masks,comp_masks,[4],os.path.join(output_dir, 'imperfect_images'))

    # ----Step2 (optional). subtract the mask with a complementary mask (uncomment the following lines)----
    comp_masks = torch.load(os.path.join(output_dir, "white_masks.pt"))
    seg_subtract_correction(
        gt_masks,
        comp_masks,
        [0, 1, 2, 3, 4],
        os.path.join(output_dir, "imperfect_images"),
    )
