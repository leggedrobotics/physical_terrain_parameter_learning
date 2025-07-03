#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import datetime
from .. import WVN_ROOT_DIR
from ..graph_manager import MainNode
from ..utils import (
    FeatureExtractor,
    MaskedPredictionData,
    plot_pred_w_overlay,
    concat_feat_dict,
    plot_overlay_image,
    plot_image,
    plot_images_side_by_side,
    plot_images_in_grid,
)
from ..model import VD_dataset
from ..config.wvn_cfg import ParamCollection
from typing import List, Dict, Tuple, Optional
import pytorch_lightning as pl
from segment_anything import SamPredictor, sam_model_registry

import seaborn as sns

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 7


def load_data(file: str) -> torch.Tensor:
    """Load data from the data folder."""
    path = os.path.join(WVN_ROOT_DIR, file)
    data = torch.load(path)
    return data


def load_one_test_image(path: str) -> torch.Tensor:
    """return img in shape (B,C,H,W)"""
    image_path = path
    if path.lower().endswith(".pt"):
        is_pt_file = True
    else:
        is_pt_file = False
    if not is_pt_file:
        np_img = cv2.imread(image_path)
        img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        img = img.permute(2, 0, 1)
        img = (img.type(torch.float32) / 255)[None]
    else:
        imgs = torch.load(image_path)
        time, img = next(iter(imgs.items()))
    return img


def load_all_test_images(folder: str) -> Dict[str, torch.Tensor]:
    """Load all images from a folder and return them"""
    if "manager" in folder:
        is_pt_file = True
    else:
        is_pt_file = False
    if not is_pt_file:
        images = {}

        for file in os.listdir(os.path.join(WVN_ROOT_DIR, folder)):
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):  # Check for image files
                image_path = os.path.join(WVN_ROOT_DIR, folder, file)
                np_img = cv2.imread(image_path)
                img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                img = img.permute(2, 0, 1)
                img = (img.type(torch.float32) / 255)[None]
                images[file] = img
    else:
        images = {}
        for file in os.listdir(os.path.join(WVN_ROOT_DIR, folder)):
            if file.lower().endswith(".pt") and file.lower().startswith("image"):
                image_path = os.path.join(WVN_ROOT_DIR, folder, file)
                imgs = torch.load(image_path)
                for time, img in imgs.items():
                    images[time] = img

                break
    return images


def find_latest_checkpoint(parent_dir: str) -> Optional[str]:
    # List all folders in the parent directory
    valid_folders = []

    # Iterate over each folder in the parent directory
    for folder in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, folder)):
            try:
                # Try parsing the folder name as a datetime object
                datetime.datetime.strptime(folder, "%Y-%m-%d_%H-%M-%S")
                # If successful, add it to the valid_folders list
                valid_folders.append(folder)
            except ValueError:
                # Skip folders that don't follow the datetime naming convention
                continue

    # Sort the valid folders based on datetime in their names
    sorted_folders = sorted(
        valid_folders,
        key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"),
        reverse=True,
    )

    # Select the latest folder
    latest_folder = sorted_folders[0] if sorted_folders else None

    if latest_folder:
        latest_folder_path = os.path.join(parent_dir, latest_folder)

        # Search for the 'last_checkpoint.pt' file in this folder
        last_checkpoint_path = os.path.join(latest_folder_path, "last_checkpoint.pt")

        if os.path.exists(last_checkpoint_path):
            return last_checkpoint_path
        else:
            print("Last checkpoint not found in the latest folder.")
            return None
    else:
        print("No folders found in the parent directory.")
        return None


def sample_furthest_points(true_coords, num_points_to_sample, given_point=None):
    """
    only support B=1 operation
    """

    B, num, _ = true_coords.shape
    if B != 1:
        raise ValueError("Only B=1 operation is supported.")

    if given_point is None:
        copy = true_coords.clone().type(torch.float32)
        # Calculate all pairwise distances
        pairwise_distances = torch.cdist(copy[0], copy[0])

        if num_points_to_sample == 2:
            # For two points, simply find the pair with the maximum distance
            max_dist, indices = torch.max(pairwise_distances, dim=1)
            furthest_pair = indices[max_dist.argmax()], max_dist.argmax()
            return true_coords[0][list(furthest_pair), :].unsqueeze(0)
    else:
        if len(given_point.shape) != 2:
            raise ValueError("given_point should be in shape (1,2)")
        # Calculate distances between the given point and all points in true_coords
        distances = torch.cdist(
            given_point.type(torch.float32), true_coords[0].type(torch.float32)
        )

        # Find the index of the furthest point
        max_dist_index = torch.argmax(distances)

        # Return the furthest point
        return true_coords[0][max_dist_index].unsqueeze(0).unsqueeze(0)


def SAM_label_mask_generate(
    param: ParamCollection, nodes: List[MainNode]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Using segment anything model to generate gt label mask
    Return: gt_masks in shape (B=node_num,1,H,W) and node images in shape (B=node_num,C,H,W)

    """
    gt_masks = []
    cor_images = []
    sam = sam_model_registry[param.offline.SAM_type](checkpoint=param.offline.SAM_ckpt)
    sam.to(param.run.device)
    predictor = SamPredictor(sam)
    for node in nodes:
        img = node.image.to(param.run.device)
        reproj_mask = node.supervision_signal_valid[0]
        # Find the indices where reproj_mask is True
        true_indices = torch.where(reproj_mask)
        true_coords = torch.stack((true_indices[1], true_indices[0]), dim=1).unsqueeze(
            0
        )  # (x, y) format
        B, num, _ = true_coords.shape

        # combine the furtherst points with the randomly sampled points
        far_coords = sample_furthest_points(true_coords, 2).to(param.run.device)
        num_points_to_sample = min(10, num)
        # sampled_true_coords = torch.zeros(B, num_points_to_sample, 2)
        rand_indices = torch.randperm(num)[:num_points_to_sample]
        sampled_true_coords = true_coords[:, rand_indices, :].to(param.run.device)
        pairs = []
        for i in range(num_points_to_sample):
            pair = sample_furthest_points(
                true_coords, 2, given_point=sampled_true_coords[:, i, :]
            ).to(param.run.device)
            pairs.append(pair)
        pairs = torch.cat(pairs, dim=1)
        true_coords = sampled_true_coords.to(param.run.device)
        true_coords = torch.cat((true_coords, far_coords, pairs), dim=1)

        true_coords_resized = predictor.transform.apply_coords_torch(
            true_coords, img.shape[-2:]
        )
        points_labels = torch.ones(
            (true_coords_resized.shape[0], true_coords_resized.shape[1]),
            dtype=torch.int64,
        ).to(param.run.device)

        # need to image--> H,W,C uint8 format
        input_img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(
            np.uint8
        )
        H, W, C = input_img.shape
        predictor.set_image(input_img)
        cor_images.append(img)
        gt_mask_pts = torch.zeros_like(reproj_mask.unsqueeze(0).unsqueeze(0)).type(
            torch.int
        )
        for i in range(true_coords_resized.shape[1]):
            current_point_coords = true_coords_resized[:, i, :].unsqueeze(1)
            current_label = points_labels[:, i].unsqueeze(0)
            masks, scores, _ = predictor.predict_torch(
                point_coords=current_point_coords,
                point_labels=current_label,
                multimask_output=True,
            )

            for j in range(masks.shape[1]):
                _, max_score_indices = torch.max(scores, dim=1)
                gt_mask = masks[:, max_score_indices, :, :]
                mask_ratio_s = gt_mask.sum() / (H * W)
                if mask_ratio_s > 0.70:
                    scores[:, max_score_indices] = 0
                else:
                    break
            gt_mask_pts += gt_mask.type(torch.int)
            gt_mask_pts[gt_mask_pts > 0] = 1
        gt_mask = gt_mask_pts.type(torch.bool)
        gt_masks.append(gt_mask)
        torch.cuda.empty_cache()

        # mask_ratio=gt_mask.sum()/(H*W)
        # plt.figure(figsize=(10,10))
        # plt.imshow(input_img)
        # print("Mask region ratio: ",mask_ratio.item())
        # show_mask(gt_mask.squeeze(0), plt.gca())
        # show_points(true_coords.squeeze(0), points_labels.squeeze(0), plt.gca())
        # plt.axis('off')
        # plt.show()
    return torch.cat(gt_masks, dim=0), torch.cat(cor_images, dim=0)


def create_dataset_from_nodes(
    param: ParamCollection,
    nodes: List[MainNode],
    feat_extractor: FeatureExtractor,
) -> VD_dataset:
    # use new_features if we want to test new feat_extractor
    # output dataset
    for node in nodes:
        img = node.image.to(param.run.device)
        if param.feat.feature_type != node.feature_type:
            trans_img, compressed_feats = feat_extractor.extract(img)
            feat_input, H, W = concat_feat_dict(compressed_feats)
            feat_input = feat_input.reshape(1, H, W, -1)
            feat_input = feat_input.permute(0, 3, 1, 2)
            new_features = {list(compressed_feats.keys())[0]: feat_input}
            node.features = new_features
    batch_list = [mnode.query_valid_batch() for i, mnode in enumerate(nodes)]
    dataset = VD_dataset(batch_list, random_num=1e10)
    return dataset


def compute_pred_phy_loss(
    ori_phy_mask: torch.Tensor,
    pred_phy_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    To calculate the mean error of predicted physical params value in confident area of a image
    phy_mask (2,H,W) H,W is the size of resized img
    """
    # check dim of phy_masks first
    if (
        ori_phy_mask.shape[-2] != pred_phy_mask.shape[-2]
        or ori_phy_mask.shape[-1] != pred_phy_mask.shape[-1]
    ):
        raise ValueError("ori_phy_mask and pred_phy_mask should have the same shape!")
    compare_regions = ~torch.isnan(ori_phy_mask)
    regions_in_pred = pred_phy_mask * compare_regions
    regions_in_pred = torch.where(regions_in_pred == 0, torch.nan, regions_in_pred)
    delta = torch.abs(regions_in_pred - ori_phy_mask)
    delta_mask = ~torch.isnan(delta[0])

    fric_dvalues = delta[0][delta_mask]
    fric_mean_deviation = torch.mean(fric_dvalues)
    fric_std_deviation = torch.std(fric_dvalues)

    stiff_dvalues = delta[1][delta_mask]
    stiff_mean_deviation = torch.mean(stiff_dvalues)
    stiff_std_deviation = torch.std(stiff_dvalues)

    return {
        "fric_mean_deviat": fric_mean_deviation,
        "fric_std_deviation": fric_std_deviation,
        "fric_loss_raw": fric_dvalues,
        "stiff_mean_deviat": stiff_mean_deviation,
        "stiff_std_deviation": stiff_std_deviation,
        "stiff_loss_raw": stiff_dvalues,
    }


def nodes_mask_generation_and_phy_pred_error_computation(
    param: ParamCollection,
    nodes: List[MainNode],
    model: pl.LightningModule,
    gt_masks: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Here we use the model to generate confidence mask for each node
    Also the loss_recon is used to compute uncertainty histograms
    And the loss of physical param prediction (mean+std) tested on all recorded nodes
    mean: mean across all pixels of all tested nodes
    std: std across all pixels of all tested nodes

    """
    conf_masks = []
    reproj_masks = []
    losses = []
    ori_imgs = []
    all_fric_losses = []
    all_stiff_losses = []
    folder_path = os.path.join(
        WVN_ROOT_DIR, param.offline.ckpt_parent_folder, model.time
    )
    for i, node in enumerate(nodes):
        img = node.image.to(param.run.device)
        reproj_mask = (
            node.supervision_signal_valid[0]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(param.run.device)
        )
        reproj_masks.append(reproj_mask)
        ori_imgs.append(img)
        trans_img, compressed_feat = model.feat_extractor.extract(img)
        res: MaskedPredictionData = (
            model.conf_mask_generator.get_confidence_masked_prediction_from_img(
                trans_img=trans_img,
                compressed_feats=compressed_feat,
                model=model.model,
                loss_fn=model.loss_fn,
            )
        )
        if param.offline.plot_nodes:
            plot_pred_w_overlay(
                data=res,
                time=model.time,
                step=i,
                image_name="node" + str(node.timestamp),
                param=param,
                save_local=True,
            )
        conf_mask = res.conf_mask
        loss_reco = res.loss_reco
        pred_phy_mask = res.masked_output_phy
        ori_phy_mask = node._supervision_mask.to(param.run.device)
        # calculate phy loss, w.r.t. the foothold labels (self-supervision)
        phy_loss_dict = compute_pred_phy_loss(
            pred_phy_mask=pred_phy_mask, ori_phy_mask=ori_phy_mask
        )
        all_fric_losses.append(phy_loss_dict["fric_loss_raw"])
        all_stiff_losses.append(phy_loss_dict["stiff_loss_raw"])
        if param.offline.plot_hist:
            calculate_and_save_uncertainty_histogram(
                loss_reco,
                conf_mask,
                reproj_mask,
                os.path.join(
                    folder_path, "hist", f"node_{i}_uncertainty_histogram.png"
                ),
            )
            calculate_and_save_uncertainty_histogram(
                loss_reco,
                gt_masks[i, :, :, :].unsqueeze(0),
                reproj_mask,
                os.path.join(
                    folder_path, "hist/gt", f"node_{i}_gt_uncertainty_histogram.png"
                ),
            )
        conf_masks.append(conf_mask)
        losses.append(loss_reco)
        torch.cuda.empty_cache()

    all_reproj_masks = torch.cat(reproj_masks, dim=0)
    torch.cuda.empty_cache()
    all_losses = torch.cat(losses, dim=0)
    all_conf_masks = torch.cat(conf_masks, dim=0)
    if param.offline.plot_hist:
        calculate_and_save_uncertainty_histogram(
            all_losses,
            all_conf_masks,
            all_reproj_masks,
            os.path.join(folder_path, "hist", "all_uncertainty_histogram.png"),
        )

    all_fric_losses = torch.cat(all_fric_losses)
    all_stiff_losses = torch.cat(all_stiff_losses)
    fric_mean = torch.mean(all_fric_losses)
    fric_std = torch.std(all_fric_losses)
    stiff_mean = torch.mean(all_stiff_losses)
    stiff_std = torch.std(all_stiff_losses)

    # Save the Results to a Text File
    file_path = os.path.join(folder_path, "overall_pred_loss_statistics.txt")
    with open(file_path, "a") as file:
        file.write(f"{param.general.name}\n")
        file.write(
            f"Overall Friction Error Mean: {round(fric_mean.item(),3)}, Standard Deviation: {round(fric_std.item(),3)}\n"
        )
        file.write(
            f"Overall Stiffness Error Mean: {round(stiff_mean.item(),3)}, Standard Deviation: {round(stiff_std.item(),3)}\n"
        )

    print("Overall loss statistics saved to overall_loss_statistics.txt")

    torch.cuda.empty_cache()
    return {
        "all_conf_masks": all_conf_masks,
        "ori_imgs": torch.cat(ori_imgs, dim=0),
        "loss_fric_mean+std": (fric_mean, fric_std),
        "loss_stiff_mean+std": (stiff_mean, stiff_std),
    }


def calculate_and_save_uncertainty_histogram(
    all_losses: torch.Tensor,
    all_conf_masks: torch.Tensor,
    all_reproj_masks: torch.Tensor = None,
    save_path: str = None,
    colormap: str = "coolwarm",
) -> np.ndarray:
    """
    Calculate a histogram of the uncertainty values (losses) from reproj_masks(should be very certain)
    and from conf_masks
    """

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # bin_size = 0.05
    num_bins = 40
    # Flatten the tensors to get the loss values of all pixels
    flattened_losses = all_losses.flatten().detach().cpu().numpy()
    flattened_conf_masks = all_conf_masks.flatten().detach().cpu().numpy().astype(bool)
    if all_reproj_masks is not None:
        flattened_reproj_masks = (
            all_reproj_masks.flatten().detach().cpu().numpy().astype(bool)
        )

    # Use boolean indexing to filter losses
    conf_mask_losses = flattened_losses[flattened_conf_masks]
    if all_reproj_masks is not None:
        reproj_mask_losses = flattened_losses[flattened_reproj_masks]

    unconf_mask_losses = flattened_losses[~flattened_conf_masks]

    if all_reproj_masks is not None:
        # Define the bin edges based on the minimum and maximum loss values and the desired bin size
        min_loss = min(
            flattened_losses.min(), conf_mask_losses.min(), reproj_mask_losses.min()
        )
        max_loss = max(
            flattened_losses.max(), conf_mask_losses.max(), reproj_mask_losses.max()
        )
    else:
        min_loss = min(
            flattened_losses.min(),
            conf_mask_losses.min()
            if conf_mask_losses.shape[0] > 0
            else flattened_losses.min(),
        )
        max_loss = max(
            flattened_losses.max(),
            conf_mask_losses.max()
            if conf_mask_losses.shape[0] > 0
            else flattened_losses.max(),
        )
    bins = np.linspace(min_loss, max_loss, num_bins)
    # Plot the histogram
    plt.figure(figsize=(2.5, 1.8))

    # cmap = plt.get_cmap('coolwarm')
    cmap = sns.color_palette(colormap, as_cmap=True)

    # Plot histograms. Since you want specific colors for conf and unconf masks, use the calculated colors
    # For the conf mask losses
    conf_n, conf_bins, conf_patches = plt.hist(
        conf_mask_losses,
        bins,
        color="orange",
        label="Confidence Mask",
        alpha=1.0,
        density=False,
    )

    # For the unconf mask losses
    unconf_n, unconf_bins, unconf_patches = plt.hist(
        unconf_mask_losses,
        bins,
        color="red",
        label="Un-confident Mask",
        alpha=1.0,
        density=False,
    )

    # calculate how many bins is non-zero in conf and unconf, use the number for splitting the colormap
    conf_bin_num = 0
    for c_i, conf_nn in enumerate(conf_n):
        if conf_nn > 0:
            conf_bin_num += 1
        else:
            if c_i < len(conf_n) - 1 and conf_n[c_i + 1] > 0:
                conf_bin_num += 1
    unconf_bin_num = num_bins - conf_bin_num
    # Split the colormap into two parts
    conf_color = cmap(np.linspace(1, 0.7, conf_bin_num))
    unconf_color = cmap(np.linspace(0.3, 0, unconf_bin_num))

    # Set the colors for the patches to make a gradient effect
    for conf_nn, conf_patch, color in zip(
        conf_n[:conf_bin_num], conf_patches[:conf_bin_num], conf_color
    ):
        conf_patch.set_facecolor(color)

    for unconf_nn, unconf_patch, color in zip(
        unconf_n[-unconf_bin_num:], unconf_patches[-unconf_bin_num:], unconf_color
    ):
        unconf_patch.set_facecolor(color)

    colored_loss_img = create_colored_lossimg(
        all_losses, bins, conf_bin_num, unconf_bin_num, conf_color, unconf_color
    )

    # Add labels and title
    plt.xlabel("Loss Value", fontsize=7)
    plt.ylabel("Frequency", fontsize=7)
    plt.title("Histogram of Uncertainty", fontsize=7)
    plt.legend(fontsize=7)
    # plt.grid(True)
    plt.tight_layout()
    base = os.path.splitext(save_path)[0]  # This removes the extension
    save_path = base + ".pdf"  # This adds the .pdf extension
    # Save the plot
    plt.savefig(save_path)
    # plt.show()
    plt.close()

    plt.imshow(colored_loss_img)
    plt.axis("off")
    plt.tight_layout()
    base = os.path.splitext(save_path)[0]
    save_path = base + "_colored_mask.pdf"
    plt.savefig(save_path)
    plt.close()
    return colored_loss_img


def create_colored_lossimg(
    loss_reco_resized: torch.Tensor,
    bins: np.ndarray,
    conf_bin_num: int,
    unconf_bin_num: int,
    conf_color: np.ndarray,
    unconf_color: np.ndarray,
) -> np.ndarray:
    H, W = loss_reco_resized.shape[-2:]
    # Convert bin edges to bin centers for accurate mapping
    bin_centers = bins[1:]
    loss_reco_resized = loss_reco_resized.detach().squeeze(0, 1).cpu().numpy()
    # Create a color map that combines both conf and unconf colors
    combined_colors = np.vstack(
        (conf_color[:conf_bin_num], unconf_color[:unconf_bin_num])
    )
    if combined_colors.shape[0] != len(bin_centers):
        combined_colors = np.vstack(
            (conf_color[:conf_bin_num], unconf_color[1:unconf_bin_num])
        )
    # Initialize an RGB image
    rgb_image = np.zeros((H, W, 3))

    # For each bin, create a mask and assign the color
    for i, center in enumerate(bin_centers):
        # Determine the color for this bin. Note: Ensure bin index does not exceed the color array.
        color = combined_colors[min(i, len(combined_colors) - 1)]

        # Create a mask for pixels in this bin
        if i == 0:
            # First bin
            bin_mask = loss_reco_resized <= bin_centers[i]
        else:
            # Middle bins
            bin_mask = (loss_reco_resized > bin_centers[i - 1]) & (
                loss_reco_resized <= bin_centers[i]
            )

        # Apply color to pixels in the current bin
        rgb_image[bin_mask] = color[:3]  # Ignore alpha channel if present

    # Ensure rgb_image values are valid for display [0-1] for floats or [0-255] for integers
    rgb_image = (
        np.clip(rgb_image, 0, 1)
        if np.issubdtype(rgb_image.dtype, np.floating)
        else np.clip(rgb_image, 0, 255)
    )

    return rgb_image


def plot_masks_compare(
    gt_masks: torch.Tensor,
    conf_masks: torch.Tensor,
    images: torch.Tensor,
    file_path: str,
    layout_type: str = "side_by_side",
    param: ParamCollection = None,
) -> None:
    """
    Plot ground truth masks, confidence masks, and images side by side and save to file.

    Args:
    gt_masks (torch.Tensor): Ground truth masks (B, 1, H, W).
    conf_masks (torch.Tensor): Confidence masks (B, 1, H, W).
    images (torch.Tensor): Corresponding images (B, 3, H, W).
    file_path (str): Path to save the file.
    """

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    all_img_list = []

    for i in range(gt_masks.shape[0]):
        img = plot_image(images[i].squeeze(0))
        output_gt = plot_overlay_image(
            images[i].unsqueeze(0),
            alpha=0.7,
            overlay_mask=gt_masks[i],
            channel=0,
            cmap="coolwarm",
            double_norm=True,
        )
        output_conf = plot_overlay_image(
            images[i].unsqueeze(0),
            alpha=0.7,
            overlay_mask=conf_masks[i],
            channel=0,
            cmap="coolwarm",
            double_norm=True,
        )
        img_list = [img, output_gt, output_conf]
        title_list = ["Original Image", "GT Mask", "Confidence Mask"]
        if layout_type == "side_by_side":
            plot_images_side_by_side(
                img_list,
                title_list,
                save_path=os.path.join(file_path, "node" + str(i) + ".png"),
            )
        elif layout_type == "grid":
            all_img_list.extend(img_list)
        torch.cuda.empty_cache()

    if layout_type == "grid":
        # Assuming all images have the same dimensions
        rows = 5
        cols = 3  # as there are 3 types of images
        num_images_per_chunk = rows * cols
        for i in range(0, len(all_img_list), num_images_per_chunk):
            chunk = all_img_list[i : i + num_images_per_chunk]
            chunk_titles = title_list * rows
            plot_images_in_grid(
                chunk,
                chunk_titles,
                rows,
                cols,
                save_path=os.path.join(
                    file_path, "chunk" + str(int(i / cols)) + ".png"
                ),
                show_plot=False,
                param=param,
            )
            print("Saved fig for chunk", int(i / cols))


def masks_iou_stats(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    output_file: str = "iou_stats.txt",
    name: str = "debug",
) -> Dict[str, float]:
    """
    Calculates the Intersection over Union (IoU) metric for each predicted mask against the ground truth.
    """
    # Ensure the masks are binary (0 or 1).
    gt_masks = gt_masks.int()
    pred_masks = pred_masks.int()

    # Calculate Intersection and Union.
    intersection = (
        (gt_masks & pred_masks).sum(dim=[2, 3]).float()
    )  # Logical AND for intersection
    union = (gt_masks | pred_masks).sum(dim=[2, 3]).float()  # Logical OR for union

    # Avoid division by zero.
    union = union.clamp(min=1)

    # Calculate IoU for each mask.
    iou = (intersection / union) * 100.0  # Percentage
    iou_mean = iou.mean().item()
    iou_std = iou.std().item()

    # Write IoU stats to file.
    with open(output_file, "a") as file:
        file.write(name + "\n")
        iou_stats = f"Average IoU: {iou_mean:.3f}%, Std. Dev: {iou_std:.3f}%\n"
        file.write(iou_stats)
        for i in range(iou.shape[0]):
            file.write(f"Node {i} IoU: {iou[i].item():.3f}%\n")

    return {"iou_mean": iou_mean, "iou_std": iou_std}


def show_mask(mask: torch.Tensor, ax: Axes, random_color: bool = False) -> None:
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(
    coords: torch.Tensor, labels: torch.Tensor, ax: Axes, marker_size: int = 375
) -> None:
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def calculate_mask_values(mask: torch.Tensor) -> Tuple[float, float]:
    if mask is None:
        return 0, 0  # Or handle this case as you see fit
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Create a mask for non-NaN values
    non_nan_mask = ~np.isnan(mask)

    # Assuming mask is a single-channel image
    max_val = np.nanmax(mask[non_nan_mask]) if np.any(non_nan_mask) else 0
    mean_val = np.nanmean(mask[non_nan_mask]) if np.any(non_nan_mask) else 0

    return max_val, mean_val
