import src.wild_visual_navigation_ros.scripts.ros_converter as rc
from tqdm import tqdm
import rosbag
import cv2
import numpy as np
import os
from base_wvn.offline.helper import plot_pred_w_overlay, calculate_mask_values
from base_wvn.config.wvn_cfg import ParamCollection
from base_wvn.offline.lightening_module import DecoderLightning
from typing import List


class RosBagPredictionVideoProducer:
    def __init__(self, model: DecoderLightning, param: ParamCollection):
        self.model = model
        self.param = param

    def produce(self) -> None:
        process_option = self.param.offline.process_option
        with rosbag.Bag(self.param.offline.img_bag_path, "r") as bag:
            total_messages = bag.get_message_count(
                topic_filters=[self.param.roscfg.camera_topic]
            )
            # Determine the number of messages to process based on the selected option
            if process_option == "all":
                messages_to_process = total_messages
            elif process_option == "first_half":
                messages_to_process = total_messages // 2
            elif process_option == "first_1000":
                messages_to_process = min(1000, total_messages)
            else:
                raise ValueError("Invalid process option")
            progress_bar = tqdm(total=messages_to_process, desc="Processing ROS Bag")
            # Set up video writer
            first_frame = True
            for _, msg, _ in bag.read_messages(topics=[self.param.roscfg.camera_topic]):
                if progress_bar.n >= messages_to_process:
                    break
                # Convert ROS Image message to OpenCV image
                img_torch = rc.ros_image_to_torch(msg, device=self.param.run.device)
                img = img_torch[None]
                trans_img, compressed_feat = self.model.feat_extractor.extract(img)
                res = self.model.conf_mask_generator.get_confidence_masked_prediction_from_img(
                    trans_img=trans_img,
                    compressed_feats=compressed_feat,
                    model=self.model.model,
                    loss_fn=self.model.loss_fn,
                )
                fric_vis_imgs, stiff_vis_imgs = plot_pred_w_overlay(
                    data=res,
                    time=self.model.time,
                    step=-1,
                    image_name="",  # irrelevant if save_local is False
                    param=self.param,
                    save_local=False,
                )
                if self.param.general.pub_which_pred == "fric":
                    overlay_img = fric_vis_imgs
                    output_phy = res.masked_output_phy[0]
                elif self.param.general.pub_which_pred == "stiff":
                    overlay_img = stiff_vis_imgs
                    output_phy = res.masked_output_phy[1]
                else:
                    raise ValueError(
                        "Invalid pub_which_pred value. Must be 'fric' or 'stiff'."
                    )
                max_val, mean_val = calculate_mask_values(output_phy)
                # Convert back to OpenCV image and store
                frame = np.concatenate(
                    (
                        np.uint8(overlay_img[0]),
                        np.uint8(overlay_img[1]),
                        np.uint8(overlay_img[2]),
                    ),
                    axis=1,
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Initialize video writer with the first frame's size
                if first_frame:
                    height, width, layers = frame.shape
                    size = (width, height)
                    output_video_path = get_output_video_path(
                        os.path.dirname(self.param.offline.img_bag_path),
                        self.param.general.pub_which_pred,
                    )
                    out = cv2.VideoWriter(
                        output_video_path, cv2.VideoWriter_fourcc(*"DIVX"), 15, size
                    )
                    first_frame = False
                # Define headers for each section
                headers = ["Original Image", "Dense Pred Raw", "Dense Pred w. Mask"]

                # Calculate the width of each section (assuming all sections are equal width)
                section_width = frame.shape[1] // len(headers)

                # Add headers to the concatenated frame
                frame = add_headers_to_frame(frame, headers, section_width)
                start_x_right_section = 2 * section_width

                # Overlay these values on the right section of the frame
                frame = overlay_values_on_section(
                    frame, max_val, mean_val, start_x_right_section
                )
                out.write(frame)
                progress_bar.update(1)
            progress_bar.close()

        out.release()


def overlay_values_on_section(
    frame: np.ndarray, max_val: float, mean_val: float, start_x: float
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    font_thickness = 2
    text_color = (0, 0, 0)  # Black text
    outline_color = (255, 255, 255)  # White outline
    outline_thickness = 8

    texts = [f"Max: {max_val:.2f}", f"Mean: {mean_val:.2f}"]
    positions = [(start_x + 10, 110), (start_x + 10, 155)]  # Text positions

    for text, position in zip(texts, positions):
        # First, draw the outline
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            outline_color,
            outline_thickness,
            lineType=cv2.LINE_AA,
        )

        # Then, draw the text
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return frame


def add_headers_to_frame(
    frame: np.ndarray, headers: List[str], section_width: float
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    font_thickness = 2
    text_color = (0, 0, 0)  # Black text
    outline_color = (255, 255, 255)  # White outline
    outline_thickness = 8

    for i, header in enumerate(headers):
        # Calculate the position of the header
        x_position = (
            i * section_width + 10
        )  # 10 pixels from the left edge of each section
        y_position = 60  # 30 pixels from the top

        # First, draw the outline
        cv2.putText(
            frame,
            header,
            (x_position, y_position),
            font,
            font_scale,
            outline_color,
            outline_thickness,
            lineType=cv2.LINE_AA,
        )

        # Then, draw the text
        cv2.putText(
            frame,
            header,
            (x_position, y_position),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return frame


# Helper function to get output video path
def get_output_video_path(directory: str, fric_or_stiff: str) -> str:
    output_video_filename = "prediction_video" + "_" + fric_or_stiff + ".avi"
    return os.path.join(directory, output_video_filename)
