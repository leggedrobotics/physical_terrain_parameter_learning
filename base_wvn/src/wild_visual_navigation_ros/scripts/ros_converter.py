#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import cv2
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from wild_visual_navigation_msgs.msg import PlaneEdge
from cv_bridge import CvBridge
from liegroups.torch import SO3, SE3
from numba import jit
import numpy as np
import torch
import torchvision.transforms as transforms

CV_BRIDGE = CvBridge()
TO_TENSOR = transforms.ToTensor()
TO_PIL_IMAGE = transforms.ToPILImage()
BASE_DIM = 7 + 6  # pose + twist


def ros_tf_to_torch(tf_pose, device="cpu"):
    assert len(tf_pose) == 2
    assert isinstance(tf_pose, tuple)
    if tf_pose[0] is None:
        return False, None
    t = torch.FloatTensor(tf_pose[0])
    q = torch.FloatTensor(tf_pose[1])
    return True, SE3(SO3.from_quaternion(q, ordering="xyzw"), t).as_matrix().to(device)


@jit(nopython=True)
def quaternion_to_rotation_matrix(quaternion):
    # Normalize the quaternion
    quaternion = quaternion / np.linalg.norm(quaternion)
    x, y, z, w = quaternion

    # Compute the rotation matrix
    R = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ],
        dtype=np.float32,
    )
    return R


@jit(nopython=True)
def ros_tf_to_numpy(tf_pose):
    assert len(tf_pose) == 2
    assert isinstance(tf_pose, tuple)
    if tf_pose[0] is None:
        return False, None
    t = np.array(tf_pose[0], dtype=np.float32)
    q = np.array(tf_pose[1], dtype=np.float32)

    R = quaternion_to_rotation_matrix(q)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t

    return True, T


def torch_tensor_to_geometry_msgs_PointArray(tensor):
    """
    Converts a torch tensor of shape (n, 3) to a list of ROS geometry_msgs/Point

    Args:
        tensor (torch.Tensor): A torch tensor of shape (n, 3) where each row represents x, y, z coordinates.

    Returns:
        list[geometry_msgs/Point]: A list of geometry_msgs/Point
    """
    # Ensure that the tensor is on the CPU and converted to a numpy array
    points_np = tensor.cpu().detach().numpy()

    # Convert the numpy array to a list of geometry_msgs/Point
    point_list = []
    for point in points_np:
        ros_point = Point(
            *point
        )  # Unpack each x, y, z coordinate into the Point constructor
        point_list.append(ros_point)

    return point_list


def np_to_geometry_msgs_PointArray(array):
    """
    Converts a torch tensor of shape (n, 3) to a list of ROS geometry_msgs/Point

    Args:
        tensor (torch.Tensor): A torch tensor of shape (n, 3) where each row represents x, y, z coordinates.

    Returns:
        list[geometry_msgs/Point]: A list of geometry_msgs/Point
    """
    # Ensure that the tensor is on the CPU and converted to a numpy array
    points_np = array

    # Convert the numpy array to a list of geometry_msgs/Point
    point_list = []
    for point in points_np:
        ros_point = Point(
            *point
        )  # Unpack each x, y, z coordinate into the Point constructor
        point_list.append(ros_point)

    return point_list


def ros_cam_info_to_tensors(caminfo_msg: CameraInfo, device="cpu"):
    K = torch.eye(4, dtype=torch.float32).to(device)
    K[:3, :3] = torch.FloatTensor(caminfo_msg.K).reshape(3, 3)
    K = K.unsqueeze(0)
    H = caminfo_msg.height  # torch.IntTensor([caminfo_msg.height]).to(device)
    W = caminfo_msg.width  # torch.IntTensor([caminfo_msg.width]).to(device)
    return K, H, W


def ros_image_to_torch(ros_img, desired_encoding="rgb8", device="cpu"):
    if type(ros_img).__name__ == "_sensor_msgs__Image" or isinstance(ros_img, Image):
        np_image = CV_BRIDGE.imgmsg_to_cv2(ros_img, desired_encoding=desired_encoding)

    elif type(ros_img).__name__ == "_sensor_msgs__CompressedImage" or isinstance(
        ros_img, CompressedImage
    ):
        np_arr = np.fromstring(ros_img.data, np.uint8)
        np_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if "bgr" in ros_img.format:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    else:
        raise ValueError("Image message type is not implemented.")

    return TO_TENSOR(np_image).to(device)


def torch_to_ros_pose(torch_pose):
    q = SO3.from_matrix(torch_pose[:3, :3].cpu(), normalize=True).to_quaternion(
        ordering="xyzw"
    )
    t = torch_pose[:3, 3].cpu()
    pose = Pose()
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    pose.position.x = t[0]
    pose.position.y = t[1]
    pose.position.z = t[2]

    return pose


def torch_to_ros_image(torch_img, desired_encoding="rgb8"):
    """

    Args:
        torch_img (torch.tensor, shape=(C,H,W)): Image to convert to ROS message
        desired_encoding (str, optional): _description_. Defaults to "rgb8".

    Returns:
        _type_: _description_
    """

    np_img = np.array(TO_PIL_IMAGE(torch_img.cpu()))
    ros_img = CV_BRIDGE.cv2_to_imgmsg(np_img, encoding=desired_encoding)
    return ros_img


def numpy_to_ros_image(np_img, desired_encoding="rgb8"):
    """

    Args:
        np_img (np.array): Image to convert to ROS message
        desired_encoding (str, optional): _description_. Defaults to "rgb8".

    Returns:
        _type_: _description_
    """
    ros_image = CV_BRIDGE.cv2_to_imgmsg(np_img, encoding=desired_encoding)
    return ros_image


def ros_pose_to_torch(ros_pose, device="cpu"):
    q = torch.FloatTensor(
        [
            ros_pose.orientation.x,
            ros_pose.orientation.y,
            ros_pose.orientation.z,
            ros_pose.orientation.w,
        ]
    )
    t = torch.FloatTensor(
        [ros_pose.position.x, ros_pose.position.y, ros_pose.position.z]
    )
    return SE3(SO3.from_quaternion(q, ordering="xyzw"), t).as_matrix().to(device)


def plane_edge_to_torch(edge: PlaneEdge, device="cpu"):
    edge = edge.edge_points
    ls = []
    for point in edge:
        point = torch.FloatTensor([point.x, point.y, point.z]).to(device)
        ls.append(point)
    return torch.stack(ls, dim=0).to(device)


def numpy_to_ros_image(np_img, desired_encoding="rgb8"):
    """

    Args:
        np_img (np.array): Image to convert to ROS message
        desired_encoding (str, optional): _description_. Defaults to "rgb8".

    Returns:
        _type_: _description_
    """
    ros_image = CV_BRIDGE.cv2_to_imgmsg(np_img, encoding=desired_encoding)
    return ros_image


def scale_intrinsic(K: torch.tensor, ratio_x, ratio_y, crop_offset_x, crop_offset_y):
    """
    scale the intrinsic matrix, first resize than crop!!
    """
    # dimension check of K
    if K.shape[2] != 4 or K.shape[1] != 4:
        raise ValueError("The dimension of the intrinsic matrix is not 4x4!")
    K_scaled = K.clone()
    K_scaled[:, 0, 0] = K[:, 0, 0] * ratio_x
    K_scaled[:, 0, 2] = K[:, 0, 2] * ratio_x - crop_offset_x
    K_scaled[:, 1, 1] = K[:, 1, 1] * ratio_y
    K_scaled[:, 1, 2] = K[:, 1, 2] * ratio_y - crop_offset_y

    return K_scaled
