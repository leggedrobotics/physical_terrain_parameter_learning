#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import cv2
from geometry_msgs.msg import Pose, PoseStamped, Point, Transform, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from wild_visual_navigation_msgs.msg import PlaneEdge
from cv_bridge import CvBridge
from liegroups.numpy import SO3
from numba import jit
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional, Union, List

CV_BRIDGE = CvBridge()
TO_TENSOR = transforms.ToTensor()
TO_PIL_IMAGE = transforms.ToPILImage()


@jit(nopython=True)
def _q_to_se3(q: np.ndarray) -> np.ndarray:
    # Normalize the quaternion
    q = q / np.linalg.norm(q)
    x, y, z, w = q

    # Compute the rotation matrix
    R = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ],
        dtype=np.float32,
    )
    # SE(3) matrix: 4x4 with R in top-left and zero translation
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R

    return T


def pq_to_se3(pq: Tuple[np.ndarray, np.ndarray]) -> Tuple[bool, Optional[np.ndarray]]:
    assert len(pq) == 2
    assert isinstance(pq, tuple)
    if pq[0] is None:
        return False, None
    t = np.array(pq[0], dtype=np.float32)
    q = np.array(pq[1], dtype=np.float32)

    T = _q_to_se3(q)
    T[:3, 3] = t

    return True, T


def np_to_geometry_msgs_PointArray(array: np.ndarray) -> List[Point]:
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


def ros_cam_info_to_tensors(
    caminfo_msg: CameraInfo, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.eye(4, dtype=torch.float32).to(device)
    K[:3, :3] = torch.FloatTensor(caminfo_msg.K).reshape(3, 3)
    K = K.unsqueeze(0)
    H = caminfo_msg.height  # torch.IntTensor([caminfo_msg.height]).to(device)
    W = caminfo_msg.width  # torch.IntTensor([caminfo_msg.width]).to(device)
    return K, H, W


def ros_image_to_torch(
    ros_img: Union[Image, CompressedImage],
    desired_encoding: str = "rgb8",
    device: str = "cpu",
) -> torch.Tensor:
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


def torch_to_ros_image(
    torch_img: torch.Tensor, desired_encoding: str = "rgb8"
) -> Image:
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


def numpy_to_ros_image(np_img: np.ndarray, desired_encoding: str = "rgb8") -> Image:
    """

    Args:
        np_img (np.array): Image to convert to ROS message
        desired_encoding (str, optional): _description_. Defaults to "rgb8".

    Returns:
        _type_: _description_
    """
    ros_image = CV_BRIDGE.cv2_to_imgmsg(np_img, encoding=desired_encoding)
    return ros_image


def se3_to_pq(matrix: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    matrix = matrix.cpu().numpy() if isinstance(matrix, torch.Tensor) else matrix
    q = SO3.from_matrix(matrix[:3, :3], normalize=True).to_quaternion(ordering="xyzw")
    p = matrix[:3, 3]
    return p, q


def se3_to_pose_msg(matrix: Union[torch.Tensor, np.ndarray]) -> Pose:
    p, q = se3_to_pq(matrix)
    pose = Pose()
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    pose.position.x = p[0]
    pose.position.y = p[1]
    pose.position.z = p[2]

    return pose


def pose_msg_to_se3_torch(msg: Pose, device: str = "cpu") -> torch.Tensor:
    se3 = msg_to_se3(msg)
    return torch.tensor(se3, device=device, dtype=torch.float32)


def plane_edge_to_torch(edge: PlaneEdge, device="cpu") -> torch.Tensor:
    edge = edge.edge_points
    ls = []
    for point in edge:
        point = torch.tensor(
            [point.x, point.y, point.z], dtype=torch.float32, device=device
        )
        ls.append(point)
    return torch.stack(ls, dim=0).to(device)


def scale_intrinsic(K: torch.Tensor, ratio_x, ratio_y, crop_offset_x, crop_offset_y):
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


""" 
below credit to https://answers.ros.org/question/332407/transformstamped-to-transformation-matrix-python/
"""


def _pose_to_pq(msg: Pose) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    )
    return p, q


def _pose_stamped_to_pq(msg: PoseStamped) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return _pose_to_pq(msg.pose)


def _transform_to_pq(msg: Transform) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
    return p, q


def _transform_stamped_to_pq(msg: TransformStamped) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return _transform_to_pq(msg.transform)


def _odometry_to_pq(msg: Odometry) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a C{nav_msgs/Odometry} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return _pose_to_pq(msg.pose.pose)


def msg_to_pq(
    msg: Union[Pose, PoseStamped, Transform, TransformStamped, Odometry],
) -> Tuple[np.ndarray, np.ndarray]:
    """Conversion from geometric ROS messages into position and quaternion

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a tuple of position and quaternion as numpy arrays
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p, q = _pose_to_pq(msg)
    elif isinstance(msg, PoseStamped):
        p, q = _pose_stamped_to_pq(msg)
    elif isinstance(msg, Transform):
        p, q = _transform_to_pq(msg)
    elif isinstance(msg, TransformStamped):
        p, q = _transform_stamped_to_pq(msg)
    elif isinstance(msg, Odometry):
        p, q = _odometry_to_pq(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)
            )
        )
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    return p, q


def msg_to_se3(
    msg: Union[Pose, PoseStamped, Transform, TransformStamped, Odometry],
) -> np.ndarray:
    """Conversion from geometric ROS messages into SE(3)

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    p, q = msg_to_pq(msg)
    _, T = pq_to_se3((p, q))
    return T
