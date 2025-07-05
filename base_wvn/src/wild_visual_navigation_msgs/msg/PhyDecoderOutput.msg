# Contains header, prediction array, 3d feet coordinates in odom frame, their plane info
# and contact info for each feet, order in list follows: "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
std_msgs/Header header
geometry_msgs/Pose base_pose
float32[] prediction
geometry_msgs/Pose[] feet_poses
bool[] feet_contact
wild_visual_navigation_msgs/PlaneEdge[] feet_planes




