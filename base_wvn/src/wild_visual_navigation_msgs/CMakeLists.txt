#                                                                               
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#                                                                               
cmake_minimum_required(VERSION 3.0.2)
project(wild_visual_navigation_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  message_runtime
  std_msgs
  sensor_msgs
)

add_message_files(
  FILES
  CustomState.msg
  RobotState.msg
  SystemState.msg
  ImageFeatures.msg
  StampedFloat32MultiArray.msg
  PlaneEdge.msg
  PhyDecoderOutput.msg
  FeatExtractorOutput.msg
  FeatInfo.msg
  ChannelInfo.msg
  AnymalState.msg
  Contact.msg
  ExtendedJointState.msg
)

add_service_files(
  FILES
  SaveCheckpoint.srv
  LoadCheckpoint.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs
                 message_generation
                 message_runtime
                 std_msgs
                 sensor_msgs
)
