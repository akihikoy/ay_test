#!/bin/bash
#\file    download1.sh
#\brief   Download data from server to avoid the GitHub limit.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.15, 2021

#TO STORE ROS IMAGE TOPICS:
#  rosbag record -j /camera/aligned_depth_to_color/image_raw /camera/color/image_raw /camera/aligned_depth_to_color/camera_info
#  rosbag record -j /camera/aligned_depth_to_color/image_raw /camera/color/image_raw /camera/aligned_depth_to_color/camera_info /camera/depth/color/points
#TO PLAY ROS IMAGE TOPICS:
#  rosbag play -l rs1.bag

server=http://akihikoy.net/p/

wget $server/rs1.bag -O rs1.bag

wget $server/rs2.bag.tar.bz2 -O rs2.bag.tar.bz2
tar jxvf rs2.bag.tar.bz2 --checkpoint=5000
