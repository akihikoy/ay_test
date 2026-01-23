//-------------------------------------------------------------------------------------------
/*! \file    cpp_srvc_main.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.31, 2025
*/
//-------------------------------------------------------------------------------------------
#include <ros2_test1/cpp_srv_client.hpp>
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ros2_test1::SrvClientNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
//-------------------------------------------------------------------------------------------
