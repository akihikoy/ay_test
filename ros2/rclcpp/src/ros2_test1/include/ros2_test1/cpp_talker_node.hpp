//-------------------------------------------------------------------------------------------
/*! \file    cpp_talker_node.hpp
    \brief   certain c++ header file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.31, 2025
*/
//-------------------------------------------------------------------------------------------
#pragma once
//-------------------------------------------------------------------------------------------
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
// NOTE: Although the interface types are defined like MsgTest1.msg, the header file names are automatically converted to snake cases as follows.
#include <ros2_test1_msgs/msg/msg_test1.hpp>
#include <ros2_test1_msgs/srv/srv_test1.hpp>
#include <ros2_test1_msgs/action/action_test1.hpp>
//-------------------------------------------------------------------------------------------
namespace ros2_test1
{
//-------------------------------------------------------------------------------------------

  class TalkerNode : public rclcpp::Node
  {
  public:
    // NOTE: Better to use the explicit specifier to avoid unexpected automatic type conversion.
    // NOTE: NodeOptions is used when making a node as a component (shared object).
    explicit TalkerNode(const rclcpp::NodeOptions & options);

  private:
    // Publisher and publisher related objects
    void on_timer();
    rclcpp::Publisher<ros2_test1_msgs::msg::MsgTest1>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    int32_t count_;

    // Service server
    void handle_service(
      const std::shared_ptr<ros2_test1_msgs::srv::SrvTest1::Request> request,
      std::shared_ptr<ros2_test1_msgs::srv::SrvTest1::Response> response);
    rclcpp::Service<ros2_test1_msgs::srv::SrvTest1>::SharedPtr srv_;

    // Action server
    using ActionTest1 = ros2_test1_msgs::action::ActionTest1;
    using GoalHandleAction = rclcpp_action::ServerGoalHandle<ActionTest1>;
    rclcpp_action::GoalResponse handle_goal(
      const rclcpp_action::GoalUUID & uuid,
      std::shared_ptr<const ActionTest1::Goal> goal);
    rclcpp_action::CancelResponse handle_cancel(
      const std::shared_ptr<GoalHandleAction> goal_handle);
    void handle_accepted(
      const std::shared_ptr<GoalHandleAction> goal_handle);
    void execute(const std::shared_ptr<GoalHandleAction> goal_handle);  // executed in a thread.
    rclcpp_action::Server<ActionTest1>::SharedPtr action_server_;
  };

//-------------------------------------------------------------------------------------------
}  // end of ros2_test1
//-------------------------------------------------------------------------------------------
