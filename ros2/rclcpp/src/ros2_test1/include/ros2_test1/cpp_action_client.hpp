//-------------------------------------------------------------------------------------------
/*! \file    cpp_action_client.hpp
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
#include <ros2_test1_msgs/action/action_test1.hpp>
//-------------------------------------------------------------------------------------------
namespace ros2_test1
{
//-------------------------------------------------------------------------------------------

  class ActionClientNode : public rclcpp::Node
  {
  public:
    explicit ActionClientNode(const rclcpp::NodeOptions & options);
  private:
    using ActionTest1 = ros2_test1_msgs::action::ActionTest1;
    using GoalHandleAction = rclcpp_action::ClientGoalHandle<ActionTest1>;

    rclcpp_action::Client<ActionTest1>::SharedPtr client_;
    rclcpp::TimerBase::SharedPtr timer_;

    void send_goal();
    void goal_response_callback(const GoalHandleAction::SharedPtr & goal_handle);
    void feedback_callback(
      GoalHandleAction::SharedPtr,
      const std::shared_ptr<const ActionTest1::Feedback> feedback);
    void result_callback(const GoalHandleAction::WrappedResult & result);
  };

//-------------------------------------------------------------------------------------------
}  // end of ros2_test1
//-------------------------------------------------------------------------------------------
