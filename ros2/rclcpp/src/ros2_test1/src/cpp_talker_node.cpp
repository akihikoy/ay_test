//-------------------------------------------------------------------------------------------
/*! \file    cpp_talker_node.cpp
    \brief   certain c++ unit file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.31, 2025
*/
//-------------------------------------------------------------------------------------------
#include <ros2_test1/cpp_talker_node.hpp>
//-------------------------------------------------------------------------------------------
#include <rclcpp_components/register_node_macro.hpp>
#include <thread>
//-------------------------------------------------------------------------------------------
namespace ros2_test1
{
//-------------------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------------------
  // class TalkerNode
  //-------------------------------------------------------------------------------------------
  TalkerNode::TalkerNode(const rclcpp::NodeOptions &options)
  : rclcpp::Node("cpp_talker_node", options), count_(0)
  {
    // Parameter
    this->declare_parameter("publish_rate", 1);
    int rate = this->get_parameter("publish_rate").as_int();

    // Publisher
    pub_ = this->create_publisher<ros2_test1_msgs::msg::MsgTest1>("topic_test", /*qos_history_depth=*/10);

    // Timer to periodically publish the counter.
    auto period = std::chrono::milliseconds(static_cast<int>(1000.0 / rate));
    timer_ = this->create_wall_timer(period, std::bind(&TalkerNode::on_timer, this));

    // Service server
    srv_ = this->create_service<ros2_test1_msgs::srv::SrvTest1>(
      "srv_test",
      std::bind(&TalkerNode::handle_service, this, std::placeholders::_1, std::placeholders::_2));

    // Action server
    action_server_ = rclcpp_action::create_server<ActionTest1>(
      this,
      "action_test",
      std::bind(&TalkerNode::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&TalkerNode::handle_cancel, this, std::placeholders::_1),
      std::bind(&TalkerNode::handle_accepted, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "cpp_talker_node: Initialized");
  }

  // Timer callback (to publish a topic)
  void TalkerNode::on_timer()
  {
    auto msg = ros2_test1_msgs::msg::MsgTest1();
    msg.count = count_++;
    msg.status_message = "Counting...";
    pub_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "cpp_talker_node: on_time: %d", msg.count);
  }

  // Service callback
  void TalkerNode::handle_service(
    const std::shared_ptr<ros2_test1_msgs::srv::SrvTest1::Request> request,
    std::shared_ptr<ros2_test1_msgs::srv::SrvTest1::Response> response)
  {
    count_ = request->target_value;
    response->success = true;
    response->message = "Counter updated to " + std::to_string(count_) + " via service";
    RCLCPP_INFO(this->get_logger(), "cpp_talker_node: handle_service: %d", request->target_value);
  }

  // Action: Goal check
  rclcpp_action::GoalResponse TalkerNode::handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const ActionTest1::Goal> goal)
  {
    (void)uuid;
    RCLCPP_INFO(this->get_logger(), "Action Goal received: %d", goal->target_count);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  // Action: Cancel check
  rclcpp_action::CancelResponse TalkerNode::handle_cancel(
    const std::shared_ptr<GoalHandleAction> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_INFO(this->get_logger(), "Action Cancel received");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  // Action: Accepted (start thread)
  void TalkerNode::handle_accepted(
    const std::shared_ptr<GoalHandleAction> goal_handle)
  {
    // As execute is time consuming code, we use a thread to avoid block the executor.
    std::thread{std::bind(&TalkerNode::execute, this, std::placeholders::_1), goal_handle}.detach();
  }

  // Action: Execution logic.
  void TalkerNode::execute(const std::shared_ptr<GoalHandleAction> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Action Execution started");
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<ActionTest1::Feedback>();
    auto result = std::make_shared<ActionTest1::Result>();
    int current = 0;

    rclcpp::Rate loop_rate(1.0);

    while (current < goal->target_count && rclcpp::ok()) {
      if (goal_handle->is_canceling()) {
        result->final_count = current;
        goal_handle->canceled(result);
        RCLCPP_INFO(this->get_logger(), "Action Canceled");
        return;
      }

      ++current;
      feedback->current_count = current;
      goal_handle->publish_feedback(feedback);
      RCLCPP_INFO(this->get_logger(), "Action Feedback: %d", current);

      loop_rate.sleep();
    }

    if(rclcpp::ok()) {
      result->final_count = current;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Action Succeeded");
    }
  }

//-------------------------------------------------------------------------------------------
}  // end of ros2_test1
//-------------------------------------------------------------------------------------------
