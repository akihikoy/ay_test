# ros2_test1

First ROS2 test.

# Build

```bash
$ colcon build --symlink-install
```

# Run

Test publisher.
```bash
$ ros2 run ros2_test1 cpp_talker_exec
  [INFO] [1767199302.980013585] [cpp_talker_node]: cpp_talker_node: Initialized
  [INFO] [1767199303.980426799] [cpp_talker_node]: cpp_talker_node: on_time: 0
  [INFO] [1767199304.980266747] [cpp_talker_node]: cpp_talker_node: on_time: 1
  [INFO] [1767199305.980362058] [cpp_talker_node]: cpp_talker_node: on_time: 2
  [INFO] [1767199306.980527797] [cpp_talker_node]: cpp_talker_node: on_time: 3
  ...

$ ros2 topic echo /topic_test
  count: 31
  status_message: Counting...
  ---
  count: 32
  status_message: Counting...
  ...
```

Test service server.
```bash
$ ros2 run ros2_test1 cpp_talker_exec
akihikoy@fv42:~/prg/ay_test/ros2/rclcpp$ ros2 run ros2_test1 cpp_talker_exec
  [INFO] [1767430712.787395978] [cpp_talker_node]: cpp_talker_node: Initialized
  [INFO] [1767430713.787524935] [cpp_talker_node]: cpp_talker_node: on_time: 0
  [INFO] [1767430714.787406624] [cpp_talker_node]: cpp_talker_node: on_time: 1
  [INFO] [1767430715.787455508] [cpp_talker_node]: cpp_talker_node: on_time: 2
  [INFO] [1767430716.611825060] [cpp_talker_node]: cpp_talker_node: handle_service: 100
  [INFO] [1767430716.787394638] [cpp_talker_node]: cpp_talker_node: on_time: 100
  [INFO] [1767430717.787432892] [cpp_talker_node]: cpp_talker_node: on_time: 101
  ...

$ ros2 service call /srv_test ros2_test1_msgs/srv/SrvTest1 "{target_value: 100}"
  waiting for service to become available...
  requester: making request: ros2_test1_msgs.srv.SrvTest1_Request(target_value=100)
  response:
  ros2_test1_msgs.srv.SrvTest1_Response(success=True, message='Counter updated to 100 via service')
```


Test action server.
```bash
$ ros2 run ros2_test1 cpp_talker_exec
akihikoy@fv42:~/prg/ay_test/ros2/rclcpp$ ros2 run ros2_test1 cpp_talker_exec
  [INFO] [1767430712.787395978] [cpp_talker_node]: cpp_talker_node: Initialized
  [INFO] [1767430713.787524935] [cpp_talker_node]: cpp_talker_node: on_time: 0
  ...
  [INFO] [1767440568.130614892] [cpp_talker_node]: cpp_talker_node: on_time: 161
  [INFO] [1767440568.969908902] [cpp_talker_node]: Action Goal received: 5
  [INFO] [1767440568.970407029] [cpp_talker_node]: Action Execution started
  [INFO] [1767440568.970698581] [cpp_talker_node]: Action Feedback: 1
  [INFO] [1767440569.130684191] [cpp_talker_node]: cpp_talker_node: on_time: 162
  [INFO] [1767440569.971002377] [cpp_talker_node]: Action Feedback: 2
  [INFO] [1767440570.130673455] [cpp_talker_node]: cpp_talker_node: on_time: 163
  [INFO] [1767440570.971018809] [cpp_talker_node]: Action Feedback: 3
  [INFO] [1767440571.130742028] [cpp_talker_node]: cpp_talker_node: on_time: 164
  [INFO] [1767440571.970988590] [cpp_talker_node]: Action Feedback: 4
  [INFO] [1767440572.130847821] [cpp_talker_node]: cpp_talker_node: on_time: 165
  [INFO] [1767440572.971005748] [cpp_talker_node]: Action Feedback: 5
  [INFO] [1767440573.130710725] [cpp_talker_node]: cpp_talker_node: on_time: 166
  [INFO] [1767440573.971188654] [cpp_talker_node]: Action Succeeded
  [INFO] [1767440574.130643805] [cpp_talker_node]: cpp_talker_node: on_time: 167
  ...

$ ros2 action send_goal /action_test ros2_test1_msgs/action/ActionTest1 "{target_count: 5}" --feedback
  Waiting for an action server to become available...
  Sending goal:
      target_count: 5

  Feedback:
      current_count: 1

  Goal accepted with ID: 658642a518f547b08a9a6f052d1bfb8e

  Feedback:
      current_count: 2

  Feedback:
      current_count: 3

  Feedback:
      current_count: 4

  Feedback:
      current_count: 5

  Result:
      final_count: 5

  Goal finished with status: SUCCEEDED
```


Test subscriber.
```bash
$ ros2 run ros2_test1 cpp_talker_exec
  [INFO] [1767199302.980013585] [cpp_talker_node]: cpp_talker_node: Initialized
  [INFO] [1767199303.980426799] [cpp_talker_node]: cpp_talker_node: on_time: 0
  [INFO] [1767199304.980266747] [cpp_talker_node]: cpp_talker_node: on_time: 1
  [INFO] [1767199305.980362058] [cpp_talker_node]: cpp_talker_node: on_time: 2
  [INFO] [1767199306.980527797] [cpp_talker_node]: cpp_talker_node: on_time: 3
  ...

$ ros2 run ros2_test1 cpp_listener_exec
  [INFO] [1767447592.356928008] [cpp_listener_node]: Received: [Counting...], count: 3
  [INFO] [1767447592.963088495] [cpp_listener_node]: Received: [Counting...], count: 4
  [INFO] [1767447593.962897127] [cpp_listener_node]: Received: [Counting...], count: 5
  [INFO] [1767447594.962873479] [cpp_listener_node]: Received: [Counting...], count: 6
  ...
```


Test service client.
```bash
$ ros2 run ros2_test1 cpp_talker_exec
  [INFO] [1767199302.980013585] [cpp_talker_node]: cpp_talker_node: Initialized
  [INFO] [1767199303.980426799] [cpp_talker_node]: cpp_talker_node: on_time: 0
  ...
  [INFO] [1767855865.889772180] [cpp_talker_node]: cpp_talker_node: on_time: 61
  [INFO] [1767855866.877067563] [cpp_talker_node]: cpp_talker_node: handle_service: 100
  [INFO] [1767855866.889488570] [cpp_talker_node]: cpp_talker_node: on_time: 100
  ...
  [INFO] [1767856280.889629599] [cpp_talker_node]: cpp_talker_node: on_time: 456
  [INFO] [1767856281.209730192] [cpp_talker_node]: cpp_talker_node: handle_service: 200
  [INFO] [1767856281.889768338] [cpp_talker_node]: cpp_talker_node: on_time: 200
  ...

$ ros2 run ros2_test1 cpp_srvc_exec
  [INFO] [1767855895.419605001] [cpp_srvc_node]: Service Result: Counter updated to 100 via service

$ ros2 run ros2_test1 cpp_srvc_exec --ros-args -p target_value:=200
  [INFO] [1767856281.210766548] [cpp_srvc_node]: Service Result: Counter updated to 200 via service
```

We cal also set the parameter externally (test this with commenting out `timer_->cancel()` in cpp_srv_client.cpp).
```bash
$ ros2 param set /cpp_srvc_node target_value 300
  Set parameter successful
```


Test action client.
```bash
$ ros2 run ros2_test1 cpp_talker_exec
  [INFO] [1767199302.980013585] [cpp_talker_node]: cpp_talker_node: Initialized
  [INFO] [1767199303.980426799] [cpp_talker_node]: cpp_talker_node: on_time: 0
  ...
  [INFO] [1767886092.223811097] [cpp_talker_node]: cpp_talker_node: on_time: 16
  [INFO] [1767886092.384431388] [cpp_talker_node]: Action Goal received: 5
  [INFO] [1767886092.385652342] [cpp_talker_node]: Action Execution started
  [INFO] [1767886092.386348871] [cpp_talker_node]: Action Feedback: 1
  [INFO] [1767886093.223779603] [cpp_talker_node]: cpp_talker_node: on_time: 17
  [INFO] [1767886093.386525228] [cpp_talker_node]: Action Feedback: 2
  [INFO] [1767886094.223794226] [cpp_talker_node]: cpp_talker_node: on_time: 18
  [INFO] [1767886094.386501309] [cpp_talker_node]: Action Feedback: 3
  [INFO] [1767886095.223812517] [cpp_talker_node]: cpp_talker_node: on_time: 19
  [INFO] [1767886095.386421537] [cpp_talker_node]: Action Feedback: 4
  [INFO] [1767886096.223840756] [cpp_talker_node]: cpp_talker_node: on_time: 20
  [INFO] [1767886096.386432748] [cpp_talker_node]: Action Feedback: 5
  [INFO] [1767886097.223898259] [cpp_talker_node]: cpp_talker_node: on_time: 21
  [INFO] [1767886097.386917834] [cpp_talker_node]: Action Succeeded
  [INFO] [1767886098.223890571] [cpp_talker_node]: cpp_talker_node: on_time: 22
  ...

$ ros2 run ros2_test1 cpp_actionc_exec
  [INFO] [1767886092.382907069] [cpp_actionc_node]: Sending goal; 5
  [INFO] [1767886092.385838285] [cpp_actionc_node]: Goal accepted by server, waiting for result
  [INFO] [1767886092.386707167] [cpp_actionc_node]: Feedback received: 1
  [INFO] [1767886093.387014637] [cpp_actionc_node]: Feedback received: 2
  [INFO] [1767886094.386931566] [cpp_actionc_node]: Feedback received: 3
  [INFO] [1767886095.386837717] [cpp_actionc_node]: Feedback received: 4
  [INFO] [1767886096.386946023] [cpp_actionc_node]: Feedback received: 5
  [INFO] [1767886097.387599758] [cpp_actionc_node]: Result received: 5
  ^C[INFO] [1767886100.718247271] [rclcpp]: signal_handler(signum=2)
```


