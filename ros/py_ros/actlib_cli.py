#!/usr/bin/python3
#\file    actlib_cli.py
#\brief   actionlib SimpleActionClient test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.21, 2018
#ref. http://wiki.ros.org/actionlib_tutorials/Tutorials/Writing%20a%20Simple%20Action%20Client%20%28Python%29

import rospy
import actionlib
import actionlib_tutorials.msg

def fibonacci_client():
  # Creates the SimpleActionClient, passing the type of the action
  # (FibonacciAction) to the constructor.
  client= actionlib.SimpleActionClient('/fibonacci', actionlib_tutorials.msg.FibonacciAction)

  # Waits until the action server has started up and started
  # listening for goals.
  client.wait_for_server()

  # Creates a goal to send to the action server.
  goal= actionlib_tutorials.msg.FibonacciGoal(order=20)

  # Sends the goal to the action server.
  client.send_goal(goal)

  # Waits for the server to finish performing the action.
  client.wait_for_result()

  # Prints out the result of executing the action
  return client.get_result()  # A FibonacciResult

if __name__=='__main__':
  try:
    rospy.init_node('fibonacci_client_py')
    result= fibonacci_client()
    print("Result:", ', '.join([str(n) for n in result.sequence]))
  except rospy.ROSInterruptException:
    print("program interrupted before completion")
