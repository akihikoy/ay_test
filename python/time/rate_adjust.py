#!/usr/bin/python3
#\file    rate_adjust.py
#\brief   Rate adjuster.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.23, 2018

import time

'''Modified rospy.Rate with standard time.
https://docs.ros.org/api/rospy/html/rospy.timer-pysrc.html#Rate '''
class TRate(object):
  """
  Convenience class for sleeping in a loop at a specified rate
  """

  def __init__(self, hz, reset=False):
    """
    Constructor.
    @param hz: hz rate to determine sleeping
    @type  hz: float
    @param reset: if True, timer is reset when time moved backward. [default: False]
    @type  reset: bool
    """
    self.last_time = time.time()
    self.sleep_dur = 1.0/hz
    self._reset = reset

  def _remaining(self, curr_time):
    """
    Calculate the time remaining for rate to sleep.
    @param curr_time: current time
    @type  curr_time: L{Time}
    @return: time remaining
    @rtype: L{Time}
    """
    # detect time jumping backwards
    if self.last_time > curr_time:
      self.last_time = curr_time

    # calculate remaining time
    elapsed = curr_time - self.last_time
    return self.sleep_dur - elapsed

  def remaining(self):
    """
    Return the time remaining for rate to sleep.
    @return: time remaining
    @rtype: L{Time}
    """
    curr_time = time.time()
    return self._remaining(curr_time)

  def sleep(self):
    """
    Attempt sleep at the specified rate. sleep() takes into
    account the time elapsed since the last successful
    sleep().
    """
    curr_time = time.time()
    time.sleep(max(0.0, self._remaining(curr_time)))
    self.last_time = self.last_time + self.sleep_dur

    # detect time jumping forwards, as well as loops that are
    # inherently too slow
    if curr_time - self.last_time > self.sleep_dur * 2:
      self.last_time = curr_time



if __name__=='__main__':
  import random

  rate= TRate(2)
  t0= time.time()

  while True:
    print(('---------',time.time()-t0))
    t0= time.time()
    s= 0
    for i in range(int(600*random.random())):
      s+= i
      time.sleep(0.001)
    print(i)
    rate.sleep()


