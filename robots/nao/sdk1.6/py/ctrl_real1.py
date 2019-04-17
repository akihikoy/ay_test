#!/usr/bin/python
# -*- coding: utf-8 -*-

# NOTE: Do following before running this;
#  1. Execute `source setup.tcsh' (if the shell is tcsh)

from ctrl_common import *

robot_IP='163.221.139.150'
#robot_IP='192.168.1.101'
is_simulation=False

test(robot_IP,is_simulation)
