#!/usr/bin/python
# -*- coding: utf-8 -*-

# NOTE: Do following before running this;
#  1. Execute `source setup.tcsh' (if the shell is tcsh)
#  2. Execute `naoqi' (in command prompt)
#  3. Execute Choregraphe
#  4. In Choregraphe, connect to the local Nao as follows:
#     Click `connect', select `HOSTNAME.local'

from ctrl_common import *

robot_IP='localhost'
is_simulation=True

test(robot_IP,is_simulation)
