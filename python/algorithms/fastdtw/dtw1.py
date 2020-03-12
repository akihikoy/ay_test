#!/usr/bin/python3
#\file    dtw1.py
#\brief   Testing fastdtw.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.11, 2020
# NOTE: Need to install fastdtw by:
#   $ sudo apt-get -f install python3-scipy
#   $ pip install fastdtw
# https://pypi.org/project/fastdtw/
# NOTE: Need to use Python 3 (#!/usr/bin/python3).

import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
y = np.array([[2,2], [3,3], [4,4]])
distance, path = fastdtw(x, y, dist=euclidean)
print(distance, path)
