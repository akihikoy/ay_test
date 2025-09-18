#!/usr/bin/python3
#\file    cyclic_2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.18, 2025

from cubic_hermite_spline import TCubicHermiteSpline
from linear_interpolator import TLinearInterpolator
import gen_data
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

if __name__=='__main__':
  # Data selection
  choice= input("Select dataset (1:Gen1d_cyc1, 2:Gen1d_cyc2(seed=10), 3:Gen1d_cyc3, 4:Gen1d_cyc2(seed=None) [default]): ").strip()
  if choice == "1":
    data= gen_data.Gen1d_cyc1()
  elif choice == "2":
    data= gen_data.Gen1d_cyc2()
  elif choice == "3":
    data= gen_data.Gen1d_cyc3()
  else:
    data= gen_data.Gen1d_cyc2(seed=None)

  # Hermite Spline
  hermite= TCubicHermiteSpline()
  #hermite.Initialize(data, tan_method=hermite.FINITE_DIFF, end_tan=hermite.CYCLIC, c=0.0)
  hermite.Initialize(data, tan_method=hermite.CARDINAL, end_tan=hermite.CYCLIC, c=0.0)

  # Linear Interpolator
  linear= TLinearInterpolator()
  linear.Initialize(data)

  # Sampling range
  t_vals= np.arange(data[0][0]-2.0, data[-1][0]+2.0, 0.001)

  # Interpolate with EvaluateC
  x_vals_hermite= [hermite.EvaluateC(t) for t in t_vals]
  x_vals_linear= [linear.EvaluateC(t) for t in t_vals]

  # Given data
  t_data= [d[0] for d in data]
  x_data= [d[1] for d in data]

  # Plot
  plt.figure()
  plt.plot(t_vals, x_vals_hermite, label='Hermite (cyclic)', color='blue')
  plt.plot(t_vals, x_vals_linear, label='Linear (cyclic)', color='orange', linestyle='--')
  plt.scatter(t_data, x_data, color='black', marker='o', s=40, label='given points')

  plt.xlabel('t')
  plt.ylabel('x')
  plt.title('Cyclic Interpolation: Hermite vs Linear')
  plt.legend()
  plt.grid(True)
  plt.show()

