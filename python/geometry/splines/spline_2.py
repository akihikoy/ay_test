#!/usr/bin/python3
#\file    spline_2.py
#\brief   Comparison of TCubicHermiteSpline and TLinearInterpolator.
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
  choice= input("Select dataset (1:Gen1d_1, 2:Gen1d_2(seed=10), 3:Gen1d_3, 4:Gen1d_2(seed=None) [default]): ").strip()
  if choice == "1":
    data= gen_data.Gen1d_1()
  elif choice == "2":
    data= gen_data.Gen1d_2()
  elif choice == "3":
    data= gen_data.Gen1d_3()
  else:
    data= gen_data.Gen1d_2(seed=None)

  # Hermite Spline
  hermite= TCubicHermiteSpline()
  hermite.Initialize(data, tan_method=hermite.CARDINAL, c=0.0)

  # Linear Interpolator
  linear= TLinearInterpolator()
  linear.Initialize(data)

  # Sampling
  t_vals= np.arange(data[0][0], data[-1][0], 0.001)
  x_vals_hermite= []
  x_vals_linear= []
  for t in t_vals:
    x_vals_hermite.append(hermite.Evaluate(t))
    x_vals_linear.append(linear.Evaluate(t))

  # Given data
  t_data= [d[0] for d in data]
  x_data= [d[1] for d in data]

  # Plot
  plt.figure()
  plt.plot(t_vals, x_vals_hermite, label='Hermite', color='blue')
  plt.plot(t_vals, x_vals_linear, label='Linear', color='orange', linestyle='--')
  plt.scatter(t_data, x_data, color='black', marker='o', s=40, label='given points')

  plt.xlabel('t')
  plt.ylabel('x')
  plt.title('Interpolation Comparison: Hermite vs Linear')
  plt.legend()
  plt.grid(True)
  plt.show()

