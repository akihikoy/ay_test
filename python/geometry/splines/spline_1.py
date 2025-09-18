#!/usr/bin/python3
#\file    spline_1.py
#\brief   Test of TCubicHermiteSpline and TLinearInterpolator.
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
  mode= sys.argv[1] if len(sys.argv)>1 else 'spline'

  if mode=='spline':
    spline= TCubicHermiteSpline()
  elif mode=='linear':
    spline= TLinearInterpolator()
  else:
    raise Exception(f'mode={mode} is specified, but can be (spline, linear).')

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

  if mode=='spline':
    spline.Initialize(data, tan_method=spline.CARDINAL, c=0.0)
  elif mode=='linear':
    spline.Initialize(data)

  # Sampling with splines
  t_vals= np.arange(data[0][0], data[-1][0], 0.001)
  x_vals= []
  dx_vals= []
  ddx_vals= []
  for t in t_vals:
    x, dx, ddx= spline.Evaluate(t, with_dd=True)
    x_vals.append(x)
    dx_vals.append(dx)
    ddx_vals.append(ddx)

  # Given data
  t_data= [d[0] for d in data]
  x_data= [d[1] for d in data]

  # Plot
  fig, ax1= plt.subplots()

  color= 'tab:blue'
  ax1.set_xlabel('t')
  ax1.set_ylabel('x', color=color)
  ax1.plot(t_vals, x_vals, color=color, label='x')
  ax1.scatter(t_data, x_data, color='black', marker='o', s=40, label='given points')
  ax1.tick_params(axis='y', labelcolor=color)

  # Plot dx, ddx with the right axis
  ax2= ax1.twinx()

  # Scale ddx
  dx_range = max(abs(np.min(dx_vals)), abs(np.max(dx_vals)))
  ddx_range = max(abs(np.min(ddx_vals)), abs(np.max(ddx_vals)))
  scale = 1
  if ddx_range > 0 and dx_range > 0:
    ratio = ddx_range / dx_range
    power = int(math.floor(math.log10(ratio)))
    scale = 10**power
  ddx_scaled = [v/scale for v in ddx_vals]

  color= 'tab:orange'
  ax2.set_ylabel(f'dx, ddx(x1/{scale})', color=color)
  ax2.plot(t_vals, dx_vals, color='orange', linestyle='--', label='dx')
  ax2.plot(t_vals, ddx_scaled, color='green', linestyle=':', label=f'ddx(x1/{scale})')
  ax2.tick_params(axis='y', labelcolor=color)

  # Gather legends
  lines1, labels1= ax1.get_legend_handles_labels()
  lines2, labels2= ax2.get_legend_handles_labels()
  ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

  plt.title('Cubic Hermite Spline')
  plt.grid(True)
  plt.show()
