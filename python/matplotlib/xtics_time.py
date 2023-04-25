#!/usr/bin/python
#\file    xtics_time.py
#\brief   Showing x major tics with elapsed time (mm:ss) format.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.25, 2023
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__=='__main__':
  fig= plt.figure()
  ax= fig.add_subplot(1,1,1,title='Test',xlabel='time[mm:ss.ss]',ylabel='y')

  X= np.linspace(0,300,20)
  ax.scatter(X, 2.0*X+100*(0.5-np.random.uniform(size=len(X))), color='blue', label='random')
  X= np.linspace(0,300,1000)
  ax.plot(X, 2.0*X, color='blue', linewidth=3, label='linear')

  ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos:'{:02}:{:05.2f}'.format(int(x//60),x%60)))
  ax.xaxis.set_major_locator(ticker.MultipleLocator(90)) #Each tick per 90 seconds
  ax.legend()

  plt.show()
