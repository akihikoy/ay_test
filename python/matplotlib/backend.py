#!/usr/bin/python
#\file    backend.py
#\brief   Test of GUI backend of matplotlib.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.24, 2023
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

if __name__=='__main__':
  gui_envs= ['TKAgg','GTKAgg','Qt4Agg','WXAgg','Agg']
  gui_env= sys.argv[1] if len(sys.argv)>1 else 'TKAgg'
  if gui_env not in gui_envs:
    raise Exception('Invalid GUI env:',gui_env,'; available:',gui_envs)

  print 'Current backend:',matplotlib.get_backend()
  matplotlib.use(gui_env, warn=False, force=True)
  import matplotlib.pyplot as plt
  print 'Current backend:',matplotlib.get_backend()

  fig= plt.figure()
  ax= fig.add_subplot(1,1,1,title='Test',xlabel='x',ylabel='y')

  X= np.linspace(0,300,20)
  ax.scatter(X, 2.0*X+100*(0.5-np.random.uniform(size=len(X))), color='blue', label='random')
  X= np.linspace(0,300,1000)
  ax.plot(X, 2.0*X, color='blue', linewidth=3, label='linear')

  plt.show()
