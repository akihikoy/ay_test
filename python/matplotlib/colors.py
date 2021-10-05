#!/usr/bin/python
#\file    colors.py
#\brief   Explore Matplotlib colors.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.02, 2021
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_cols

if __name__=='__main__':
  fig= plt.figure(figsize=(8,8))
  ax= fig.add_subplot(1,1,1)

  for i in range(0,10):
    X= np.linspace(0.1,5,200)
    ax.plot(X, (1+i)*np.log(X), linewidth=1, label='Default[{}]'.format(i))
  for i in range(0,10):
    X= np.linspace(-0.1,-5,200)
    ax.plot(X, (1+i)*np.log(-X), color=plt_cols.BASE_COLORS.values()[i%len(plt_cols.BASE_COLORS)], linewidth=1, linestyle='dashdot', label='BASE_COLORS[{}]'.format(i))
  for i in range(0,10):
    X= np.linspace(0.1,5,200)
    ax.plot(X, -(1+i)*np.log(X), color=plt_cols.TABLEAU_COLORS.values()[i%len(plt_cols.TABLEAU_COLORS)], linewidth=1, linestyle='dashed',  label='TABLEAU_COLORS[{}]'.format(i))
  for i in range(0,10):
    X= np.linspace(-0.1,-5,200)
    ax.plot(X, -(1+i)*np.log(-X), color=plt_cols.CSS4_COLORS.values()[i%len(plt_cols.CSS4_COLORS)], linewidth=1, linestyle='solid',  label='CSS4_COLORS[{}]'.format(i))

  fig2= plt.figure(figsize=(8,8))
  ax2= fig2.add_subplot(1,1,1)
  sorted_css4= [c_rgb for c_hsv,c_rgb in sorted((tuple(plt_cols.rgb_to_hsv(plt_cols.to_rgb(c))), c) for c in plt_cols.CSS4_COLORS.itervalues())]
  for i in range(0,10):
    X= np.linspace(0.1,5,200)
    ax2.plot(X, (1+i)*np.log(X), color=sorted_css4[i%len(sorted_css4)], linewidth=1, linestyle='solid',  label='sorted_css4[{}]'.format(i))
  for i in range(0,10):
    X= np.linspace(-0.1,-5,200)
    ax2.plot(X, (1+i)*np.log(-X), color=sorted_css4[(i+14)%len(sorted_css4)], linewidth=1, linestyle='solid',  label='sorted_css4[{}]'.format(i+14))
  for i in range(0,10):
    X= np.linspace(0.1,5,200)
    ax2.plot(X, -(1+i)*np.log(X), color=sorted_css4[(i+67)%len(sorted_css4)], linewidth=1, linestyle='solid',  label='sorted_css4[{}]'.format(i+67))
  for i in range(0,10):
    X= np.linspace(-0.1,-5,200)
    ax2.plot(X, -(1+i)*np.log(-X), color=sorted_css4[(i+100)%len(sorted_css4)], linewidth=1, linestyle='solid',  label='sorted_css4[{}]'.format(i))

  fig3= plt.figure(figsize=(8,8))
  ax3= fig3.add_subplot(1,1,1)
  grad= {'blue':[plt_cols.hsv_to_rgb((0.56823266, s, 0.9)) for s in np.linspace(1.0,0.3,5)],
         'red':[plt_cols.hsv_to_rgb((0.99904762, s, 0.9)) for s in np.linspace(1.0,0.3,5)],
         'green':[plt_cols.hsv_to_rgb((0.33333333, s, 0.9)) for s in np.linspace(1.0,0.3,5)],
         'orange':[plt_cols.hsv_to_rgb((0.07814661, s, 0.9)) for s in np.linspace(1.0,0.3,5)]}
  for i in range(0,10):
    X= np.linspace(0.1,5,200)
    ax3.plot(X, (1+i)*np.log(X), color=grad['blue'][i%len(grad['blue'])], linewidth=1, linestyle='solid',  label='grad["blue"][{}]'.format(i))
  for i in range(0,10):
    X= np.linspace(-0.1,-5,200)
    ax3.plot(X, (1+i)*np.log(-X), color=grad['red'][i%len(grad['red'])], linewidth=1, linestyle='solid',  label='grad["red"][{}]'.format(i+14))
  for i in range(0,10):
    X= np.linspace(0.1,5,200)
    ax3.plot(X, -(1+i)*np.log(X), color=grad['green'][i%len(grad['green'])], linewidth=1, linestyle='solid',  label='grad["green"][{}]'.format(i+67))
  for i in range(0,10):
    X= np.linspace(-0.1,-5,200)
    ax3.plot(X, -(1+i)*np.log(-X), color=grad['orange'][i%len(grad['orange'])], linewidth=1, linestyle='solid',  label='grad["orange"][{}]'.format(i))

  box= ax.get_position()
  ax.set_position([box.x0, box.y0, box.width*0.85, box.height])
  ax.set_title('Colors-1')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=5)
  box= ax2.get_position()
  ax2.set_position([box.x0, box.y0, box.width*0.85, box.height])
  ax2.set_title('Colors-2')
  ax2.set_xlabel('x')
  ax2.set_ylabel('y')
  ax2.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=5)
  box= ax3.get_position()
  ax3.set_position([box.x0, box.y0, box.width*0.85, box.height])
  ax3.set_title('Colors-3')
  ax3.set_xlabel('x')
  ax3.set_ylabel('y')
  ax3.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=5)
  plt.rcParams['keymap.quit'].append('q')
  plt.show()
