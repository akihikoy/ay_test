#!/usr/bin/python
#\file    exp_maf.py
#\brief   Exponential moving average filter.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.20, 2015
import math

#Exponential moving average filter for one-dimensional variable.
class TExpMovingAverage1(object):
  #mean: initial mean. If None, the first value is used.
  #init_sd: initial standard deviation.
  #alpha: weight of new value.
  def __init__(self, mean=None, init_sd=0.0, alpha=0.5):
    self.Mean= mean
    self.SqMean= None
    self.InitSD= init_sd
    self.Alpha= alpha
    self.sd_= None

  def Update(self, value):
    if self.Mean==None:  self.Mean= value
    else:  self.Mean= self.Alpha*value + (1.0-self.Alpha)*self.Mean
    if self.SqMean==None:  self.SqMean= self.InitSD*self.InitSD + self.Mean*self.Mean
    else:  self.SqMean= self.Alpha*(value*value) + (1.0-self.Alpha)*self.SqMean
    self.sd_= None

  @property
  def StdDev(self):
    if self.sd_==None:  self.sd_= math.sqrt(max(0.0,self.SqMean-self.Mean*self.Mean))
    return self.sd_


def Main():
  import random
  def Rand(xmin=-0.5,xmax=0.5):
    return random.random()*(xmax-xmin)+xmin

  exp_maf= TExpMovingAverage1(init_sd=0.1)
  fp= open('/tmp/exp_maf1.dat','w')
  for n in range(20):
    x= 0.2/(n+1.0) + Rand()*0.03
    exp_maf.Update(x)
    fp.write('%f %f %f %f\n' % (n, x, exp_maf.Mean, exp_maf.StdDev))
  fp.close()

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa
        '/tmp/exp_maf1.dat' u 1:3 w l t '"Mean"'
        '/tmp/exp_maf1.dat' u 1:3:4 w yerrorbar t '"Mean+/-SD"'
        '/tmp/exp_maf1.dat' u 1:2 w p t '"Samples"' &''',
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print '###',cmd
      os.system(cmd)

  print '##########################'
  print '###Press enter to close###'
  print '##########################'
  raw_input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
