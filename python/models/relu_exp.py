#!/usr/bin/python3
#\file    relu_exp.py
#\brief   Expectation of ReLU (rectified linear unit).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.17, 2015

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Main():
  import math
  data_x= FRange1(-5.0, 5.0, 100)

  relu= lambda x: max(0.0, x)

  #ReLU whose input is normal distribution variable.
  #  mu: mean, var: variance (square of std-dev).
  #  cut_sd: if abs(mu)>cut_sd*sigma, an approximation is used.  Set None to disable this.
  def relu_gauss(mu, var, epsilon=1.0e-6, cut_sd=4.0):
    cast= type(mu)
    sigma= math.sqrt(var)
    if sigma<epsilon:  return cast(max(0.0,mu)), cast(0.0)
    #Approximation to speedup for abs(mu)>cut_sd*sigma.
    if cut_sd!=None and mu>cut_sd*sigma:   return cast(mu), cast(var)
    if cut_sd!=None and mu<-cut_sd*sigma:  return cast(0.0), cast(0.0)
    sqrt2= math.sqrt(2.0)
    sqrt2pi= math.sqrt(2.0*math.pi)
    z= mu/(sqrt2*sigma)
    E= math.erf(z)
    X= math.exp(-z*z)
    mu_out= sigma/sqrt2pi*X + mu/2.0*(1.0+E)
    var_out= (1.0+E)/4.0*(mu*mu*(1.0-E)+2.0*var) - sigma*X/sqrt2pi*(sigma*X/sqrt2pi+mu*E)
    if var_out<0.0:
      if var_out>-epsilon:  return mu_out, 0.0
      else:
        msg= 'ERROR in relu_gauss: %f, %f, %f, %f'%(mu, sigma, mu_out, var_out)
        print(msg)
        raise Exception(msg)
    return cast(mu_out), cast(var_out)

  fp= open('/tmp/relu.dat','w')
  for x in data_x:
    fp.write('%f %f\n' % (x, relu(x)))
  fp.close()

  var_in= 1.0**2
  fp= open('/tmp/relu_exp100.dat','w')
  for x in data_x:
    expec,var= relu_gauss(x,var_in)
    fp.write('%f %f %f\n' % (x, expec, math.sqrt(var)))
  fp.close()

  var_in= 0.5**2
  fp= open('/tmp/relu_exp050.dat','w')
  for x in data_x:
    expec,var= relu_gauss(x,var_in)
    fp.write('%f %f %f\n' % (x, expec, math.sqrt(var)))
  fp.close()

  var_in= 0.01**2
  fp= open('/tmp/relu_exp001.dat','w')
  for x in data_x:
    expec,var= relu_gauss(x,var_in)
    fp.write('%f %f %f\n' % (x, expec, math.sqrt(var)))
  fp.close()


  #Comparison with numerical expectation

  from num_expec_g import NumExpec2
  import numpy as np
  relu_numexp= lambda x,sigma: NumExpec2(relu, x, sigma*sigma)

  data_x= FRange1(-5.0, 5.0, 40)

  sigma= 1.0
  fp= open('/tmp/relu_numexp100.dat','w')
  for x in data_x:
    fp.write('%f %f\n' % (x, relu_numexp(x,sigma)))
  fp.close()

  sigma= 0.5
  fp= open('/tmp/relu_numexp050.dat','w')
  for x in data_x:
    fp.write('%f %f\n' % (x, relu_numexp(x,sigma)))
  fp.close()

  sigma= 0.01
  fp= open('/tmp/relu_numexp001.dat','w')
  for x in data_x:
    fp.write('%f %f\n' % (x, relu_numexp(x,sigma)))
  fp.close()


def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa
        '/tmp/relu.dat' w l t '"ReLU"'
        '/tmp/relu_exp100.dat' w yerrorbar t '"E[ReLU]; sigma=1.00"'
        '/tmp/relu_exp050.dat' w yerrorbar t '"E[ReLU]; sigma=0.50"'
        '/tmp/relu_exp001.dat' w yerrorbar t '"E[ReLU]; sigma=0.01"'
        '/tmp/relu_numexp100.dat' w p t '"NumE[ReLU]; sigma=1.00"'
        '/tmp/relu_numexp050.dat' w p t '"NumE[ReLU]; sigma=0.50"'
        '/tmp/relu_numexp001.dat' w p t '"NumE[ReLU]; sigma=0.01"' &''',
    '''qplot -x2 aaa
        -s 'set xrange [-5:5]; set xlabel "x"; set ylabel "y";'
        -s 'set size ratio 0.5;'
        '/tmp/relu.dat' w l lw 2 t '"ReLU"'
        '/tmp/relu_exp100.dat' w yerrorbar pt 0 lt 3 t '"E[ReLU]"'
        '/tmp/relu_exp100.dat' w l         lw 2 lt 3 t '""'
        '/tmp/relu_numexp100.dat' w p pt 6 ps 1.5 lt 2 t '"NumE[ReLU]"'
        &''',
        #-o res/relu_exp.svg
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print('###',cmd)
      os.system(cmd)

  print('##########################')
  print('###Press enter to close###')
  print('##########################')
  input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
