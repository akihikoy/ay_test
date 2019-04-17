#!/usr/bin/python
#\file    regression2d.py
#\brief   Expectation and covariance of NN.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.21, 2015
import random,math
from regression2c import FRange1, Rand, TrueFunc, Bound, GenData


def Main():
  import argparse
  import numpy as np
  from chainer import cuda, Variable, FunctionSet, optimizers, utils
  import chainer.functions  as F
  from loss_for_error import loss_for_error1, loss_for_error2
  import six.moves.cPickle as pickle

  parser = argparse.ArgumentParser(description='Chainer example: regression')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                      help='GPU ID (negative value indicates CPU)')
  args = parser.parse_args()

  n_units   = 200  #NOTE: should be the same as ones in regression2c

  N_test= 100
  x_test= np.array([[x] for x in FRange1(*Bound,num_div=N_test)]).astype(np.float32)
  y_test= np.array([[TrueFunc(x[0])] for x in x_test]).astype(np.float32)
  y_err_test= np.array([[0.0] for x in x_test]).astype(np.float32)

  # Dump data for plot:
  fp1= file('/tmp/smpl_test.dat','w')
  for x,y in zip(x_test,y_test):
    fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y))))
  fp1.close()

  # Prepare multi-layer perceptron model
  model = FunctionSet(l1=F.Linear(1, n_units),
                      l2=F.Linear(n_units, n_units),
                      l3=F.Linear(n_units, 1))
  # Error model
  model_err = FunctionSet(l1=F.Linear(1, n_units),
                          l2=F.Linear(n_units, n_units),
                          l3=F.Linear(n_units, 1))
  # Load parameters from file:
  model.copy_parameters_from(pickle.load(open('datak/reg2c_mean.dat', 'rb')))
  model_err.copy_parameters_from(pickle.load(open('datak/reg2c_err.dat', 'rb')))
  #model.copy_parameters_from(map(lambda e:np.array(e,np.float32),pickle.load(open('/tmp/nn_model.dat', 'rb')) ))
  #model_err.copy_parameters_from(map(lambda e:np.array(e,np.float32),pickle.load(open('/tmp/nn_model_err.dat', 'rb')) ))
  if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()
    model_err.to_gpu()

  # Neural net architecture
  def forward(x_data, y_data, train=True):
    #train= False  #TEST: Turn off dropout
    dratio= 0.2  #0.5  #TEST: Dropout ratio
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    y  = model.l3(h2)
    return F.mean_squared_error(y, t), y

  # Neural net architecture
  def forward_err(x_data, y_data, train=True):
    #train= False  #TEST: Turn off dropout
    dratio= 0.2  #0.5  #TEST: Dropout ratio
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model_err.l1(x)),  ratio=dratio, train=train)
    h2 = F.dropout(F.relu(model_err.l2(h1)), ratio=dratio, train=train)
    y  = model_err.l3(h2)
    #return F.mean_squared_error(y, t), y
    #return loss_for_error1(y, t, 0.1), y  #TEST
    return loss_for_error2(y, t, 0.1), y  #TEST

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
        print msg
        raise Exception(msg)
    return cast(mu_out), cast(var_out)

  relu_gaussv= np.vectorize(relu_gauss)  #Vector version

  #Gradient of ReLU whose input is normal distribution variable.
  #  mu: mean, var: variance (square of std-dev).
  #  cut_sd: if abs(mu)>cut_sd*sigma, an approximation is used.  Set None to disable this.
  def relu_gauss_grad(mu, var, epsilon=1.0e-6, cut_sd=4.0):
    cast= type(mu)
    sigma= math.sqrt(var)
    if sigma<epsilon:  return cast(1.0 if mu>0.0 else 0.0)
    #Approximation to speedup for abs(mu)>cut_sd*sigma.
    if cut_sd!=None and mu>cut_sd*sigma:   return cast(1.0)
    if cut_sd!=None and mu<-cut_sd*sigma:  return cast(0.0)
    sqrt2= math.sqrt(2.0)
    z= mu/(sqrt2*sigma)
    return cast(0.5*(1.0+math.erf(z)))

  relu_gauss_gradv= np.vectorize(relu_gauss_grad)  #Vector version


  #Forward computation of neural net considering input distribution.
  def forward_x(x, x_var=None):
    zero= np.float32(0)
    x= np.array(x,np.float32); x= x.reshape(x.size,1)

    #Error model:
    h0= x
    for l in (model_err.l1, model_err.l2):
      hl1= l.W.dot(h0) + l.b.reshape(l.b.size,1)  #W h0 + b
      h1= np.maximum(zero, hl1)  #ReLU(hl1)
      h0= h1
    l= model_err.l3
    y_err0= l.W.dot(h0) + l.b.reshape(l.b.size,1)
    y_var0= np.diag((y_err0*y_err0).ravel())

    if x_var in (0.0, None):
      g= None  #Gradient
      h0= x
      for l in (model.l1, model.l2):
        hl1= l.W.dot(h0) + l.b.reshape(l.b.size,1)  #W h0 + b
        h1= np.maximum(zero, hl1)  #ReLU(hl1)
        g2= l.W.T.dot(np.diag((hl1>0.0).ravel().astype(np.float32)))  #W diag(step(hl1))
        g= g2 if g==None else g.dot(g2)
        h0= h1
      l= model.l3
      y= l.W.dot(h0) + l.b.reshape(l.b.size,1)
      g= g2 if g==None else g.dot(l.W.T)
      return y, y_var0, g

    else:
      if isinstance(x_var, (float, np.float_, np.float16, np.float32, np.float64)):
        x_var= np.diag(np.array([x_var]*x.size).astype(np.float32))
      elif x_var.size==x.size:
        x_var= np.diag(np.array(x_var.ravel(),np.float32))
      else:
        x_var= np.array(x_var,np.float32); x_var= x_var.reshape(x.size,x.size)
      g= None  #Gradient
      h0= x
      h0_var= x_var
      for l in (model.l1, model.l2):
        hl1= l.W.dot(h0) + l.b.reshape(l.b.size,1)  #W h0 + b
        #print 'l.W',l.W.shape
        #print 'h0_var',h0_var.shape
        hl1_dvar= np.diag( l.W.dot(h0_var.dot(l.W.T)) ).reshape(hl1.size,1)  #diag(W h0_var W^T)
        #print 'hl1',hl1.shape
        #print 'hl1_dvar',hl1_dvar.shape
        h1,h1_dvar= relu_gaussv(hl1,hl1_dvar)  #ReLU_gauss(hl1,hl1_dvar)
        #print 'h1_dvar',h1_dvar.shape
        h1_var= np.diag(h1_dvar.ravel())  #To a full matrix
        #print 'h1_var',h1_var.shape
        #print 'relu_gauss_gradv(hl1,hl1_dvar)',relu_gauss_gradv(hl1,hl1_dvar).shape
        g2= l.W.T.dot(np.diag(relu_gauss_gradv(hl1,hl1_dvar).ravel()))
        g= g2 if g==None else g.dot(g2)
        h0= h1
        h0_var= h1_var
      l= model.l3
      y= l.W.dot(h0) + l.b.reshape(l.b.size,1)
      y_var= l.W.dot(h0_var.dot(l.W.T))
      g= g2 if g==None else g.dot(l.W.T)
      return y, y_var+y_var0, g

  '''
  # testing all data
  preds = []
  x_batch = x_test[:]
  y_batch = y_test[:]
  y_err_batch = y_err_test[:]
  if args.gpu >= 0:
    x_batch = cuda.to_gpu(x_batch)
    y_batch = cuda.to_gpu(y_batch)
    y_err_batch = cuda.to_gpu(y_err_batch)
  loss, pred = forward(x_batch, y_batch, train=False)
  loss_err, pred_err = forward_err(x_batch, y_err_batch, train=False)
  preds = cuda.to_cpu(pred.data)
  preds_err = cuda.to_cpu(pred_err.data)
  sum_loss = float(cuda.to_cpu(loss.data)) * len(y_test)
  sum_loss_err = float(cuda.to_cpu(loss_err.data)) * len(y_test)
  pearson = np.corrcoef(np.asarray(preds).reshape(len(preds),), np.asarray(y_test).reshape(len(preds),))

  print 'test mean loss={}, corrcoef={}, error loss={}'.format(
      sum_loss / N_test, pearson[0][1], sum_loss_err / N_test)

  # Dump data for plot:
  fp1= file('/tmp/nn_test0001.dat','w')
  for x,y,yerr in zip(x_test,preds,preds_err):
    fp1.write('%s #%i# %s %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y)),' '.join(map(str,yerr))))
  fp1.close()
  #'''

  # Dump data for plot:
  fp1= file('/tmp/nn_test0001.dat','w')
  for x in x_test:
    y, var, g= forward_x(x, 0.0)
    y, var, g= y.ravel(), var.ravel(), g.ravel()
    yerr= np.sqrt(var)
    fp1.write('%s %s %s %s\n' % (' '.join(map(str,x)),' '.join(map(str,y)),' '.join(map(str,yerr)),' '.join(map(str,g))))
  fp1.close()

  # Dump data for plot:
  fp1= file('/tmp/nn_test0002.dat','w')
  for x in x_test:
    y, var, g= forward_x(x, 0.5**2)
    y, var, g= y.ravel(), var.ravel(), g.ravel()
    yerr= np.sqrt(var)
    fp1.write('%s %s %s %s\n' % (' '.join(map(str,x)),' '.join(map(str,y)),' '.join(map(str,yerr)),' '.join(map(str,g))))
  fp1.close()


def PlotGraphs():
  print 'Plotting graphs..'
  import os,sys
  opt= sys.argv[2:]
  commands=[
    '''qplot -x2 aaa {opt} -s 'set xlabel "x";set ylabel "y";set key right top'
          /tmp/smpl_test.dat u 1:3 w l lw 3 t '"true"'
          /tmp/nn_test0001.dat u 1:2   w l                       lw 3 lt 4 t '"Final epoch(x/sd=0.0)"'
          /tmp/nn_test0001.dat u 1:2:3 w yerrorbar               lw 2 lt 4 t '"+/- 1 SD"'
          /tmp/nn_test0001.dat u 1:2:'(0.1)':'(0.1*$4)' w vector lw 2 lt 5 t '"Gradient(x/sd=0.0)"'
          /tmp/nn_test0002.dat u 1:2   w l                       lw 3 lt 2 t '"Final epoch(x/sd=0.5)"'
          /tmp/nn_test0002.dat u 1:2:3 w yerrorbar               lw 2 lt 2 t '"+/- 1 SD"'
          /tmp/nn_test0002.dat u 1:2:'(0.1)':'(0.1*$4)' w vector lw 2 lt 3 t '"Gradient(x/sd=0.5)"'
          /tmp/smpl_train.dat u 1:3 w p pt 6 ps 2 t '"sample"'
    &''',
          #/tmp/nn_test0001.dat u 1:2:3 w yerrorbar lt 3 lw 2 t '""'
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.format(opt=' '.join(opt)).splitlines())
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
