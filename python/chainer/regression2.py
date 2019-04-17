#!/usr/bin/python
#\file    regression2.py
#\brief   Chainer for 1-D regression.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.04, 2015
import random,math

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

TrueFunc= lambda x: 1.2+math.sin(x)
#TrueFunc= lambda x: 2.0*x**2
#TrueFunc= lambda x: 4.0-x if x>0.0 else 0.0

def GenData(n=100, noise=0.3):
  #data_x= [[x+1.0*Rand()] for x in FRange1(-3.0,5.0,n)]
  data_x= [[Rand(-3.0,5.0)] for k in range(n)]
  data_y= [[TrueFunc(x[0])+noise*Rand()] for x in data_x]
  return data_x, data_y

def Main():
  import argparse
  import numpy as np
  from chainer import cuda, Variable, FunctionSet, optimizers
  import chainer.functions  as F

  parser = argparse.ArgumentParser(description='Chainer example: regression')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                      help='GPU ID (negative value indicates CPU)')
  args = parser.parse_args()

  batchsize = 10
  n_epoch   = 100
  n_units   = 50

  # Prepare dataset
  n_test= 300
  data_x, data_y = GenData(n_test+300, noise=0.3)  #TEST: n samples, noise
  #dx2,dy2=GenData(300, noise=0.0); data_x.extend(dx2); data_y.extend(dy2)
  data = np.array(data_x).astype(np.float32)
  target = np.array(data_y).astype(np.float32).reshape(len(data_y), 1)

  N = min(batchsize * 30, len(data_x)-n_test)  #Number of training data
  x_train, x_test = np.split(data, [N])
  y_train, y_test = np.split(target, [N])
  N_test = y_test.size

  print 'Num of samples for train:',len(y_train)
  print 'Num of samples for test:',len(y_test)
  # Dump data for plot:
  fp1= file('/tmp/smpl_train.dat','w')
  for x,y in zip(x_train,y_train):
    fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y))))
  fp1.close()
  # Dump data for plot:
  fp1= file('/tmp/smpl_test.dat','w')
  for x,y in zip(x_test,y_test):
    fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y))))
  fp1.close()

  # Prepare multi-layer perceptron model
  model = FunctionSet(l1=F.Linear(1, n_units),
                      l2=F.Linear(n_units, n_units),
                      l3=F.Linear(n_units, 1))
  #TEST: Random bias initialization
  #, bias=Rand()
  model.l1.b[:]= [Rand() for k in range(n_units)]
  model.l2.b[:]= [Rand() for k in range(n_units)]
  model.l3.b[:]= [Rand() for k in range(1)]
  #print model.l2.__dict__
  if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

  # Neural net architecture
  def forward(x_data, y_data, train=True):
    #train= False  #TEST: Turn off dropout
    dratio= 0.0  #0.5  #TEST: Dropout ratio
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    #h1 = F.dropout(F.leaky_relu(model.l1(x),slope=-0.2),  ratio=dratio, train=train)
    #h2 = F.dropout(F.leaky_relu(model.l2(h1),slope=-0.2), ratio=dratio, train=train)
    #h1 = F.dropout(F.sigmoid(model.l1(x)),  ratio=dratio, train=train)
    #h2 = F.dropout(F.sigmoid(model.l2(h1)), ratio=dratio, train=train)
    #h1 = F.dropout(F.tanh(model.l1(x)),  ratio=dratio, train=train)
    #h2 = F.dropout(F.tanh(model.l2(h1)), ratio=dratio, train=train)
    #h1 = F.dropout(model.l1(x),  ratio=dratio, train=train)
    #h2 = F.dropout(model.l2(h1), ratio=dratio, train=train)
    #h1 = F.relu(model.l1(x))
    #h2 = F.relu(model.l2(h1))
    #h1 = model.l1(x)
    #h2 = model.l2(h1)
    y  = model.l3(h2)
    return F.mean_squared_error(y, t), y

  # Setup optimizer
  optimizer = optimizers.AdaDelta(rho=0.9)
  #optimizer = optimizers.AdaGrad(lr=0.5)
  #optimizer = optimizers.RMSprop()
  #optimizer = optimizers.MomentumSGD()
  #optimizer = optimizers.SGD(lr=0.8)
  optimizer.setup(model.collect_parameters())

  # Learning loop
  for epoch in xrange(1, n_epoch+1):
    print 'epoch', epoch

    # training
    perm = np.random.permutation(N)
    sum_loss = 0

    for i in xrange(0, N, batchsize):
      x_batch = x_train[perm[i:i+batchsize]]
      y_batch = y_train[perm[i:i+batchsize]]
      if args.gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)
        y_batch = cuda.to_gpu(y_batch)

      optimizer.zero_grads()
      loss, pred = forward(x_batch, y_batch)
      loss.backward()  #Computing gradients
      optimizer.update()

      sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

    print 'train mean loss={}'.format(
        sum_loss / N)

    '''
    # testing per batch
    sum_loss     = 0
    preds = []
    for i in xrange(0, N_test, batchsize):
      x_batch = x_test[i:i+batchsize]
      y_batch = y_test[i:i+batchsize]
      if args.gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)
        y_batch = cuda.to_gpu(y_batch)

      loss, pred = forward(x_batch, y_batch, train=False)
      preds.extend(cuda.to_cpu(pred.data))
      sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
    pearson = np.corrcoef(np.asarray(preds).reshape(len(preds),), np.asarray(y_test).reshape(len(preds),))
    #'''

    #'''
    # testing all data
    preds = []
    x_batch = x_test[:]
    y_batch = y_test[:]
    if args.gpu >= 0:
      x_batch = cuda.to_gpu(x_batch)
      y_batch = cuda.to_gpu(y_batch)
    loss, pred = forward(x_batch, y_batch, train=False)
    preds = cuda.to_cpu(pred.data)
    sum_loss = float(cuda.to_cpu(loss.data)) * len(y_test)
    pearson = np.corrcoef(np.asarray(preds).reshape(len(preds),), np.asarray(y_test).reshape(len(preds),))
    #'''

    print 'test  mean loss={}, corrcoef={}'.format(
        sum_loss / N_test, pearson[0][1])

    # Dump data for plot:
    fp1= file('/tmp/nn_test%04i.dat'%epoch,'w')
    for x,y in zip(x_test,preds):
      fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y))))
    fp1.close()


def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa -cs 'u 1:3' /tmp/smpl_train.dat /tmp/smpl_test.dat
          /tmp/nn_test0001.dat
          /tmp/nn_test0005.dat
          /tmp/nn_test0020.dat
          /tmp/nn_test0050.dat
          /tmp/nn_test0075.dat
          /tmp/nn_test0100.dat &''',
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
