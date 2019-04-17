#!/usr/bin/python
#\file    regression2c.py
#\brief   Chainer for 1-D -> (1-D, error) regression.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.18, 2015
import random,math

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

#TrueFunc= lambda x: 1.2+math.sin(x)
#TrueFunc= lambda x: 1.2+math.sin(3*x)
#TrueFunc= lambda x: 2.0*x**2
#TrueFunc= lambda x: 4.0-x if x>0.0 else 0.0
TrueFunc= lambda x: 4.0 if 0.0<x and x<2.5 else 0.0

#TEST: NN's estimation is bad where |x| is far from zero.
Bound= [-3.0,5.0]
#Bound= [-5.0,3.0]
#Bound= [-3.0,3.0]
#Bound= [1.0,5.0]
#Bound= [-5.0,-1.0]
#Bound= [-5.0,5.0]

def GenData(n=100, noise=0.3):
  #data_x= [[x+1.0*Rand()] for x in FRange1(*Bound,num_div=n)]
  data_x= [[Rand(*Bound)] for k in range(n)]
  data_y= [[TrueFunc(x[0])+noise*Rand()] for x in data_x]
  #data_y= [[TrueFunc(x[0])+(noise if abs(x[0])<2.0 else 0.0)*Rand()] for x in data_x]
  return data_x, data_y

NEpoch= 200  #TEST

def Main():
  import argparse
  import numpy as np
  from chainer import cuda, Variable, FunctionSet, optimizers
  import chainer.functions  as F
  from loss_for_error import loss_for_error1, loss_for_error2
  import six.moves.cPickle as pickle

  parser = argparse.ArgumentParser(description='Chainer example: regression')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                      help='GPU ID (negative value indicates CPU)')
  args = parser.parse_args()

  batchsize = 20
  n_epoch   = NEpoch
  n_units   = 200  #TEST

  # Prepare dataset
  data_x, data_y = GenData(200, noise=0.5)  #TEST: n samples, noise
  #batchsize= max(1,min(batchsize, len(data_y)/10))  #TEST: adjust batchsize
  batchsize= max(1,min(batchsize, len(data_y)))  #TEST: adjust batchsize
  #dx2,dy2=GenData(300, noise=0.0); data_x.extend(dx2); data_y.extend(dy2)
  data = np.array(data_x).astype(np.float32)
  target = np.array(data_y).astype(np.float32)

  N= len(data) #batchsize * 30
  x_train= data
  y_train= target
  y_err_train= np.array([[0.0]*y_train.shape[1]]*y_train.shape[0]).astype(np.float32)
  N_test= 50
  x_test= np.array([[x] for x in FRange1(*Bound,num_div=N_test)]).astype(np.float32)
  y_test= np.array([[TrueFunc(x[0])] for x in x_test]).astype(np.float32)
  y_err_test= np.array([[0.0] for x in x_test]).astype(np.float32)

  print 'Num of samples for train:',len(y_train)
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
  b2= [b for b in Bound]
  model.l1.b[:]= [Rand(*b2) for k in range(n_units)]
  model.l2.b[:]= [Rand(*b2) for k in range(n_units)]
  model.l3.b[:]= [Rand(*b2) for k in range(1)]
  # Error model
  model_err = FunctionSet(l1=F.Linear(1, n_units),
                          l2=F.Linear(n_units, n_units),
                          l3=F.Linear(n_units, 1))
  #TEST: Random bias initialization
  #b2= [b for b in Bound]
  b2= [0.0,0.5]
  model_err.l1.b[:]= [Rand(*b2) for k in range(n_units)]
  model_err.l2.b[:]= [Rand(*b2) for k in range(n_units)]
  model_err.l3.b[:]= [Rand(*b2) for k in range(1)]
  #TEST: load parameters from file:
  #model.copy_parameters_from(pickle.load(open('datak/reg2c_mean.dat', 'rb')))
  #model_err.copy_parameters_from(pickle.load(open('datak/reg2c_err.dat', 'rb')))
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
    #h1 = F.dropout(F.softplus(model.l1(x),beta=10.0),  ratio=dratio, train=train)
    #h2 = F.dropout(F.softplus(model.l2(h1),beta=10.0), ratio=dratio, train=train)
    #h1 = F.dropout(F.leaky_relu(model.l1(x),slope=0.2),  ratio=dratio, train=train)
    #h2 = F.dropout(F.leaky_relu(model.l2(h1),slope=0.2), ratio=dratio, train=train)
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

  # Setup optimizer
  optimizer = optimizers.AdaDelta(rho=0.9)
  #optimizer = optimizers.AdaGrad(lr=0.5)
  #optimizer = optimizers.RMSprop()
  #optimizer = optimizers.MomentumSGD()
  #optimizer = optimizers.SGD(lr=0.8)
  optimizer.setup(model.collect_parameters())

  optimizer_err = optimizers.AdaDelta(rho=0.9)
  optimizer_err.setup(model_err.collect_parameters())

  # Learning loop
  for epoch in xrange(1, n_epoch+1):
    print 'epoch', epoch

    # training
    perm = np.random.permutation(N)
    sum_loss = 0

    # Train model
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

      #D= y_train.shape[1]
      #y_err_train[perm[i:i+batchsize]] = np.array([[(y1[d]-y2[d])**2 for d in range(D)] for y1,y2 in zip(cuda.to_cpu(pred.data), y_train[perm[i:i+batchsize]]) ]).astype(np.float32)

      sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

    print 'train mean loss={}'.format(sum_loss / N)

    # Generate training data for error model
    preds = []
    x_batch = x_train[:]
    y_batch = y_train[:]
    if args.gpu >= 0:
      x_batch = cuda.to_gpu(x_batch)
      y_batch = cuda.to_gpu(y_batch)
    loss, pred = forward(x_batch, y_batch, train=False)
    D= y_train.shape[1]
    #y_err_train = np.array([[abs(y2[d]-y1[d]) for d in range(D)] for y1,y2 in zip(cuda.to_cpu(pred.data), y_train) ]).astype(np.float32)
    y_err_train = np.abs(cuda.to_cpu(pred.data) - y_train)


    # Learning loop
    if epoch in (5,99,NEpoch,):
      for epoch2 in xrange(1, 200+1):
        print '--epoch2', epoch2

        # training
        perm = np.random.permutation(N)
        sum_loss = 0

        # Train error model
        for i in xrange(0, N, batchsize):
          x_batch = x_train[perm[i:i+batchsize]]
          y_batch = y_err_train[perm[i:i+batchsize]]
          if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

          optimizer_err.zero_grads()
          loss, pred = forward_err(x_batch, y_batch)
          loss.backward()  #Computing gradients
          optimizer_err.update()

          sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

        print 'train(error) mean loss={}'.format(sum_loss / N)


    #'''
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
    #'''

    print 'test mean loss={}, corrcoef={}, error loss={}'.format(
        sum_loss / N_test, pearson[0][1], sum_loss_err / N_test)

    # Dump data for plot:
    fp1= file('/tmp/nn_test%04i.dat'%epoch,'w')
    for x,y,yerr in zip(x_test,preds,preds_err):
      fp1.write('%s #%i# %s %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y)),' '.join(map(str,yerr))))
    fp1.close()

    fp1= file('/tmp/smpl_train2.dat','w')
    for x,y,yerr in zip(x_train,y_train,y_err_train):
      fp1.write('%s #%i# %s %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y)),' '.join(map(str,yerr))))
    fp1.close()

  #Save parameters
  pickle.dump(model.parameters, open('datak/reg2c_mean.dat', 'wb'), -1)
  pickle.dump(model_err.parameters, open('datak/reg2c_err.dat', 'wb'), -1)


def PlotGraphs():
  print 'Plotting graphs..'
  import os,sys
  opt= sys.argv[2:]
  commands=[
    '''qplot -x2 aaa {opt} -s 'set xlabel "x";set ylabel "y";'
          -cs 'u 1:3' /tmp/smpl_train.dat pt 6 ps 2 t '"sample"'
          /tmp/smpl_test.dat w l lw 3 t '"true"'
          /tmp/nn_test0001.dat w l t '"1st epoch"'
          /tmp/nn_test0005.dat w l t '"5th epoch"'
          /tmp/nn_test0020.dat w l t '"20th epoch"'
          /tmp/nn_test0050.dat w l t '"50th epoch"'
          /tmp/nn_test0075.dat w l t '"75th epoch"'
          /tmp/nn_test0099.dat w l t '"99th epoch"'
          /tmp/nn_test{NEpoch:04d}.dat w l lw 3 t '"Final({NEpoch}) epoch"' &''',
    '''qplot -x2 aaa {opt} -s 'set xlabel "x";set ylabel "y";'
          /tmp/smpl_train2.dat u 1:3:4 w yerrorbar pt 6 lw 2 ps 2 t '"sample"'
          /tmp/smpl_test.dat u 1:3 w l lw 3 t '"true"'
          -cs 'u 1:3:4'
          /tmp/nn_test0005.dat w yerrorbar t '"5st epoch"'
          /tmp/nn_test0099.dat w yerrorbar t '"99th epoch"'
          /tmp/nn_test{NEpoch:04d}.dat w yerrorbar lw 3 t '"Final({NEpoch}) epoch"' &''',
    '''''',
    '''qplot -x2 aaa {opt} -s 'set xlabel "x";set ylabel "y";'
          /tmp/smpl_train2.dat u 1:4 w p pt 6 lw 2 ps 2 t '"sample"'
          , '0.0' w l t '"true"'
          -cs 'u 1:4'
          /tmp/nn_test0005.dat w l t '"5st epoch"'
          /tmp/nn_test0099.dat w l t '"99th epoch"'
          /tmp/nn_test{NEpoch:04d}.dat w l lw 3 t '"Final({NEpoch}) epoch"' &''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.format(opt=' '.join(opt),NEpoch=NEpoch).splitlines())
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
