#!/usr/bin/python
#\file    regression3.py
#\brief   Chainer for 2-D -> 1-D regression.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.04, 2015
import random,math

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

#TrueFunc= lambda x: 1.2+math.sin(2.0*(x[0]+x[1]))
#TrueFunc= lambda x: 0.1*(x[0]*x[0]+x[1]*x[1])
#TrueFunc= lambda x: 4.0-x[0]*x[1]
TrueFunc= lambda x: 4.0-x[0]-x[1] if x[0]**2+x[1]**2<2.0 else 0.0

def GenData(n=100, noise=0.3):
  #data_x= [[x+1.0*Rand()] for x in FRange1(-3.0,5.0,n)]
  data_x= [[Rand(-3.0,3.0), Rand(-3.0,3.0)] for k in range(n)]
  data_y= [[TrueFunc(x)+noise*Rand()] for x in data_x]
  return data_x, data_y

NEpoch= 100  #TEST

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
  n_epoch   = NEpoch
  n_units   = 200  #TEST

  # Prepare dataset
  data_x, data_y = GenData(20, noise=0.0)  #TEST: n samples, noise
  batchsize= max(1,min(batchsize, len(data_y)/20))  #TEST: adjust batchsize
  #dx2,dy2=GenData(300, noise=0.0); data_x.extend(dx2); data_y.extend(dy2)
  data = np.array(data_x).astype(np.float32)
  target = np.array(data_y).astype(np.float32)

  N= len(data) #batchsize * 30
  x_train= data
  y_train= target
  nt= 25
  N_test= nt*nt
  x_test= np.array(sum([[[x1,x2] for x2 in FRange1(-3.0,3.0,nt)] for x1 in FRange1(-3.0,3.0,nt)],[])).astype(np.float32)
  y_test= np.array([[TrueFunc(x)] for x in x_test]).astype(np.float32)

  print 'Num of samples for train:',len(y_train)
  # Dump data for plot:
  fp1= file('/tmp/smpl_train.dat','w')
  for x,y in zip(x_train,y_train):
    fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y))))
  fp1.close()
  # Dump data for plot:
  fp1= file('/tmp/smpl_test.dat','w')
  for x,y,i in zip(x_test,y_test,range(len(y_test))):
    if i%(nt+1)==0:  fp1.write('\n')
    fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y))))
  fp1.close()

  # Prepare multi-layer perceptron model
  model = FunctionSet(l1=F.Linear(2, n_units),
                      l2=F.Linear(n_units, n_units),
                      l3=F.Linear(n_units, 1))
  #TEST: Random bias initialization
  #, bias=Rand()
  #model.l1.b[:]= [Rand() for k in range(n_units)]
  #model.l2.b[:]= [Rand() for k in range(n_units)]
  #model.l3.b[:]= [Rand() for k in range(1)]
  #print model.l2.__dict__
  if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

  # Neural net architecture
  def forward(x_data, y_data, train=True):
    #train= False  #TEST: Turn off dropout
    dratio= 0.2  #0.5  #TEST: Dropout ratio
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
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
    for x,y,i in zip(x_test,preds,range(len(preds))):
      if i%(nt+1)==0:  fp1.write('\n')
      fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y))))
    fp1.close()


def PlotGraphs():
  print 'Plotting graphs..'
  import os,sys
  opt= sys.argv[2:]
  commands=[
    '''qplot -x2 aaa -3d {opt} -s 'set xlabel "x";set ylabel "y";set ticslevel 0;'
          -cs 'u 1:2:4' /tmp/smpl_train.dat pt 6 ps 2 t '"sample"'
          /tmp/smpl_test.dat w l lw 1 t '"true"'
          /tmp/nn_test{NEpoch:04d}.dat w l lw 3 t '"Final({NEpoch}) epoch"' &''',
          #/tmp/nn_test0001.dat w l t '"1st epoch"'
          #/tmp/nn_test0005.dat w l t '"5th epoch"'
          #/tmp/nn_test0020.dat w l t '"20th epoch"'
          #/tmp/nn_test0050.dat w l t '"50th epoch"'
          #/tmp/nn_test0075.dat w l t '"75th epoch"'
          #/tmp/nn_test0099.dat w l t '"99th epoch"'
    '''''',
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
