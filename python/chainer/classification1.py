#!/usr/bin/python
#\file    classification1.py
#\brief   Chainer for 2-D --> Multi-class classification test.
#         Based on regression4a.py
#         Search DIFF_REG for the difference.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.16, 2017
import random,math,copy

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

def Median(array):
  if len(array)==0:  return None
  a_sorted= copy.deepcopy(array)
  a_sorted.sort()
  return a_sorted[len(a_sorted)/2]

def LoadData():
  #NOTE: we use only [x0,x1] (delete [:2] to use all x)
  data_x= [map(float,d.split()[1:3]) for d in open('data/iris_x.dat','r').read().split('\n') if d!='']
  data_y= [float(d) for d in open('data/iris_y.dat','r').read().split('\n') if d!='']  #DIFF_REG
  return data_x, data_y

#Return min, max, median vectors of data.
def GetStat(data):
  mi= [min([x[d] for x in data]) for d in range(len(data[0]))]
  ma= [max([x[d] for x in data]) for d in range(len(data[0]))]
  me= [Median([x[d] for x in data]) for d in range(len(data[0]))]
  return mi,ma,me

#Dump data with dimension reduction f_reduce.
#Each row of dumped data: reduced x, original x, original y
def DumpData(file_name, data_x, data_y, f_reduce, lb=0):
  fp1= file(file_name,'w')
  for x,y,i in zip(data_x,data_y,range(len(data_y))):
    if lb>0 and i%lb==0:  fp1.write('\n')
    fp1.write('%s  %s  %s\n' % (' '.join(map(str,f_reduce(x))), ' '.join(map(str,x)), ' '.join(map(str,y))))
  fp1.close()


NEpoch= 500  #TEST

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
  n_units   = 300  #TEST

  # Prepare dataset
  data_x, data_y = LoadData()
  batchsize= max(1,min(batchsize, len(data_y)/20))  #TEST: adjust batchsize
  #dx2,dy2=GenData(300, noise=0.0); data_x.extend(dx2); data_y.extend(dy2)
  data = np.array(data_x).astype(np.float32)
  target = np.array(data_y).astype(np.int32)  #DIFF_REG

  N= len(data) #batchsize * 30
  x_train= data
  y_train= target

  #For test:
  mi,ma,me= GetStat(data_x)
  f_reduce=lambda xa:[xa[0],xa[1]]
  f_repair=lambda xa:[xa[0],xa[1]]
  nt= 20+1
  N_test= nt*nt
  x_test= np.array(sum([[f_repair([x1,x2]) for x2 in FRange1(f_reduce(mi)[1],f_reduce(ma)[1],nt)] for x1 in FRange1(f_reduce(mi)[0],f_reduce(ma)[0],nt)],[])).astype(np.float32)
  y_test= np.array([0.0 for x in x_test]).astype(np.int32)  #DIFF_REG
  #No true test data (just for plotting)

  print 'Num of samples for train:',len(y_train),'batchsize:',batchsize
  # Dump data for plot:
  DumpData('/tmp/nn/smpl_train.dat', x_train, [[y] for y in y_train], f_reduce)  #DIFF_REG

  # Prepare multi-layer perceptron model
  model = FunctionSet(l1=F.Linear(2, n_units),
                      l2=F.Linear(n_units, n_units),
                      l3=F.Linear(n_units, 3))
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
    #return F.mean_squared_error(y, t), y
    return F.softmax_cross_entropy(y, t), F.softmax(y)  #DIFF_REG

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


    if epoch%10==0:
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
      #'''

      print 'test  mean loss={}'.format(
          sum_loss / N_test)

      # Dump data for plot:
      y_pred= [[y.index(max(y))]+y for y in preds.tolist()]  #DIFF_REG
      DumpData('/tmp/nn/nn_test%04i.dat'%epoch, x_test, y_pred, f_reduce, lb=nt+1)


def PlotGraphs():
  print 'Plotting graphs..'
  import os,sys
  opt= sys.argv[2:]
  commands=[
    '''qplot -x2 aaa {opt}
          -s 'set xlabel "x1";set ylabel "x2";set title "";'
          -s 'set encoding utf8;symbol(z)="+xo%#"[int(z):int(z)];'
          /tmp/nn/nn_test{NEpoch:04d}.dat u 1:2:'(symbol($5+1))' w labels textcolor lt 3 t '"Final({NEpoch}) epoch"'
          /tmp/nn/smpl_train.dat u 1:2:'(symbol($5+1))' w labels textcolor lt 1 t '"sample"'  &''',
          #/tmp/nn/lwr/f1_3_est.dat w l lw 1 t '"LWR"'
          #/tmp/nn/nn_test0001.dat w l t '"1st epoch"'
          #/tmp/nn/nn_test0005.dat w l t '"5th epoch"'
          #/tmp/nn/nn_test0020.dat w l t '"20th epoch"'
          #/tmp/nn/nn_test0050.dat w l t '"50th epoch"'
          #/tmp/nn/nn_test0075.dat w l t '"75th epoch"'
          #/tmp/nn/nn_test0099.dat w l t '"99th epoch"'
    '''qplot -x2 aaa {opt} -3d
          -s 'set xlabel "x1";set ylabel "x2";set title "Class 0";'
          -s 'set pm3d;unset surface;set view map;'
          -s 'set encoding utf8;symbol(z)="+xo%#"[int(z):int(z)];'
          /tmp/nn/nn_test{NEpoch:04d}.dat u 1:2:6 t '""'
          /tmp/nn/smpl_train.dat u 1:2:'(0.0)':'(symbol($5+1))' w labels textcolor lt 1 t '"sample"'  &''',
    '''qplot -x2 aaa {opt} -3d
          -s 'set xlabel "x1";set ylabel "x2";set title "Class 1";'
          -s 'set encoding utf8;symbol(z)="+xo%#"[int(z):int(z)];'
          -s 'set pm3d;unset surface;set view map;'
          /tmp/nn/nn_test{NEpoch:04d}.dat u 1:2:7 t '""'
          /tmp/nn/smpl_train.dat u 1:2:'(0.0)':'(symbol($5+1))' w labels textcolor lt 1 t '"sample"'  &''',
    '''qplot -x2 aaa {opt} -3d
          -s 'set xlabel "x1";set ylabel "x2";set title "Class 2";'
          -s 'set pm3d;unset surface;set view map;'
          -s 'set encoding utf8;symbol(z)="+xo%#"[int(z):int(z)];'
          /tmp/nn/nn_test{NEpoch:04d}.dat u 1:2:8 t '""'
          /tmp/nn/smpl_train.dat u 1:2:'(0.0)':'(symbol($5+1))' w labels textcolor lt 1 t '"sample"'  &''',
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
