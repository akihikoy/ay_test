#!/usr/bin/python
#\file    fk_learn.py
#\brief   Learning forward kinematics with Chainer's regression.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.05, 2015
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F

ModelCodesWithXAll= ('3d','3e','3f','3g','7c','7d')

def LoadData(src_file,c1=0,c2=None):
  data= []
  fp= file(src_file)
  while True:
    line= fp.readline()
    if not line: break
    data_s= line.split()
    data.append(map(float,data_s[c1:c2]))
  return data

def CreateModel(dof, Dx, Dy, n_units, n_units2, n_units3):
  if dof=='2':
    model= FunctionSet(l1=F.Linear(Dx, n_units),
                       l2=F.Linear(n_units, Dy))
  elif dof=='3':
    model= FunctionSet(l1=F.Linear(Dx, n_units),
                       l2=F.Linear(n_units, n_units),
                       l3=F.Linear(n_units, Dy))
    #TEST: Random bias initialization
    #, bias=Rand()
    #model.l1.b[:]= [Rand() for k in range(n_units)]
    #model.l2.b[:]= [Rand() for k in range(n_units)]
    #model.l3.b[:]= [Rand() for k in range(1)]
    #print model.l2.__dict__
  elif dof=='3b':
    model= FunctionSet(l1=F.Linear(Dx, n_units),
                       l2=F.Linear(n_units, n_units2),
                       l3=F.Linear(n_units2, n_units),
                       l4=F.Linear(n_units, Dy))
  elif dof=='3c':
    model= FunctionSet(l1=F.Linear(Dx-1, n_units),
                       l2=F.Linear(n_units, n_units2),
                       l3=F.Linear(1+n_units2, n_units),
                       l4=F.Linear(n_units, Dy))
  elif dof=='3d':
    model= FunctionSet(l1=F.Linear(1, n_units),
                       l2=F.Linear(n_units, 7),
                       l3=F.Linear(1+7, n_units),
                       l4=F.Linear(n_units, 7),
                       l5=F.Linear(1+7, n_units),
                       l6=F.Linear(n_units, 7))
  elif dof=='3e':
    model= FunctionSet(l1=F.Linear(1, n_units),
                       l2=F.Linear(n_units, 7+n_units3),
                       l3=F.Linear(1+7+n_units3, n_units),
                       l4=F.Linear(n_units, 7+n_units3),
                       l5=F.Linear(1+7+n_units3, n_units),
                       l6=F.Linear(n_units, 7))
  elif dof=='3f':
    model= FunctionSet(l1=F.Linear(Dx, n_units),
                       l2=F.Linear(n_units, n_units),
                       l3=F.Linear(n_units, Dy))
  elif dof=='3g':
    model= FunctionSet(l1=F.Linear(Dx, n_units),
                       l2=F.Linear(n_units, n_units),
                       l3=F.Linear(n_units, n_units),
                       l4=F.Linear(n_units, Dy))
  elif dof=='3twin':
    model= FunctionSet(l11=F.Linear(Dx, n_units),
                       l12=F.Linear(n_units, n_units),
                       l13=F.Linear(n_units, 3),
                       l21=F.Linear(Dx, n_units),
                       l22=F.Linear(n_units, n_units),
                       l23=F.Linear(n_units, 4))
  elif dof=='7':
    model= FunctionSet(l1=F.Linear(Dx, n_units),
                       l2=F.Linear(n_units, n_units),
                       l3=F.Linear(n_units, n_units),
                       l4=F.Linear(n_units, n_units),
                       l5=F.Linear(n_units, n_units),
                       l6=F.Linear(n_units, n_units),
                       l7=F.Linear(n_units, Dy))
  elif dof=='7b':
    model= FunctionSet(l1=F.Linear(Dx, n_units),
                       l2=F.Linear(n_units, n_units),
                       l3=F.Linear(n_units, n_units),
                       l4=F.Linear(n_units, n_units),
                       l5=F.Linear(n_units, n_units),
                       l6=F.Linear(n_units, n_units),
                       l7=F.Linear(n_units, n_units),
                       l8=F.Linear(n_units, n_units),
                       l9=F.Linear(n_units, n_units),
                       l10=F.Linear(n_units, Dy))
  elif dof=='7c':
    model= FunctionSet(l1=F.Linear(Dx, n_units),
                       l2=F.Linear(n_units, n_units),
                       l3=F.Linear(n_units, n_units),
                       l4=F.Linear(n_units, n_units),
                       l5=F.Linear(n_units, n_units),
                       l6=F.Linear(n_units, n_units),
                       l7=F.Linear(n_units, n_units),
                       l8=F.Linear(n_units, n_units),
                       l9=F.Linear(n_units, n_units),
                       l10=F.Linear(n_units, Dy))
  elif dof=='7d':
    model= FunctionSet(l1=F.Linear(Dx, n_units),
                       l2=F.Linear(n_units, n_units),
                       l3=F.Linear(n_units, n_units),
                       l4=F.Linear(n_units, n_units),
                       l5=F.Linear(n_units, n_units),
                       l6=F.Linear(n_units, n_units),
                       l7=F.Linear(n_units, n_units),
                       l8=F.Linear(n_units, n_units),
                       l9=F.Linear(n_units, n_units),
                       l10=F.Linear(n_units, n_units),
                       l11=F.Linear(n_units, n_units),
                       l12=F.Linear(n_units, n_units),
                       l13=F.Linear(n_units, n_units),
                       l14=F.Linear(n_units, Dy))
  return model

# Neural net architecture
def ForwardModel(dof, model, x_data, y_data, train=True):
  #train= False  #TEST: Turn off dropout
  dratio= 0.2  #0.5  #TEST: Dropout ratio
  if dof=='2':
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    y  = model.l2(h1)
    loss= F.mean_squared_error(y, t)
  elif dof=='3':
    x, t= Variable(x_data), Variable(y_data)
    h1= F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    #h1= F.dropout(F.leaky_relu(model.l1(x),slope=0.2),  ratio=dratio, train=train)
    #h2= F.dropout(F.leaky_relu(model.l2(h1),slope=0.2), ratio=dratio, train=train)
    #h1= F.dropout(F.sigmoid(model.l1(x)),  ratio=dratio, train=train)
    #h2= F.dropout(F.sigmoid(model.l2(h1)), ratio=dratio, train=train)
    #h1= F.dropout(F.tanh(model.l1(x)),  ratio=dratio, train=train)
    #h2= F.dropout(F.tanh(model.l2(h1)), ratio=dratio, train=train)
    #h1= F.dropout(model.l1(x),  ratio=dratio, train=train)
    #h2= F.dropout(model.l2(h1), ratio=dratio, train=train)
    #h1= F.relu(model.l1(x))
    #h2= F.relu(model.l2(h1))
    #h1= model.l1(x)
    #h2= model.l2(h1)
    y= model.l3(h2)
    loss= F.mean_squared_error(y, t)
  elif dof=='3b':
    x, t= Variable(x_data), Variable(y_data)
    h1= F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    h3= F.dropout(F.relu(model.l3(h2)), ratio=dratio, train=train)
    y= model.l4(h3)
    loss= F.mean_squared_error(y, t)
  elif dof=='3c':
    x, t= Variable(x_data), Variable(y_data)
    x1,x2= F.split_axis(x, [2], 1)
    h1= F.dropout(F.relu(model.l1(x1)), ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    h3= F.dropout(F.relu(model.l3(F.concat((x2,h2),1))), ratio=dratio, train=train)
    y= model.l4(h3)
    loss= F.mean_squared_error(y, t)
  elif dof=='3d':
    x, t= Variable(x_data), Variable(y_data)
    x1,x2,x3= F.split_axis(x, range(1,3), 1)
    #print x1.data[0],x2.data[0],x3.data[0]
    tt= F.split_axis(t, range(7,21,7), 1)
    #print tt[0].data[0],tt[1].data[0],tt[2].data[0]
    h1= F.dropout(F.relu(model.l1(x1)), ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    h3= F.dropout(F.relu(model.l3(F.concat((x2,h2),1))), ratio=dratio, train=train)
    h4= F.dropout(F.relu(model.l4(h3)), ratio=dratio, train=train)
    h5= F.dropout(F.relu(model.l5(F.concat((x3,h4),1))), ratio=dratio, train=train)
    y= model.l6(h5)
    #print h2.data[0],h4.data[0],y.data[0]
    loss1= F.mean_squared_error(h2, tt[0])
    loss2= F.mean_squared_error(h4, tt[1])
    loss3= F.mean_squared_error(y, tt[2])
    loss= loss1 + loss2 + loss3
    #print loss1.data, loss2.data, loss3.data, loss.data
  elif dof=='3e':
    x, t= Variable(x_data), Variable(y_data)
    x1,x2,x3= F.split_axis(x, range(1,3), 1)
    #print x1.data[0],x2.data[0],x3.data[0]
    tt= F.split_axis(t, range(7,21,7), 1)
    #print tt[0].data[0],tt[1].data[0],tt[2].data[0]
    h1= F.dropout(F.relu(model.l1(x1)), ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    h3= F.dropout(F.relu(model.l3(F.concat((x2,h2),1))), ratio=dratio, train=train)
    h4= F.dropout(F.relu(model.l4(h3)), ratio=dratio, train=train)
    h5= F.dropout(F.relu(model.l5(F.concat((x3,h4),1))), ratio=dratio, train=train)
    y= model.l6(h5)
    #print h2.data[0],h4.data[0],y.data[0]
    h2_1,h2_2= F.split_axis(h2, [7], 1)
    h4_1,h4_2= F.split_axis(h4, [7], 1)
    #print h2_1.data[0],h4_1.data[0]
    loss1= F.mean_squared_error(h2_1, tt[0])
    loss2= F.mean_squared_error(h4_1, tt[1])
    loss3= F.mean_squared_error(y, tt[2])
    loss= loss1 + loss2 + loss3 + 0.0*F.mean_squared_error(h2_2,h4_2)  #dummy
    #print loss1.data, loss2.data, loss3.data, loss.data
  elif dof=='3f':
    x, t= Variable(x_data), Variable(y_data)
    h1= F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    h3= model.l3(h2)
    loss= F.mean_squared_error(h3, t)
    #print h3.data[0]
    y= F.split_axis(h3, [7*(3-1)], 1)[1]
  elif dof=='3g':
    x, t= Variable(x_data), Variable(y_data)
    h1= F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    h3= F.dropout(F.relu(model.l3(h2)), ratio=dratio, train=train)
    h4= model.l4(h3)
    loss= F.mean_squared_error(h4, t)
    y= F.split_axis(h4, [7*(3-1)], 1)[1]
  elif dof=='3twin':
    x, t= Variable(x_data), Variable(y_data)
    h11= F.dropout(F.relu(model.l11(x)),  ratio=dratio, train=train)
    h12= F.dropout(F.relu(model.l12(h11)), ratio=dratio, train=train)
    y1= model.l13(h12)
    h21= F.dropout(F.relu(model.l21(x)),  ratio=dratio, train=train)
    h22= F.dropout(F.relu(model.l22(h21)), ratio=dratio, train=train)
    y2= model.l23(h22)
    y= F.concat((y1,y2),1)
    loss= F.mean_squared_error(y, t)
  elif dof=='7':
    x, t= Variable(x_data), Variable(y_data)
    h1= F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    h3= F.dropout(F.relu(model.l3(h2)), ratio=dratio, train=train)
    h4= F.dropout(F.relu(model.l4(h3)), ratio=dratio, train=train)
    h5= F.dropout(F.relu(model.l5(h4)), ratio=dratio, train=train)
    h6= F.dropout(F.relu(model.l6(h5)), ratio=dratio, train=train)
    y= model.l7(h6)
    loss= F.mean_squared_error(y, t)
  elif dof=='7b':
    x, t= Variable(x_data), Variable(y_data)
    h1= F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    h3= F.dropout(F.relu(model.l3(h2)), ratio=dratio, train=train)
    h4= F.dropout(F.relu(model.l4(h3)), ratio=dratio, train=train)
    h5= F.dropout(F.relu(model.l5(h4)), ratio=dratio, train=train)
    h6= F.dropout(F.relu(model.l6(h5)), ratio=dratio, train=train)
    h7= F.dropout(F.relu(model.l7(h6)), ratio=dratio, train=train)
    h8= F.dropout(F.relu(model.l8(h7)), ratio=dratio, train=train)
    h9= F.dropout(F.relu(model.l9(h8)), ratio=dratio, train=train)
    y= model.l10(h9)
    loss= F.mean_squared_error(y, t)
  elif dof=='7c':
    x, t= Variable(x_data), Variable(y_data)
    h1= F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    h3= F.dropout(F.relu(model.l3(h2)), ratio=dratio, train=train)
    h4= F.dropout(F.relu(model.l4(h3)), ratio=dratio, train=train)
    h5= F.dropout(F.relu(model.l5(h4)), ratio=dratio, train=train)
    h6= F.dropout(F.relu(model.l6(h5)), ratio=dratio, train=train)
    h7= F.dropout(F.relu(model.l7(h6)), ratio=dratio, train=train)
    h8= F.dropout(F.relu(model.l8(h7)), ratio=dratio, train=train)
    h9= F.dropout(F.relu(model.l9(h8)), ratio=dratio, train=train)
    h10= model.l10(h9)
    loss= F.mean_squared_error(h10, t)
    y= F.split_axis(h10, [7*(7-1)], 1)[1]
  elif dof=='7d':
    x, t= Variable(x_data), Variable(y_data)
    h1= F.dropout(F.relu(model.l1(x)),  ratio=dratio, train=train)
    h2= F.dropout(F.relu(model.l2(h1)), ratio=dratio, train=train)
    h3= F.dropout(F.relu(model.l3(h2)), ratio=dratio, train=train)
    h4= F.dropout(F.relu(model.l4(h3)), ratio=dratio, train=train)
    h5= F.dropout(F.relu(model.l5(h4)), ratio=dratio, train=train)
    h6= F.dropout(F.relu(model.l6(h5)), ratio=dratio, train=train)
    h7= F.dropout(F.relu(model.l7(h6)), ratio=dratio, train=train)
    h8= F.dropout(F.relu(model.l8(h7)), ratio=dratio, train=train)
    h9= F.dropout(F.relu(model.l9(h8)), ratio=dratio, train=train)
    h10= F.dropout(F.relu(model.l10(h9)), ratio=dratio, train=train)
    h11= F.dropout(F.relu(model.l11(h10)), ratio=dratio, train=train)
    h12= F.dropout(F.relu(model.l12(h11)), ratio=dratio, train=train)
    h13= F.dropout(F.relu(model.l13(h12)), ratio=dratio, train=train)
    h14= model.l14(h13)
    loss= F.mean_squared_error(h14, t)
    y= F.split_axis(h14, [7*(7-1)], 1)[1]
  return loss, y

NEpoch= 100  #TEST
#NEpoch= 200  #TEST
TestNEpochs= (1,5,20,75,99,NEpoch)

def Main():
  import argparse
  import os,re
  import numpy as np
  import six.moves.cPickle as pickle

  #from fk_test import TFKTester

  parser = argparse.ArgumentParser(description='Chainer example: regression')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                      help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--dof', '-D', default='3', type=str,
                      help='DoF code')
  parser.add_argument('--sdof', '-SD', default='', type=str,
                      help='DoF code of samples. Blank uses the same one as --dof.')
  parser.add_argument('--mdof', '-MD', default='', type=str,
                      help='DoF code of model file. Blank uses the same one as --dof.')
  args = parser.parse_args()

  batchsize = 10  #TEST; 20
  n_epoch   = NEpoch
  n_units   = 200  #TEST
  n_units2  = 20
  n_units3  = 50
  #n_units   = 500
  #batchsize = 20

  #dof = 3
  dofc= args.dof
  sdofc= args.sdof if args.sdof!='' else dofc
  mdofc= args.mdof if args.mdof!='' else dofc
  dof= int(re.search('^[0-9]+',dofc).group())
  data_x = LoadData('datak/chain%s_q.dat'%sdofc)
  if dofc not in ModelCodesWithXAll:
    data_y = LoadData('datak/chain%s_x.dat'%sdofc)
  else:
    data_y = LoadData('datak/chain%s_xall.dat'%sdofc, c1=7)  #Skip the first pose.
  # Prepare dataset
  batchsize= max(1,min(batchsize, len(data_y)/20))  #TEST: adjust batchsize
  #dx2,dy2=GenData(300, noise=0.0); data_x.extend(dx2); data_y.extend(dy2)
  data = np.array(data_x).astype(np.float32)
  target = np.array(data_y).astype(np.float32)

  N= len(data) #batchsize * 30
  x_train= data
  y_train= target

  print 'Num of samples for train:',len(y_train)
  # Dump data for plot:
  #DumpData('/tmp/smpl_train.dat', x_train, y_train, f_reduce)

  # Prepare multi-layer perceptron model
  Dx = len(data_x[0])
  Dy = len(data_y[0])
  model = CreateModel(dofc, Dx, Dy, n_units, n_units2, n_units3)
  if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

  # Neural net architecture
  def forward(x_data, y_data, train=True):
    return ForwardModel(dofc, model, x_data, y_data, train)

  # Predict for a single query x
  def predict(x):
    x_batch = np.array([x]).astype(np.float32)
    y_batch = np.array([[0.0]*Dy]).astype(np.float32)  #Dummy
    if args.gpu >= 0:
      x_batch = cuda.to_gpu(x_batch)
      y_batch = cuda.to_gpu(y_batch)
    loss, pred = forward(x_batch, y_batch, train=False)
    y= cuda.to_cpu(pred.data)[0]
    return y

  # Setup optimizer
  optimizer = optimizers.AdaDelta(rho=0.9)
  #optimizer = optimizers.AdaGrad(lr=0.5)
  #optimizer = optimizers.RMSprop()
  #optimizer = optimizers.MomentumSGD()
  #optimizer = optimizers.SGD(lr=0.8)
  optimizer.setup(model.collect_parameters())

  #tester= TFKTester(3)

  file_names= {'l':'result/fklog%s.dat'%mdofc,
               'm':'result/fknn%s.dat'%mdofc}
  if os.path.exists(file_names['l']) or os.path.exists(file_names['m']):
    print 'File(s) already exists.'
    print 'Check:',file_names
    return

  fp_log= open(file_names['l'],'w')

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

      bloss = float(cuda.to_cpu(loss.data))
      sum_loss += bloss * batchsize
      fp_log.write('%f %f\n'%(epoch-1.0+float(i)/float(N), bloss))

    print 'train mean loss={}'.format(
        sum_loss / N)
    fp_log.write('%f %f %f\n'%(float(epoch), bloss, sum_loss / N))


    print predict(np.array([0.0]*dof).astype(np.float32))
    #if epoch in TestNEpochs:
      #tester.Test(f_fwdkin=predict, n_samples=100)

    #if args.gpu >= 0:  model.to_cpu()
    model2= CreateModel(dofc, Dx, Dy, n_units, n_units2, n_units3)
    model2.copy_parameters_from(model.parameters)
    pickle.dump(model2, open(file_names['m'], 'wb'), -1)
    #pickle.dump(model.parameters, open(file_names['m'], 'wb'), -1)
    #if args.gpu >= 0:  model.to_gpu()
    print 'Model file is dumped to:',file_names['m']

    '''
    if epoch in TestNEpochs:
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

      print 'test  mean loss={}'.format(
          sum_loss / N_test)

      # Dump data for plot:
      DumpData('/tmp/nn_test%04i.dat'%epoch, x_test, preds, f_reduce, lb=nt+1)
    #'''

  #tester.Cleanup()
  fp_log.close()


def PlotGraphs():
  import sys
  argv= sys.argv[2:]
  dof= argv[0] if len(argv)>0 else '3'  #input mdofc

  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa
          -s 'set logscale y; set title "Learning curve (DoF:{dof})"; set xlabel "epoch"; set ylabel "error";'
          result/fklog{dof}.dat w l  t '"error per batch"'
          result/fklog{dof}.dat u 1:3 w l lw 3  t '"error per epoch"'  &''',
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.format(dof=dof).splitlines())
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
