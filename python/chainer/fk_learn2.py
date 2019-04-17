#!/usr/bin/python
#\file    fk_learn2.py
#\brief   Learning forward kinematics with Chainer's regression with decomposed nets.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.13, 2015
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F

ModelCodesWithXAll= ()

def LoadData(src_file,c1=0,c2=None):
  data= []
  fp= file(src_file)
  while True:
    line= fp.readline()
    if not line: break
    data_s= line.split()
    data.append(map(float,data_s[c1:c2]))
  return data

def CreateModel(dof, Dx, Dy1, Dy2, n_units, n_units2, n_units3):
  if dof=='2':
    model1= FunctionSet(l1=F.Linear(Dx, n_units),
                        l2=F.Linear(n_units, Dy1))
    model2= FunctionSet(l1=F.Linear(Dx, n_units),
                        l2=F.Linear(n_units, Dy2))
  elif dof=='3':
    model1= FunctionSet(l1=F.Linear(Dx, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, Dy1))
    model2= FunctionSet(l1=F.Linear(Dx, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, Dy2))
  elif dof=='7':
    model1= FunctionSet(l1=F.Linear(Dx, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, n_units),
                        l4=F.Linear(n_units, n_units),
                        l5=F.Linear(n_units, n_units),
                        l6=F.Linear(n_units, n_units),
                        l7=F.Linear(n_units, Dy1))
    model2= FunctionSet(l1=F.Linear(Dx, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, n_units),
                        l4=F.Linear(n_units, n_units),
                        l5=F.Linear(n_units, n_units),
                        l6=F.Linear(n_units, n_units),
                        l7=F.Linear(n_units, Dy2))
  return model1, model2

# Neural net architecture
def ForwardModel(dof, model1, model2, x_data, y1_data, y2_data, train=True):
  #train= False  #TEST: Turn off dropout
  dratio= 0.2  #0.5  #TEST: Dropout ratio
  if dof=='2':
    x, t1, t2= Variable(x_data), Variable(y1_data), Variable(y2_data)
    h11 = F.dropout(F.relu(model1.l1(x)),  ratio=dratio, train=train)
    y1  = model1.l2(h11)
    loss1= F.mean_squared_error(y1, t1)
    h21 = F.dropout(F.relu(model2.l1(x)),  ratio=dratio, train=train)
    y2  = model2.l2(h21)
    loss2= F.mean_squared_error(y2, t2)
  elif dof=='3':
    x, t1, t2= Variable(x_data), Variable(y1_data), Variable(y2_data)
    h11 = F.dropout(F.relu(model1.l1(x)),   ratio=dratio, train=train)
    h12 = F.dropout(F.relu(model1.l2(h11)), ratio=dratio, train=train)
    y1  = model1.l3(h12)
    loss1= F.mean_squared_error(y1, t1)
    h21 = F.dropout(F.relu(model2.l1(x)),   ratio=dratio, train=train)
    h22 = F.dropout(F.relu(model2.l2(h21)), ratio=dratio, train=train)
    y2  = model2.l3(h22)
    loss2= F.mean_squared_error(y2, t2)
  elif dof=='7':
    x, t1, t2= Variable(x_data), Variable(y1_data), Variable(y2_data)
    h11= F.dropout(F.relu(model1.l1(x)),   ratio=dratio, train=train)
    h12= F.dropout(F.relu(model1.l2(h11)), ratio=dratio, train=train)
    h13= F.dropout(F.relu(model1.l3(h12)), ratio=dratio, train=train)
    h14= F.dropout(F.relu(model1.l4(h13)), ratio=dratio, train=train)
    h15= F.dropout(F.relu(model1.l5(h14)), ratio=dratio, train=train)
    h16= F.dropout(F.relu(model1.l6(h15)), ratio=dratio, train=train)
    y1= model1.l7(h16)
    loss1= F.mean_squared_error(y1, t1)
    h21= F.dropout(F.relu(model2.l1(x)),   ratio=dratio, train=train)
    h22= F.dropout(F.relu(model2.l2(h21)), ratio=dratio, train=train)
    h23= F.dropout(F.relu(model2.l3(h22)), ratio=dratio, train=train)
    h24= F.dropout(F.relu(model2.l4(h23)), ratio=dratio, train=train)
    h25= F.dropout(F.relu(model2.l5(h24)), ratio=dratio, train=train)
    h26= F.dropout(F.relu(model2.l6(h25)), ratio=dratio, train=train)
    y2= model2.l7(h26)
    loss2= F.mean_squared_error(y2, t2)
  return loss1, loss2, y1, y2

NEpoch= 100  #TEST
#NEpoch= 200  #TEST

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
    data_y1 = LoadData('datak/chain%s_x.dat'%sdofc, c1=0, c2=3)
    data_y2 = LoadData('datak/chain%s_x.dat'%sdofc, c1=3, c2=None)
  else:
    data_y = LoadData('datak/chain%s_xall.dat'%sdofc, c1=7)  #Skip the first pose.
  # Prepare dataset
  batchsize= max(1,min(batchsize, len(data_x)/20))  #TEST: adjust batchsize
  data = np.array(data_x).astype(np.float32)
  target1 = np.array(data_y1).astype(np.float32)
  target2 = np.array(data_y2).astype(np.float32)

  N= len(data) #batchsize * 30
  x_train= data
  y1_train= target1
  y2_train= target2

  print 'Num of samples for train:',len(x_train)

  Dx = len(data_x[0])
  Dy1 = len(data_y1[0])
  Dy2 = len(data_y2[0])
  model1, model2 = CreateModel(dofc, Dx, Dy1, Dy2, n_units, n_units2, n_units3)
  if args.gpu >= 0:
    cuda.init(args.gpu)
    model1.to_gpu()
    model2.to_gpu()

  # Neural net architecture
  def forward(x_data, y1_data, y2_data, train=True):
    return ForwardModel(dofc, model1, model2, x_data, y1_data, y2_data, train)

  # Predict for a single query x
  def predict(x):
    x_batch = np.array([x]).astype(np.float32)
    y1_batch = np.array([[0.0]*Dy1]).astype(np.float32)  #Dummy
    y2_batch = np.array([[0.0]*Dy2]).astype(np.float32)  #Dummy
    if args.gpu >= 0:
      x_batch = cuda.to_gpu(x_batch)
      y1_batch = cuda.to_gpu(y1_batch)
      y2_batch = cuda.to_gpu(y2_batch)
    loss1,loss2, pred1,pred2 = forward(x_batch, y1_batch, y2_batch, train=False)
    y1= cuda.to_cpu(pred1.data)[0]
    y2= cuda.to_cpu(pred2.data)[0]
    y= np.concatenate((y1,y2))
    return y

  # Setup optimizer
  optimizer1 = optimizers.AdaDelta(rho=0.9)
  optimizer2 = optimizers.AdaDelta(rho=0.9)
  #optimizer = optimizers.AdaGrad(lr=0.5)
  #optimizer = optimizers.RMSprop()
  #optimizer = optimizers.MomentumSGD()
  #optimizer = optimizers.SGD(lr=0.8)
  optimizer1.setup(model1.collect_parameters())
  optimizer2.setup(model2.collect_parameters())

  #tester= TFKTester(3)

  file_names= {'l':'result/fk2log%s.dat'%mdofc,
               'm':'result/fk2nn%s.dat'%mdofc}
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
      y1_batch = y1_train[perm[i:i+batchsize]]
      y2_batch = y2_train[perm[i:i+batchsize]]
      if args.gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)
        y1_batch = cuda.to_gpu(y1_batch)
        y2_batch = cuda.to_gpu(y2_batch)

      optimizer1.zero_grads()
      optimizer2.zero_grads()
      loss1, loss2, pred1, pred2 = forward(x_batch, y1_batch, y2_batch)
      loss1.backward()  #Computing gradients
      loss2.backward()  #Computing gradients
      optimizer1.update()
      optimizer2.update()

      loss= loss1 + loss2
      bloss = float(cuda.to_cpu(loss.data))
      sum_loss += bloss * batchsize
      fp_log.write('%f %f\n'%(epoch-1.0+float(i)/float(N), bloss))

    print 'train mean loss={}'.format(
        sum_loss / N)
    fp_log.write('%f %f %f\n'%(float(epoch), bloss, sum_loss / N))


    print predict(np.array([0.0]*dof).astype(np.float32))
    #if epoch in TestNEpochs:
      #tester.Test(f_fwdkin=predict, n_samples=100)

    model= {'model1':model1, 'model2':model2}
    pickle.dump(model, open(file_names['m'], 'wb'), -1)
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
