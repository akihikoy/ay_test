#!/usr/bin/python
#\file    fk_learn3.py
#\brief   Learning forward kinematics with Chainer's regression.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.29, 2015

def LoadData(src_file,c1=0,c2=None):
  data= []
  fp= file(src_file)
  while True:
    line= fp.readline()
    if not line: break
    data_s= line.split()
    data.append(map(float,data_s[c1:c2]))
  return data

def Main():
  from dnn_reg1 import TNNRegression, DumpPlot, SaveYAML, LoadYAML
  import os,re
  import argparse

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

  #dof = 3
  dofc= args.dof
  sdofc= args.sdof if args.sdof!='' else dofc
  mdofc= args.mdof if args.mdof!='' else dofc
  dof= int(re.search('^[0-9]+',dofc).group())
  data_x= LoadData('datak/chain%s_q.dat'%sdofc)
  data_y1= LoadData('datak/chain%s_x.dat'%sdofc, c1=0, c2=3)
  data_y2= LoadData('datak/chain%s_x.dat'%sdofc, c1=3, c2=None)

  file_prefix= 'result/fk3nn%s_'%mdofc
  file_names= {
    '1': '{base}1_model.yaml'.format(base=file_prefix),
    '2': '{base}2_model.yaml'.format(base=file_prefix) }
  if os.path.exists(file_names['1']) or os.path.exists(file_names['2']):
    print 'File(s) already exists.'
    print 'Check:',file_names
    return

  if dofc=='3':
    n_hidden= [200,200,200,200]
  elif dofc=='7':
    n_hidden= [200,200,200,200,200,200,200]

  options1= {}
  options1['n_units']= [dof]+n_hidden+[3]
  options1['base_dir']= '/tmp/fknn1/'
  options1['data_file_name']= '{base}1_{label}.dat'
  options1['loss_stddev_stop']= 1.0e-4
  options1['num_max_update']= 50000
  options2= {}
  options2['n_units']= [dof]+n_hidden+[4]
  options2['base_dir']= '/tmp/fknn2/'
  options2['data_file_name']= '{base}2_{label}.dat'
  options2['loss_stddev_stop']= 1.0e-4
  options2['num_max_update']= 50000

  model1= TNNRegression()
  model1.Load(data={'options':options1})
  model1.Init()
  model2= TNNRegression()
  model2.Load(data={'options':options2})
  model2.Init()

  model1.UpdateBatch(data_x,data_y1)
  SaveYAML(model1.Save(file_prefix), file_names['1'])

  model2.UpdateBatch(data_x,data_y2)
  SaveYAML(model2.Save(file_prefix), file_names['2'])


def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa 'sin(x)' w l &''',
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
