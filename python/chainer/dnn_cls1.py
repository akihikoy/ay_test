#!/usr/bin/python
#\file    dnn_cls1.py
#\brief   Deep neural networks for classification.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.29, 2017
import os
import math
import random
import copy
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
from loss_for_error import loss_for_error1, loss_for_error2
from dnn_reg1 import ReLUGaussV, ReLUGaussGradV

import six.moves.cPickle as pickle

#Speedup YAML using CLoader/CDumper
from yaml import load as yamlload
from yaml import dump as yamldump
try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

def AskYesNo():
  while 1:
    sys.stdout.write('  (y|n) > ')
    ans= sys.stdin.readline().strip()
    if ans=='y' or ans=='Y':  return True
    elif ans=='n' or ans=='N':  return False

def IfNone(x,y):
  return x if x!=None else y

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

'''Regularize a given covariance matrix (scalar and None are acceptable) to a [D,D] form.
    cov, D: input covariance and requested dimensionality (DxD).
    req_type: requesting the type of numpy.array elements.
    req_diag: requesting the output matrix is diagonal matrix (i.e. shape=[D,]).
    Return cov2, is_zero
      cov2: regularized covariance or None (error).
      is_zero: given covariance was zero or None.  '''
def RegularizeCov(cov, D, req_type=float, req_diag=False):
  is_scalar= False
  is_zero= None
  if cov is None:
    is_scalar= True
    cov= 0.0
    is_zero= True
  if isinstance(cov, (float, np.float_, np.float16, np.float32, np.float64)):
    is_scalar= True
    is_zero= (cov==0.0)
  if is_scalar:
    if req_diag:  cov2= np.array([cov]*D).astype(req_type)
    else:         cov2= np.diag(np.array([cov]*D).astype(req_type))
    return cov2, is_zero
  elif isinstance(cov,np.ndarray):
    is_zero= (cov==0.0).all()
    if cov.size==D:
      if req_diag:  cov2= np.array(cov.ravel(),req_type)
      else:         cov2= np.diag(np.array(cov.ravel(),req_type))
    else:
      cov2= np.array(cov,req_type)
      cov2= cov2.reshape(D,D)
      if req_diag:  cov2= np.diag(cov2)
    return cov2, is_zero
  elif isinstance(cov,(list,tuple)):
    if len(cov)==D:
      if req_diag:  cov2= np.array(cov,req_type)
      else:         cov2= np.diag(np.array(cov,req_type))
    else:
      cov2= np.array(cov,req_type)
      cov2= cov2.reshape(D,D)
      if req_diag:  cov2= np.diag(cov2)
    return cov2, (cov2==0.0).all()
  raise Exception('RegularizeCov: Unacceptable type:',type(cov))

def Median(array):
  if len(array)==0:  return None
  a_sorted= copy.deepcopy(array)
  a_sorted.sort()
  return a_sorted[len(a_sorted)/2]

def ToStr(*lists):
  s= ''
  delim= ''
  for v in lists:
    s+= delim+' '.join(map(str,list(v)))
    delim= ' '
  return s

def ToList(x):
  if x==None:  return []
  elif isinstance(x,list):  return x
  elif isinstance(x,(np.ndarray,np.matrix)):
    if len(x.shape)==1:  return x.tolist()
    if len(x.shape)==2:
      if x.shape[0]==1:  return x.tolist()[0]
      if x.shape[1]==1:  return x.T.tolist()[0]
      if x.shape[0]==0 and x.shape[1]==0:  return []
  raise Exception('ToList: Impossible to serialize:',x)

def Len(x):
  if x==None:  return 0
  elif isinstance(x,list):  return len(x)
  elif isinstance(x,(np.ndarray,np.matrix)):
    if len(x.shape)==1:  return x.shape[0]
    if len(x.shape)==2:
      if x.shape[0]==1:  return x.shape[1]
      if x.shape[1]==1:  return x.shape[0]
      if x.shape[0]==0 or x.shape[1]==0:  return 0
  raise Exception('Len: Impossible to serialize:',x)

#Convert a data into a standard python object
def ToStdType(x, except_cnv=lambda y:y):
  npbool= (np.bool_)
  npint= (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)
  npuint= (np.uint8, np.uint16, np.uint32, np.uint64)
  npfloat= (np.float_, np.float16, np.float32, np.float64)
  if isinstance(x, npbool):   return bool(x)
  if isinstance(x, npint):    return int(x)
  if isinstance(x, npuint):   return int(x)
  if isinstance(x, npfloat):  return float(x)
  if isinstance(x, (int,long,float,bool,str)):  return x
  if isinstance(x, np.ndarray):  return x.tolist()
  if isinstance(x, (list,tuple,set)):  return map(lambda x2:ToStdType(x2,except_cnv), x)
  if isinstance(x, dict):  return {ToStdType(k,except_cnv):ToStdType(v,except_cnv) for k,v in x.iteritems()}
  try:
    return {ToStdType(k,except_cnv):ToStdType(v,except_cnv) for k,v in x.__dict__.iteritems()}
  except AttributeError:
    return except_cnv(x)
    #pass
  print 'Failed to convert:',x
  print 'Type:',type(x)
  raise

#Insert a new dictionary to the base dictionary
def InsertDict(d_base, d_new):
  for k_new,v_new in d_new.iteritems():
    if k_new in d_base and (type(v_new)==dict and type(d_base[k_new])==dict):
      InsertDict(d_base[k_new], v_new)
    else:
      d_base[k_new]= v_new

#Print with a color (col can be a code or an int)
def CPrint(col,*s):
  if len(s)==0:
    print ''
  else:
    print s[0],
    for ss in s[1:]:
      print ss,
    print ''

#Open a file in 'w' mode.
#If the parent directory does not exist, we create it.
#  mode: 'w' in default.
#  interactive: ask user if creating the parent dir.
#    If False, we create the parent dir without asking.
def OpenW(file_name, mode='w', interactive=True):
  parent= os.path.dirname(file_name)
  if parent!='' and not os.path.exists(parent):
    if interactive:
      CPrint(2,'OpenW: Parent directory does not exist:',parent)
      CPrint(2,'Create? (if No, raise IOError)')
      if AskYesNo():  os.makedirs(parent)
      else:  raise IOError(2,'Can not open the file',file_name)
    else:
      CPrint(2,'OpenW: Creating parent directory:',parent)
      os.makedirs(parent)
  if os.path.exists(file_name) and mode[0]=='w' and interactive:
    CPrint(2,'OpenW: File exists:',file_name)
    CPrint(2,'Overwrite? (if No, raise IOError)')
    if not AskYesNo():  raise IOError(2,'Canceled to open file as file exists:',file_name)
  return open(file_name,mode)


#Load a YAML and return a dictionary
def LoadYAML(file_name):
  return yamlload(file(file_name).read(), Loader=Loader)

#Save a dictionary as a YAML
def SaveYAML(d, file_name, except_cnv=lambda y:y):
  file(file_name,'w').write(yamldump(ToStdType(d,except_cnv), Dumper=Dumper))



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



'''
Interface class of a function approximator.
We assume a data takes a form like:
  X=[[x1^T],  Y=[[y1^T],
     [x2^T],     [y2^T],
     [... ]]     [... ]]
where xn is an input vector, yn is an output vector (n=1,2,...).
'''
class TFunctionApprox(object):
  @staticmethod
  def DefaultOptions():
    Options= {}
    return Options
  @staticmethod
  def DefaultParams():
    Params= {}
    return Params

  #Number of samples
  @property
  def NSamples(self):
    return len(self.DataX)

  #Number of x-dimensions
  @property
  def Dx(self):
    return len(self.DataX[0]) if self.NSamples>0 else 0

  #Number of y-dimensions
  @property
  def Dy(self):
    return len(self.DataY[0]) if self.NSamples>0 else 0

  def __init__(self):
    self.Options= {}
    self.Params= {}
    self.Load(data={'options':self.DefaultOptions(), 'params':self.DefaultParams()})
    self.is_predictable= False
    self.load_base_dir= None

  #Load options and parameters from a dictionary.
  #base_dir: where external data file(s) are stored; None for a default value.
  #  Note: data may contain a filename like '{base}/func.dat'
  #        where {base} is supposed be replaced by base_dir.
  #        Use self.Locate to get the actual path (e.g. self.Locate('{base}/func.dat')).
  def Load(self, data=None, base_dir=None):
    if data!=None and 'options' in data: InsertDict(self.Options, data['options'])
    if data!=None and 'params' in data: InsertDict(self.Params, data['params'])
    self.load_base_dir= base_dir

  def Locate(self, filename):
    if filename.find('{base}')>=0 and self.load_base_dir==None:
      raise Exception('Use Load with specifying base_dir argument. Otherwise Locate() can not return the correct location for the filename: %s'%filename)
    return filename.format(base=self.load_base_dir)

  #Save options and parameters into a dictionary.
  #base_dir: used to store data into external data file(s); None for a default value.
  #  Note: returned dict may contain file path(s) containing data.
  #        Such path(s) my contain {base} which is actually base_dir.
  #        Those {base} will be replaced by base_dir when using Load().
  #        This is useful to move the data files and load them.
  def Save(self, base_dir=None):
    self.SyncParams(base_dir)
    data= {}
    data['options']= ToStdType(self.Options)
    data['params']= ToStdType(self.Params)
    return copy.deepcopy(data)

  #Synchronize Params (and maybe Options) with an internal learner to be saved.
  #base_dir: used to store data into external data file(s); None for a default value.
  def SyncParams(self, base_dir):
    pass

  #Whether prediction is available (False if the model is not learned).
  def IsPredictable(self):
    return self.is_predictable

  #Initialize approximator.  Should be executed before Update/UpdateBatch.
  def Init(self):
    self.DataX= []
    self.DataY= []
    self.is_predictable= False

  #Incrementally update the internal parameters with a single I/O pair (x,y).
  #If x and/or y are None, only updating internal parameters is done.
  def Update(self, x=None, y=None, not_learn=False):
    if x!=None or y!=None:
      self.DataX.append(list(x))
      self.DataY.append(list(y))
    if not_learn:  return

  #Incrementally update the internal parameters with I/O data (X,Y).
  #If x and/or y are None, only updating internal parameters is done.
  def UpdateBatch(self, X=None, Y=None, not_learn=False):
    if X!=None or Y!=None:
      self.DataX.extend(X)
      self.DataY.extend(Y)
    if not_learn:  return

  #Prediction result class.
  class TPredRes:
    def __init__(self):
      self.Y= None  #Prediction result.
      self.Var= None  #Covariance matrix.
      self.Grad= None  #Gradient.

  '''
  Do prediction.
    Return a TPredRes instance.
    x_var: Covariance of x.  If a scholar is given, we use diag(x_var,x_var,..).
    with_var: Whether compute a covariance matrix of error at the query point as well.
    with_grad: Whether compute a gradient at the query point as well.
  '''
  def Predict(self, x, x_var=0.0, with_var=False, with_grad=False):
    raise Exception('FIXME: Implement')


'''
#Dump function approximator to file for plot.
def DumpPlot(fa, f_reduce=lambda xa:xa, f_repair=lambda xa,mi,ma,me:xa, file_prefix='/tmp/f', x_var=0.0, n_div=50, bounds=None):
  if len(fa.DataX)==0:  print 'DumpPlot: No data'; return
  if not fa.IsPredictable():  print 'DumpPlot: Not predictable'; return
  if bounds!=None:
    xamin0,xamax0= bounds
  else:
    xamin0= [min([x[d] for x in fa.DataX]) for d in range(len(fa.DataX[0]))]
    xamax0= [max([x[d] for x in fa.DataX]) for d in range(len(fa.DataX[0]))]
  xamin= f_reduce(xamin0)
  xamax= f_reduce(xamax0)
  xmed= [Median([x[d] for x in fa.DataX]) for d in range(len(fa.DataX[0]))]
  if len(xamin)>=3 or len(xamin)!=len(xamax) or len(xamin)<=0:
    print 'DumpPlot: Invalid f_reduce function'
    return

  fp= open('%s_est.dat'%(file_prefix),'w')
  if len(xamin)==2:
    for xa1_1 in FRange1(xamin[0],xamax[0],n_div):
      for xa1_2 in FRange1(xamin[1],xamax[1],n_div):
        xa1r= [xa1_1,xa1_2]
        xa1= f_repair(xa1r, xamin0, xamax0, xmed)
        fp.write('%s\n' % ToStr(xa1r,xa1,ToList(fa.Predict(xa1,x_var).Y)))
      fp.write('\n')
  else:  #len(xamin)==1:
    for xa1_1 in FRange1(xamin[0],xamax[0],n_div):
      xa1r= [xa1_1]
      xa1= f_repair(xa1r, xamin0, xamax0, xmed)
      fp.write('%s\n' % ToStr(xa1r,xa1,ToList(fa.Predict(xa1,x_var).Y)))
  fp.close()
  fp= open('%s_smp.dat'%(file_prefix),'w')
  for xa1,x2 in zip(fa.DataX, fa.DataY):
    fp.write('%s\n' % ToStr(f_reduce(xa1),xa1,x2))
  fp.close()
'''












'''
Neural networks for classification..
We assume a data takes a form like:
  X=[[x1^T],  Y=[y1,
     [x2^T],     y2,
     [... ]]     ...]
where xn is an input vector, yn is an output class-label (n=1,2,...).
'''
class TNNClassification(TFunctionApprox):
  @staticmethod
  def DefaultOptions():
    Options= {}
    Options['name']= '' #Arbitrary name.
    Options['gpu']= -1  #Device ID of GPU (-1: not use).
    Options['n_units']= [1,200,200,1]  #Number of input/hidden/output units.

    Options['num_min_predictable']= 3  #Number of minimum samples necessary to train NNs.

    Options['init_bias_randomly']= True  #Initialize bias of linear models randomly.
    Options['bias_rand_init_bound']= [-1.0, 1.0]  #Bound used in random bias initialization.

    Options['dropout']= True  #If use dropout.
    Options['dropout_ratio']= 0.01  #Ratio of dropout.

    Options['AdaDelta_rho']= 0.9  #Parameter for AdaDelta.

    Options['batchsize']= 10  #Size of mini-batch.
    Options['num_max_update']= 5000  #Maximum number of updates with mini-batch.
    Options['num_check_stop']= 50  #Stop condition is checked for every this number of updates w mini-batch.
    Options['loss_maf_alpha']= 0.4  #Update ratio of moving average filter for loss.
    Options['loss_stddev_init']= 2.0  #Initial value of loss std-dev (unit (1.0) is 'loss_stddev_stop').
    Options['loss_stddev_stop']= 1.0e-3  #If std-dev of loss is smaller than this value, iteration stops.

    Options['base_dir']= '/tmp/dnn/'  #Base directory.  Last '/' matters.
    '''Some data (model.parameters, DataX, DataY)
        are saved into this file name when Save() is executed.
        label: 'model_mean', 'data_x', or 'data_y'.
        base: Options['base_dir'] or base_dir argument of Save method.'''
    Options['data_file_name']= '{base}nn_{label}.dat'
    '''Template of filename to store the training log.
        name: Options['name'].
        n: number of training executions.
        code: 'mean'.
        base: Options['base_dir'].'''
    Options['train_log_file']= '{base}train/nn_log-{n:05d}-{name}{code}.dat'

    Options['verbose']= True

    return Options
  @staticmethod
  def DefaultParams():
    Params= {}
    Params['nn_params']= None
    Params['nn_data_x']= None
    Params['nn_data_y']= None
    Params['num_train']= 0  #Number of training executions.
    return Params

  @staticmethod
  def ToVec(x):
    if x is None:  return np.array([],np.float32)
    elif isinstance(x,list):  return np.array(x,np.float32)
    elif isinstance(x,(np.ndarray,np.matrix)):
      return x.ravel().astype(np.float32)
    raise Exception('ToVec: Impossible to serialize:',x)

  def __init__(self):
    TFunctionApprox.__init__(self)

  '''
  NOTE
  In order to save and load model parameters,
    save: p=ToStdType(model.parameters)
    load: model.copy_parameters_from(map(lambda e:np.array(e,np.float32),p))
  '''

  #Synchronize Params (and maybe Options) with an internal learner to be saved.
  #base_dir: used to store data into external data file(s); None for a default value.
  def SyncParams(self, base_dir):
    TFunctionApprox.SyncParams(self, base_dir)
    if base_dir is None:  base_dir= self.Options['base_dir']
    L= lambda f: f.format(base=base_dir)
    if self.IsPredictable():
      #self.Params['nn_params']= ToStdType(self.model.parameters)
      #self.Params['nn_params_err']= ToStdType(self.model_err.parameters)
      self.Params['nn_params']= self.Options['data_file_name'].format(label='model_mean',base='{base}')
      pickle.dump(ToStdType(self.model.parameters), OpenW(L(self.Params['nn_params']), 'wb'), -1)

    if self.NSamples>0:
      self.Params['nn_data_x']= self.Options['data_file_name'].format(label='data_x',base='{base}')
      #fp= OpenW(L(self.Params['nn_data_x']), 'w')
      #for x in self.DataX:
        #fp.write('%s\n'%(' '.join(map(str,x))))
      #fp.close()
      pickle.dump(ToStdType(self.DataX), OpenW(L(self.Params['nn_data_x']), 'wb'), -1)

      self.Params['nn_data_y']= self.Options['data_file_name'].format(label='data_y',base='{base}')
      #fp= OpenW(L(self.Params['nn_data_y']), 'w')
      #for y in self.DataY:
        #fp.write('%s\n'%(' '.join(map(str,y))))
      #fp.close()
      pickle.dump(ToStdType(self.DataY), OpenW(L(self.Params['nn_data_y']), 'wb'), -1)

  #Initialize approximator.  Should be executed before Update/UpdateBatch.
  def Init(self):
    TFunctionApprox.Init(self)
    L= self.Locate
    if self.Params['nn_data_x'] != None:
      self.DataX= np.array(pickle.load(open(L(self.Params['nn_data_x']), 'rb')), np.float32)
    else:
      self.DataX= np.array([],np.float32)
    if self.Params['nn_data_y'] != None:
      self.DataY= np.array(pickle.load(open(L(self.Params['nn_data_y']), 'rb')), np.int32)
    else:
      self.DataY= np.array([],np.int32)

    self.CreateNNs()

    if self.Params['nn_params'] != None:
      #self.model.copy_parameters_from(map(lambda e:np.array(e,np.float32),self.Params['nn_params']))
      self.model.copy_parameters_from(map(lambda e:np.array(e,np.float32),pickle.load(open(L(self.Params['nn_params']), 'rb')) ))
      self.is_predictable= True
    else:
      if self.Options['init_bias_randomly']:
        self.InitBias(m='mean')

    if self.Options['gpu'] >= 0:
      cuda.init(self.Options['gpu'])
      self.model.to_gpu()
      self.model_err.to_gpu()

    self.optimizer= optimizers.AdaDelta(rho=self.Options['AdaDelta_rho'])
    self.optimizer.setup(self.model.collect_parameters())

  #Create neural networks.
  def CreateNNs(self):
    assert(len(self.Options['n_units'])>=2)
    #Mean model
    n_units= self.Options['n_units']
    self.f_names= ['l%d'%i for i in range(len(n_units)-1)]
    funcs= {}
    for i in range(len(n_units)-1):
      funcs[self.f_names[i]]= F.Linear(n_units[i],n_units[i+1])
    self.model= FunctionSet(**funcs)

  #Randomly initialize bias of linear models.
  def InitBias(self, m='both'):
    if m in ('both','mean'):
      for l in self.f_names:
        getattr(self.model,l).b[:]= [Rand(*self.Options['bias_rand_init_bound'])
                                     for d in range(getattr(self.model,l).b.size)]


  #Compute output (mean) for a set of x.
  def Forward(self, x_data, train):
    if not self.Options['dropout']:  train= False
    dratio= self.Options['dropout_ratio']
    x= Variable(x_data)
    h0= x
    for l in self.f_names[:-1]:
      h1= F.dropout(F.relu(getattr(self.model,l)(h0)), ratio=dratio, train=train)
      h0= h1
    y= getattr(self.model,self.f_names[-1])(h0)
    return y

  #Compute output (mean) and loss for sets of x and y.
  def FwdLoss(self, x_data, y_data, train):
    y= self.Forward(x_data, train)
    t= Variable(y_data)
    return F.softmax_cross_entropy(y, t), y


  #Forward computation of neural net considering input distribution.
  def ForwardX(self, x, x_var=None, with_var=False, with_grad=False):
    zero= np.float32(0)
    x= np.array(x,np.float32); x= x.reshape(x.size,1)

    #Error model:
    if with_var:
      raise NotImplementedError('TNNClassification.ForwardX: with_var==True')
    else:
      y_var0= None

    if with_grad:
      raise NotImplementedError('TNNClassification.ForwardX: with_grad==True')

    x_var, var_is_zero= RegularizeCov(x_var, x.size, np.float32)
    if var_is_zero:
      g= None  #Gradient
      h0= x
      for ln in self.f_names[:-1]:
        l= getattr(self.model,ln)
        hl1= l.W.dot(h0) + l.b.reshape(l.b.size,1)  #W h0 + b
        h1= np.maximum(zero, hl1)  #ReLU(hl1)
        if with_grad:
          g2= l.W.T.dot(np.diag((hl1>0.0).ravel().astype(np.float32)))  #W diag(step(hl1))
          g= g2 if g is None else g.dot(g2)
        h0= h1
      l= getattr(self.model,self.f_names[-1])
      y= l.W.dot(h0) + l.b.reshape(l.b.size,1)
      if with_grad:
        g= g2 if g is None else g.dot(l.W.T)
      return y, y_var0, g

    else:
      g= None  #Gradient
      h0= x
      h0_var= x_var
      for ln in self.f_names[:-1]:
        l= getattr(self.model,ln)
        hl1= l.W.dot(h0) + l.b.reshape(l.b.size,1)  #W h0 + b
        hl1_dvar= np.diag( l.W.dot(h0_var.dot(l.W.T)) ).reshape(hl1.size,1)  #diag(W h0_var W^T)
        h1,h1_dvar= ReLUGaussV(hl1,hl1_dvar)  #ReLU_gauss(hl1,hl1_dvar)
        h1_var= np.diag(h1_dvar.ravel())  #To a full matrix
        if with_grad:
          g2= l.W.T.dot(np.diag(ReLUGaussGradV(hl1,hl1_dvar).ravel()))
          g= g2 if g is None else g.dot(g2)
        h0= h1
        h0_var= h1_var
      l= getattr(self.model,self.f_names[-1])
      y= l.W.dot(h0) + l.b.reshape(l.b.size,1)
      y_var= None
      if with_var:
        y_var= l.W.dot(h0_var.dot(l.W.T)) + y_var0
      if with_grad:
        g= g2 if g is None else g.dot(l.W.T)
      return y, y_var, g

  #Training code common for mean model and error model.
  @staticmethod
  def TrainNN(**opt):
    N= len(opt['x_train'])
    loss_maf= TExpMovingAverage1(init_sd=opt['loss_stddev_init']*opt['loss_stddev_stop'],
                                 alpha=opt['loss_maf_alpha'])
    batchsize= min(opt['batchsize'], N)  #Adjust mini-batch size for too small N
    num_max_update= opt['num_max_update']
    n_epoch= num_max_update/(N/batchsize)+1
    is_updating= True
    n_update= 0
    sum_loss= 0.0
    fp= OpenW(opt['log_filename'],'w')
    for epoch in xrange(n_epoch):
      perm= np.random.permutation(N)
      # Train model per batch
      for i in xrange(0, N, batchsize):
        x_batch= opt['x_train'][perm[i:i+batchsize]]
        y_batch= opt['y_train'][perm[i:i+batchsize]]
        if opt['gpu'] >= 0:
          x_batch= cuda.to_gpu(x_batch)
          y_batch= cuda.to_gpu(y_batch)

        opt['optimizer'].zero_grads()
        loss, pred= opt['fwd_loss'](x_batch, y_batch, train=True)
        loss.backward()  #Computing gradients
        opt['optimizer'].update()
        n_update+= 1

        sum_loss+= float(cuda.to_cpu(loss.data))
        if n_update % opt['num_check_stop'] == 0:
          #loss_maf.Update(float(cuda.to_cpu(loss.data)))
          loss_maf.Update(sum_loss / opt['num_check_stop'])
          sum_loss= 0.0
          if opt['verb']:  print 'Training %s:'%opt['code'], epoch, n_update, loss_maf.Mean, loss_maf.StdDev
          fp.write('%d %d %f %f\n' % (epoch, n_update, loss_maf.Mean, loss_maf.StdDev))
          if loss_maf.StdDev < opt['loss_stddev_stop']:
            is_updating= False
            break
        if n_update >= num_max_update:
          is_updating= False
          break
      if not is_updating:  break
    fp.close()

  #Main update code in which we train the mean model, generate y-error data, train the error model.
  def UpdateMain(self):
    if self.NSamples < self.Options['num_min_predictable']:  return

    #Train mean model
    opt={
      'code': '{code}-{n:05d}'.format(n=self.Params['num_train'], code=self.Options['name']+'mean'),
      'log_filename': self.Options['train_log_file'].format(n=self.Params['num_train'], name=self.Options['name'], code='mean', base=self.Options['base_dir']),
      'verb': self.Options['verbose'],
      'gpu': self.Options['gpu'],
      'fwd_loss': self.FwdLoss,
      'optimizer': self.optimizer,
      'x_train': self.DataX,
      'y_train': self.DataY,
      'batchsize': self.Options['batchsize'],
      'num_max_update': self.Options['num_max_update'],
      'num_check_stop': self.Options['num_check_stop'],
      'loss_maf_alpha': self.Options['loss_maf_alpha'],
      'loss_stddev_init': self.Options['loss_stddev_init'],
      'loss_stddev_stop': self.Options['loss_stddev_stop'],
      }
    self.TrainNN(**opt)

    self.Params['num_train']+= 1

    #End of training NNs
    self.is_predictable= True


  #Incrementally update the internal parameters with a single I/O pair (x,y).
  #If x and/or y are None, only updating internal parameters is done.
  def Update(self, x=None, y=None, not_learn=False):
    #TFunctionApprox.Update(self, x, y, not_learn)
    if x!=None or y!=None:
      if len(self.DataX)==0:
        self.DataX= np.array([self.ToVec(x)],np.float32)
        self.DataY= np.array([y],np.int32)
      else:
        self.DataX= np.vstack((self.DataX, self.ToVec(x)))
        self.DataY= np.hstack((self.DataY, y)).astype(np.int32)
    if not_learn:  return
    self.UpdateMain()

  #Incrementally update the internal parameters with I/O data (X,Y).
  #If x and/or y are None, only updating internal parameters is done.
  def UpdateBatch(self, X=None, Y=None, not_learn=False):
    #TFunctionApprox.UpdateBatch(self, X, Y, not_learn)
    if X!=None or Y!=None:
      if len(self.DataX)==0:
        self.DataX= np.array(X, np.float32)
        self.DataY= np.array(Y, np.int32)
      else:
        self.DataX= np.vstack((self.DataX, np.array(X, np.float32)))
        self.DataY= np.hstack((self.DataY, np.array(Y, np.int32)))
    if not_learn:  return
    self.UpdateMain()

  '''
  Do prediction.
    Return a TPredRes instance.
    x_var: Covariance of x.  If a scholar is given, we use diag(x_var,x_var,..).
    with_var: Whether compute a covariance matrix of error at the query point as well.
    with_grad: Whether compute a gradient at the query point as well.
  '''
  def Predict(self, x, x_var=0.0, with_var=False, with_grad=False):
    res= self.TPredRes()
    #x_batch= np.array([self.ToVec(x)],np.float32)
    #if self.Options['gpu'] >= 0:
      #x_batch= cuda.to_gpu(x_batch)
    #pred= self.Forward(x_batch, train=False)
    #res.Y= cuda.to_cpu(pred.data)[0]
    #if with_var:
      #pred_err= self.ForwardErr(x_batch, train=False)
      #res.Var= np.diag(cuda.to_cpu(pred_err.data)[0])
      #res.Var= res.Var*res.Var
    y, y_var, g= self.ForwardX(x, x_var, with_var, with_grad)
    res.Y= F.softmax(Variable(y.reshape(1,y.size))).data[0]
    res.Var= None  #y_var NOTE: Not implemented.
    res.Grad= g
    return res


def TNNClassificationExample1():
  def GenData(ix1=None,ix2=None):
    #From IRIS
    data_x= [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2], [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3], [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1.0, 0.2], [5.1, 3.3, 1.7, 0.5], [4.8, 3.4, 1.9, 0.2], [5.0, 3.0, 1.6, 0.2], [5.0, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.0, 3.2, 1.2, 0.2], [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3.0, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2], [5.0, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5.0, 3.5, 1.6, 0.6], [5.1, 3.8, 1.9, 0.4], [4.8, 3.0, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2], [5.3, 3.7, 1.5, 0.2], [5.0, 3.3, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5], [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4], [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1], [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4.0, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2], [6.4, 2.9, 4.3, 1.3], [6.6, 3.0, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3.0, 5.0, 1.7], [6.0, 2.9, 4.5, 1.5], [5.7, 2.6, 3.5, 1.0], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1.0], [5.8, 2.7, 3.9, 1.2], [6.0, 2.7, 5.1, 1.6], [5.4, 3.0, 4.5, 1.5], [6.0, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5], [6.3, 2.3, 4.4, 1.3], [5.6, 3.0, 4.1, 1.3], [5.5, 2.5, 4.0, 1.3], [5.5, 2.6, 4.4, 1.2], [6.1, 3.0, 4.6, 1.4], [5.8, 2.6, 4.0, 1.2], [5.0, 2.3, 3.3, 1.0], [5.6, 2.7, 4.2, 1.3], [5.7, 3.0, 4.2, 1.2], [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3.0, 1.1], [5.7, 2.8, 4.1, 1.3], [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2], [7.6, 3.0, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8], [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2.0], [6.4, 2.7, 5.3, 1.9], [6.8, 3.0, 5.5, 2.1], [5.7, 2.5, 5.0, 2.0], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3.0, 5.5, 1.8], [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6.0, 2.2, 5.0, 1.5], [6.9, 3.2, 5.7, 2.3], [5.6, 2.8, 4.9, 2.0], [7.7, 2.8, 6.7, 2.0], [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1], [7.2, 3.2, 6.0, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3.0, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1], [7.2, 3.0, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2.0], [6.4, 2.8, 5.6, 2.2], [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3.0, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4], [6.4, 3.1, 5.5, 1.8], [6.0, 3.0, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4], [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5], [6.7, 3.0, 5.2, 2.3], [6.3, 2.5, 5.0, 1.9], [6.5, 3.0, 5.2, 2.0], [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 5.1, 1.8]]
    data_x= [x[ix1:ix2] for x in data_x]
    data_y= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    return data_x, data_y

  #Dump data with dimension reduction f_reduce.
  #Each row of dumped data: reduced x, original x, original y
  def DumpData(file_name, data_x, data_y, f_reduce, lb=0):
    fp1= file(file_name,'w')
    for x,y,i in zip(data_x,data_y,range(len(data_y))):
      if lb>0 and i%lb==0:  fp1.write('\n')
      fp1.write('%s  %s  %s\n' % (' '.join(map(str,f_reduce(x))), ' '.join(map(str,x)), ' '.join(map(str,y))))
    fp1.close()

  #Return min, max, median vectors of data.
  def GetStat(data):
    mi= [min([x[d] for x in data]) for d in range(len(data[0]))]
    ma= [max([x[d] for x in data]) for d in range(len(data[0]))]
    me= [Median([x[d] for x in data]) for d in range(len(data[0]))]
    return mi,ma,me

  #load_model,train_model= False,True
  load_model,train_model= True,False

  if True or train_model:
    x_train,y_train= GenData(1,3)
    mi,ma,me= GetStat(x_train)
    f_reduce=lambda xa:[xa[0],xa[1]]
    f_repair=lambda xa:[xa[0],xa[1]]

    print 'Num of samples for train:',len(y_train)
    DumpData('/tmp/dnn/smpl_train.dat', x_train, [[y] for y in y_train], f_reduce)  #DIFF_REG

  #x_test= np.array([[x] for x in FRange1(*Bound,num_div=100)]).astype(np.float32)
  #y_test= np.array([[TrueFunc(x[0])] for x in x_test]).astype(np.int32)
  nt= 20+1
  x_test= np.array(sum([[f_repair([x1,x2]) for x2 in FRange1(f_reduce(mi)[1],f_reduce(ma)[1],nt)] for x1 in FRange1(f_reduce(mi)[0],f_reduce(ma)[0],nt)],[])).astype(np.float32)

  ## Dump data for plot:
  #fp1= file('/tmp/dnn/smpl_test.dat','w')
  #for x,y in zip(x_test,y_test):
    #fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,str(y)))
  #fp1.close()

  batch_train= True
  #batch_train= False
  options= {}
  options['n_units']= [len(x_train[0]),200,200,200,3]
  #options['AdaDelta_rho']= 0.5
  #options['AdaDelta_rho']= 0.9
  #options['dropout']= False
  #options['dropout_ratio']= 0.01
  options['loss_stddev_stop']= 1.0e-4
  options['num_max_update']= 5000
  #options['batchsize']= 5
  #options['batchsize']= 10
  #options['num_check_stop']= 50
  #options['loss_maf_alpha']= 0.4
  options['loss_stddev_stop']= 1.0e-4
  model= TNNClassification()
  #print 'model.Options=',model.Options
  model.Load({'options':options})
  if load_model:
    model.Load(LoadYAML('/tmp/dnn/nn_model.yaml'), '/tmp/dnn/')
  model.Init()
  #print 'model.Options=',model.Options
  if train_model:
    if not batch_train:
      for x,y,n in zip(x_train,y_train,range(len(x_train))):
        print '========',n,'========'
        model.Update(x,y,not_learn=((n+1)%min(10,len(x_train))!=0))
      #model.Update()
    else:
      #model.Options['dropout_ratio']= options['dropout_ratio']
      model.UpdateBatch(x_train,y_train)
      #model.Options['dropout_ratio']= 0.0
      #model.UpdateBatch()

  if not load_model:
    SaveYAML(model.Save('/tmp/dnn/'), '/tmp/dnn/nn_model.yaml')


  # Dump data for plot:
  preds= []
  for x in x_test:
    with_var,with_grad= False, False
    pred= model.Predict(x,x_var=0.0**2,with_var=with_var,with_grad=with_grad)
    preds.append(pred.Y.tolist())
  y_pred= [[y.index(max(y))]+y for y in preds]
  DumpData('/tmp/dnn/nn_test1.dat', x_test, y_pred, f_reduce, lb=nt+1)

  # Dump data for plot:
  preds= []
  for x in x_test:
    with_var,with_grad= False, False
    pred= model.Predict(x,x_var=0.5**2,with_var=with_var,with_grad=with_grad)
    preds.append(pred.Y.tolist())
  y_pred= [[y.index(max(y))]+y for y in preds]
  DumpData('/tmp/dnn/nn_test2.dat', x_test, y_pred, f_reduce, lb=nt+1)

def TNNClassificationExample1PlotGraphs():
  print 'Plotting graphs..'
  import os,sys
  opt= sys.argv[2:]
  NEpoch= 1
  commands=[
    '''qplot -x2 aaa {opt}
          -s 'set xlabel "x1";set ylabel "x2";set title "";'
          -s 'set encoding utf8;symbol(z)="+xo%#"[int(z):int(z)];'
          /tmp/dnn/nn_test{NEpoch:d}.dat u 1:2:'(symbol($5+1))' w labels textcolor lt 3 t '"Final({NEpoch}) epoch"'
          /tmp/dnn/smpl_train.dat u 1:2:'(symbol($5+1))' w labels textcolor lt 1 t '"sample"'  &''',
          #/tmp/dnn/lwr/f1_3_est.dat w l lw 1 t '"LWR"'
          #/tmp/dnn/nn_test0001.dat w l t '"1st epoch"'
          #/tmp/dnn/nn_test0005.dat w l t '"5th epoch"'
          #/tmp/dnn/nn_test0020.dat w l t '"20th epoch"'
          #/tmp/dnn/nn_test0050.dat w l t '"50th epoch"'
          #/tmp/dnn/nn_test0075.dat w l t '"75th epoch"'
          #/tmp/dnn/nn_test0099.dat w l t '"99th epoch"'
    '''qplot -x2 aaa {opt} -3d
          -s 'set xlabel "x1";set ylabel "x2";set title "Class 0";'
          -s 'set pm3d;unset surface;set view map;'
          -s 'set encoding utf8;symbol(z)="+xo%#"[int(z):int(z)];'
          /tmp/dnn/nn_test{NEpoch:d}.dat u 1:2:6 t '""'
          /tmp/dnn/smpl_train.dat u 1:2:'(0.0)':'(symbol($5+1))' w labels textcolor lt 1 t '"sample"'  &''',
    '''qplot -x2 aaa {opt} -3d
          -s 'set xlabel "x1";set ylabel "x2";set title "Class 1";'
          -s 'set encoding utf8;symbol(z)="+xo%#"[int(z):int(z)];'
          -s 'set pm3d;unset surface;set view map;'
          /tmp/dnn/nn_test{NEpoch:d}.dat u 1:2:7 t '""'
          /tmp/dnn/smpl_train.dat u 1:2:'(0.0)':'(symbol($5+1))' w labels textcolor lt 1 t '"sample"'  &''',
    '''qplot -x2 aaa {opt} -3d
          -s 'set xlabel "x1";set ylabel "x2";set title "Class 2";'
          -s 'set pm3d;unset surface;set view map;'
          -s 'set encoding utf8;symbol(z)="+xo%#"[int(z):int(z)];'
          /tmp/dnn/nn_test{NEpoch:d}.dat u 1:2:8 t '""'
          /tmp/dnn/smpl_train.dat u 1:2:'(0.0)':'(symbol($5+1))' w labels textcolor lt 1 t '"sample"'  &''',
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
    TNNClassificationExample1PlotGraphs()
    sys.exit(0)
  TNNClassificationExample1()


