#! /usr/bin/env python
#This is a collection of optimization tools, based on: pr2_lfd_trick/src/base/base_opt.py
#Including (thus, newer than) disc_opt.py
#
#Finally the new codes are copied into pr2_lfd_trick/src/base/base_opt2.py

import numpy as np
import numpy.linalg as la
import math
import random
import copy
import os
import time
import cma

#------------------------------------------------
#copied from base_const:
#------------------------------------------------

#------------------------------------------------
#copied from base_util:
#------------------------------------------------

#Float version of range
def FRange1(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

#Check if a is between [a_range[0],a_range[1]]
def IsIn(a, a_range):
  if a_range[0]<a_range[1]:
    return a_range[0]<=a and a<=a_range[1]
  else:
    return a_range[1]<=a and a<=a_range[0]

#Check if a is between [a_range[0][0],a_range[0][1]] or [a_range[1][0],a_range[1][1]] or ...
def IsIn2(a, a_range):
  for a_r in a_range:
    if IsIn(a,a_r):  return True
  return False


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

#ASCII colors
class ACol:
  Fail    = '\033[91m'
  OKGreen = '\033[92m'
  Warning = '\033[93m'
  OKBlue  = '\033[94m'
  Header  = '\033[95m'
  EndC    = '\033[0m'
  #Get a color code from an index:
  @staticmethod
  def I(index=None):
    if   index==0:   return ACol.Warning
    elif index==1:   return ACol.OKBlue
    elif index==2:   return ACol.OKGreen
    elif index==3:   return ACol.Header
    elif index==4:   return ACol.Fail
    return ACol.EndC
  #Get two color codes (start and stop) by guessing:
  @staticmethod
  def X2(col=None):
    if col is None or col=='':  return '',''
    if isinstance(col,int):  return ACol.I(col),ACol.EndC
    return col,ACol.EndC

#Get a colored string
def CStr(col,*s):
  c1,c2= ACol.X2(col)
  return c1 + ' '.join(map(str,s)) + c2

#Print with a color (col can be a code or an int)
def CPrint(col,*s):
  if len(s)==0:
    print ''
  else:
    c1,c2= ACol.X2(col)
    print c1+str(s[0]),
    for ss in s[1:]:
      print ss,
    print c2


#------------------------------------------------
#copied from base_geom:
#------------------------------------------------

#------------------------------------------------
#copied from base_adv:
#------------------------------------------------


#------------------------------------------------
#copied from base_opt:
#------------------------------------------------

#Return a Boltzmann policy (probabilities of selecting each action)
def BoltzmannPolicy(tau, values):
  if len(values)==0: return []
  max_v= max(values)
  sum_v= 0.0
  for q in values:
    sum_v+= math.exp((q-max_v)/tau)
  if sum_v<1.0e-10:
    return [1.0/float(len(values))]*len(values)
  probs= [0.0]*len(values)
  for d in range(len(values)):
    probs[d]= math.exp((values[d]-max_v)/tau)/sum_v
  return probs

#Return an action selected w.r.t. the policy (probabilities of selecting each action)
def SelectFromPolicy(probs):
  p= random.random()  #Random number in [0,1]
  action= 0
  for prob in probs:
    if p<=prob:  return action
    p-= prob
    action+= 1
  return action-1


#------------------------------------------------
#NEW:
#------------------------------------------------

#Return a Boltzmann policy (probabilities of selecting each action), considering valid options.
#  valid_options: a set of valid options.
#  The probabilities of options that not contained in valid_options are zero.
def BoltzmannPolicy2(tau, values, valid_options):
  if len(valid_options)==0: return []
  max_v= max([values[o] for o in valid_options])
  sum_v= 0.0
  for o in valid_options:
    q= values[o]
    sum_v+= math.exp((q-max_v)/tau)
  if sum_v<1.0e-10:
    return [1.0/float(len(valid_options))]*len(values)
  probs= [0.0]*len(values)
  for o in valid_options:
    probs[o]= math.exp((values[o]-max_v)/tau)/sum_v
  return probs

def UpdateMean(mean, new_value, alpha):
  return alpha*new_value + (1.0-alpha)*mean
def UpdateSqMean(sqmean, new_value, alpha):
  return alpha*(new_value**2) + (1.0-alpha)*sqmean
def GetSTD(mean, sqmean):
  return math.sqrt(max(0.0,sqmean-mean**2))




#Optimizer interface
class TOptimizerInterface:
  @staticmethod
  def DefaultOptions():
    Options= {}
    return Options
  @staticmethod
  def DefaultParams():
    Params= {}
    return Params

  def __init__(self):
    self.Options= {}
    self.Params= {}
    self.Init(data={'options':self.DefaultOptions(), 'params':self.DefaultParams()})

  #Initialize learner.  data: set to continue from previous state, generated by Save.
  def Init(self, data=None):
    if data<>None and 'options' in data: InsertDict(self.Options, data['options'])
    if data<>None and 'params' in data: InsertDict(self.Params, data['params'])
    #InsertDict(self.__dict__, self.Options)  #Map options to the dict for speed up

  #Save internal parameters as a dictionary.
  def Save(self):
    self.SyncParams()
    data= {}
    data['options']= ToStdType(self.Options)
    data['params']= ToStdType(self.Params)
    return copy.deepcopy(data)

  #Synchronize Params (and maybe Options) with an internal optimizer to be saved.
  def SyncParams(self):
    pass

  #Set an initial parameter. Should be executed after Init.
  def SetParam0(self, param0):
    return None

  #Returns an initial parameter.
  def Param0(self):
    return None

  #Return the obtained parameter, its score.
  def Result(self):
    return None, None

  #Check the stop condition.
  def Stopped(self):
    return True

  #Returns the latest selected parameter.
  def Param(self):
    return None

  #Select a parameter.  Use Param() to refer to the selected parameter.
  def Select(self):
    return None

  #Update with a selected parameter and a score (positive is better) which can be None.
  def Update(self,score):
    pass

  #Update with a given parameter and a score (positive is better) which can be None.
  def UpdateWith(self,param,score):
    pass

  #Generate a random number inside a bound. Should be executed after Init.
  def Rand(self):
    return None

  #Search [[f,x]*num] such that f=fobj(x) is not None and x is in the bound.
  #Should be executed after Init.
  #max_count: Algorithm stops after max_count evaluations of fobj.
  def InitialGuess(self, fobj, num=1, max_count=None):
    fx_data= []
    if num<=0:  return fx_data
    while max_count==None or max_count>0:
      x= self.Rand()
      f= fobj(x)
      if f is not None:
        fx_data.append([f,x])
        if len(fx_data)>=num:  return fx_data
      if max_count is not None:  max_count-= 1
    return fx_data

#------------------------------------------------
'''NOTE: class TDiscOptProb is a new version of class TDiscParam.
Example:
  def fobj(x,f_none=None):
    if 100<x and x<120:
      return 1.0-float((x-110)**2)/10.0
    return f_none

  opt= TDiscOptProb()
  options= {}
  options['mapped_values']= range(200)
  options['init_std_dev']= 0.5
  opt.Init({'options':options})

  count= 0
  while not opt.Stopped():
    x= opt.Select()
    f= fobj(x)
    opt.Update(f)
    count+= 1
  print 'Result=',opt.Result(),'obtained in',count
'''
#------------------------------------------------

#Optimizing a discrete parameter with a probabilistic fashion
class TDiscOptProb (TOptimizerInterface):
  @staticmethod
  def DefaultOptions():
    Options= {}
    Options['mapped_values']= [] #Mapping from an index to a value of any type
    Options['boltzmann_tau']= 0.1  #Temparature parameter for the Boltzmann selection.
    Options['ucb_nsd']= 1.0
    Options['alpha']= 0.2
    Options['none_to_score']= -1.0  #Treating score==None as this value
    Options['init_std_dev']= 1.0
    Options['tol_same_opt']= 5  #Stop condition of choosing the same option (index)
    Options['tol_score']= 1.0e-6  #Stop condition of score
    Options['maxfevals']= 1000000
    Options['param0_mean']= 0.1  #Assigned to a mean in SetParam0
    return Options
  @staticmethod
  def DefaultParams():
    Params= {}
    Params['means']= []
    Params['sq_means']= []
    return Params

  def __init__(self):
    TOptimizerInterface.__init__(self)

  #Initialize learner.  data: set to continue from previous state, generated by Save.
  def Init(self, data=None):
    TOptimizerInterface.Init(self,data)
    #Index of the parameter vector lastly selected:
    self.index= -1
    self.same_opt_cnt= 0
    self.prev_index= None
    self.score_mean= None
    self.score_sqmean= None
    self.fevals= self.Options['maxfevals']
    if len(self.Options['mapped_values'])!=len(self.Params['means']):
      self.Params['means']= [0.0]*len(self.Options['mapped_values'])
      self.Params['sq_means']= [self.Options['init_std_dev']**2]*len(self.Options['mapped_values'])

  #Set an initial parameter. Should be executed after Init.
  def SetParam0(self, param0):
    index= self.Options['mapped_values'].index(param0)
    self.Params['means'][index]= self.Options['param0_mean']
    return None

  #Returns an initial parameter.
  def Param0(self):
    return self.Result()[0]

  def UCB(self):
    return [self.Params['means'][d] + self.Options['ucb_nsd']*GetSTD(self.Params['means'][d], self.Params['sq_means'][d]) for d in range(len(self.Options['mapped_values']))]

  #Return the best parameter, mean-score.
  def Result(self):
    best_s,best_o= max(zip(self.Params['means'], range(len(self.Options['mapped_values']))))
    return best_o, best_s

  #Check the stop condition.
  def Stopped(self):
    if self.fevals<=0:  return True
    if self.same_opt_cnt>=self.Options['tol_same_opt']:  return True
    if self.score_mean<>None and self.score_sqmean<>None and GetSTD(self.score_mean,self.score_sqmean)<self.Options['tol_score']:  return True
    return False

  #Returns the latest selected parameter.
  def Param(self):
    if len(self.Options['mapped_values'])>0 and self.index>=0:
      return self.Options['mapped_values'][self.index]
    return None

  #Select a parameter.  Use Param() to refer to the selected parameter.
  def Select(self):
    if len(self.Options['mapped_values'])>0:
      if len(self.Options['mapped_values'])!=len(self.Params['means']):
        self.Params['means']= [0.0]*len(self.Options['mapped_values'])
        self.Params['sq_means']= [self.Options['init_std_dev']**2]*len(self.Options['mapped_values'])
        #print 'Warning: Means is not initialized.  Using %r.' % (self.Params['means'])
      ucb= self.UCB()
      probs= BoltzmannPolicy(self.Options['boltzmann_tau'],ucb)

      self.prev_index= self.index
      self.index= SelectFromPolicy(probs)
      if self.prev_index<>None and self.index==self.prev_index: self.same_opt_cnt+=1
      else:  self.same_opt_cnt= 0
      #CPrint(1,'TDiscOptProb:DEBUG: Param:%r Index:%i UCB:%f' % (self.Options['mapped_values'][self.index],self.index,ucb[self.index]))
    else:
      self.index= -1
    return self.Param()

  #Update with a selected parameter and a score (positive is better) which can be None.
  def Update(self,score):
    if self.index>=0:
      self.UpdateWithIS(self.index,score)

  #Update with a given parameter and a score (positive is better) which can be None.
  def UpdateWith(self,param,score):
    index= self.Options['mapped_values'].index(param)
    self.UpdateWithIS(index,score)

  #Update with a given index and a score (positive is better) which can be None.
  def UpdateWithIS(self,index,score):
    self.fevals-= 1
    if score==None:  score= self.Options['none_to_score']
    self.Params['means'][index]= UpdateMean(self.Params['means'][index], score, self.Options['alpha'])
    self.Params['sq_means'][index]= UpdateSqMean(self.Params['sq_means'][index], score, self.Options['alpha'])
    if self.score_mean==None:
      self.score_mean= 0.0
      self.score_sqmean= self.Options['init_std_dev']**2
    else:
      self.score_mean= UpdateMean(self.score_mean, score, self.Options['alpha'])
      self.score_sqmean= UpdateSqMean(self.score_sqmean, score, self.Options['alpha'])
    #CPrint(1,'TDiscOptProb:DEBUG: Index:%i Score:%f New-Mean:%f' % (index,score,self.Params['means'][index]))

  #Generate a random number inside a bound. Should be executed after Init.
  def Rand(self):
    mv= self.Options['mapped_values']
    return mv[random.randint(0,len(mv)-1)]


#------------------------------------------------
'''NOTE: class TContOptNoGrad is a new version of class TContParamNoGrad.
Example:
  def fobj(x,f_none=None):
    assert len(x)==2
    if (x[0]-0.5)**2+(x[1]+0.5)**2<0.2:  return f_none
    return 1.0 - 3.0*(x[0]-1.2)**2 - 2.0*(x[1]+2.0)**2

  opt= TContOptNoGrad()
  options= {}
  options['bounds']= [[-3.0,-3.0],[3.0,3.0]]
  options['tolfun']= 1.0e-4
  options['scale0']= 0.5
  options['parameters0']= [1.6,2.5]
  opt.Init({'options':options})

  count= 0
  while not opt.Stopped():
    x= opt.Select()
    f= fobj(x)
    opt.Update(f)
    count+= 1
  print 'Result=',opt.Result(),'obtained in',count
'''
#------------------------------------------------

#Optimizer (wrapper of CMA-ES) for a continuous value vector whose gradient is unknown:
class TContOptNoGrad (TOptimizerInterface):
  @staticmethod
  def DefaultOptions():
    #Options of TContOptNoGrad includes ones for CMA-ES
    Options= {}
    #Only for TContOptNoGrad:
    Options['parameters0']= []  #Initial parameters
    Options['scale0']= 1.0  #Initial scale
    #Common for CMA-ES:
    Options['verb_time']= 0
    Options['verb_log']= False
    Options['CMA_diagonal']= 1
    Options['bounds']= [[],[]]  #[Min_vector, Max_vector]
    Options['maxfevals']= 1000000
    return Options
  @staticmethod
  def DefaultParams():
    Params= {}
    Params['xmean']= None  #Has more priority than Options['parameters0']
    Params['stds']= None
    Params['solutions']= []
    Params['scores']= []
    Params['generation']= 0
    return Params

  def __init__(self):
    TOptimizerInterface.__init__(self)
    #self.Logger= '%s/tmp/cmaes%02i%02i%02i%02i%02i%02i.dat' % (os.environ['HOME'],t.tm_year%100,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)

  #Initialize learner.  data: set to continue from previous state, generated by Save.
  def Init(self, data=None):
    TOptimizerInterface.Init(self,data)
    self.es= None
    self.es_gen= None
    self.curr_param= None
    #self.tmpfp= file(self.Logger,'a')
    #self.tmpfp= file(self.Logger,'w')
    self.CreateES()

  #Synchronize Params (and maybe Options) with an internal optimizer to be saved.
  def SyncParams(self):
    if self.es:
      self.Params['xmean']= self.es.result()[5]
      self.Params['stds']= self.es.result()[6]

  #Create a CMA-ES learner.
  def CreateES(self):
    if self.Params['xmean']<>None and self.Params['stds']<>None:
      #Continue from previous learning stage:
      self.Options['parameters0']= self.Params['xmean']
      self.Options['scaling_of_variables']= self.Params['stds']
      self.es= None
      self.es_gen= lambda:cma.CMAEvolutionStrategy(self.Options['parameters0'], self.Options['scale0'], self.Options)
    else:
      #Start new learning:
      self.es= None
      self.es_gen= lambda:cma.CMAEvolutionStrategy(self.Options['parameters0'], self.Options['scale0'], self.Options)
    self.curr_param= None
    self.fevals= self.Options['maxfevals']

  def lazy_es_gen(self):
    if self.es_gen==None:  self.CreateES()
    self.es= self.es_gen()

  #Set an initial parameter. Should be executed after Init.
  def SetParam0(self, param0):
    if self.Params['xmean']<>None and self.Params['stds']<>None:
      self.Params['xmean']= param0
      #self.Options['scale0']= self.Options['scale0'] * self.Options['scale0_ratio']
    else:
      self.Options['parameters0']= param0
      #self.Options['scale0']= self.Options['scale0'] * self.Options['scale0_ratio']
    self.CreateES()

  #Returns an initial parameter.
  def Param0(self):
    if self.Params['xmean']<>None and self.Params['stds']<>None:
      return self.Params['xmean']
    else:
      return self.Options['parameters0']

  #Return the parameter, score.
  def Result(self):
    res= self.es.result()
    if res[0]<>None and res[1]<>np.inf:
      return res[0], -res[1]
    elif len(self.Params['scores'])>0:
      score,params= min(zip(self.Params['scores'],self.Params['solutions']))
      return params, -score
    else:
      return None, None

  #Check the stop condition.
  def Stopped(self):
    #TEST
    if self.es==None:  return self.es_gen==None
    #if es.result()[1]<>np.inf and -es.result()[1]>0.8:  break True
    return self.es.stop() or self.fevals<=0

  #Returns the latest selected parameter.
  def Param(self):
    return self.curr_param

  #Select a parameter.  Use Param() to refer to the selected parameter.
  def Select(self):
    if self.es==None:  self.lazy_es_gen()
    self.curr_param= self.es.ask(1)[0]
    #CPrint(1,'TContOptNoGrad:DEBUG: Param:%r' % (self.curr_param))
    return self.curr_param

  #Update with a selected parameter and a score (positive is better) which can be None.
  def Update(self,score):
    self.UpdateWith(self.curr_param,score)

  #Update with a given parameter and a score (positive is better) which can be None.
  def UpdateWith(self,param,score):
    if self.es==None:  self.lazy_es_gen()
    self.fevals-= 1
    if score<>None and param<>None:
      self.Params['solutions'].append(param)
      self.Params['scores'].append(-score)
      if len(self.Params['scores'])>=self.es.popsize:
        self.es.tell(self.Params['solutions'], self.Params['scores'])
        self.es.disp()
        #for i in range(len(self.Params['solutions'])):
          #self.tmpfp.write('%i %s %f\n'%(self.Params['generation'],' '.join(map(str,self.Params['solutions'][i])),self.Params['scores'][i]))
        #self.tmpfp.write('\n\n')
        self.Params['solutions']= []
        self.Params['scores']= []
        self.Params['generation']+= 1
    #CPrint(1,'TContOptNoGrad:DEBUG: Score:%r' % (score,))

  #Generate a random number inside a bound. Should be executed after Init.
  def Rand(self):
    xmin= self.Options['bounds'][0]
    xmax= self.Options['bounds'][1]
    dim= len(xmin)
    return [xmin[d]+(xmax[d]-xmin[d])*random.random() for d in range(dim)]


#------------------------------------------------
'''Representation of composite parameter structure:
  param_struct= {}
  param_struct['name']= <str>
  param_struct['kind']= 'disc_prob' or 'cont_no_grad' or 'composite'
  'composite':
    param_struct['sub']= [<param_struct>]
  or  [<str(name)>, <str(kind)>] if kind in ('disc_prob', 'cont_no_grad')
      [<str(name)>, <str(kind)>, <list[](sub)>] if kind == 'composite'
  abbreviation: 'd':'disc_prob', 'c':'cont_no_grad', 'x':'composite'

  e.g.
  param_struct= {}
  param_struct['name']= 'c1'
  param_struct['kind']= 'cont_no_grad'
  or ['c1','c']
  actual parameter is like: [0.1, 0.2]

  e.g.
  param_struct= {}
  param_struct['name']= 'd1'
  param_struct['kind']= 'disc_prob'
  or ['d1','d']
  actual parameter is like: 102 or 'aaa'
  #The discrete optimizer can map a discrete value to any value.

  e.g.
  param_struct= {}
  param_struct['name']= 'x1'
  param_struct['kind']= 'composite'
  param_struct['sub']= [
      {'name':'c1','kind':'cont_no_grad'},
      {'name':'c2','kind':'cont_no_grad'}]
  or ['x1','x',[['c1','c'],['c2','c']]]
  actual parameter is like: [0, [0.1, 0.2]] or [1, [1.0, 2.0]]
  #The first element is an index of composite's sub item.

  e.g.
  param_struct= {}
  param_struct['name']= 'x1'
  param_struct['kind']= 'composite'
  param_struct['sub']= []
  param_struct['sub'].append({'name':'c1','kind':'cont_no_grad'})
  param_struct['sub'].append({})
  param_struct['sub'][-1]['name']= 'x2'
  param_struct['sub'][-1]['kind']= 'composite'
  param_struct['sub'][-1]['sub']= [
      {'name':'c2','kind':'cont_no_grad'},
      {'name':'c3','kind':'cont_no_grad'}]
  or ['x1','x',[ ['c1','c'], ['x2','x',[['c2','c'],['c3','c']]] ]]
  actual parameter is like: [0, [0.1, 0.2]] or [1, [0, [0.1, 0.2]] ] or [1, [1, [1.0, 2.0]] ]

class TCompositeOpt recognizes the abbreviated representation like ['c1','c'].
Example:
  #Discrete parameter is a type of function
  #Note that each function assumes different size of x
  def fobj1(x,f_none=None):
    assert len(x)==2
    if not IsIn(x[0], [-3.0,-0.5]):  return f_none
    if (x[0]+2.0)**2+(x[1]+0.5)**2<0.2:  return f_none
    return -0.01*(3.0*(x[0]-1.2)**2 + 2.0*(x[1]+2.0)**2)

  def fobj2(x,f_none=None):
    assert len(x)==2
    if not IsIn(x[0], [0.0,1.0]):  return f_none
    return -0.01*(0.5*(x[0]+2.0)**2 + 0.5*(x[1]+2.0)**2 + 20.0)

  def fobj3(x,f_none=None):
    assert len(x)==1
    if not IsIn(x[0], [1.5,3.0]):  return f_none
    return -0.01*(20.0*(x[0]-2.0)**2 - 15.0)

  #x[0]: discrete parameter to select function
  #x[1]: continuous parameters for each function; size depends on x[0]
  def fobj(x,f_none=None):
    if x[0]==0:    f= fobj1(x[1])
    elif x[0]==1:  f= fobj2(x[1])
    elif x[0]==2:  f= fobj3(x[1])
    if f is None:  return f_none
    return f

  opt= TCompositeOpt()
  options= {}
  options['param_struct']= ['x1','x',[ ['c1','c'], ['c2','c'], ['c3','c'] ]]
  options['x1']= {}
  options['x1']['alpha']= 0.2
  options['c1']= {}
  options['c1']['bounds']= [[-3.0,-3.0],[3.0,3.0]]
  options['c1']['tolfun']= 1.0e-4
  options['c1']['scale0']= 1.0
  options['c1']['parameters0']= [-2.0,0.0]
  options['c2']= {}
  options['c2']['bounds']= [[-3.0,-3.0],[3.0,3.0]]
  options['c2']['tolfun']= 1.0e-4
  options['c2']['scale0']= 1.0
  options['c2']['parameters0']= [0.5,0.0]
  options['c3']= {}
  options['c3']['bounds']= [[-3.0],[3.0]]
  options['c3']['tolfun']= 1.0e-4
  options['c3']['scale0']= 1.0
  options['c3']['parameters0']= [2.0]
  opt.Init({'options':options})

  count= 0
  while not opt.Stopped():
    x= opt.Select()
    #print count,'#',x
    f= fobj(x)
    opt.Update(f)
    count+= 1
  print 'Result=',ToStdType(opt.Result()),'obtained in',count
'''
#------------------------------------------------

#Optimizer for a composite parameter vector:
class TCompositeOpt (TOptimizerInterface):
  @staticmethod
  def DefaultOptions():
    Options= {}
    Options['param_struct']= []  #See above
    Options['maxfevals']= 1000000
    #Options[name]= {Options for each optimizer-Set TDiscOptProb's parameters for composite}
    return Options
  @staticmethod
  def DefaultParams():
    Params= {}
    #Params[name]= {Parameters for each optimizer other than composite}
    return Params

  def __init__(self):
    TOptimizerInterface.__init__(self)

  #Initialize learner.  data: set to continue from previous state, generated by Save.
  def Init(self, data=None):
    TOptimizerInterface.Init(self,data)
    self.Opts= {}  #name to optimizer
    self.CreateOptimizers()

  #Synchronize Params (and maybe Options) with an internal optimizer to be saved.
  def SyncParams(self):
    for name,opt in self.Opts.iteritems():
      opt.SyncParams()
      self.Params[name]= opt.Params
      self.Options[name]= opt.Options

  #Create a optimizers.
  def CreateOptimizers(self):
    self.Opts= {}
    def sub_parse(sub):
      if len(sub)==0:  return
      name= sub[0]
      kind= sub[1]
      if name in self.Opts:
        raise Exception('TCompositeOpt: invalid param_struct where the same name is used twice: '+name)
      if kind=='c':
        self.Opts[name]= TContOptNoGrad()
        options= self.Options[name] if name in self.Options else {}
        params= self.Params[name] if name in self.Params else {}
        self.Opts[name].Init({'options':options,'params':params})
      elif kind=='d':
        self.Opts[name]= TDiscOptProb()
        options= self.Options[name] if name in self.Options else {}
        params= self.Params[name] if name in self.Params else {}
        self.Opts[name].Init({'options':options,'params':params})
      elif kind=='x':
        #This optimizer is a selector:
        self.Opts[name]= TDiscOptProb()
        options= self.Options[name] if name in self.Options else {}
        options['mapped_values']= range(len(sub[2]))
        params= self.Params[name] if name in self.Params else {}
        self.Opts[name].Init({'options':options,'params':params})
        for sub2 in sub[2]:
          sub_parse(sub2)
    sub_parse(self.Options['param_struct'])
    self.curr_param= None
    self.fevals= self.Options['maxfevals']

  #Set an initial parameter. Should be executed after Init.
  def SetParam0(self, param0):
    def sub_parse(sub, sub_param0):
      if len(sub)==0:  return
      name= sub[0]
      kind= sub[1]
      if kind=='c':
        self.Opts[name].SetParam0(sub_param0)
      elif kind=='d':
        self.Opts[name].SetParam0(sub_param0)
      elif kind=='x':
        assert(len(sub_param0)==2)  #sub_param0==[index,sub_parameter]
        self.Opts[name].SetParam0(sub_param0[0])
        sub_parse(sub[2][sub_param0[0]], sub_param0[1])
    sub_parse(self.Options['param_struct'], param0)
    self.SyncParams()  #Because each SetParam0 modifies their Options and Params
    self.CreateOptimizers()

  #Returns an initial parameter.
  def Param0(self):
    def sub_parse(sub):
      if len(sub)==0:  return None
      name= sub[0]
      kind= sub[1]
      if kind=='c':
        return self.Opts[name].Param0()
      elif kind=='d':
        return self.Opts[name].Param0()
      elif kind=='x':
        sub2_idx= self.Opts[name].Param0()
        if sub2_idx==None:  return None
        param2= sub_parse(sub[2][sub2_idx])
        return [sub2_idx,param2]
    return sub_parse(self.Options['param_struct'])

  #Return the parameter, score.
  def Result(self):
    def sub_parse(sub):
      if len(sub)==0:  return None,None
      name= sub[0]
      kind= sub[1]
      if kind=='c':
        return self.Opts[name].Result()
      elif kind=='d':
        return self.Opts[name].Result()
      elif kind=='x':
        sub2_idx,score= self.Opts[name].Result()
        if sub2_idx==None:  return sub2_idx,score
        param2,score2= sub_parse(sub[2][sub2_idx])
        return [sub2_idx,param2], score2
    return sub_parse(self.Options['param_struct'])

  #Check the stop condition.
  def Stopped(self):
    if self.fevals<=0:  return True
    def sub_parse(sub):
      if len(sub)==0:  return True
      name= sub[0]
      kind= sub[1]
      if kind=='c':
        return self.Opts[name].Stopped()
      elif kind=='d':
        return self.Opts[name].Stopped()
      elif kind=='x':
        if not self.Opts[name].Stopped():  return False
        #else, the selector stopped
        sub2_idx,score= self.Opts[name].Result()
        if sub2_idx==None:  return True
        return sub_parse(sub[2][sub2_idx])
    return sub_parse(self.Options['param_struct'])

  #Returns the latest selected parameter.
  def Param(self):
    #def sub_parse(sub):
      #if len(sub)==0:  return None
      #name= sub[0]
      #kind= sub[1]
      #if kind=='c':
        #return self.Opts[name].Param()
      #elif kind=='d':
        #return self.Opts[name].Param()
      #elif kind=='x':
        #sub2_idx= self.Opts[name].Param()
        #if sub2_idx==None:  return None
        #param2= sub_parse(sub[2][sub2_idx])
        #return [sub2_idx,param2]
    #return sub_parse(self.Options['param_struct'])
    return self.curr_param

  #Select a parameter.  Use Param() to refer to the selected parameter.
  def Select(self):
    def sub_parse(sub):
      if len(sub)==0:  return None
      name= sub[0]
      kind= sub[1]
      if kind=='c':
        return self.Opts[name].Select()
      elif kind=='d':
        return self.Opts[name].Select()
      elif kind=='x':
        sub2_idx= self.Opts[name].Select()
        if sub2_idx==None:  return None
        param2= sub_parse(sub[2][sub2_idx])
        return [sub2_idx,param2]
    self.curr_param= sub_parse(self.Options['param_struct'])
    return self.curr_param

  #Update with a selected parameter and a score (positive is better) which can be None.
  def Update(self,score):
    self.fevals-= 1
    def sub_parse(sub):
      if len(sub)==0:  return
      name= sub[0]
      kind= sub[1]
      if kind=='c':
        self.Opts[name].Update(score)
      elif kind=='d':
        self.Opts[name].Update(score)
      elif kind=='x':
        self.Opts[name].Update(score)
        sub2_idx= self.Opts[name].Param()
        if sub2_idx==None:  return
        sub_parse(sub[2][sub2_idx])
    sub_parse(self.Options['param_struct'])

  #Update with a given parameter and a score (positive is better) which can be None.
  def UpdateWith(self,param,score):
    self.fevals-= 1
    def sub_parse(sub, sub_param):
      if len(sub)==0:  return
      name= sub[0]
      kind= sub[1]
      if kind=='c':
        self.Opts[name].UpdateWith(sub_param, score)
      elif kind=='d':
        self.Opts[name].UpdateWith(sub_param, score)
      elif kind=='x':
        assert(len(sub_param)==2)  #sub_param==[index,sub_parameter]
        self.Opts[name].UpdateWith(sub_param[0], score)
        sub_parse(sub[2][sub_param[0]], sub_param[1])
    sub_parse(self.Options['param_struct'], param)

  #Return a list of optimizer names used in param whose type is key.
  #Note: kind=='*' matches any type.
  def NamesInParam(self,param,key='*'):
    def sub_parse(sub, sub_param, key):
      if len(sub)==0:  return []
      name= sub[0]
      kind= sub[1]
      if kind in ('c','d'):
        if kind==key or key=='*':  return [name]
        else:                      return []
      elif kind=='x':
        assert(len(sub_param)==2)  #sub_param==[index,sub_parameter]
        sub_names= sub_parse(sub[2][sub_param[0]], sub_param[1], key)
        if kind==key or key=='*':  return [name]+sub_names
        else:                      return sub_names
    return sub_parse(self.Options['param_struct'], param, key)

  #Generate a random number inside a bound. Should be executed after Init.
  def Rand(self):
    def sub_parse(sub):
      if len(sub)==0:  return None
      name= sub[0]
      kind= sub[1]
      if kind=='c':
        return self.Opts[name].Rand()
      elif kind=='d':
        return self.Opts[name].Rand()
      elif kind=='x':
        sub2_idx= self.Opts[name].Rand()
        if sub2_idx==None:  return None
        param2= sub_parse(sub[2][sub2_idx])
        return [sub2_idx,param2]
    return sub_parse(self.Options['param_struct'])




if __name__=='__main__':

  #Test of continuous parameter (2d) function
  def TestC1():
    def fobj(x,f_none=None):
      assert len(x)==2
      if (x[0]-0.5)**2+(x[1]+0.5)**2<0.2:  return f_none
      return 1.0 - 3.0*(x[0]-1.2)**2 - 2.0*(x[1]+2.0)**2

    opt= TContOptNoGrad()
    options= {}
    options['bounds']= [[-3.0,-3.0],[3.0,3.0]]
    options['tolfun']= 1.0e-4
    options['scale0']= 0.5
    options['parameters0']= [1.6,2.5]
    opt.Init({'options':options})

    fx_data= opt.InitialGuess(fobj, 10)
    fx_data.sort()
    print 'InitialGuess=',fx_data

    count= 0
    while not opt.Stopped():
      x= opt.Select()
      f= fobj(x)
      opt.Update(f)
      count+= 1
    print 'Result=',opt.Result(),'obtained in',count
    #print 'Saved params=\n',opt.Save()

    fp= file('/tmp/outcmaes_obj.dat','w')
    for x1 in FRange1(-4.0,4.0,100):
      for x2 in FRange1(-4.0,4.0,100):
        x= np.array([x1,x2])
        fp.write('%s %f\n' % (' '.join(map(str,x)),fobj(x,-100)))
      fp.write('\n')
    fp.close()

    fp= file('/tmp/outcmaes_res.dat','w')
    x,f= opt.Result()
    fp.write('%s %f\n' % (' '.join(map(str,x)),f))
    fp.close()

    print 'Plot by:'
    print 'qplot -x -3d /tmp/outcmaes_obj.dat w l /tmp/outcmaes_res.dat w p pt 7'

  #Test of discrete parameter function
  def TestD1():
    def fobj(x,f_none=None):
      if 100<x and x<120:
        return 1.0-float((x-110)**2)/10.0
      return f_none

    opt= TDiscOptProb()
    options= {}
    options['mapped_values']= range(200)
    options['init_std_dev']= 0.5
    opt.Init({'options':options})

    fx_data= opt.InitialGuess(fobj, 10)
    fx_data.sort()
    print 'InitialGuess=',fx_data

    count= 0
    while not opt.Stopped():
      x= opt.Select()
      f= fobj(x)
      opt.Update(f)
      count+= 1
    print 'Result=',opt.Result(),'obtained in',count
    #print 'Saved params=\n',opt.Save()

    fp= file('/tmp/outcmaes_obj.dat','w')
    for x in range(200):
      fp.write('%s %f\n' % (str(x),fobj(x,-1)))
    fp.close()

    fp= file('/tmp/outcmaes_res.dat','w')
    x,f= opt.Result()
    fp.write('%s %f\n' % (str(x),f))
    fp.close()

    print 'Plot by:'
    print 'qplot -x /tmp/outcmaes_obj.dat /tmp/outcmaes_res.dat w p pt 7'

  #Test of composite parameter function
  def TestX1():
    #Discrete parameter is a type of function
    #Note that each function assumes different size of x
    def fobj1(x,f_none=None):
      assert len(x)==2
      if not IsIn(x[0], [-3.0,-0.5]):  return f_none
      if (x[0]+2.0)**2+(x[1]+0.5)**2<0.2:  return f_none
      return -0.01*(3.0*(x[0]-1.2)**2 + 2.0*(x[1]+2.0)**2)

    def fobj2(x,f_none=None):
      assert len(x)==2
      if not IsIn(x[0], [0.0,1.0]):  return f_none
      return -0.01*(0.5*(x[0]+2.0)**2 + 0.5*(x[1]+2.0)**2 + 20.0)

    def fobj3(x,f_none=None):
      assert len(x)==1
      if not IsIn(x[0], [1.5,3.0]):  return f_none
      return -0.01*(20.0*(x[0]-2.0)**2 - 15.0)

    #x[0]: discrete parameter to select function
    #x[1]: continuous parameters for each function; size depends on x[0]
    def fobj(x,f_none=None):
      if x[0]==0:    f= fobj1(x[1])
      elif x[0]==1:  f= fobj2(x[1])
      elif x[0]==2:  f= fobj3(x[1])
      if f is None:  return f_none
      return f

    t0=time.time()
    opt= TCompositeOpt()
    options= {}
    options['param_struct']= ['x1','x',[ ['c1','c'], ['c2','c'], ['c3','c'] ]]
    options['x1']= {}
    #options['x1']['ucb_nsd']= 2.0
    options['x1']['alpha']= 0.2
    options['c1']= {}
    options['c1']['bounds']= [[-3.0,-3.0],[3.0,3.0]]
    options['c1']['tolfun']= 1.0e-4
    options['c1']['scale0']= 1.0
    options['c1']['parameters0']= [0.0,0.0]
    #options['c1']['parameters0']= [-2.0,0.0]
    options['c2']= {}
    options['c2']['bounds']= [[-3.0,-3.0],[3.0,3.0]]
    options['c2']['tolfun']= 1.0e-4
    options['c2']['scale0']= 1.0
    options['c2']['parameters0']= [0.0,0.0]
    #options['c2']['parameters0']= [0.5,0.0]
    options['c3']= {}
    options['c3']['bounds']= [[-3.0],[3.0]]
    options['c3']['tolfun']= 1.0e-4
    options['c3']['scale0']= 1.0
    options['c3']['parameters0']= [0.0]
    #options['c3']['parameters0']= [2.0]
    opt.Init({'options':options})

    ##Initial guess:
    #fx_data= opt.InitialGuess(fobj, 10)
    #fx_data.sort()
    #print 'InitialGuess=',fx_data
    #opt.SetParam0(fx_data[-1][1])
    #for name in opt.NamesInParam(fx_data[-1][1],key='c'):
      #opt.Opts[name].Options['scale0']*= 0.01
    #opt.SyncParams()
    #opt.CreateOptimizers()

    #New initial parameters:
    opt.SetParam0([0,[-2.0,0.0]])
    opt.SetParam0([1,[0.5,0.0]])
    opt.SetParam0([2,[2.0]])
    #CPrint(3,'Param0= ',opt.Param0(),'/',opt.Opts['x1'].Params)
    #Modify scale0 (initial scales):
    for x in ([0,[-2.0,0.0]], [1,[0.5,0.0]], [2,[2.0]]):
      for name in opt.NamesInParam(x,key='c'):
        opt.Opts[name].Options['scale0']*= 0.1
    opt.SyncParams()
    opt.CreateOptimizers()

    CPrint(2,'generation time=',time.time()-t0)

    count= 0
    t0=time.time()
    while not opt.Stopped():
      x= opt.Select()
      #print count,'#',x
      #CPrint(3,'DEBUG: ',opt.NamesInParam(x,'*'))
      f= fobj(x)
      opt.Update(f)
      count+= 1
    CPrint(1,'optimization time=',time.time()-t0)
    print 'Result=',ToStdType(opt.Result()),'obtained in',count
    print 'opt.Opts["x1"].Params["means"]=',opt.Opts['x1'].Params['means']
    print 'opt.Opts["x1"].UCB()=',opt.Opts['x1'].UCB()
    #print 'Saved params=\n',opt.Save()

    fp= file('/tmp/outcmaes_obj.dat','w')
    for x1 in FRange1(-4.0,4.0,100):
      for x2 in FRange1(-4.0,4.0,100):
        for i in range(3):
          if i in (0,1):  f= fobj(x=[i,[x1,x2]])
          elif i in (2,):  f= fobj(x=[i,[x1]])
          if f is not None:
            fp.write('%f %f %f\n' % (x1,x2,f))
      fp.write('\n')
    fp.close()

    fp= file('/tmp/outcmaes_res.dat','w')
    x,f= opt.Result()
    if x[0]<>2:  fp.write('%s %f\n' % (' '.join(map(str,x[1])),f))
    else:        fp.write('%s %f\n' % (' '.join(map(str,list(x[1])+[0.0])),f))
    fp.close()

    print 'Plot by:'
    print 'qplot -x -3d /tmp/outcmaes_obj.dat w l /tmp/outcmaes_res.dat w p pt 7'


  #TestC1()
  #TestD1()
  TestX1()


