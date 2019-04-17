#! /usr/bin/env python
#Basic tools (utility).
import numpy as np
import numpy.linalg as la
import math
import os
import sys
import copy
import threading
import Queue  #For thread communication
import time
import datetime
import random
import traceback
import importlib

#Speedup YAML using CLoader/CDumper
from yaml import load as yamlload
from yaml import dump as yamldump
try:
  from yaml import CLoader as YLoader, CDumper as YDumper
except ImportError:
  from yaml import YLoader, YDumper

def AskYesNo():
  while 1:
    sys.stdout.write('  (y|n) > ')
    ans= sys.stdin.readline().strip()
    if ans=='y' or ans=='Y':  return True
    elif ans=='n' or ans=='N':  return False

#Usage: AskGen('y','n','c')
def AskGen(*argv):
  assert(len(argv)>0)
  while 1:
    sys.stdout.write('  (%s) > ' % '|'.join(argv))
    ans= sys.stdin.readline().strip()
    for a in argv:
      if ans==a:  return a

def TimeStr(fmt='short',now=None):
  if now is None: now= time.localtime()
  if fmt=='short':
    return '%04i%02i%02i%02i%02i%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  elif fmt=='short2':
    return '%04i%02i%02i-%02i%02i%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  elif fmt=='normal':
    return '%04i.%02i.%02i-%02i.%02i.%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  return '%r'%now

def IfNone(x,y):
  return x if x!=None else y

#Unique list
def LUnique(a):
  return list(set(a))
#Intersection of two lists
def LIntersection(a, b):
  return list(set(a) & set(b))
#Union of two lists
def LUnion(a, b):
  return list(set(a) | set(b))
#Difference of two lists (a-b)
def LDifference(a, b):
  return list(set(a) - set(b))

def Vec(x):
  return np.array(x)

def Mat(x):
  return np.mat(x)

#Return row vector of np.matrix.
def MRVec(x):
  if x is None:  return np.mat([])
  elif isinstance(x,(list,tuple)):  return np.mat(x).ravel()
  elif isinstance(x,(np.ndarray,np.matrix)):  return np.mat(x.ravel())
  raise Exception('Len: Impossible to serialize:',x)

#Return column vector of np.matrix.
def MCVec(x):
  return MRVec(x).T

def Eye(N=3):
  return np.eye(N)

def NormSq(x):
  #s= 0.0
  #for xd in x:  s+= xd**2
  #return s
  return la.norm(x)**2

#L2 norm of a vector x
def Norm(x):
  return la.norm(x)

#Max norm of a vector x
def MaxNorm(x):
  return max(map(abs,x))


#Return a normalized vector with L2 norm
def Normalize(x):
  return np.array(x)/la.norm(x)

#Distance of two vectors: L2 norm of their difference
def Dist(p1,p2):
  return la.norm(np.array(p2)-p1)

#Distance of two vectors: Max norm of their difference
def DistM(p1,p2):
  return np.abs(np.array(p2)-p1).max()

#Distance of two composite vectors, e.g. [0.1,2,[0.1,0.2]]
#Returns d_struct_diff is their structures are different.
def CompDist(p1,p2, d_struct_diff=-1.0):
  def is_same_type_list(v1,v2):
    if isinstance(v1,(int,float)):  return isinstance(v2,(int,float))
    if isinstance(v2,(int,float)):  return isinstance(v1,(int,float))
    if len(v1)!=len(v2):  return False
    if False in [isinstance(v1[i],(int,float))==isinstance(v2[i],(int,float)) for i in range(len(v1))]:
      return False
    return True
  if not is_same_type_list(p1,p2):
    return d_struct_diff
  if isinstance(p1,(int,float)):  return abs(p2-p1)
  def subserialize(tree, s):
    for e in tree:
      if isinstance(e,(int,float)):  s.append(e)
      else:  subserialize(e, s)
  def serialize(tree):
    s= []
    subserialize(tree,s)
    return s
  return Dist(serialize(p1),serialize(p2))

#Check if a composite vector has a specified structure.
#cstruct is a compared structure that should consist of standard types (int, float)
def CompHasStruct(cvec, cstruct):
  comp_types= (list,tuple,np.ndarray)
  derivables= {}
  derivables[int]= (int,)
  derivables[float]= derivables[int]+(float,np.float_, np.float16, np.float32, np.float64)
  def subcheck(cv,cs):
    if not isinstance(cv,comp_types):
      if cs not in (float,int):  return False
      else:  return type(cv) in derivables[cs]
    else:
      if len(cv)!=len(cs):  return False
      for i in range(len(cs)):
        if not subcheck(cv[i],cs[i]):  return False
      return True
  return subcheck(cvec, cstruct)

#Convert a vector to string
def VecToStr(vec,delim=', '):
  return delim.join(map(str,vec))

#Join lists where each list is converted to a string
def ToStr(*lists):
  delim=' '
  s= ''
  delim2= ''
  for v in lists:
    s+= delim2 + (delim.join(map(str,list(v))) if v is not None else '')
    delim2= delim
  return s

def Sign(x):
  if x==0.0:  return 0
  if x>0.0:   return 1
  if x<0.0:   return -1

#Return a median of an array
def Median(array,pos=0.5):
  if len(array)==0:  return None
  a_sorted= copy.deepcopy(array)
  a_sorted.sort()
  return a_sorted[int(len(a_sorted)*pos)]

# Matlab-like mod function that returns always positive
def Mod(x, y):
  if y==0:  return x
  return x-y*math.floor(x/y)

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

def ConstrainN(bound,x):
  if isinstance(x,(np.ndarray,np.matrix)):
    if len(x.shape)==1:
      return np.array([min(max(x[d],bound[0][d]),bound[1][d]) for d in range(len(x))])
    if len(x.shape)==2:
      if x.shape[0]==1:
        return np.mat([min(max(x[0,d],bound[0][d]),bound[1][d]) for d in range(x.shape[1])])
      if x.shape[1]==1:
        return np.mat([min(max(x[d,0],bound[0][d]),bound[1][d]) for d in range(x.shape[0])]).T

#Generalized version of len() that can take a list, np.ndarray, or single row/column np.matrix.
def Len(x):
  if x is None:  return 0
  elif isinstance(x,list):  return len(x)
  elif isinstance(x,(np.ndarray,np.matrix)):
    if len(x.shape)==1:  return x.shape[0]
    if len(x.shape)==2:
      if x.shape[0]==1:  return x.shape[1]
      if x.shape[1]==1:  return x.shape[0]
      if x.shape[0]==0 or x.shape[1]==0:  return 0
  raise Exception('Len: Impossible to serialize:',x)

#Convert a np.ndarray or single row/column np.matrix to a list.
def ToList(x):
  if x is None:  return []
  elif isinstance(x,list):  return x
  elif isinstance(x,(np.ndarray,np.matrix)):
    if len(x.shape)==1:  return x.tolist()
    if len(x.shape)==2:
      if x.shape[0]==1:  return x.tolist()[0]
      if x.shape[1]==1:  return x.T.tolist()[0]
      if x.shape[0]==0 and x.shape[1]==0:  return []
  raise Exception('ToList: Impossible to serialize:',x)


#Generate a random number of uniform distribution of specified bound.
def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

#Generate a random integer.
def RandI(*args):
  if len(args)==1:
    start= 0
    end= args[0]
  elif len(args)==2:
    start= args[0]
    end= args[1]
  return random.randint(start,end-1)

#Generate a random vector of uniform distribution; each dim has the same bound.
def RandVec(nd,xmin=-0.5,xmax=0.5):
  return Vec([Rand(xmin,xmax) for d in range(nd)])

#Generate a random vector of uniform distribution; each dim has different bound.
def RandN(xmins,xmaxs):
  assert(len(xmins)==len(xmaxs))
  return [Rand(xmins[d],xmaxs[d]) for d in range(len(xmins))]

#Generate a random vector of uniform distribution; each dim has different bound.
def RandB(bounds):
  return RandN(bounds[0],bounds[1])


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


#Automatically estimate the type of input and convert to it
def EstStrConvert(v_str):
  try:
    return int(v_str)
  except ValueError:
    pass
  try:
    return float(v_str)
  except ValueError:
    pass
  if v_str=='True' or v_str=='true' :  return True
  if v_str=='False' or v_str=='false':  return False
  try:
    x=[]
    for v in v_str.split(' '):
      x.append(float(v))
    return x
  except ValueError:
    pass
  try:
    x=[]
    for v in v_str.split(','):
      x.append(float(v))
    return x
  except ValueError:
    pass
  try:
    x=[]
    for v in v_str.split('\t'):
      x.append(float(v))
    return x
  except ValueError:
    pass
  return v_str

class Types:
  stdprim= (int,long,float,bool,str)
  npbool= (np.bool_)
  npint= (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)
  npuint= (np.uint8, np.uint16, np.uint32, np.uint64)
  npfloat= (np.float_, np.float16, np.float32, np.float64)

#Convert a data into a standard python object
def ToStdType(x, except_cnv=lambda y:y):
  if isinstance(x, Types.npbool):   return bool(x)
  if isinstance(x, Types.npint):    return int(x)
  if isinstance(x, Types.npuint):   return int(x)
  if isinstance(x, Types.npfloat):  return float(x)
  if isinstance(x, Types.stdprim):  return x
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

#Add a sub-dictionary with the key into a dictionary d
def AddSubDict(d,key):
  if not key in d:  d[key]= {}

#Print a dictionary with a nice format
#max_level: maximum depth level to be displayed
#level: current level
#keyonly: display key only, value is shown as type
#col: color code
#c1,c2: internal variable (DO NOT USE)
def PrintDict(d,max_level=-1,level=0,keyonly=False,col=None,c1='',c2=''):
  if col is not None:  c1,c2= ACol.I(col,None)
  for k,v in d.iteritems():
    if type(v)==dict:
      print '%s%s[%s]%s= ...' % ('  '*level, c1, str(k), c2)
      if max_level<0 or level<max_level:
        PrintDict(v,max_level=max_level,level=level+1,keyonly=keyonly,col=None,c1=c1,c2=c2)
    elif keyonly:
      print '%s%s[%s]%s= %s' % ('  '*level, c1, str(k), c2, type(v))
    else:
      print '%s%s[%s]%s= %r' % ('  '*level, c1, str(k), c2, v)

#Insert a new dictionary to the base dictionary
def InsertDict(d_base, d_new):
  for k_new,v_new in d_new.iteritems():
    if k_new in d_base and (type(v_new)==dict and type(d_base[k_new])==dict):
      InsertDict(d_base[k_new], v_new)
    else:
      d_base[k_new]= v_new

#If default is None, this behaves like d[key],
#else it returns d[key] if key in d (default is not evaluated)
#  otherwise set d[key]=default() then return d[key]
#Note: default should be None or a function
def GetOrSet(d, key, default=None):
  if default is None:
    return d[key]
  else:
    if key in d:
      return d[key]
    else:
      d[key]= default()
      return d[key]

#Open a file in 'w' mode.
#If the parent directory does not exist, we create it.
#  mode: 'w' in default.
#  interactive: Interactive mode.
#    If True, this asks user:
#      If creating the parent dir.
#      If overwriting the file when it exists.
#    If False, this does not ask user to:
#      Create the parent dir.
#      Overwrite the file when it exists.
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

#Modify file name from xxx.pyc to xxx.py.
#This does nothing to xxx.py or other extensions.
def PycToPy(file_name):
  path,ext= os.path.splitext(file_name)
  if ext=='.pyc':  return path+'.py'
  return file_name

'''Import/reload a module named mod_id (string).
If mod_id was already loaded, the time stamp of the module (note: xxx.pyc is considered as xxx.py)
is compared with the time when the mod_id was loaded previously.
Only when the time stamp is newer than the loaded time, this function reloads the module.
return: module '''
def SmartImportReload(mod_id, __loaded={}):
  if mod_id in __loaded:
    loaded_time,mod= __loaded[mod_id]
    file_time= datetime.datetime.fromtimestamp(os.path.getmtime(PycToPy(mod.__file__)))
    #Reload if the file is modified:
    if file_time>loaded_time:
      reload(mod)
      __loaded[mod_id]= (datetime.datetime.now(), mod)  #Loaded time, module
    return mod
  else:
    mod= importlib.import_module(mod_id)
    __loaded[mod_id]= (datetime.datetime.now(), mod)  #Loaded time, module
    return mod

#Write down a variable to file in human-understandable format
def WriteVarToFile(x, file_name, except_cnv=lambda y:y, interactive=True):
  x2= ToStdType(x, except_cnv)
  so= sys.stdout
  sys.stdout= OpenW(file_name,interactive=interactive)
  PrintDict(x2)
  sys.stdout= so

#Load a YAML and return a dictionary
def LoadYAML(file_name):
  return yamlload(open(file_name).read(), Loader=YLoader)

#Save a dictionary as a YAML
def SaveYAML(d, file_name, except_cnv=lambda y:y, interactive=True):
  OpenW(file_name,interactive=interactive).write(yamldump(ToStdType(d,except_cnv), Dumper=YDumper))

#Load a YAML and insert the data into a dictionary
def InsertYAML(d_base, file_name):
  LoadYAML(file_name)
  InsertDict(d_base, d_new)

#Another simple print function to be used as an action
def Print(*s):
  for ss in s:
    print ss,
  print ''

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
  def I(*index):
    if len(index)==0:  return ACol.EndC
    elif len(index)==1:
      index= index[0]
      if   index==0:   return ACol.Warning
      elif index==1:   return ACol.OKBlue
      elif index==2:   return ACol.OKGreen
      elif index==3:   return ACol.Header
      elif index==4:   return ACol.Fail
      else:   return ACol.EndC
    else:
      return [ACol.I(i) for i in index]
  #def X2: #DEPRECATED

#Get a colored string
def CStr(col,*s):
  c1,c2= ACol.I(col,None)
  return c1 + ' '.join(map(str,s)) + c2

#Print with a color (col can be a code or an int)
def CPrint(col,*s):
  if len(s)==0:
    print ''
  else:
    c1,c2= ACol.I(col,None)
    print c1+str(s[0]),
    for ss in s[1:]:
      print ss,
    print c2

#Print an exception with a good format
def PrintException(e, msg=''):
  c1,c2,c3,ce= ACol.I(4,1,0,None)
  print '%sException( %s%r %s)%s:' % (c1, c2,type(e), c1, msg)
  print '%r' % (e)
  #print '  type: ',type(e)
  #print '  args: ',e.args
  #print '  message: ',e.message
  #print '  sys.exc_info(): ',sys.exc_info()
  print '  %sTraceback: ' % (c3)
  print '{'
  traceback.print_tb(sys.exc_info()[2])
  print '}'
  print '%s# Exception( %s%r %s)%s:' % (c1, c2,type(e), c1, msg)
  print '# %r%s' % (e, ce)

#Container class that can hold any variables
#ref. http://blog.beanz-net.jp/happy_programming/2008/11/python-5.html
class TContainerCore(object):
  def __init__(self):
    pass
  def __del__(self):
    pass
  def __str__(self):
    return str(self.__dict__)
  def __repr__(self):
    return str(self.__dict__)
  def __iter__(self):
    return self.__dict__.itervalues()
  def items(self):
    return self.__dict__.items()
  def iteritems(self):
    return self.__dict__.iteritems()
  def keys(self):
    return self.__dict__.keys()
  def values(self):
    return self.__dict__.values()
  def __getitem__(self,key):
    return self.__dict__[key]
  def __setitem__(self,key,value):
    self.__dict__[key]= value
  def __delitem__(self,key):
    del self.__dict__[key]
  def __contains__(self,key):
    return key in self.__dict__
  def Cleanup(self):
    keys= self.__dict__.keys()
    for k in keys:
      self.__dict__[k]= None
      del self.__dict__[k]
class TContainerDebug(TContainerCore):
  def __init__(self):
    super(TContainerDebug,self).__init__()
    print 'Created TContainer object',hex(id(self))
  def __del__(self):
    super(TContainerDebug,self).__del__()
    print 'Deleting TContainer object',hex(id(self))
'''Helper function to generate a container object.
  Note: the function name is like a class name;
    this is because originally TContainer was a class
    where TContainerCore and TContainerDebug are unified.'''
def TContainer(debug=False):
  return TContainerCore() if not debug else TContainerDebug()

#Managing thread objects
class TThreadManager:
  class TItem:
    def __init__(self):
      self.running= False
      self.thread= None
    def __del__(self):
      self.Stop()
    def Start(self):
      self.running= True
      self.thread.start()
    def Stop(self):
      self.running= False
      if self.thread and self.thread.is_alive():
        self.thread.join()
      self.thread= None
    def StopRequest(self):
      self.running= False
    def Join(self):
      if self.thread and self.thread.is_alive():
        self.thread.join()
      self.thread= None

  class TThreadInfo:
    def __init__(self):
      self.Manager= None  #Reference to the thread manager
      self.Name= None  #Thread name
      self.IsRunning= None  #Function to check if the thread is running (you may use in the target function)
      self.Stop= None  #Function to stop the thread (do not use in the target function)

  @staticmethod
  def ThreadHelper(th_mngr, th_name, target):
    #target(th_mngr, th_name)
    #th_mngr.Stop(th_name)
    try:
      th_info= th_mngr.TThreadInfo()
      th_info.Manager= th_mngr
      th_info.Name= th_name
      th_info.IsRunning= lambda:th_mngr.IsRunning(th_name)
      th_info.Stop= lambda:th_mngr.Stop(th_name)
      target(th_info)
    except Exception as e:
      PrintException(e,' caught in ThreadHelper')
      raise e
    finally:  #Executed after try section or before "raise e"
      if th_name in th_mngr.thread_list:
        th_mngr.thread_list[th_name].thread= None  #To avoid thread.join() is executed in the next step
      th_mngr.Stop(th_name)
      #th_mngr.stop_messenger.put(th_name)

  def __init__(self):
    self.thread_list= {}
    #self.main_running= True
    #self.stop_messenger= Queue.Queue()
    #self.thread_stop_msg= threading.Thread(name='StopMessageHandler', target=lambda:self.StopMessageHandler())
    #self.thread_stop_msg.start()
  def __del__(self):
    self.StopAll()
    #self.main_running= False
    #self.thread_stop_msg.join()
  #def __getitem__(self,key):
    #return self.thread_list[key]
  def __contains__(self,key):
    return key in self.thread_list

  #def StopMessageHandler(self):
    #while self.main_running:
      #try:
        #req= self.stop_messenger.get(timeout=5.0)
        #self.Stop(req)
      #except Queue.Empty:
        #pass

  def Add(self,name,target,start=True):
    self.Stop(name)
    self.thread_list[name]= self.TItem()
    th= self.thread_list[name]
    th.thread= threading.Thread(name=name, target=lambda:self.ThreadHelper(self,name,target))
    if start:  th.Start()
    return th

  def IsRunning(self,name):
    if name in self.thread_list:
      return self.thread_list[name].running
    return False

  def Stop(self,name):
    if name in self.thread_list:
      del self.thread_list[name]

  def StopRequest(self,name):
    if name in self.thread_list:
      self.thread_list[name].StopRequest()

  def StopAll(self):
    for k in self.thread_list.keys():
      print 'Stop thread %r...' % k,
      del self.thread_list[k]
      print 'ok'

  def Join(self,name):
    if name in self.thread_list:
      self.thread_list[name].Join()
      #del self.thread_list[name]  #After joining the thread, self.thread_list[name] is automatically deleted.


'''TSignal class for a thread to send a message to several threads.
This is an extension of Queue.Queue.  The idea is queue-per-thread.
Usage:
  In a wider scope, define this object, like a queue.
    signal= TSignal()
  In receivers, you can write either a with-statement form or a normal form.
    with signal.NewQueue() as queue:
      #use queue normally; e.g. data= queue.get()
  Or:
    queue= signal.NewQueue()
    #use queue normally; e.g. data= queue.get()
    #at the end of this scope, queue is automatically released,
    #but if you want to do it explicitly, you can use either:
    del queue
    #or
    queue= None
  In sender(s), you can write normally:
    signal.put(data)
'''
class TSignal:
  def __init__(self):
    self.queues= {}  #Map from index to queue
    self.counter= 0
    self.locker= threading.Lock()
  def NewQueue(self):
    idx= self.counter
    self.counter+= 1
    with self.locker:
      self.queues[idx]= Queue.Queue()
    queue= self.TQueue(self,idx,self.queues[idx])
    return queue
  def DeleteQueue(self,idx):
    with self.locker:
      if idx in self.queues:
        del self.queues[idx]
  def put(self,item,block=True,timeout=None):
    with self.locker:
      items= self.queues.items()
    for idx,queue in items:
      queue.put(item,block,timeout)

  class TQueue:
    def __init__(self,parent,idx,queue):
      self.parent= parent
      self.idx= idx
      self.queue= queue
    def __del__(self):
      self.parent.DeleteQueue(self.idx)
    def __enter__(self):
      return self
    def __exit__(self,e_type,e_value,e_traceback):
      self.parent.DeleteQueue(self.idx)
    def get(self,block=True,timeout=None):
      return self.queue.get(block,timeout)


'''Modified rospy.Rate with standard time.
https://docs.ros.org/api/rospy/html/rospy.timer-pysrc.html#Rate '''
class TRate(object):
  """
  Convenience class for sleeping in a loop at a specified rate
  """

  def __init__(self, hz, reset=False):
    """
    Constructor.
    @param hz: hz rate to determine sleeping
    @type  hz: float
    @param reset: if True, timer is reset when time moved backward. [default: False]
    @type  reset: bool
    """
    self.last_time = time.time()
    self.sleep_dur = 1.0/hz
    self._reset = reset

  def _remaining(self, curr_time):
    """
    Calculate the time remaining for rate to sleep.
    @param curr_time: current time
    @type  curr_time: L{Time}
    @return: time remaining
    @rtype: L{Time}
    """
    # detect time jumping backwards
    if self.last_time > curr_time:
      self.last_time = curr_time

    # calculate remaining time
    elapsed = curr_time - self.last_time
    return self.sleep_dur - elapsed

  def remaining(self):
    """
    Return the time remaining for rate to sleep.
    @return: time remaining
    @rtype: L{Time}
    """
    curr_time = time.time()
    return self._remaining(curr_time)

  def sleep(self):
    """
    Attempt sleep at the specified rate. sleep() takes into
    account the time elapsed since the last successful
    sleep().
    """
    curr_time = time.time()
    time.sleep(max(0.0, self._remaining(curr_time)))
    self.last_time = self.last_time + self.sleep_dur

    # detect time jumping forwards, as well as loops that are
    # inherently too slow
    if curr_time - self.last_time > self.sleep_dur * 2:
      self.last_time = curr_time
