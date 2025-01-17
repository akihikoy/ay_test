#!/usr/bin/python3
'''
Incremental version of LWR; locally weighted regression.
- Gaussian kernel with max norm is available.
- Kernel width can vary in each dimension.
'''

import copy
import numpy as np
import numpy.linalg as la

#Distance of two vectors: L2 norm of their difference
def Dist(p1,p2):
  return la.norm(np.array(p2)-p1)

#Distance of two vectors: Max norm of their difference
def DistM(p1,p2):
  return np.abs(np.array(p2)-p1).max()

#L2 norm of a vector
def Norm(p1):
  return la.norm(p1)

#Max norm of a vector
def NormM(p1):
  return np.abs(p1).max()

def Mat(x):
  return np.mat(x)

#Gaussian function
def Gaussian(xd, var, a=1.0):
  assert(xd.shape[0]==1)
  return a * np.exp(-0.5*xd / var * xd.T)

#Gaussian function with max norm
def GaussianM(xd, var, a=1.0):
  assert(xd.shape[0]==1)
  return a * np.exp(-0.5*(np.multiply(xd,xd)/var).max())  #Max norm

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


class TLWR:
  def __init__(self,kernel='l2g'):
    self.AddOne= True  #Automatically add one to the end of x
    self.LazyCopy= False
    self.Importance= None
    if   kernel=='l2g':  #L2 norm Gaussian
      self.kernel= Gaussian
      self.norm= Norm
    elif kernel=='maxg':  #Max norm Gaussian
      self.kernel= GaussianM
      self.norm= NormM

  #Get weights W(x)=diag([k(x,x1),k(x,x2),...])
  def Weights(self,x,x_var):
    N= self.X.shape[0]
    D= self.X.shape[1] - (1 if self.AddOne else 0)
    W= np.diag([1.0]*N)
    for n in range(N):
      W[n,n]= self.kernel((x-self.X[n])[:,:D], self.C[n]*self.C[n]+x_var)
    if self.Importance!=None:
      for k,v in self.Importance.items():
        W[k,k]*= v
    return W

  #Whether prediction is available (False if the model is not learned at all).
  def Available(self):
    return len(self.DataX)>=2

  #Train with data X,Y.
  #   [x1^T]    [y1^T]
  # X=[x2^T]  Y=[y2^T]
  #   [... ]    [... ]
  # c: Gaussian kernel width.
  # f_reg: Reguralization factor.
  def Train(self,X,Y,c_min=0.01,f_reg=0.01):
    self.DataX= X
    self.DataY= Y
    self.C= self.AutoWidth(c_min)
    self.FReg= f_reg
    self.LazyCopy= True

  #Initialization for incremental learning
  def Init(self,c_min=0.01, c_max=1.0e6, c_gain=0.7, f_reg=0.01):
    self.DataX= []
    self.DataY= []
    self.C= []
    self.Closests= []
    self.CDists= []  #Distance to the closest point
    self.CMin= c_min
    self.CMax= c_max
    self.CGain= c_gain
    self.FReg= f_reg
    self.LazyCopy= False

  #Incrementally update the internal parameters
  def Update(self,x,y):
    #FIXME: it's better to use an efficient nearest neighbor like KD tree.
    self.DataX.append(list(x))
    self.DataY.append(list(y))
    if len(self.DataX)==1:
      self.Closests.append(-1)
      self.C.append(None)
      self.CDists.append(None)
      return
    diff_to_c= lambda diff: np.array([ min(max(self.CMin,self.CGain*abs(d)),self.CMax) for d in diff])
    #diff_to_c= lambda diff: np.array([max([ min(max(self.CMin,self.CGain*d),self.CMax) for d in diff])]*len(diff))
    if len(self.Closests)==1:
      diff= np.array(self.DataX[0])-self.DataX[1]
      self.Closests.append(0)
      self.Closests[0]= 1
      self.CDists= [self.norm(diff)]*2
      self.C= [diff_to_c(diff)]*2
    else:
      n= len(self.Closests)
      dc= 1.0e100
      diffc= None
      nc= None #Closest point
      for k in range(n):
        diff= np.array(x)-self.DataX[k]
        d= self.norm(diff)
        if d<dc:
          dc=d
          diffc= diff
          nc=k
        if d<self.CDists[k]:
          self.Closests[k]= n
          self.CDists[k]= d
          self.C[k]= diff_to_c(diff)
      self.Closests.append(nc)
      self.CDists.append(dc)
      self.C.append(diff_to_c(diffc))
      #for k in range(n+1):
        #self.C[k]= min(self.C[k], self.C[self.Closests[k]])
    self.LazyCopy= True

  #Prediction result class.
  class TPredRes:
    def __init__(self):
      self.Y= None  #Prediction result.
      self.Var= None  #Variance matrix.
      self.Grad= None  #Gradient.
  #Do prediction.
  # Return a TPredRes instance.
  # x_var: Variance of x (TEST).
  # with_var: Whether compute a variance matrix of error at the query point as well.
  # with_grad: Whether compute a gradient at the query point as well.
  def Predict(self,x,x_var=0.0,with_var=False,with_grad=False):
    if self.LazyCopy:
      self.X= Mat(AddOnes(copy.deepcopy(self.DataX)) if self.AddOne else self.DataX)
      self.Y= Mat(self.DataY)
      self.LazyCopy= False
    self.DiffQueryX= False
    if self.DiffQueryX:
      subx= [x[d] for d in range(len(x))] + ([0.0] if self.AddOne else [])
      X= self.X - subx
    else:
      X= self.X
    xx= AddOne(x) if self.AddOne else x
    res= self.TPredRes()
    D= X.shape[1]  #Num of dimensions of x
    W= self.Weights(xx,x_var)
    beta= (X.T*(W*X) + self.FReg*np.eye(D,D)).I * (X.T*(W*self.Y))
    if self.DiffQueryX:
      res.Y= beta[-1].T
    else:
      res.Y= (xx * beta).T
    if with_var:
      N= X.shape[0]  #Num of samples
      div= W.trace()
      div*= (1.0 - float(D)/float(N)) if N>D else 1.0e-4
      Err= X * beta - self.Y
      div= max(div,1.0e-4)
      res.Var= (Err.T * W * Err) / div
    if with_grad:
      res.Grad= beta[:-1]
      #res.Grad= self.NumDeriv(x,x_var)
    return res

  #Compute derivative at x numerically.
  def NumDeriv(self,x,x_var=0.0,h=0.01):
    Dx= Len(x)
    Dy= Len(self.DataY[0])
    delta= lambda dd: np.array([0.0 if d!=dd else h for d in range(Dx)])
    dy= Mat([[0.0]*Dy]*Dx)
    for d in range(Dx):
      dy[d,:]= (self.Predict(x+delta(d),x_var).Y - self.Predict(x-delta(d),x_var).Y).T/(2.0*h)
      maxd=abs(dy[d,:]).max()
      if maxd>1.0:  dy[d,:]*= 1.0/maxd
    return dy

  #Compute Gaussian kernel width for each data point automatically.
  def AutoWidth(self, c_min=0.01, c_max=1.0e6, c_gain=0.7):
    N= len(self.DataX)
    diff_to_c= lambda diff: np.array([ min(max(c_min,c_gain*abs(d)),c_max) for d in diff])
    C= [0.0]*N
    for n in range(N):
      closest,cdist= min([(d,self.norm(np.array(self.DataX[n])-self.DataX[d])) for d in range(N) if d!=n], key=lambda x:x[1])
      C[n]= diff_to_c(np.array(self.DataX[n])-self.DataX[closest])
    #print C
    return C


  #Dump data to file for plot.
  def DumpPlot(self,bounds,f_reduce,f_repair,file_prefix='/tmp/f',x_var=0.0):
    if len(self.DataX)==0:
      print('No data')
      return
    xamin0,xamax0= bounds
    xamin= f_reduce(xamin0)
    xamax= f_reduce(xamax0)
    xmed= [Median([x[d] for x in self.DataX]) for d in range(len(self.DataX[0]))]
    if len(xamin)>=3 or len(xamin)!=len(xamax) or len(xamin)<=0:
      print('DumpPlot: Invalid f_reduce function')
      return

    fp= open('%s_est.dat'%(file_prefix),'w')
    if len(xamin)==2:
      for xa1_1 in FRange1(xamin[0],xamax[0],50):
        for xa1_2 in FRange1(xamin[1],xamax[1],50):
          xa1r= [xa1_1,xa1_2]
          xa1= f_repair(xa1r, xamin0, xamax0, xmed)
          fp.write('%s\n' % ToStr(xa1r,xa1,ToList(self.Predict(xa1,x_var).Y)))
        fp.write('\n')
    else:  #len(xamin)==1:
      for xa1_1 in FRange1(xamin[0],xamax[0],50):
        xa1r= [xa1_1]
        xa1= f_repair(xa1r, xamin0, xamax0, xmed)
        fp.write('%s\n' % ToStr(xa1r,xa1,ToList(self.Predict(xa1,x_var).Y)))
    fp.close()
    fp= open('%s_smp.dat'%(file_prefix),'w')
    for xa1,x2 in zip(self.DataX, self.DataY):
      fp.write('%s\n' % ToStr(f_reduce(xa1),xa1,x2))
    fp.close()


import random

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

def AddOne(x):
  if isinstance(x,list):
    if len(x)==0:  return [1.0]
    if isinstance(x[0],list): return [x[0]+[1.0]]
    return x+[1.0]
  if isinstance(x,np.ndarray):
    return x.tolist()+[1.0]
  if x.shape[0]==1:  return Mat(x.tolist()[0]+[1.0])
  if x.shape[1]==1:  return Mat(x.T.tolist()[0]+[1.0]).T
  return None

def AddOnes(X):
  if isinstance(X,list):  Y= X
  else:  Y= X.tolist()
  for x in Y:
    x.append(1.0)
  return Y


if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))
  import math

  #example= 1
  example= 2

  if example==1:
    true_func= lambda x: 1.2+math.sin(x)
    data_x= [[x+1.0*Rand()] for x in FRange1(-3.0,5.0,10)]
    data_y= [[true_func(x[0])+0.3*Rand()] for x in data_x]

    fp1= open('/tmp/smpl.dat','w')
    for x,y in zip(data_x,data_y):
      fp1.write('%f %f\n' % (x[0],y[0]))
    fp1.close()

    lwr= TLWR()
    #lwr.Train(data_x, data_y, c_min=0.3)
    lwr.Init(c_min=0.3)
    for x,y in zip(data_x,data_y):
      lwr.Update(x,y)

    fp1= open('/tmp/true.dat','w')
    fp2= open('/tmp/est.dat','w')
    for x in FRange1(-7.0,10.0,200):
      pred= lwr.Predict([x],with_var=True,with_grad=True)
      #print 'pred.Grad=',pred.Grad
      fp1.write('%f %f\n' % (x,true_func(x)))
      fp2.write('%f %f %f %f %f\n' % (x,pred.Y,np.sqrt(pred.Var),0.2,0.2*pred.Grad[0]))
    fp1.close()
    fp2.close()

    print('Plot by:')
    print('''qplot -x /tmp/est.dat u 1:2:3 w errorbars /tmp/true.dat w l /tmp/smpl.dat w p''')
    print('''qplot -x /tmp/est.dat u 1:2:3 w errorbars /tmp/true.dat w l /tmp/smpl.dat w p /tmp/est.dat u 1:2:4:5 ev 2 w vector lt 3''')

  elif example==2:
    true_func= lambda x: 1.2+math.sin(2.0*(x[0]+x[1]))
    data_x= [[2.0*Rand(),10.0*Rand()] for i in range(10)]
    data_y= [[true_func(x)+0.3*Rand()] for x in data_x]

    fp1= open('/tmp/smpl.dat','w')
    for x,y in zip(data_x,data_y):
      fp1.write('%f %f %f\n' % (x[0],x[1],y[0]))
    fp1.close()

    lwr= TLWR('maxg')
    #lwr.Train(data_x, data_y, c_min=0.3)
    lwr.Init(c_min=0.3)
    for x,y in zip(data_x,data_y):
      lwr.Update(x,y)

    fp1= open('/tmp/true.dat','w')
    fp2= open('/tmp/est.dat','w')
    for x1 in FRange1(-4.0,4.0,50):
      for x2 in FRange1(-4.0,4.0,50):
        pred= lwr.Predict([x1,x2],with_grad=True)
        fp1.write('%f %f %f\n' % (x1,x2,true_func([x1,x2])))
        fp2.write('%f %f %f %f %f\n' % (x1,x2,pred.Y,0.2*pred.Grad[0],0.2*pred.Grad[1]))
      fp1.write('\n')
      fp2.write('\n')
    fp1.close()
    fp2.close()

    print('Plot by:')
    print('''qplot -x -3d /tmp/est.dat w l /tmp/true.dat w l /tmp/smpl.dat w p''')
    print('''qplot -x -3d /tmp/est.dat w l /tmp/true.dat w l /tmp/smpl.dat w p /tmp/est.dat u 1:2:3:4:5:'(0.0)' w vector''')
