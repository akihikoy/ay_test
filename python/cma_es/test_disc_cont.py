#!/usr/bin/python
#qplot -x -3d outcmaes_obj.dat w l outcmaes_res.dat w p
#TEST to optimize discrete+continuous parameters

import cma

#Optimizer for a continuous vector.
class TContOpt:
  def __init__(self, options):
    self.maxfevals= options['maxfevals'] if 'maxfevals' in options else 1000000
    self.es= cma.CMAEvolutionStrategy(options['parameters0'], options['scale0'], options)
    self.has_solution= False
    #self.curr_param= options['parameters0']

    self.solutions= []
    self.scores= []
    #self.count= 0

  def Param(self):
    return self.curr_param

  def Result(self):
    res= self.es.result()
    return res[0], -res[1]

  def Stopped(self):
    #TEST
    #if es.result()[1]<>np.inf and -es.result()[1]>0.8:  break True
    return self.es.stop() or self.maxfevals<=0

  def Select(self):
    self.curr_param= self.es.ask(1)[0]

  def Update(self, score):
    #if score0 is not None:
      #self.has_solution= True
      #self.solutions= [parameters0]
      #self.scores= [score0]
    #else:
      #self.solutions= []
      #self.scores= []

    self.maxfevals-= 1
    if score is not None:
      self.has_solution= True
      self.solutions.append(self.curr_param)
      self.scores.append(-score)
      if len(self.scores)>=self.es.popsize:
        self.es.tell(self.solutions, self.scores)
        self.es.disp()
        self.solutions= []
        self.scores= []

        #fp= file('data/res%04i.dat'%(count),'w')
        #count+=1
        #for x in solutions:
          #fp.write('%s %f\n' % (' '.join(map(str,x)),fobj(x,-10)))
        #fp.close()



import disc_opt

#Optimizer for a (discrete variable + continuous vector).
class TDiscContOpt:
  def __init__(self, cont_opt_set):
    #Set of continuous optimizers, should be defined by user
    self.ContOptSet= cont_opt_set
    self.DiscOpt= disc_opt.TProbDiscOpt(N=len(self.ContOptSet), using_none_set=False)

  #Return the selected parameter (discrete, continuous)
  def Param(self):
    x_d= self.DiscOpt.Param()
    x_c= self.ContOptSet[x_d].Param()
    return x_d, x_c

  #Return the best parameter (discrete, continuous), the score
  def Result(self):
    best_x_d, best_s_d= self.DiscOpt.Result()
    best_x_c, best_s_c= self.ContOptSet[best_x_d].Result()
    return best_x_d, best_x_c, best_s_c

  def Stopped(self):
    #FIXME: maxfevals
    if not self.DiscOpt.Stopped():  return False
    best_x_d, best_s_d= self.DiscOpt.Result()
    if not self.ContOptSet[best_x_d].Stopped():  return False
    return True

  def Select(self):
    self.DiscOpt.Select()
    self.ContOptSet[self.DiscOpt.Param()].Select()

  def Update(self, score):
    self.DiscOpt.Update(score)
    self.ContOptSet[self.DiscOpt.Param()].Update(score)



import copy

if __name__=='__main__':

  import numpy as np

  #Check if a is between [a_range[0],a_range[1]]
  def IsIn(a, a_range):
    if a_range[0]<a_range[1]:
      return a_range[0]<=a and a<=a_range[1]
    else:
      return a_range[1]<=a and a<=a_range[0]

  def frange(xmin,xmax,num_div):
    return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]


  #Discrete parameter is a type of function
  #Note that each function assumes different size of x

  def fobj1(x,f_none=None):
    assert len(x)==2
    if not IsIn(x[0], [-3.0,-0.5]):  return f_none
    if (x[0]+2.0)**2+(x[1]+0.5)**2<0.2:  return f_none
    return -(3.0*(x[0]-1.2)**2 + 2.0*(x[1]+2.0)**2)

  def fobj2(x,f_none=None):
    assert len(x)==2
    if not IsIn(x[0], [0.0,1.0]):  return f_none
    return -(0.5*(x[0]+2.0)**2 + 0.5*(x[1]+2.0)**2 + 20.0)

  def fobj3(x,f_none=None):
    assert len(x)==1
    if not IsIn(x[0], [1.5,3.0]):  return f_none
    return -(20.0*(x[0]-2.0)**2 - 15.0)


  #x[0]: discrete parameter to select function
  #x[1:]: continuous parameters for each function; size depends on x[0]
  def composite_fobj(x,f_none=None):
    if isinstance(x[0], float):
      print '###x[0] is float:',x[0], int(round(x[0]))
      x[0]= int(round(x[0]))
    if x[0]==2 and len(x[1:])<>1:
      print '$$$len(x[1:]) is not 1:',len(x[1:])
      x= x[0:2]
    if x[0]==0:  f= fobj1(x[1:])
    elif x[0]==1:  f= fobj2(x[1:])
    elif x[0]==2:  f= fobj3(x[1:])
    if f is None:  return f_none
    return 0.01*f


  fobj= composite_fobj

  options= {'CMA_diagonal':1, 'verb_time':0}
  #options['bounds']= [[0, -3.0,-3.0],[2.49, 3.0,3.0]]
  options['bounds']= [[-3.0,-3.0],[3.0,3.0]]
  options['tolfun']= 1.0e-4 # 1.0e-4
  #options['verb_log']= False
  options['scaling_of_variables']= np.array([0.5,1.0])
  #options['popsize']= 200
  #typical_x= [0.0,0.0]
  #options['typical_x']= np.array(typical_x)
  options['scale0']= 2.0
  options['parameters0']= [0.0,0.0]
  #options['maxfevals']= 30

  options3= {'CMA_diagonal':1, 'verb_time':0}
  options3['bounds']= [[-3.0],[3.0]]
  options3['tolfun']= 1.0e-4 # 1.0e-4
  options3['scaling_of_variables']= np.array([1.0])
  options3['scale0']= 2.0
  options3['parameters0']= [0.0]

  #copt= TContOpt(options)
  #while not copt.Stopped():
    #copt.Select()
    ##print copt.Param()
    #f= fobj(copt.Param())
    #copt.Update(f)
  #print 'copt.Result():',copt.Result()

  it_count= 0
  copts= [TContOpt(options),TContOpt(options),TContOpt(options3)]
  dcopt= TDiscContOpt(copts)
  while not dcopt.Stopped():
    dcopt.Select()
    #print copt.Param()
    x_d, x_c= dcopt.Param()
    f= fobj([x_d]+list(x_c))
    dcopt.Update(f)
    it_count+= 1
    print it_count, x_d, x_c, f
  print 'dcopt.Result():',dcopt.Result()


  fp= file('outcmaes_obj.dat','w')
  for x1 in frange(-4.0,4.0,100):
    for x2 in frange(-4.0,4.0,100):
      for i in range(3):
        x= [i,x1,x2]
        if i in (0,1):  f= fobj(x)
        elif i in (2,):  f= fobj(x[0:2])
        if f is not None:
          fp.write('%f %f %f\n' % (x1,x2,f))
    fp.write('\n')
  fp.close()

  fp= file('outcmaes_res.dat','w')
  #x,f= copt.Result()
  x_d,x_c,f= dcopt.Result()
  #fp.write('%f %f %f\n' % (x[1],x[2],f if f is not None else -10.0))
  if x_d<>2:
    fp.write('%f %f %f\n' % (x_c[0],x_c[1],f if f is not None else -10.0))
  else:
    fp.write('%f %f %f\n' % (x_c[0],0.0,f if f is not None else -10.0))
  fp.close()

  #cma.plot();
  #print 'press a key to exit > ',
  #raw_input()

  ##cma.savefig('outcmaesgraph')

