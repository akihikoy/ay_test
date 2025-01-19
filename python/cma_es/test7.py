#!/usr/bin/python3
import cma
import numpy as np
import os
if not os.path.exists('data'):
  os.makedirs('data')  #make data directory

def fobj1(x):
  assert len(x)==2
  return 3.0*np.dot(x-np.array([1.5,0.5]),x-np.array([1.5,0.5]))+2.0
  #return 3.0*np.dot(x-np.array([1.5,0.5]),x-np.array([1.5,0.5]))+1.2*x[0]-2.5*x[1]+2.0

def fobj2(x):
  assert len(x)==2
  return 3.0*(x[0]-1.2)**2+2.0

def frange(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

#fobj= cma.fcts.rosen
fobj= fobj1

using_bounds= False
if not using_bounds:
  #options= {'CMA_diagonal':100, 'verb_time':0}
  options= {'CMA_diagonal':1, 'verb_time':0}
  #res= cma.fmin(fobj, [0.1] * 2, 0.5, options)
  es= cma.CMAEvolutionStrategy([0.1] * 2, 0.5, options)
else:
  options= {'CMA_diagonal':1, 'verb_time':0, 'bounds':[[-1.0,-1.0],[0.0,0.0]]}
  es= cma.CMAEvolutionStrategy([-0.1] * 2, 0.5, options)

print('es.result():',es.result())

count= 0
while not es.stop():
  solutions= es.ask()
  scores= [fobj(x) for x in solutions]
  es.tell(solutions,scores)
  es.disp()
  #print 'es.result():',es.result()

  fp= open('data/res%04i.dat'%(count),'w')
  count+=1
  for x in solutions:
    fp.write('%f %f %f\n' % (x[0],x[1],fobj(x)))
  fp.close()

res= es.result()
'''
- ``res[0]`` (``xopt``) -- best evaluated solution
- ``res[1]`` (``fopt``) -- respective function value
- ``res[2]`` (``evalsopt``) -- respective number of function evaluations
- ``res[3]`` (``evals``) -- number of overall conducted objective function evaluations
- ``res[4]`` (``iterations``) -- number of overall conducted iterations
- ``res[5]`` (``xmean``) -- mean of the final sample distribution
- ``res[6]`` (``stds``) -- effective stds of the final sample distribution
- ``res[-3]`` (``stop``) -- termination condition(s) in a dictionary
- ``res[-2]`` (``cmaes``) -- class `CMAEvolutionStrategy` instance
- ``res[-1]`` (``logger``) -- class `CMADataLogger` instance
'''

print(('best solutions fitness = %f' % (res[1])))

print(res)


fp= open('outcmaes_obj.dat','w')
for x1 in frange(-2.0,2.0,100):
  for x2 in frange(-2.0,2.0,100):
    x= np.array([x1,x2])
    fp.write('%f %f %f\n' % (x[0],x[1],fobj(x)))
  fp.write('\n')
fp.close()

fp= open('outcmaes_res.dat','w')
#for x in res[0]:
x= res[0]
fp.write('%f %f %f\n' % (x[0],x[1],fobj(x)))
fp.close()

#cma.plot();
#print 'press a key to exit > ',
#raw_input()

#cma.savefig('outcmaesgraph')

print('''qplot -x -3d -s 'set xlabel "x";set ylabel "y";set view equal xy' outcmaes_obj.dat w l outcmaes_res.dat ps 3 data/res00{00,02,05,10,15}.dat''')
