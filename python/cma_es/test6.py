#!/usr/bin/python
import cma
import numpy as np

def fobj1(x):
  assert len(x)==2
  return 3.0*np.dot(x-np.array([1.5,0.5]),x-np.array([1.5,0.5]))+2.0
  #return 3.0*np.dot(x-np.array([1.5,0.5]),x-np.array([1.5,0.5]))+1.2*x[0]-2.5*x[1]+2.0

def frange(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

#fobj= cma.fcts.rosen
fobj= fobj1

#options = {'CMA_diagonal':100, 'verb_time':0}
options = {'CMA_diagonal':1, 'verb_time':0}
res = cma.fmin(fobj, [0.1] * 2, 0.5, options)
#res = cma.CMAEvolutionStrategy([0.1] * 10, 0.5, options).optimize(fobj)
print('best solutions fitness = %f' % (res[1]))

print res


fp= file('outcmaes_obj.dat','w')
for x1 in frange(-2.0,2.0,100):
  for x2 in frange(-2.0,2.0,100):
    x= np.array([x1,x2])
    fp.write('%f %f %f\n' % (x[0],x[1],fobj(x)))
  fp.write('\n')
fp.close()

fp= file('outcmaes_res.dat','w')
#for x in res[0]:
x= res[0]
fp.write('%f %f %f\n' % (x[0],x[1],fobj(x)))
fp.close()

cma.plot();
print 'press a key to exit > ',
raw_input()

#cma.savefig('outcmaesgraph')
