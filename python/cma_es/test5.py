#!/usr/bin/python
import cma
options = {'CMA_diagonal':100, 'seed':1234, 'verb_time':0}
res = cma.fmin(cma.fcts.rosen, [0.1] * 2, 0.5, options)
#res = cma.CMAEvolutionStrategy([0.1] * 10, 0.5, options).optimize(cma.fcts.rosen)
print('best solutions fitness = %f' % (res[1]))

print res

import numpy as np
def frange(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

fp= file('outcmaes_obj.dat','w')
for x1 in frange(-2.0,2.0,100):
  for x2 in frange(-2.0,2.0,100):
    x= np.array([x1,x2])
    fp.write('%f %f %f\n' % (x[0],x[1],cma.fcts.rosen(x)))
  fp.write('\n')
fp.close()

fp= file('outcmaes_res.dat','w')
#for x in res[0]:
x= res[0]
fp.write('%f %f %f\n' % (x[0],x[1],cma.fcts.rosen(x)))
fp.close()

cma.plot();
print 'press a key to exit > ',
raw_input()

#cma.savefig('outcmaesgraph')
