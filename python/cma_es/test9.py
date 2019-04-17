#!/usr/bin/python
import cma
import numpy as np

def fobj1(x):
  assert len(x)==1
  return 3.0*(x[0]-1.2)**2+2.0

def frange(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

#fobj= cma.fcts.rosen
fobj= fobj1

using_bounds= False
if not using_bounds:
  #options = {'CMA_diagonal':100, 'verb_time':0}
  #options = {'CMA_diagonal':1, 'verb_time':0}
  #options = {'verb_time':0}
  options={}
  #options['popsize']= 4
  #res = cma.fmin(fobj, [0.1], 0.5, options)
  es = cma.CMAEvolutionStrategy([0.1], 0.5, options)
else:
  #options = {'CMA_diagonal':1, 'verb_time':0, 'bounds':[[-1.0],[0.0]]}
  #options = {'CMA_diagonal':1, 'verb_time':0, 'bounds':[[-1.0],[]]}
  #options = {'CMA_diagonal':1, 'verb_time':0, 'bounds':[[-1.0],None]}
  options = {'CMA_diagonal':1, 'verb_time':0, 'bounds':[[],[0.0]]}
  es = cma.CMAEvolutionStrategy([-0.1], 0.5, options)


#solutions= es.ask()
#solutions= [np.array([ 1.29323333]), np.array([ 1.33494294]), np.array([ 1.2478004]), np.array([ 1.34619473])]
#scores= [fobj(x) for x in solutions]
#es.tell(solutions,scores)

print 'es.result():',es.result()

count= 0
while not es.stop():
  solutions, scores = [], []
  #while len(solutions) < es.popsize+3:  #This is OK
  while len(solutions) < es.popsize:
    #curr_fit = None
    #while curr_fit in (None, np.NaN):
    x = es.ask(1)[0]
    #curr_fit = cma.fcts.somenan(x, cma.fcts.elli) # might return np.NaN
    solutions.append(x)
    scores.append(fobj(x))
  es.tell(solutions, scores)
  es.disp()
  #print 'es.result():',es.result()
  #print solutions

  fp= file('data/res%04i.dat'%(count),'w')
  count+=1
  for x in solutions:
    fp.write('%f %f\n' % (x[0],fobj(x)))
  fp.close()


res= es.result()

print('best solutions fitness = %f' % (res[1]))

print res


fp= file('outcmaes_obj.dat','w')
for x1 in frange(-2.0,2.0,100):
  x= np.array([x1])
  fp.write('%f %f\n' % (x[0],fobj(x)))
fp.close()

fp= file('outcmaes_res.dat','w')
#for x in res[0]:
x= res[0]
fp.write('%f %f\n' % (x[0],fobj(x)))
fp.close()

cma.plot();
print 'press a key to exit > ',
raw_input()

#cma.savefig('outcmaesgraph')
