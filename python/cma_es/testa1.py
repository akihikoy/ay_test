#!/usr/bin/python
#qplot -x -3d -s 'set xlabel "x";set ylabel "y";set view equal xy' outcmaes_obj.dat w l outcmaes_res.dat ps 3 data/res000{0,1,2,3,5,6}.dat -showerr
import cma
import numpy as np

def fobj1(x,f_none=None):
  assert len(x)==2
  if (x[0]-0.5)**2+(x[1]+0.5)**2<0.2:  return f_none
  return 3.0*(x[0]-1.2)**2 + 2.0*(x[1]+2.0)**2

def frange(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

#fobj= cma.fcts.rosen
fobj= fobj1

options = {'CMA_diagonal':1, 'verb_time':0}
options['bounds']= [[-3.0,-3.0],[3.0,3.0]]
options['tolfun']= 1.0e-4 # 1.0e-4
#options['verb_log']= False
#options['scaling_of_variables']= np.array([0.5,1.0])
options['scaling_of_variables']= np.array([0.00001,1.0])
options['popsize']= 200
#typical_x= [0.0,0.0]
#options['typical_x']= np.array(typical_x)
scale0= 0.5
#parameters0= [0.0,0.0]
#parameters0= [1.19,-1.99]
parameters0= [1.2,0.0]
es= cma.CMAEvolutionStrategy(parameters0, scale0, options)


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
    f= fobj(x)
    if f is not None:
      solutions.append(x)
      scores.append(f)
  es.tell(solutions, scores)
  es.disp()
  #print 'es.result():',es.result()
  #print solutions

  #if count%5==0:
    #print '[%i]'%count, ' '.join(map(str,solutions[0]))

  fp= file('data/res%04i.dat'%(count),'w')
  count+=1
  for x in solutions:
    fp.write('%s %f\n' % (' '.join(map(str,x)),fobj(x,-10)))
  fp.close()


res= es.result()

print 'best solutions = ', res[0]
print 'best solutions fitness = %f' % (res[1])

print res


fp= file('outcmaes_obj.dat','w')
for x1 in frange(-4.0,4.0,100):
  for x2 in frange(-4.0,4.0,100):
    x= np.array([x1,x2])
    fp.write('%s %f\n' % (' '.join(map(str,x)),fobj(x,-10)))
  fp.write('\n')
fp.close()

fp= file('outcmaes_res.dat','w')
#for x in res[0]:
x= res[0]
fp.write('%s %f\n' % (' '.join(map(str,x)),fobj(x,-10)))
fp.close()

cma.plot();
print 'press a key to exit > ',
raw_input()

#cma.savefig('outcmaesgraph')
