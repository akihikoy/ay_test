#!/usr/bin/python
#qplot -x -3d -s 'set xlabel "x";set ylabel "y";set view equal xy' outcmaes_obj.dat w l outcmaes_res.dat ps 3 data/res000{0,1,2,3,5,6}.dat -showerr
#TEST to use stored solutions and scores data
import cma
import numpy as np

def fobj1(x,f_none=None):
  assert len(x)==2
  #if (x[0]-0.5)**2+(x[1]+0.5)**2<0.2:  return f_none
  return 3.0*(x[0]-1.2)**2 + 2.0*(x[1]+2.0)**2

def frange(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

#fobj= cma.fcts.rosen
fobj= fobj1


using_init= True
#init_x0= [ 1.35091082, -1.64772063]
#init_sigma0= [ 0.28612504,  1.11171478]
##popsize-1:
#init_solutions0= [
  #np.array([1.19997522269, -0.91075646271]),
  #np.array([1.20312824671, -1.87495850026]),
  #np.array([1.41625771099, -1.65410726357]),
  #np.array([1.74571448699, -1.73080568194]),
  #np.array([1.1541920869, -0.648801324059])]
#init_score0= [
  #2.372903,
  #0.031300,
  #0.379586,
  #1.038344,
  #3.657771]
##1.62588423219 0.185635118507 10.098134
init_x0= [1.6,2.5]
init_sigma0= [0.25,0.5]
#popsize-1:
init_solutions0= [
  np.array([1.15577068683, 2.78306227797]),
  np.array([1.97053342159, 1.51965484274]),
  np.array([1.56030150378, 2.90707053744]),
  np.array([1.5699010727, 2.90721893587 ]),
  np.array([1.99722431459, 2.95065252211])]
init_score0= [
  45.761238,
  26.557106,
  48.548134,
  48.572076,
  50.924621]
#1.17593719841, 2.33528811379 37.591183

options = {'CMA_diagonal':1, 'verb_time':0}
options['bounds']= [[-3.0,-3.0],[3.0,3.0]]
options['tolfun']= 1.0e-4 # 1.0e-4
#options['verb_log']= False
options['scaling_of_variables']= np.array([0.5,1.0])
#options['scaling_of_variables']= np.array([0.00001,1.0])
#options['popsize']= 200
#typical_x= [0.0,0.0]
#options['typical_x']= np.array(typical_x)
scale0= 0.5
#parameters0= [0.0,0.0]
#parameters0= [1.19,-1.99]
parameters0= [1.6,2.5]
if using_init:
  parameters0= init_x0
  scale0= 1.0
  options['scaling_of_variables']= init_sigma0
es= cma.CMAEvolutionStrategy(parameters0, scale0, options)


solutions, scores = [], []
if using_init:
  assert(len(init_solutions0)==len(init_score0))
  es.ask(len(init_solutions0))
  solutions= init_solutions0
  scores= init_score0
  #es.tell(solutions,scores)

print 'es.result():',es.result()

count= 0
while not es.stop():
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
  print 'es.result()@%i:'%(count),es.result()
  #print solutions

  #if count%5==0:
    #print '[%i]'%count, ' '.join(map(str,solutions[0]))

  fp= file('data/res%04i.dat'%(count),'w')
  count+=1
  for x in solutions:
    fp.write('%s %f\n' % (' '.join(map(str,x)),fobj(x,-10)))
  fp.close()

  solutions, scores = [], []


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
