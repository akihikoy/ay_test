#!/usr/bin/python
import cma
options = {'CMA_diagonal':100, 'seed':1234, 'verb_time':0}
res = cma.fmin(cma.fcts.rosen, [0.1] * 10, 0.5, options)
#res = cma.CMAEvolutionStrategy([0.1] * 10, 0.5, options).optimize(cma.fcts.rosen)
print('best solutions fitness = %f' % (res[1]))

cma.plot();
print 'press a key to exit > ',
raw_input()

cma.show()
print 'press a key to exit > ',
raw_input()

cma.savefig('outcmaesgraph')
