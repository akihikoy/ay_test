#!/usr/bin/python3
import cma
#help(cma)  # "this" help message, use cma? in ipython
#help(cma.fmin)
#help(cma.CMAEvolutionStrategy)
#help(cma.CMAOptions)
for k,v in list(cma.CMAOptions().items()):
  print('[%s]=%s'%(k,v))
#cma.CMAOptions('tol')  # display 'tolerance' termination options
#cma.CMAOptions('verb') # display verbosity options
#res = cma.fmin(cma.Fcts.tablet, 15 * [1], 1)
#res[0]  # best evaluated solution
#res[5]  # mean solution, presumably better with noise

