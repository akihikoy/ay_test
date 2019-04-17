#!/usr/bin/python
import random
import barecmaes2 as cma
random.seed(5)
x = cma.fmin(cma.Fcts.rosenbrock, 4 * [0.5], 0.5, verb_plot=0)

