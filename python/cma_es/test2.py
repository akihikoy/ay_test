#!/usr/bin/python
import barecmaes2 as cma
es = cma.CMAES(3 * [0.1], 1)
logger = cma.CMAESDataLogger().register(es)
while not es.stop():
    X = es.ask()
    es.tell(X, [cma.Fcts.elli(x) for x in X])
    logger.add()
logger.plot()

keytoexit= True
if keytoexit:
  print 'press a key to exit > ',
  raw_input()
