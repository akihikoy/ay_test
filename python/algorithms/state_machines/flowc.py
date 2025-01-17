#!/usr/bin/python3
from state_machine import *
from flow.flow_dyn import *

import time,sys

class TLocal:
  amount_trg= 0.3
  amount_prev= 0.0
  amount= 0.0

l= TLocal()

fdyn= TFlowDyn(a_bottle=0.5,show_state=False)
fdyn.Start()

l.amount= fdyn.a_cup


def AskYesNo():
  while 1:
    sys.stdout.write('  (y|n) > ')
    ans= sys.stdin.readline().strip()
    if ans=='y' or ans=='Y':  return True
    elif ans=='n' or ans=='N':  return False

def ControlStep(dtheta=1.0):
  l.amount_prev= l.amount
  l.amount= fdyn.a_cup

  fdyn.Control(dtheta)
  time.sleep(fdyn.time_step)

def IsTimeout():
  return fdyn.time > 20.0

def IsPoured():
  if fdyn.a_cup > l.amount_trg:
    print('Poured (%f: %f / %f)' % (fdyn.time, fdyn.a_cup, l.amount_trg))
    return True
  return False

def IsFlowObserved(self, sensitivity=0.02):
  threshold= l.amount_trg * sensitivity
  #print 'DEBUG: %f %f' % (fdyn.a_cup-l.amount_prev, threshold)
  #return fdyn.a_cup-l.amount_prev > threshold
  #print 'DEBUG: %f %f' % (fdyn.flow_obs, threshold)
  return fdyn.flow_obs > threshold


sm= TStateMachine()
sm.Debug= True

timeout_action= TFSMConditionedAction()
timeout_action.Condition= IsTimeout
timeout_action.NextState= 'stop'

poured_action= TFSMConditionedAction()
poured_action.Condition= IsPoured
poured_action.NextState= 'stop'

sm.StartState= 'start'
sm['start']= TFSMState()
sm['start'].NewAction()
sm['start'].Actions[-1]= poured_action
sm['start'].NewAction()
sm['start'].Actions[-1]= timeout_action
sm['start'].NewAction()
sm['start'].Actions[-1].Condition= lambda: IsFlowObserved(0.05)  #FIXME:magic number
sm['start'].Actions[-1].NextState= 'pour'
sm['start'].ElseAction.Condition= lambda: True
sm['start'].ElseAction.Action= lambda: ControlStep(5.0)
sm['start'].ElseAction.NextState= 'start'

sm['pour']= TFSMState()
sm['pour'].NewAction()
sm['pour'].Actions[-1]= poured_action
sm['pour'].NewAction()
sm['pour'].Actions[-1]= timeout_action
sm['pour'].NewAction()
sm['pour'].Actions[-1].Condition= lambda: IsFlowObserved(0.02)
sm['pour'].Actions[-1].Action= lambda: ControlStep(1.5)  #FIXME: magic number
sm['pour'].Actions[-1].NextState= 'pour'
sm['pour'].ElseAction.Condition= lambda: True
sm['pour'].ElseAction.Action= lambda: ControlStep(2.5)  #FIXME: magic number
sm['pour'].ElseAction.NextState= 'pour'

sm['stop']= TFSMState()
sm['stop'].EntryAction= lambda: (ControlStep(0.0), Print('Move back to init'))
#sm['stop'].NewAction()
#sm['stop'].Actions[-1].Condition= lambda: fdyn.theta>0.0
#sm['stop'].Actions[-1].Action= lambda: ControlStep(-5.0)
#sm['stop'].Actions[-1].NextState= 'stop'
sm['stop'].ElseAction.Condition= lambda: True
sm['stop'].ElseAction.Action= lambda: (time.sleep(0.5), Print('End of pouring'))
sm['stop'].ElseAction.NextState= EXIT_STATE


sm.Run()

fdyn.Stop()

print('Plot by:')
print("cat data/flow.dat | qplot -x -s 'set y2tics' - u 1:3 - u 1:4 - u 1:5 - u 1:6 ax x1y2")

