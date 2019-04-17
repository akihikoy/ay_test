#!/usr/bin/python
from state_machine_rl import *
from flow.flow_dyn import *

import time,sys

class TLocal:
  amount_trg= 0.3
  amount_prev= 0.0
  amount= 0.0

l= TLocal()

#fdyn= TFlowDyn(a_bottle=0.5,show_state=False)
#fdyn.Start()

#l.amount= fdyn.a_cup
l.amount= 0.0


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

def IsTimeout(st):
  return fdyn.time > 20.0

def IsPoured(st):
  if fdyn.a_cup > l.amount_trg:
    print 'Poured (%f: %f / %f)' % (fdyn.time, fdyn.a_cup, l.amount_trg)
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
sm['start'].Actions[-1].Condition= lambda st: IsFlowObserved(0.05)  #FIXME:magic number
sm['start'].Actions[-1].NextState= 'pour'
sm['start'].ElseAction.Condition= lambda st: True
sm['start'].ElseAction.Action= lambda st: ControlStep(5.0)
sm['start'].ElseAction.NextState= 'start'

sm['pour']= TFSMState()
sm['pour'].ParamLearner= TDiscLearner()
sm['pour'].ParamLearner.Candidates= [0.0,1.5,3.0]
sm['pour'].ParamLearner.BoltzmannTau= 5.0
#sm['pour'].ParamLearner= TQuadLearner()
#sm['pour'].ParamLearner.Min= 0.0
#sm['pour'].ParamLearner.Max= 3.0
#sm['pour'].ParamLearner.C= [1.0,0.0,0.0]
sm['pour'].NewAction()
sm['pour'].Actions[-1]= poured_action
sm['pour'].NewAction()
sm['pour'].Actions[-1]= timeout_action
sm['pour'].NewAction()
sm['pour'].Actions[-1].Condition= lambda st: IsFlowObserved(0.02)
sm['pour'].Actions[-1].Action= lambda st: (Print(st.Param()),ControlStep(st.Param()))  #FIXME: magic number
sm['pour'].Actions[-1].NextState= 'pour'
sm['pour'].ElseAction.Condition= lambda st: True
sm['pour'].ElseAction.Action= lambda st: ControlStep(2.5)  #FIXME: magic number
sm['pour'].ElseAction.NextState= 'pour'
sm['pour'].Evaluator= lambda st: 5.0-fdyn.time

sm['stop']= TFSMState()
sm['stop'].EntryAction= lambda st: (ControlStep(0.0), Print('Move back to init'))
#sm['stop'].NewAction()
#sm['stop'].Actions[-1].Condition= lambda st: fdyn.theta>0.0
#sm['stop'].Actions[-1].Action= lambda st: ControlStep(-5.0)
#sm['stop'].Actions[-1].NextState= 'stop'
sm['stop'].ElseAction.Condition= lambda st: True
sm['stop'].ElseAction.Action= lambda st: (time.sleep(0.5), Print('End of pouring'))
sm['stop'].ElseAction.NextState= EXIT_STATE

for i in range(50):
  fdyn= TFlowDyn(a_bottle=0.5,show_state=False)
  fdyn.Start()
  sm.Run()
  print sm['pour'].ParamLearner.ParamValues
  #print sm['pour'].ParamLearner.C
  fdyn.Stop()
  print 'Continue?'
  #if AskYesNo():
    #continue
  #else:
    #break

print 'Plot by:'
print "cat data/flow.dat | qplot -x -s 'set y2tics' - u 1:3 - u 1:4 - u 1:5 - u 1:6 ax x1y2"

