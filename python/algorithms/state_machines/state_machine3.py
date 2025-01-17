#!/usr/bin/python3
#\file    state_machine3.py
#\brief   State machine ver.3
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.01, 2021

def Print(*args):
  print(' '.join(map(str,args)))
CPrint= Print


class TStateMachine(object):
  def __init__(self, states, start_state, debug=False):
    self.States= states
    self.StartState= start_state
    self.Debug= debug

  class TAction(object):
    def __init__(self):
      self.Entry= None
      self.Actions= []
      self.Else= None

  def LoadState(self, state):
    actions= self.TAction()
    for action in self.States[state]:
      if len(action)==0:  raise Exception('Invalid state description.')
      if action[0]=='entry':
        if len(action)!=2:  raise Exception('Invalid state description.')
        actions.Entry= action[1]  #Action
      elif action[0]=='else':
        if len(action)==2:  actions.Else= (action[1],None)  #Next state, no action
        elif len(action)==3:  actions.Else= (action[1],action[2])  #Next state, action
        else:  raise Exception('Invalid state description.')
      else:
        if len(action)==2:  actions.Actions.append((action[0],action[1],None))  #Condition, next state, no action
        elif len(action)==3:  actions.Actions.append((action[0],action[1],action[2]))  #Condition, next state, action
        else:  raise Exception('Invalid state description.')
    return actions

  def Run(self):
    self.prev_state= ''
    self.curr_state= self.StartState
    count= 0
    while self.curr_state!='':
      count+=1
      if self.Debug: CPrint(2, '@',count,self.curr_state)
      if self.curr_state not in self.States:
        raise Exception('State',self.curr_state,'does not exist.')
      actions= self.LoadState(self.curr_state)
      if actions.Entry and self.prev_state!=self.curr_state:
        if self.Debug: CPrint(2, '@',count,self.curr_state,'Entry')
        actions.Entry()

      a_id_satisfied= -1
      next_state= ''
      for a_id,(condition,next_st,action) in enumerate(actions.Actions):
        if condition():
          if a_id_satisfied>=0:
            print('Warning: multiple conditions are satisfied in ',self.curr_state)
            print('  First satisfied condition index, next state:',a_id_satisfied, next_state)
            print('  Additionally satisfied condition index, next state:',a_id, next_st)
            print('  First conditioned action is used')
          else:
            a_id_satisfied= a_id
            next_state= next_st

      #Execute action
      if a_id_satisfied>=0:
        if self.Debug: CPrint(2, '@',count,self.curr_state,'Condition satisfied:',a_id_satisfied)
        action= actions.Actions[a_id_satisfied][2]
        if action:
          if self.Debug: CPrint(2, '@',count,self.curr_state,'Action',a_id_satisfied)
          action()
      else:
        if actions.Else:
          next_st,action= actions.Else
          if action:
            if self.Debug: CPrint(2, '@',count,self.curr_state,'Else')
            action()
          next_state= next_st

      if self.Debug: CPrint(2, '@',count,self.curr_state,'Next state:',next_state)

      if next_state=='':
        raise Exception('Next state is not defined at',self.curr_state,'; Hint: use else action to specify the case where no conditions are satisfied.')
      elif next_state=='.exit':
        self.prev_state= self.curr_state
        self.curr_state= ''
      else:
        self.prev_state= self.curr_state
        self.curr_state= next_state


if __name__=='__main__':
  import time,sys

  def AskYesNo():
    while 1:
      sys.stdout.write('  (y|n) > ')
      ans= sys.stdin.readline().strip()
      if ans=='y' or ans=='Y':  return True
      elif ans=='n' or ans=='N':  return False

  start_time= 0
  def GetStartTime():
    global start_time
    start_time= int(time.time())

  states= {
    'start': [
      ('entry',lambda: Print('Hello state machine!')),
      (lambda: Print("Want to move?") or AskYesNo(),'count', lambda:(Print('-->'), GetStartTime())),
      ('else','start',lambda: Print('Keep to stay in start\n')),
      ],
    'count': [
      ('entry',lambda: Print('Counting...')),
      (lambda: (int(time.time())-start_time)>=3,'stop',lambda: Print('Hit!: '+str(int(time.time())-start_time))),
      ('else','count',lambda: (Print(str(int(time.time())-start_time)), time.sleep(0.2))),
      ],
    'stop': [
      ('entry',lambda: Print('Finishing state machine')),
      ('else','.exit'),
      ],
    }

  sm= TStateMachine(states,'start')

  sm.Run()

