#!/usr/bin/python3
#\file    state_machine2.py
#\brief   State machine ver.2
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.01, 2021

def Print(*args):
  print(' '.join(map(str,args)))
CPrint= Print

#Exit the state machine:
EXIT_STATE= '.exit'
#Stay the same state:
ORIGIN_STATE= '.origin'

#Code for exit status:
SUCCESS_CODE= 'success'
FAILURE_SM= 'failure..sm'  #Failure of state machine
FAILURE_PREFIX= 'failure.'  #User defined failure
FAILURE_PRECOND= 'failure.precond'  #Failure because preconditions are not satisfied
FAILURE_FORCEDQUIT= 'failure.forcedquit'  #Forced quit (Internal use only; if you want to quit forcibly, use SetExceptionFlag)
FAILURE_OTHER= 'failure.other'  #Other type of failure

#Event code:
EVENT_SM_ENTRY= 0
EVENT_SM_EXIT= 1
EVENT_STATE_ENTRY= 2
EVENT_STATE_STAY= 3
EVENT_STATE_EXIT= 4

#Return a failure code.
def FailureCode(code):
  return FAILURE_PREFIX+code

#Check if the exit status is success.
def IsSuccess(status):
  return status==SUCCESS_CODE

def EventCodeToStr(code):
  if   code==EVENT_SM_ENTRY   :  return 'sm_entry'
  elif code==EVENT_SM_EXIT    :  return 'sm_exit'
  elif code==EVENT_STATE_ENTRY:  return 'state_entry'
  elif code==EVENT_STATE_STAY :  return 'state_stay'
  elif code==EVENT_STATE_EXIT :  return 'state_exit'

#Check if the exit status is a failure.
#  If code is given, a specific failure code is checked.
def IsFailure(status, code=None):
  l= len(FAILURE_PREFIX)
  if code is None:
    return len(status)>=l and status[:l]==FAILURE_PREFIX
  else:
    return status==FailureCode(code)

class TFSMConditionedAction:
  def __init__(self):
    #Pointer to a function that returns if a condition is satisfied.
    self.Condition= lambda: False
    #Pointer to a function to be executed when the condition is satisfied.
    self.Action= None
    #Next state identifier after the action is executed.
    #If this is EXIT_STATE, the state machine is terminated.
    #If this is ORIGIN_STATE, the next state is the original state (i.e. state does not change).
    self.NextState= ''

class TFSMState:
  def __init__(self):
    #Pointer to a function to be executed when entering to this state.
    self.EntryAction= None
    #Pointer to a function to be executed when exiting from this state.
    self.ExitAction= None
    #Set of conditioned actions (list of TFSMConditionedAction).
    self.Actions= []
    #Pointer to a function to be executed when all conditions are not satisfied.
    #To activate ElseAction, assign: ElseAction.Condition= lambda: True
    self.ElseAction= TFSMConditionedAction()
  def NewAction(self):
    self.Actions.append(TFSMConditionedAction())
    return self.Actions[-1]  #Reference to the new action

class TStateMachine:
  def __init__(self,start_state='', debug=False, local_obj=None):
    #Set of states (dictionary of TFSMState).
    self.States= {}
    #Store some adjustable parameters of the state machine.
    #Each value should be an instance of TDiscParam or TContParam*
    #Usage: execute Select and Update as an action, access by Param.
    self.Params= {}
    #Start state identifier.
    self.StartState= start_state
    #Debug mode
    self.Debug= debug
    #Exit status.
    #SUCCESS_CODE: exit in success.
    #FAILURE_SM: exit in failure of state machine error.
    self.ExitStatus= SUCCESS_CODE
    #Event callback function.
    #EventCallback(sm, event_type, state, action)
    #  sm: this object (self)
    #  event_type: one of EVENT_*
    #  state: state or None (EVENT_SM_*)
    #  action: action or None (now takes always None; for future use)
    self.EventCallback= None
    #True when the state machine is running.
    self.Running= False
    #If this flag is set True, the state machine will quit.
    #ExitStatus is set FAILURE_FORCEDQUIT.
    self.exception_flag= False
    #Local object to store variables used in the state machine.
    #State machine does not use this variable, so it's user defined.
    #Use local_obj=TContainer() for a flexible container.
    #Use local_obj=None for small memory use.
    #NOTE: The default value of local_obj was TContainer() before.
    self.l= local_obj
    #Thread executing this state machine can store the thread information to:
    self.ThreadInfo= None

  def __getitem__(self,key):
    return self.States[key]

  def __setitem__(self,key,value):
    self.States[key]= value

  def Cleanup(self):
    self.l= None
    self.ThreadInfo= None
    del self.States

  #TODO: check the name (allowed only: [_a-zA-Z][_a-zA-Z0-9]*)
  def NewState(self,st):
    if st in self.States:
      print('Error: state ',st,' already exists')
      raise
    self.States[st]= TFSMState()

  def Show(self):
    for id,st in list(self.States.items()):
      print('[%s].EntryAction= %r' % (id,st.EntryAction))
      print('[%s].ExitAction= %r' % (id,st.ExitAction))
      print('[%s].ElseAction= %r' % (id,st.ElseAction))
      a_id= 0
      for a in st.Actions:
        print('[%s].Actions[%i].Condition= %r' % (id,a_id,a.Condition))
        print('[%s].Actions[%i].Action= %r' % (id,a_id,a.Action))
        print('[%s].Actions[%i].NextState= %r' % (id,a_id,a.NextState))
        a_id+=1
      print('')
    print('StartState=',self.StartState)
    print('Debug=',self.Debug)

  def SetStartState(self,start_state=''):
    self.StartState= start_state

  def SetExitStatus(self,code='success'):
    self.ExitStatus= code
  def SetFailure(self):
    self.SetExitStatus(FailureCode(self.curr_state))
  def SetExceptionFlag(self):
    self.exception_flag= True
    CPrint(4,'ExceptionFlag is set. Will be terminated...')

  def Run(self):
    self.Running= True
    self.prev_state= ''
    self.curr_state= self.StartState
    count= 0
    if self.EventCallback: self.EventCallback(self, EVENT_SM_ENTRY, None, None)
    while self.curr_state!='':
      count+=1
      if self.Debug: CPrint(2, '@',count,self.curr_state)
      st= self.States[self.curr_state]
      if self.EventCallback:
        if self.prev_state!=self.curr_state:
          self.EventCallback(self, EVENT_STATE_ENTRY, self.curr_state, None)
        else:
          self.EventCallback(self, EVENT_STATE_STAY, self.curr_state, None)
      if st.EntryAction and self.prev_state!=self.curr_state:
        if self.Debug: CPrint(2, '@',count,self.curr_state,'EntryAction')
        st.EntryAction()

      a_id= 0
      a_id_satisfied= -1
      next_state= ''
      for ca in st.Actions:
        if ca.Condition():
          if a_id_satisfied>=0:
            print('Warning: multiple conditions are satisfied in ',self.curr_state)
            print('  First satisfied condition index & next state:',a_id_satisfied, next_state)
            print('  Additionally satisfied condition index & next state:',a_id, ca.NextState)
            print('  First conditioned action is activated')
          else:
            a_id_satisfied= a_id
            next_state= ca.NextState
        a_id+=1

      #Execute action
      if a_id_satisfied>=0:
        if self.Debug: CPrint(2, '@',count,self.curr_state,'Condition satisfied:',a_id_satisfied)
        if st.Actions[a_id_satisfied].Action:
          if self.Debug: CPrint(2, '@',count,self.curr_state,'Action',a_id_satisfied)
          st.Actions[a_id_satisfied].Action()
      else:
        if st.ElseAction.Condition():
          if st.ElseAction.Action:
            if self.Debug: CPrint(2, '@',count,self.curr_state,'ElseAction')
            st.ElseAction.Action()
          next_state= st.ElseAction.NextState

      if self.Debug: CPrint(2, '@',count,self.curr_state,'Next state:',next_state)

      if next_state==ORIGIN_STATE or  next_state==self.curr_state:
        self.prev_state= self.curr_state
        #Note: we do not change self.curr_state
      else:
        if st.ExitAction:
          if self.Debug: CPrint(2, '@',count,self.curr_state,'ExitAction')
          st.ExitAction()
        if self.EventCallback: self.EventCallback(self, EVENT_STATE_EXIT, self.curr_state, None)
        if next_state=='':
          CPrint(4,'Next state is not defined at %s. Hint: use ElseAction to specify the case where no conditions are satisfied.' % (self.curr_state))
          self.SetExitStatus(FAILURE_SM)
        if next_state==EXIT_STATE:
          self.prev_state= self.curr_state
          self.curr_state= ''
        else:
          self.prev_state= self.curr_state
          self.curr_state= next_state

      if self.exception_flag:
        self.SetExitStatus(FAILURE_FORCEDQUIT)
        self.curr_state= ''

    if self.Debug: CPrint(2, '@ Finished in',self.ExitStatus)
    if self.EventCallback: self.EventCallback(self, EVENT_SM_EXIT, None, None)
    self.Running= False
    return self.ExitStatus

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

  sm= TStateMachine()

  sm.StartState= 'start'
  sm['start']= TFSMState()
  sm['start'].EntryAction= lambda: Print('Hello state machine!')
  sm['start'].NewAction()
  sm['start'].Actions[-1].Condition= lambda: Print("Want to move?") or AskYesNo()
  sm['start'].Actions[-1].NextState= 'count'
  sm['start'].ElseAction.Condition= lambda: True
  sm['start'].ElseAction.Action= lambda: Print('Keep to stay in start\n')
  sm['start'].ElseAction.NextState= 'start'
  sm['start'].ExitAction= lambda: (Print('-->'), GetStartTime())

  sm['count']= TFSMState()
  sm['count'].EntryAction= lambda: Print('Counting...')
  sm['count'].NewAction()
  sm['count'].Actions[-1].Condition= lambda: (int(time.time())-start_time)>=3
  sm['count'].Actions[-1].Action= lambda: Print('Hit!: '+str(int(time.time())-start_time))
  sm['count'].Actions[-1].NextState= 'stop'
  sm['count'].ElseAction.Condition= lambda: True
  sm['count'].ElseAction.Action= lambda: (Print(str(int(time.time())-start_time)), time.sleep(0.2))
  sm['count'].ElseAction.NextState= 'count'

  sm['stop']= TFSMState()
  sm['stop'].EntryAction= lambda: Print('Finishing state machine')
  sm['stop'].ElseAction.Condition= lambda: True
  sm['stop'].ElseAction.NextState= EXIT_STATE

  sm.Run()

