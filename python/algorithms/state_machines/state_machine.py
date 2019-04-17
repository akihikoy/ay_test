#!/usr/bin/python
#Simple state machine

EXIT_STATE= '__exit__'
ORIGIN_STATE= '__origin__'

#Another simple print function to be used as an action
def Print(*s):
  for ss in s:
    print ss,
  print ''

class TAsciiColors:
  Header  = '\033[95m'
  OKBlue  = '\033[94m'
  OKGreen = '\033[92m'
  Warning = '\033[93m'
  Fail    = '\033[91m'
  EndC    = '\033[0m'

def DPrint(*s):
  first=True
  for ss in s:
    if not first:
      print ss,
    else:
      print TAsciiColors.OKGreen+str(ss),
  print TAsciiColors.EndC+''

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
  def __init__(self,start_state=''):
    #Set of states (dictionary of TFSMState).
    self.States= {}
    #Start state identifier.
    self.StartState= start_state
    #Debug mode
    self.Debug= False

  def __getitem__(self,key):
    return self.States[key]

  def __setitem__(self,key,value):
    self.States[key]= value

  def Show(self):
    for id,st in self.States.items():
      print '[%s].EntryAction= %r' % (id,st.EntryAction)
      print '[%s].ExitAction= %r' % (id,st.ExitAction)
      print '[%s].ElseAction= %r' % (id,st.ElseAction)
      a_id= 0
      for a in st.Actions:
        print '[%s].Actions[%i].Condition= %r' % (id,a_id,a.Condition)
        print '[%s].Actions[%i].Action= %r' % (id,a_id,a.Action)
        print '[%s].Actions[%i].NextState= %r' % (id,a_id,a.NextState)
        a_id+=1
      print ''
    print 'StartState=',self.StartState
    print 'Debug=',self.Debug

  def SetStartState(self,start_state=''):
    self.StartState= start_state

  def Run(self):
    self.prev_state= ''
    self.curr_state= self.StartState
    count= 0
    while self.curr_state!='':
      count+=1
      if self.Debug: DPrint('@',count,self.curr_state)
      st= self.States[self.curr_state]
      if st.EntryAction and self.prev_state!=self.curr_state:
        if self.Debug: DPrint('@',count,self.curr_state,'EntryAction')
        st.EntryAction()

      a_id= 0
      a_id_satisfied= -1
      next_state= ''
      for ca in st.Actions:
        if ca.Condition():
          if a_id_satisfied>=0:
            print 'Warning: multiple conditions are satisfied in ',self.curr_state
            print '  First satisfied condition index & next state:',a_id_satisfied, next_state
            print '  Additionally satisfied condition index & next state:',a_id, ca.NextState
            print '  First conditioned action is activated'
          else:
            a_id_satisfied= a_id
            next_state= ca.NextState
        a_id+=1

      #Execute action
      if a_id_satisfied>=0:
        if self.Debug: DPrint('@',count,self.curr_state,'Condition satisfied:',a_id_satisfied)
        if st.Actions[a_id_satisfied].Action:
          if self.Debug: DPrint('@',count,self.curr_state,'Action',a_id_satisfied)
          st.Actions[a_id_satisfied].Action()
      else:
        if st.ElseAction.Condition():
          if st.ElseAction.Action:
            if self.Debug: DPrint('@',count,self.curr_state,'ElseAction')
            st.ElseAction.Action()
          next_state= st.ElseAction.NextState

      if self.Debug: DPrint('@',count,self.curr_state,'Next state:',next_state)

      if next_state==ORIGIN_STATE or  next_state==self.curr_state:
        self.prev_state= self.curr_state
        #self.curr_state= self.curr_state
      else:
        if st.ExitAction:
          if self.Debug: DPrint('@',count,self.curr_state,'ExitAction')
          st.ExitAction()
        if next_state=='':
          print 'Next state is not defined at %s. Hint: use ElseAction to specify the case where no conditions are satisfied.' % (self.curr_state)
        self.prev_state= self.curr_state
        if next_state==EXIT_STATE:
          self.curr_state= ''
        else:
          self.curr_state= next_state

    if self.Debug: DPrint('@ Finished')


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

