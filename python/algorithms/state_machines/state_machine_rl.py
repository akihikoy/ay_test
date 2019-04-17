#!/usr/bin/python
#State machine with reinforcement learning
import time,math,random

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

#Return a Boltzmann policy (probabilities of selecting each action)
def BoltzmannPolicy(tau, values):
  if len(values)==0: return []
  max_v= max(values)
  sum_v= 0.0
  for q in values:
    sum_v+= math.exp((q-max_v)/tau)
  if sum_v<1.0e-10:
    return [1.0/float(len(values))]*len(values)
  probs= [0.0]*len(values)
  for d in range(len(values)):
    probs[d]= math.exp((values[d]-max_v)/tau)/sum_v
  return probs

#Return an action selected w.r.t. the policy (probabilities of selecting each action)
def SelectFromPolicy(probs):
  p= random.random()  #Random number in [0,1]
  action= 0
  for prob in probs:
    if p<=prob:  return action
    p-= prob
    action+= 1
  return action-1


class TFSMConditionedAction:
  def __init__(self):
    #Pointer to a function that returns if a condition is satisfied.
    #[NEW]Its argument is a reference to TFSMState object that used this function.
    self.Condition= lambda st: False
    #Pointer to a function to be executed when the condition is satisfied.
    #[NEW]Its argument is a reference to TFSMState object that used this function.
    self.Action= None
    #Next state identifier after the action is executed.
    #If this is EXIT_STATE, the state machine is terminated.
    #If this is ORIGIN_STATE, the next state is the original state (i.e. state does not change).
    self.NextState= ''

class TDefaultLearner:
  def __init__(self):
    self.ParamValue= 0.0
  #Returns the latest selected parameter.
  def Param(self):
    return None
  #Returns the value of the latest selected parameter or the state.
  def Value(self):
    return self.ParamValue
  #Select a parameter
  def Select(self):
    return
  #Update
  def Update(self,discounted_td_err):
    #FIXME: REPLACE THE MAGIC NUMBERS!!
    alpha= 0.4
    self.ParamValue+= alpha*discounted_td_err

#Learning to select a vector from discrete set
class TDiscLearner:
  def __init__(self):
    #[NEW]Parameter candidates which is a set of parameter vectors.
    self.Candidates= []
    #[NEW]Values of the parameter candidates.
    #Its size is len(Candidates) if Candidates is not empty.
    #Otherwise it stores the value of the state as a list of size 1.
    self.ParamValues= [0.0]
    #Index of the parameter vector lastly executed.
    self.Index= -1
    #[NEW]Temparature parameter for the Boltzmann selection.
    self.BoltzmannTau= 1.0
  #Returns the latest selected parameter.
  def Param(self):
    if len(self.Candidates)>0 and self.Index>=0:
      return self.Candidates[self.Index]
    return None
  #Returns the value of the latest selected parameter or the state.
  def Value(self):
    if self.Index>=0:
      return self.ParamValues[self.Index]
    return None
  def Select(self):
    if len(self.Candidates)>0:
      if len(self.Candidates)!=len(self.ParamValues):
        self.ParamValues= [0.0]*len(self.Candidates)
        print 'Warning: ParamValues is not initialized.  Using %r.' % (self.ParamValues)
      probs= BoltzmannPolicy(self.BoltzmannTau,self.ParamValues)
      self.Index= SelectFromPolicy(probs)
    else:
      self.Index= 0
  def Update(self,discounted_td_err):
    if self.Index>=0:
      #FIXME: REPLACE THE MAGIC NUMBERS!!
      alpha= 0.4
      self.ParamValues[self.Index]+= alpha*discounted_td_err
      print 'DEBUG: %i, %f' % (self.Index,discounted_td_err)

#Learning a 1-dim parameter using quadratic function
class TQuadLearner:
  def __init__(self):
    self.Min= -1.0
    self.Max= 1.0
    #Parameter of quadratic function (C[1]: center)
    self.C= [1.0,0.0,0.0]
    #[NEW]Temparature parameter for the Boltzmann selection.
    #self.GaussianSigmaFactor= 0.5
    self.GaussianSigmaFactor= 5.0
    self.LastParam= None
  #Returns the latest selected parameter.
  def Param(self):
    return self.LastParam
  #Returns the value of the latest selected parameter or the state.
  def Value(self):
    return self.ValueAt(self.LastParam)
  def ValueAt(self,param):
    return -self.C[0]*(param-self.C[1])**2+self.C[2]
  def Select(self):
    self.LastParam= random.gauss(self.C[1],self.GaussianSigmaFactor/math.sqrt(self.C[0]))
    if self.LastParam<self.Min:  self.LastParam= self.Min
    elif self.LastParam>self.Max:  self.LastParam= self.Max
  def Update(self,discounted_td_err):
    alpha= 0.1
    self.C[0]+= 0.01*alpha*discounted_td_err*(-(self.LastParam-self.C[1])**2)
    self.C[1]+= alpha*discounted_td_err*(2.0*self.C[0]*(self.LastParam-self.C[1]))
    self.C[2]+= alpha*discounted_td_err*1.0
    if self.C[1]<self.Min:  self.C[1]= self.Min
    elif self.C[1]>self.Max:  self.C[1]= self.Max
    if self.C[0]<=1.0e-2: self.C[0]= 1.0e-2
    print 'DEBUG: %f, %f, %f' % ( discounted_td_err, (-(self.LastParam-self.C[1])**2), (2.0*self.C[0]*(self.LastParam-self.C[1])) )

#[NEW]Store a log of executed parameter
class TFSMParameterLog:
  def __init__(self):
    #Value of the parameter
    self.Value= 0.0
    #Time of the execution
    self.Time= -1.0

class TFSMState:
  def __init__(self):
    #Pointer to a function to be executed when entering to this state.
    #[NEW]Its argument is a reference to TFSMState object that used this function.
    self.EntryAction= None
    #Pointer to a function to be executed when exiting from this state.
    #[NEW]Its argument is a reference to TFSMState object that used this function.
    self.ExitAction= None
    #Set of conditioned actions (list of TFSMConditionedAction).
    self.Actions= []
    #Pointer to a function to be executed when all conditions are not satisfied.
    #To activate ElseAction, assign: ElseAction.Condition= lambda st: True
    self.ElseAction= TFSMConditionedAction()

    #[NEW]Pointer to a function to evaluate the result of the actions.
    #This function should return a float value.
    #This function is executed after ExitAction.
    #Its argument is a reference to TFSMState object that used this function.
    self.Evaluator= None

    #FIXME:REMOVE
    ##[NEW]Parameter candidates which is a set of parameter vectors.
    #self.Candidates= []
    ##[NEW]Values of the parameter candidates.
    ##Its size is len(Candidates) if Candidates is not empty.
    ##Otherwise it stores the value of the state as a list of size 1.
    #self.ParamValues= [0.0]

    self.ParamLearner= TDefaultLearner()
    #[NEW]Last parameter log.
    #If Candidates is empty, it stores the time of execution and Index==0.
    self.ParamLog= TFSMParameterLog()
  #Add a new action.
  def NewAction(self):
    self.Actions.append(TFSMConditionedAction())
    return self.Actions[-1]  #Reference to the new action
  def Reset(self):
    self.ParamLog= TFSMParameterLog()
  def Param(self):
    return self.ParamLearner.Param()
  def Value(self):
    return self.ParamLearner.Value()

class TStateMachine:
  def __init__(self,start_state=''):
    #Set of states (dictionary of TFSMState).
    self.States= {}
    #Start state identifier.
    self.StartState= start_state
    #Debug mode
    self.Debug= False
    #[NEW]Function to get the current time
    self.GetTime= time.time
    #[NEW]Start time
    self.start_time= -1.0

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

  def Time(self):
    if self.start_time>=0.0:
      return self.GetTime()-self.start_time
    else:
      print 'Warning: not started'
      return -1.0

  def Reset(self):
    self.start_time= -1.0
    self.prev_state= ''
    self.curr_state= ''
    for id,st in self.States.items():
      st.Reset()

  def Run(self):
    self.prev_state= ''
    self.curr_state= self.StartState
    self.start_time= self.GetTime()
    count= 0
    self.IsUpdating= True
    evaluation= 0.0
    while self.curr_state!='':
      count+=1
      if self.Debug: DPrint('@',count,self.curr_state)
      st= self.States[self.curr_state]

      #Entering to a new state:
      if self.prev_state!=self.curr_state:
        #[NEW]Choose a parameter when entering to a new state
        st.ParamLearner.Select()
        st.ParamLog.Value= st.Value()
        st.ParamLog.Time= self.Time()
        if self.Debug: DPrint('@',count,self.curr_state,'Parameter:',st.Param())

        #FIXME:REMOVE>>>
        #if len(st.Candidates)>0:
          #if len(st.Candidates)!=len(st.ParamValues):
            #st.ParamValues= [0.0]*len(st.Candidates)
            #print 'Warning: [%s].ParamValues is not initialized.  Using %r.' % (self.curr_state, st.ParamValues)
          #probs= BoltzmannPolicy(self.BoltzmannTau,st.ParamValues)
          #st.ParamLog.Index= SelectFromPolicy(probs)
          #st.ParamLog.Time= self.Time()
          #if self.Debug: DPrint('@',count,self.curr_state,'Parameter:',st.ParamLog.Index,'=',st.Param())
        #else:
          #st.ParamLog.Index= 0
          #st.ParamLog.Time= self.Time()
        #FIXME:REMOVE<<<

        #[NEW]Update the states
        if self.IsUpdating and self.prev_state!='':
          pst= self.States[self.prev_state]
          prev_value= pst.Value()
          curr_value= st.Value()
          #FIXME: REPLACE THE MAGIC NUMBER!!
          gamma= math.exp(-0.01*(st.ParamLog.Time-pst.ParamLog.Time))
          td_err= evaluation + gamma*curr_value - prev_value
          print 'DEBUG: %s %f, %s %f, %f, %f' % (self.prev_state,prev_value,self.curr_state,curr_value,evaluation,td_err)
          self.UpdateStates(td_err)

        if st.EntryAction:
          if self.Debug: DPrint('@',count,self.curr_state,'EntryAction')
          st.EntryAction(st)

      a_id= 0
      a_id_satisfied= -1
      next_state= ''
      for ca in st.Actions:
        if ca.Condition(st):
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
          st.Actions[a_id_satisfied].Action(st)
      else:
        if st.ElseAction.Condition(st):
          if st.ElseAction.Action:
            if self.Debug: DPrint('@',count,self.curr_state,'ElseAction')
            st.ElseAction.Action(st)
          next_state= st.ElseAction.NextState

      if self.Debug: DPrint('@',count,self.curr_state,'Next state:',next_state)

      evaluation= 0.0

      if next_state==ORIGIN_STATE or  next_state==self.curr_state:
        self.prev_state= self.curr_state
        #self.curr_state= self.curr_state
      else:
        if st.ExitAction:
          if self.Debug: DPrint('@',count,self.curr_state,'ExitAction')
          st.ExitAction(st)
        if st.Evaluator:
          evaluation= st.Evaluator(st)
          if self.Debug: DPrint('@',count,self.curr_state,'Evaluation:',evaluation)
          if self.prev_state!='':
            print 'DEBUG: Evaluation:',evaluation,' / prev.value:',self.States[self.prev_state].ParamLearner.Value()
        if next_state=='':
          print 'Next state is not defined at %s. Hint: use ElseAction to specify the case where no conditions are satisfied.' % (self.curr_state)
        self.prev_state= self.curr_state
        if next_state==EXIT_STATE:
          self.curr_state= ''
        else:
          self.curr_state= next_state

    if self.Debug: DPrint('@ Finished')

  def UpdateStates(self,td_err):
    time= self.States[self.prev_state].ParamLog.Time
    for id,st in self.States.items():
      if id==self.curr_state:
        continue
      if st.ParamLog.Time>=0.0:
        discount= math.exp(-0.05*(time-st.ParamLog.Time))
        st.ParamLearner.Update(discount*td_err)


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
  sm.Debug= True

  sm.StartState= 'start'
  sm['start']= TFSMState()
  sm['start'].EntryAction= lambda st: Print('Hello state machine!')
  sm['start'].NewAction()
  sm['start'].Actions[-1].Condition= lambda st: Print("Want to move?") or AskYesNo()
  sm['start'].Actions[-1].NextState= 'count'
  sm['start'].ElseAction.Condition= lambda st: True
  sm['start'].ElseAction.Action= lambda st: Print('Keep to stay in start\n')
  sm['start'].ElseAction.NextState= 'start'
  sm['start'].ExitAction= lambda st: (Print('-->'), GetStartTime())

  sm['count']= TFSMState()
  #sm['count'].ParamLearner= TDiscLearner()
  #sm['count'].ParamLearner.Candidates= [1.0,1.2,1.4,1.6,1.8,2.0]
  sm['count'].ParamLearner= TQuadLearner()
  sm['count'].ParamLearner.Min= 1.0
  sm['count'].ParamLearner.Max= 2.0
  sm['count'].ParamLearner.C= [1.0,1.0,0.0]
  sm['count'].EntryAction= lambda st: Print('Counting...',st.Param())
  sm['count'].NewAction()
  sm['count'].Actions[-1].Condition= lambda st: (time.time()-start_time)>=st.Param()
  sm['count'].Actions[-1].Action= lambda st: Print('Hit!: '+str(time.time()-start_time))
  sm['count'].Actions[-1].NextState= 'stop'
  sm['count'].ElseAction.Condition= lambda st: True
  sm['count'].ElseAction.Action= lambda st: (Print(str(int(time.time())-start_time)), time.sleep(0.1))
  sm['count'].ElseAction.NextState= 'count'
  sm['count'].Evaluator= lambda st: 1.0 if abs(time.time()-start_time-1.6)<0.2 else 0.0

  sm['stop']= TFSMState()
  sm['stop'].EntryAction= lambda st: Print('Finishing state machine')
  sm['stop'].ElseAction.Condition= lambda st: True
  sm['stop'].ElseAction.NextState= EXIT_STATE

  for i in range(20):
    sm.Reset()
    sm.Run()
    #print sm['count'].ParamLearner.ParamValues
    print sm['count'].ParamLearner.C

