#!/usr/bin/python
#State machine with a parameter adjustment architecture
import math,random
#CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
import cma

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


#Learning to select a vector from discrete set
class TDiscParam:
  def __init__(self):
    #Parameter candidates which is a set of parameter vectors.
    self.Candidates= []
    self.Means= []
    self.SqMeans= []
    #Index of the parameter vector lastly selected.
    self.index= -1
    #Temparature parameter for the Boltzmann selection.
    self.BoltzmannTau= 0.1
    self.UCBNsd= 1.0
    self.Alpha= 0.2
    self.InitStdDev= 1.0
  def UCB(self):
    return [self.Means[d] + self.UCBNsd*math.sqrt(max(0.0,self.SqMeans[d]-self.Means[d]**2)) for d in range(len(self.Candidates))]
  def Init(self):
    pass
  #Returns the latest selected parameter.
  def Param(self):
    if len(self.Candidates)>0 and self.index>=0:
      return self.Candidates[self.index]
    return None
  def Select(self):
    if len(self.Candidates)>0:
      if len(self.Candidates)!=len(self.Means):
        self.Means= [0.0]*len(self.Candidates)
        self.SqMeans= [self.InitStdDev**2]*len(self.Candidates)
        #print 'Warning: Means is not initialized.  Using %r.' % (self.Means)
      ucb= self.UCB()
      probs= BoltzmannPolicy(self.BoltzmannTau,ucb)
      self.index= SelectFromPolicy(probs)
      print 'TDiscParam:DEBUG: Param:%r Index:%i UCB:%f' % (self.Candidates[self.index],self.index,ucb[self.index])
    else:
      self.index= -1
  def Update(self,score):
    if self.index>=0:
      self.Means[self.index]= self.Alpha*score + (1.0-self.Alpha)*self.Means[self.index]
      self.SqMeans[self.index]= self.Alpha*(score**2) + (1.0-self.Alpha)*self.SqMeans[self.index]
      print 'TDiscParam:DEBUG: Index:%i Score:%f New-Mean:%f' % (self.index,score,self.Means[self.index])

#Learning to a continuous value vector whose gradient is known
class TContParamGrad:
  def __init__(self):
    self.Mean= []
    self.Min= []
    self.Max= []
    #Function to compute a gradient from argv (parameter of Update)
    self.Gradient= None
    self.Alpha= 0.2
  def Init(self):
    pass
  #Returns the latest selected parameter.
  def Param(self):
    return self.Mean
  def Select(self):
    print 'TContParamGrad:DEBUG: Param:%r' % (self.Mean)
  def Update(self,argv):
    if not self.Gradient:
      print 'TContParamGrad:Error: No gradient function'
      return
    gradient= self.Gradient(argv)
    assert len(gradient)==len(self.Mean)
    self.Mean= [self.Mean[d] + self.Alpha*gradient[d] for d in range(len(gradient))]
    if len(self.Min)>0:
      self.Mean= [max(self.Mean[d],self.Min[d]) for d in range(len(gradient))]
    if len(self.Max)>0:
      self.Mean= [min(self.Mean[d],self.Max[d]) for d in range(len(gradient))]
    print 'TContParamGrad:DEBUG: Grad:%r New-Mean:%r' % (gradient,self.Mean)

#Learning to a continuous value vector whose gradient is unknown
class TContParamNoGrad:
  def __init__(self):
    self.Mean= []
    self.Std= 1.0
    self.Min= []
    self.Max= []
    self.CMAESOptions= {}
  def Init(self):
    if len(self.Min)>0 or len(self.Max)>0:
      self.CMAESOptions['bounds']= [self.Min,self.Max]
    self.es= cma.CMAEvolutionStrategy(self.Mean, self.Std, self.CMAESOptions)
    self.solutions= []
    self.scores= []
  #Returns the latest selected parameter.
  def Param(self):
    if len(self.solutions)>0:
      return self.solutions[-1]
    return None
  def Select(self):
    #For the previous selection was not evaluated...
    while len(self.scores)<len(self.solutions):
      self.solutions.pop()
    self.solutions.append(self.es.ask(1)[0])
    print 'TContParamNoGrad:DEBUG: Param:%r' % (self.solutions[-1])
  def Update(self,score):
    if len(self.scores)==len(self.solutions)-1:
      self.scores.append(-score)
    if len(self.solutions)==self.es.popsize:
      self.es.tell(self.solutions, self.scores)
      self.solutions= []
      self.scores= []
      self.es.disp()
    print 'TContParamNoGrad:DEBUG: Score:%f' % (score,)

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
    #Store some adjustable parameters of the state machine.
    #Each value should be an instance of TDiscParam or TContParam*
    #Usage: execute Select and Update as an action, access by Param.
    self.Params= {}
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

  param_kind= 3
  if param_kind==1:
    sm.Params['wait_time']= TDiscParam()
    wait_time= sm.Params['wait_time']
    wait_time.Candidates= [[1.0],[1.2],[1.4],[1.6],[1.8],[2.0]]
  elif param_kind==2:
    sm.Params['wait_time']= TContParamGrad()
    wait_time= sm.Params['wait_time']
    wait_time.Mean= [1.2]
    wait_time.Min= [1.0]
    wait_time.Max= [2.0]
    #argv[0]: displacement from the target time
    wait_time.Gradient= lambda argv: [argv[0]]
  elif param_kind==3:
    sm.Params['wait_time']= TContParamNoGrad()
    wait_time= sm.Params['wait_time']
    wait_time.Mean= [1.2]
    wait_time.Std= 0.25
    wait_time.Min= [1.0]
    wait_time.Max= [2.0]
  wait_time.Init()

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
  sm['count'].EntryAction= lambda: ( Print('Counting...'), wait_time.Select() )
  sm['count'].NewAction()
  sm['count'].Actions[-1].Condition= lambda: (time.time()-start_time)>=wait_time.Param()[0]
  sm['count'].Actions[-1].Action= lambda: Print('Stop @'+str(time.time()-start_time))
  sm['count'].Actions[-1].NextState= 'stop'
  sm['count'].ElseAction.Condition= lambda: True
  sm['count'].ElseAction.Action= lambda: (Print(str(time.time()-start_time)), time.sleep(0.2))
  sm['count'].ElseAction.NextState= 'count'

  sm['stop']= TFSMState()
  if isinstance(wait_time,TDiscParam) or isinstance(wait_time,TContParamNoGrad):
    sm['stop'].EntryAction= lambda: ( Print('Finishing state machine'), wait_time.Update(1.0-abs(time.time()-start_time-1.6)**2 if abs(time.time()-start_time-1.6)<0.2 else 0.0) )
  elif isinstance(wait_time,TContParamGrad):
    sm['stop'].EntryAction= lambda: ( Print('Finishing state machine'), wait_time.Update([1.6-(time.time()-start_time)]) )
  sm['stop'].ElseAction.Condition= lambda: True
  sm['stop'].ElseAction.NextState= EXIT_STATE

  #sm.Run()

  for i in range(40):
    #sm.Reset()
    sm.Run()
    if isinstance(wait_time,TDiscParam):
      print wait_time.Means
      print wait_time.UCB()
    elif isinstance(wait_time,TContParamGrad):
      print wait_time.Mean
