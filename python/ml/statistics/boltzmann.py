#!/usr/bin/python3
#Boltzmann action selection
import math,random

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


if __name__=='__main__':
  values= [1.2,-3.2,0.5,2.12,7.8,-3.0]
  probs= BoltzmannPolicy(5.0,values)
  hist=[0]*len(values)
  for i in range(10000):
    a= SelectFromPolicy(probs)
    hist[a]+=1
  for i in range(len(values)):
    print(i,values[i],probs[i],hist[i]/10000.0)
