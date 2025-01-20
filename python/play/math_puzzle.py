#!/usr/bin/python3
def Check(x, final=66):
  if len(x)!=9:  return None
  if set(x)!=set([1,2,3,4,5,6,7,8,9]):  return None
  value= x[0]+13*x[1]/x[2]+x[3]+12*x[4]-x[5]-11+x[6]*x[7]/x[8]-10
  return value==final, value

def PrintCheck(x, final=66):
  print(x,'-->',Check(x, final))

def ForEachCombination(collection, operation, pre_seq=[]):
  if len(collection)==0:
    return
  if len(collection)==1:
    operation(pre_seq+list(collection))
    return
  for item in collection:
    ForEachCombination(set(collection)-set([item]), operation, pre_seq+[item])

if __name__=='__main__':
  print('Answer-1 (by akihikoy):')
  PrintCheck([5,9,3,6,2,1,7,8,4])
  print('Answer-2 (by The Guardian):')
  PrintCheck([3,2,1,5,4,7,9,8,6])

  answers= []
  ForEachCombination([1,2,3,4,5,6,7,8,9], lambda x:answers.append(x) if Check(x)[0] else None)

  factorial= lambda n:1 if n==1 else n*factorial(n-1)
  print('Num of answers: %i / %i (%f %%)' % (len(answers), factorial(9), len(answers)*100.0/factorial(9)))
  for a in answers:
    PrintCheck(a)

  #for final in range(0,100):
  #for final in range(-20,10):
    #answers= []
    #ForEachCombination([1,2,3,4,5,6,7,8,9], lambda x:answers.append(x) if Check(x,final)[0] else None)
    #print final,len(answers)

  #NOTE: final=-5 is the most difficult case. There are only 4 possible answers.

