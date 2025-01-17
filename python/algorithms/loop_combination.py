#!/usr/bin/python3

#Apply operation for every combination in collection.
def ForEachCombination(collection, operation, pre_seq=[]):
  if len(collection)==0:
    return
  if len(collection)==1:
    operation(pre_seq+list(collection))
    return
  for item in collection:
    ForEachCombination(set(collection)-set([item]), operation, pre_seq+[item])

if __name__=='__main__':
  def Operation(x):
    print(x)
  ForEachCombination([1,2,3], Operation)

  cmb= []
  ForEachCombination([1,2,3,4], lambda x:cmb.append(x) if 2*x[1]+x[2]==7 else None)
  print(cmb)
