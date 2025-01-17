#!/usr/bin/python3
import copy

def Median(array):
  if len(array)==0:  return None
  a_sorted= copy.deepcopy(array)
  a_sorted.sort()
  return a_sorted[len(a_sorted)//2]

array= [2,10,5,1,55,7,48,103,22,6,3,99,45,99]

print("median:",Median(array))
print("  in:",array)

array= [[2,2],[10,8],[5,3],[1,1],[55,0],[7,55],[48,13],[103,0],[22,2],[6,2],[3,21],[99,32],[45,2],[99,9]]

print("median:",Median([a[0] for a in array]),Median([a[1] for a in array]))
print("  in:",array)
