#!/usr/bin/python
#\file    loop_combination2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.25, 2017
import itertools

#For a given list of lists, generate all combinations.
#e.g. If d=[[v1,v2],[v3,v4]], return is [[v1,v3],[v1,v4],[v2,v3],[v2,v4]]
def AllCombinations(d):
  return itertools.product(*d)

if __name__=='__main__':
  print list(AllCombinations([['v1','v2'],['v3','v4']]))
  print list(AllCombinations([['v1','v2'],['v5'],['v3','v4']]))
  print list(AllCombinations([['v1','v2'],['v3','v4'],['v5','v6','v7']]))
