#!/usr/bin/python
#\file    dict_aggregate.py
#\brief   We consider aggregating a list of dictionaries that have the same structure.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.11, 2020

#For each non-dict element in dict_data including sub-dict, apply v'=op(v).
#e.g. SubMapDict({'a':{'b':1},'c':2}, lambda v:v+1)  returns {'a':{'b':2},'c':3}.
def SubMapDict(dict_data, op):
  empty_dict= {}
  def sub(d,a):
    for k,v in d.iteritems():
      if isinstance(v,dict):
        a[k]= {}
        sub(v,a[k])
      else:
        a[k]= op(v)
  sub(dict_data, empty_dict)
  return empty_dict

#Aggregate a list of dictionaries.
#e.g. AggregateListOfDicts([{'a':{'b':1},'c':2},{'a':{'b':3},'c':4}])  returns {'a':{'b':[1,3]},'c':[2,4]}
def AggregateListOfDicts(data):
  aggregated= SubMapDict(data[0],lambda v:[])
  def sub(d,a):
    for k,v in d.iteritems():
      if isinstance(v,dict):
        sub(v,a[k])
      else:
        a[k].append(v)
  for dict_data in data:
    sub(dict_data, aggregated)
  return aggregated

if __name__=='__main__':
  print SubMapDict({'a':{'b':1},'c':2}, lambda v:v+1)
  print AggregateListOfDicts([{'a':{'b':1},'c':2},{'a':{'b':3},'c':4}])

  data= [
    {'a':{'aa':1,'ab':2},'b':{'ba':3,'bb':4},'c':5},
    {'a':{'aa':2,'ab':3},'b':{'ba':5,'bb':8},'c':15},
    {'a':{'aa':3,'ab':4},'b':{'ba':7,'bb':12},'c':25},
    ]
  print AggregateListOfDicts(data)
  print SubMapDict(AggregateListOfDicts(data), lambda v:float(sum(v))/len(v))
