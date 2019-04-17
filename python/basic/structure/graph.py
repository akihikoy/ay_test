#!/usr/bin/python

#class TVertex:
  #def __init__(self,name):
    #self.name= name
    #self.connected= []

#class TGraph:
  #def __init__(self):
    #self.vertices= {}  #Map: name(str) --> vertex (TVertex)
  #def

#if __name__=='__main__':
  #graph= TGraph()
  #graph['container_types']= TVertex('container_types')
  #graph['container_types']['b1']= TVertex('container_types_b1')


#Print a dictionary with a nice format
def PrintDict(d,indent=0):
  for k,v in d.items():
    if type(v)==dict or isinstance(v,TContainer):
      print '  '*indent,'[',k,']=...'
      PrintDict(v,indent+1)
    else:
      print '  '*indent,'[',k,']=',v

#Print a graph with a nice format
def PrintGraph(d,indent=0,displayed=set()):
  for k,v in d.items():
    if id(v) in displayed:
      print '  '*indent,'[',k,']=',('<<%i>>' % id(v))
    else:
      displayed= displayed.union({id(v)})
      if type(v)==dict or isinstance(v,TContainer):
        print '  '*indent,'[',k,']=...'
        PrintGraph(v,indent+1,displayed)
      else:
        print '  '*indent,'[',k,']=',v

#Container class that can hold any variables
#ref. http://blog.beanz-net.jp/happy_programming/2008/11/python-5.html
class TContainer:
  def __init__(self,debug=False):
    self.debug= debug
    if self.debug:  print 'Created TContainer object',hex(id(self))
  def __del__(self):
    if self.debug:  print 'Deleting TContainer object',hex(id(self))
  def __str__(self):
    return str(self.__dict__)
  def __repr__(self):
    return str(self.__dict__)
  def __iter__(self):
    return self.__dict__.itervalues()
  def items(self):
    return self.__dict__.items()
  def iteritems(self):
    return self.__dict__.iteritems()
  def keys(self):
    return self.__dict__.keys()
  def values(self):
    return self.__dict__.values()
  def __getitem__(self,key):
    return self.__dict__[key]
  def __setitem__(self,key,value):
    self.__dict__[key]= value
  def __delitem__(self,key):
    del self.__dict__[key]
  def __contains__(self,key):
    return key in self.__dict__


if __name__=='__main__':
  graph= TContainer()
  graph['c_type']= TContainer()
  graph['c_type']['b1']= TContainer()
  graph['c_type']['b1']['color']= [0,0,255]
  graph['c_type']['b1']['size']= [1.2,2.0,1.5]
  graph['c_type']['b2']= TContainer()
  graph['c_type']['b2']['color']= [255,0,0]
  graph['c_type']['b2']['size']= [1.5,1.5,0.5]
  graph['scene']= TContainer()
  graph['scene']['objects']= TContainer()
  graph['scene']['objects']['b1']= TContainer()
  graph['scene']['objects']['b1']['pose']= [0.0,1.0,0.0]
  graph['scene']['objects']['b1']['c_type']= graph['c_type']['b1']
  graph['c_type']['b1']['instance']= [graph['scene']['objects']['b1']]
  #graph['scene']['objects']['b1']['c_type']= ['c_type','b1']
  #graph['c_type']['b1']['instance']= ['scene','objects','b1']

  PrintGraph(graph)
  #for i in graph:
    #i.Print()
  #del i
  print '----'
