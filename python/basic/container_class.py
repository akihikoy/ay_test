#!/usr/bin/python

#Print a dictionary with a nice format
def PrintDict(d,indent=0):
  for k,v in d.items():
    if type(v)==dict:
      print '  '*indent,'[',k,']=...'
      PrintDict(v,indent+1)
    else:
      print '  '*indent,'[',k,']=',v

#Container class that can hold any variables
#ref. http://blog.beanz-net.jp/happy_programming/2008/11/python-5.html
class TContainer:
  def __str__(self):
    return str(self.__dict__)
  def __repr__(self):
    return str(self.__dict__)
  def __iter__(self):
    return self.__dict__.itervalues()
    #return self.__dict__.iteritems()
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

class TTest:
  def __init__(self,x):
    self.x= x
    print 'Test class',self.x
  def __del__(self):
    print 'Deleted',self.x
  def Print(self):
    print 'This is',self.x

def main():
  cont= TContainer()
  cont.var_1= TTest(1.23)
  cont.var_2= TTest('hoge')
  cont.var_3= TTest('aa aa')
  #print cont
  if 'var_2' in cont:  print 'var_2 is contained'
  PrintDict(cont)
  for i in cont:
    i.Print()
  del i
  #for v in cont:
    #del v
  #for k,v in cont.items():
    #v.Print()
    #del cont[k]
  #del k,v
  for k in cont.keys():
    print 'del',k,cont[k].x
    del cont[k]
  #del cont.var_1
  #del cont.var_2
  #del cont.var_3
  PrintDict(cont)
  print '----'

if __name__=='__main__':
  main()
  print '===='
