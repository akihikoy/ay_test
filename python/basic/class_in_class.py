#!/usr/bin/python3

class TTest:
  class TItem:
    def __init__(self,name):
      self.Running= True
      self.Name= name
      print('Init',self.Name,self.Running)
    def __del__(self):
      self.Running= False
      print('Delete',self.Name,self.Running)

  def __init__(self):
    self.item_list= {}

  def Add(self,name):
    self.item_list[name]= self.TItem(name)

  def Del(self,name):
    del self.item_list[name]

  def __del__(self):
    for k in list(self.item_list.keys()):
      print('Deleting %r...' % k)
      del self.item_list[k]
    print('self.item_list=',self.item_list)


test= TTest()
print('------------')
test.Add('a')
test.Add('b')
test.Add('c')
print('------------')
test.Del('b')
print('test.item_list=',test.item_list)
print('------------')
del test
print('============')
