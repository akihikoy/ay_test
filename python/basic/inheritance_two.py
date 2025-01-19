#!/usr/bin/python3
#\file    inheritance_two.py
#\brief   inheritance from two classes
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.06, 2016

'''Define a space.
min and max should be list or None. '''
class TSpaceDef(object):
  def __init__(self,dim=0,min=None,max=None):
    self.D= dim
    self.Min= min
    self.Max= max

  @property
  def Bounds(self):
    return [self.Min if self.Min is not None else [], self.Max if self.Max is not None else []]

'''Define a discrete action (selection) space. '''
class TSelectDef(object):
  def __init__(self,num=None):
    self.N= num


'''Define a composite space.
    'state': state space.
    'action': (continuous) action space.
    'select': discrete action space (selection).
'''
class TCompSpaceDef(TSpaceDef,TSelectDef):
  def __init__(self,type=None,dim=0,min=None,max=None,num=0):
    self.Type= type
    if self.Type in ('state','action'):
      TSpaceDef.__init__(self,dim=dim,min=min,max=max)
      TSelectDef.__init__(self)
    elif self.Type==('select'):
      TSpaceDef.__init__(self,dim=1,min=0 if num>0 else None,max=num-1 if num>0 else None)
      TSelectDef.__init__(self,num=num)

  #Check if self.Type is type.
  def Is(self,type):
    return self.Type==type

if __name__=='__main__':
  cspdef= TCompSpaceDef('state',2,[-1,-2],[1,2])
  print('cspdef:',cspdef.__dict__,cspdef.D,cspdef.Is('state'),cspdef.Is('action'),cspdef.Is('select'),cspdef.Bounds)
  cspdef= TCompSpaceDef('action',1,[-1.5],[1.5])
  print('cspdef:',cspdef.__dict__,cspdef.D,cspdef.Is('state'),cspdef.Is('action'),cspdef.Is('select'),cspdef.Bounds)
  cspdef= TCompSpaceDef('select',num=5)
  print('cspdef:',cspdef.__dict__,cspdef.D,cspdef.Is('state'),cspdef.Is('action'),cspdef.Is('select'),cspdef.Bounds)

