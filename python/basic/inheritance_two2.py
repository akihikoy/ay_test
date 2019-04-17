#!/usr/bin/python
#\file    inheritance_two2.py
#\brief   import test of inheritance_two
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.01, 2016
import inheritance_two
reload(inheritance_two)
from inheritance_two import *

SP= TCompSpaceDef

cspdef= SP('state',2,[-1,-2],[1,2])
print 'cspdef:',cspdef.__dict__,cspdef.D,cspdef.Is('state'),cspdef.Is('action'),cspdef.Is('select'),cspdef.Bounds

import inheritance_two
reload(inheritance_two)
from inheritance_two import *

'''NOTE: If the following statement `SP= TCompSpaceDef` is commented out, there will be an error:
-----
Traceback (most recent call last):
  File "./inheritance_two2.py", line 33, in <module>
    cspdef= SP('action',1,[-1.5],[1.5])
  File "/home/akihiko/prg/testl/python/inheritance_two.py", line 35, in __init__
    TSpaceDef.__init__(self,dim=dim,min=min,max=max)
TypeError: unbound method __init__() must be called with TSpaceDef instance as first argument (got TCompSpaceDef instance instead)
-----
This is because SP (defined in line 11 with an old TCompSpaceDef) is different from the current TCompSpaceDef (which is updated in line 18).
'''
#SP= TCompSpaceDef

cspdef= SP('action',1,[-1.5],[1.5])
print 'cspdef:',cspdef.__dict__,cspdef.D,cspdef.Is('state'),cspdef.Is('action'),cspdef.Is('select'),cspdef.Bounds
cspdef= SP('select',num=5)
print 'cspdef:',cspdef.__dict__,cspdef.D,cspdef.Is('state'),cspdef.Is('action'),cspdef.Is('select'),cspdef.Bounds
