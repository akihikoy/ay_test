#!/usr/bin/python3
#\file    smart_reload.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.15, 2017
import datetime
import os
import importlib

import sys

def AskYesNo():
  while 1:
    sys.stdout.write('  (y|n) > ')
    sys.stdout.flush()
    ans= sys.stdin.readline().strip()
    if ans=='y' or ans=='Y':  return True
    elif ans=='n' or ans=='N':  return False

def Reload(mod):
  importlib.reload(mod)

def Import(mod_id):
  #return __import__(mod_id)
  return importlib.import_module(mod_id)

#def SmartReload(mod, __load_time=[None]):

#Modify file name from xxx.pyc to xxx.py.
#This does nothing to xxx.py or other extensions.
def PycToPy(file_name):
  path,ext= os.path.splitext(file_name)
  if ext=='.pyc':  return path+'.py'
  return file_name

'''Import/reload a module named mod_id (string).
If mod_id was already loaded, the time stamp of the module (note: xxx.pyc is considered as xxx.py)
is compared with the time when the mod_id was loaded previously.
Only when the time stamp is newer than the loaded time, this function reloads the module.
return: module '''
def SmartImportReload(mod_id, __loaded={}):
  if mod_id in __loaded:
    loaded_time,mod= __loaded[mod_id]
    file_time= datetime.datetime.fromtimestamp(os.path.getmtime(PycToPy(mod.__file__)))
    #Reload if the file is modified:
    if file_time>loaded_time:
      importlib.reload(mod)
      __loaded[mod_id]= (datetime.datetime.now(), mod)  #Loaded time, module
    return mod
  else:
    mod= importlib.import_module(mod_id)
    __loaded[mod_id]= (datetime.datetime.now(), mod)  #Loaded time, module
    return mod


'''
print '======Doing: TEST A======'

#import sample_mod
sample_mod= Import('sample_mod')
from sample_mod import *

print '1: Done: import sample_mod'
print '2: id(sample_mod.F),id(F)=',id(sample_mod.F),id(F)
sample_mod.F()
F()

print ''

Reload(sample_mod)
#sample_mod= Import('sample_mod')

print '3: Done: Reload(sample_mod)'
print '4: id(sample_mod.F),id(F)=',id(sample_mod.F),id(F)
sample_mod.F()
F()

print ''

from sample_mod import *

print '5: Done: "from sample_mod import *"'
print '6: id(sample_mod.F),id(F)=',id(sample_mod.F),id(F)
sample_mod.F()
F()

print '======Done: TEST A======'
#'''



#'''
print('======Doing: TEST B======')

sample_mod= SmartImportReload('sample_mod')
from sample_mod import *

print(0,': Done: SmartImportReload')
print(0,': id(sample_mod.F),id(F)=',id(sample_mod.F),id(F))
sample_mod.F()
F()

for i in range(1,1000):
  print('Quit?')
  if AskYesNo():  break

  sample_mod= SmartImportReload('sample_mod')
  print(i,': Done: SmartImportReload')
  print(i,': id(sample_mod.F),id(F)=',id(sample_mod.F),id(F))
  sample_mod.F()
  F()

print('======Done: TEST B======')
#'''

