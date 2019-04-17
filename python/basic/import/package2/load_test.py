#!/usr/bin/python
#\file    load_test.py
#\brief   Load module
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.28, 2015

class TCore:
  #Load external motion script written in python,
  #which is imported as a module to this script, so we can share the memory
  def LoadMod(self,fileid):
    print '#__package__:',__package__
    print '#__file__:',__file__
    modname= fileid
    try:
      sub= __import__(modname,globals(),locals(),modname,-1)
      #if modname in self.loaded_motions:
        #reload(sub)
      #else:
        #self.loaded_motions.append(modname)
    except ImportError:
      print 'Cannot import motion file: ',modname
      sub= None
    return sub

  #Execute external motion script written in python,
  #which is imported as a module to this script, so we can share the memory
  def ExecuteMod(self,fileid, *args):
    sub= self.LoadMod(fileid)
    if sub:
      return sub.Run(self, *args)

if __name__=='__main__':
  t= TCore()
  #t.ExecuteMod('lib', 3, 'hoge')  #Try to load lib/__init__.py
  #t.ExecuteMod('lib.py', 3, 'hoge')  #Cannot import motion file:  lib.py
  t.ExecuteMod('lib.lib1', 3, 'hoge')
  t.ExecuteMod('lib.sub.lib2', 10, 'haha')
