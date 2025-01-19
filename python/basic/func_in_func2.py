#!/usr/bin/python3
#\file    func_in_func2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.31, 2018

class TTest(object):
  def __init__(self):
    self.x= 101
  def Run(self):
    def RunInRun1():
      print('x is',self.x)
    def RunInRun2(self):
      print('x is',self.x)
    RunInRun1()
    RunInRun2(self)

if __name__=='__main__':
  t= TTest()
  t.Run()
