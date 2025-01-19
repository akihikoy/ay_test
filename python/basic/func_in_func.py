#!/usr/bin/python3
t= 10
def Run():
  #global t
  print('t=',t)
  #t= None

def Run2():
  t2= 100
  def RunInRun2():
    #global t2  #ERROR
    print('t2=',t2)
    #t2= None
  RunInRun2()

Run()
Run2()
