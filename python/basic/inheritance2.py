#!/usr/bin/python3
import importlib

#class TApproachToX:
  #def __init__(self):
    ##Following variables should be assigned:
    #self.core_tool= None  #Ref to TCoreTool
    #self.get_x_trg= None  #Function to get the target
    #self.l_x_ext= None  #Control frame
    #self.arm= None  #Arm id / hand id

    ##Parameters:
    #self.time_step= 0.01  #Control step in sec
    #self.max_speed= [0.05, 0.1]  #Max speed; 5cm/s, 0.1rad/s

    ##Temporary variables:
    #self.x_curr= None

  #def Init(self):
    #self.reached= False

  #def Check(self):
    #return not self.reached

  #def Step(self):
    #t= self.core_tool
    #dt= self.time_step
    #print 'process...'

  #def Exit(self):
    #pass


#class TMoveDiffX(TApproachToX):
  #def __init__(self):
    #TApproachToX.__init__(self)

    ##Following variables should be assigned:
    #self.core_tool= None  #Ref to TCoreTool
    #self.change_x= None  #Function to change current x to a target
    #self.l_x_ext= None  #Control frame
    #self.arm= None  #Arm id / hand id

    ##Parameters inherited from TApproachToX:
    ##self.time_step
    ##self.max_speed

  #def Init(self):
    #TApproachToX.Init(self)

    #t= self.core_tool
    #self.x_curr= [1,2,3]
    #self.x_trg= self.change_x(self.x_curr)
    ##Setup for TApproachToX; function to get the target
    #self.get_x_trg= lambda:self.x_trg


#test= TMoveDiffX()
#test.change_x= lambda x: x

#holder= test.Init

#holder()


#import inheritance
if __name__=='__main__':
  #test= __import__('inheritance',globals(),locals(),'inheritance',-1).TBase()
  #test= __import__('inheritance',globals(),locals(),'inheritance',-1).TTest()
  test= importlib.import_module('inheritance').TTest()
  test.a= 999

  print('=====')
  test.Print()
  print('=====')
  del test

