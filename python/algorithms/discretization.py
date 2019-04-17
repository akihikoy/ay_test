#!/usr/bin/python
import numpy as np

class TDiscretizer:

  def __init__(self,vmin,vmax,ndiv):
    self.vmin= vmin
    self.vmax= vmax
    self.ndiv= ndiv

    self.step= [0.0]*len(self.ndiv)
    self.index_base= [1]*len(self.ndiv)
    idb= 1
    for d in range(len(self.ndiv)):
      if self.ndiv[d]>0:
        self.step[d]= (self.vmax[d]-self.vmin[d]) / float(self.ndiv[d])
      self.index_base[d]= idb
      idb= idb*(self.ndiv[d]+1)
    self.size= idb

  def Size(self):
    return self.size

  def VecSet(self):
    vec_set= []
    iv= [0]*(len(self.ndiv)+1)
    for i in range(self.size):
      vec= np.array(self.vmin) + np.array(self.step)*np.array(iv[:-1])
      vec_set.append(list(vec))
      iv[0]+=1
      for d in range(len(self.ndiv)-1):
        if iv[d] >= self.ndiv[d]+1:
          iv[d]= 0
          iv[d+1]+=1
        else:
          break
    return vec_set

  #Get an index of a given vector
  def VecToIndex(self,vec):
    iv= [0]*len(self.ndiv)
    for d in range(len(self.ndiv)):
      if self.ndiv[d]==0:  iv[d]= 0
      elif vec[d]<=self.vmin[d]:  iv[d]= 0
      elif vec[d]>=self.vmax[d]:  iv[d]= self.ndiv[d]
      else:  iv[d]= int( (vec[d]-self.vmin[d]+0.5*self.step[d]) / self.step[d] )
    return np.dot(iv, self.index_base)

  #Get a value of a given index
  def IndexToVec(self,idx):
    iv= [0]*len(self.ndiv)
    for d in range(len(self.ndiv)-1):
      iv[d]= idx % (self.ndiv[d]+1)
      idx= (idx-iv[d]) / (self.ndiv[d]+1)
    iv[-1]= idx
    vec= np.array(self.vmin) + np.array(self.step)*np.array(iv)
    return vec


if __name__=='__main__':
  import sys

  dim= 3
  fp= file('data/disc.dat','w')

  print 'dim=',dim

  if dim==1:
    vmin=[-1.0]
    vmax=[1.0]
    ndiv=[10]
    disc= TDiscretizer(vmin,vmax,ndiv)
    print 'disc.Size()=',disc.Size()
    print 'disc.VecSet()=',disc.VecSet()
    print 'Verified: ',map(disc.VecToIndex, disc.VecSet())==range(disc.Size())

    for x in map(lambda i:2.0*(i/100.0-1.0),range(201)):
      vec= [x]
      idx= disc.VecToIndex(vec)
      vec2= disc.IndexToVec(idx)
      fp.write('%f %i %f\n' % (vec[0], idx, vec2[0]))

      #Verification:
      idx2= disc.VecToIndex(vec2)
      if idx!=idx2:
        print 'ERROR:'
        print 'vec=',vec
        print 'idx=',idx
        print 'vec2=',vec2
        print 'idx2=',idx2
        sys.exit(1)

    #Plot: cat data/disc.dat | qplot -x - - u 3:2

  if dim==2:
    vmin=[-1.5,-1.0]
    vmax=[1.5,1.0]
    ndiv=[4,2]
    disc= TDiscretizer(vmin,vmax,ndiv)
    print 'disc.Size()=',disc.Size()
    print 'disc.VecSet()=',disc.VecSet()
    print 'Verified: ',map(disc.VecToIndex, disc.VecSet())==range(disc.Size())

    for x in map(lambda i:2.0*(i/100.0-1.0),range(201)):
      for y in map(lambda i:2.0*(i/100.0-1.0),range(201)):
        vec= [x,y]
        idx= disc.VecToIndex(vec)
        vec2= disc.IndexToVec(idx)
        fp.write('%f %f %i %f %f\n' % (vec[0],vec[1], idx, vec2[0],vec2[1]))

        #Verification:
        idx2= disc.VecToIndex(vec2)
        if idx!=idx2:
          print 'ERROR:'
          print 'vec=',vec
          print 'idx=',idx
          print 'vec2=',vec2
          print 'idx2=',idx2
          sys.exit(1)

    #Plot: cat data/disc.dat | qplot -x -3d - - u 4:5:3

  if dim==3:
    vmin=[-1.5,-1.0,-0.5]
    vmax=[1.5,1.0,0.5]
    ndiv=[4,2,3]
    disc= TDiscretizer(vmin,vmax,ndiv)
    print 'disc.Size()=',disc.Size()
    print 'disc.VecSet()=',disc.VecSet()
    print 'Verified: ',map(disc.VecToIndex, disc.VecSet())==range(disc.Size())

    for x in map(lambda i:2.0*(i/15.0-1.0),range(31)):
      for y in map(lambda i:2.0*(i/15.0-1.0),range(31)):
        for z in map(lambda i:2.0*(i/15.0-1.0),range(31)):
          vec= [x,y,z]
          idx= disc.VecToIndex(vec)
          vec2= disc.IndexToVec(idx)
          fp.write('%f %f %f %i %f %f %f\n' % (vec[0],vec[1],vec[2], idx, vec2[0],vec2[1],vec2[2]))

          #Verification:
          idx2= disc.VecToIndex(vec2)
          if idx!=idx2:
            print 'ERROR:'
            print 'vec=',vec
            print 'idx=',idx
            print 'vec2=',vec2
            print 'idx2=',idx2
            sys.exit(1)

    #Plot: cat data/disc.dat | qplot -x -3d - - u 4:5:3

