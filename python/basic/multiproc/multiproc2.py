#!/usr/bin/python
#\file    multiproc2.py
#\brief   multiprocessing test 2
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.02, 2016
import multiprocessing as mp

class TTest:
  def __init__(self,q,n):
    self.q= q
    self.n= n

  def Func1(self,p):
    assert(p>0)
    s= 0
    for x in xrange(0,self.q,p):
      s+= x
    return p,s

  def Func1S(self):
    for pp in xrange(self.n):
      p,s= self.Func1(pp+1)
      print '{p}: {s}'.format(p=p,s=s)

  #ERROR: Doesn't work
  def Func1M(self):
    pool= mp.Pool(self.n)
    callback= pool.map(self.Func1, (pp+1 for pp in xrange(self.n)))
    for p,s in callback:
      print '{p}: {s}'.format(p=p,s=s)

  def Func2(self,p):
    p,s= self.Func1(p)
    self.queue.put((p,s))

  def Func2M(self):
    self.queue= mp.Queue()
    ps= [mp.Process(target=self.Func2, args=(pp+1,)) for pp in xrange(self.n)]
    for p in ps:  p.start()
    for i in xrange(self.n):
      p,s= self.queue.get()
      print '{p}: {s}'.format(p=p,s=s)
    for p in ps:  p.join()

if __name__=='__main__':
  t= TTest(q=10**8,n=8)
  #t.Func1S()
  #t.Func1M()  #ERROR: Doesn't work
  t.Func2M()

