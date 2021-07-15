#!/usr/bin/python
#\file    floyd_apsp.py
#\brief   Floyd's all-pairs-shortest-path (APSP) algorithm.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.15, 2021

import numpy as np


#Floyd's all-pairs-shortest-path (APSP) algorithm.
#  N: Number of nodes.
#  edges: List of edge and cost tuples: ((node1,node2),cost).
def FloydAPSP(N, edges, inf_cost=1.0e9):
  #C: cost matrix.
  C= np.ones((N,N))*inf_cost
  for (n1,n2),c in edges:
    c= min(C[n1,n2],C[n2,n1],c)
    C[n1,n2]= c
    C[n2,n1]= c
  #D: matrix containing lowest cost.
  #P: matrix containing a via point on the shortest path.
  D= np.zeros((N,N))
  P= np.zeros((N,N),np.int32)
  for i in range(N):
    for j in range(N):
      D[i,j]= C[i,j]
      P[i,j]= -1 if C[i,j]!=inf_cost else -2
    D[i,i]= 0.0
    P[i,i]= -1
  for k in range(N):
    for i in range(N):
      for j in range(N):
        if D[i,k] + D[k,j] < D[i,j]:
          D[i,j]= D[i,k] + D[k,j]
          P[i,j]= k
  return D,P

#From the output path matrix P of FloydAPSP, find a shortest path between two nodes.
def ShortestPath(n1, n2, P):
  k= P[n1,n2]
  if k==-1:  return [n1,n2]
  if k==-2:  return None
  path1,path2= ShortestPath(n1,k,P),ShortestPath(k,n2,P)
  if path1!=None and path2!=None:  return path1+path2[1:]
  return None

if __name__=='__main__':
  '''
  Definition of a graph:
              |--2--|       |-----3------|
              |     |       |            |
      n1--1--n3--1--n4--2--n5--1--n6--1--n7--1--n9
             |                    |
      n2--2--|             n8--2--|
  '''
  N= 9  #Number of nodes
  edges= [  #List of edge and cost tuples: ((node1,node2),cost)
    ((0,2),1.0),
    ((1,2),2.0),
    ((2,3),1.0),
    ((2,3),2.0),
    #((3,4),2.0),  #Comment out this edge to make the graph two isolated parts.
    ((4,5),1.0),
    ((5,6),1.0),
    ((4,6),3.0),
    ((5,7),2.0),
    ((6,8),1.0),
    ]

  D,P= FloydAPSP(N, edges)
  print 'Lowest cost matrix:'
  print D
  print 'Via points on the shortest path matrix:'
  print P

  while True:
    print 'Type two node indexes (starting from 1) separating with space (0 0 to quit):'
    n1,n2= map(lambda s:int(s)-1,raw_input(' > ').split(' '))
    if n1<0 or n2<0 or n1>=N or n2>=N:  break
    path= ShortestPath(n1,n2,P)
    print '  Shortest path:', map(lambda n:n+1,path) if path is not None else None, 'Cost:', D[n1,n2]

