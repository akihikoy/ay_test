#!/usr/bin/python3
#\file    graph_search1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.13, 2016

from hash1 import TStrInt

'''
Example:
  node= TGNode('n1', ('n2','n3'))
'''
class TGNode(object):
  def __init__(self,parent=None,nexts=()):
    self.Parent= parent
    self.Next= []
    for key in nexts:
      self.Next.append(key)

class TGraph(object):
  def __init__(self):
    self.Graph= None       #{key:node,...}, key is a str, node is a TGNode

  def PrintStructure(self):
    for key,node in self.Graph.items():
      print(key,'-->',node.Next)

class TTree(object):
  def __init__(self):
    self.Start= None      #key of a start node
    self.Tree= {}         #{key:node,...}, key is TStrInt(key_graph,num_visits), node is a TGNode
    #Note: key_graph is a key of TGraph.Graph (str), num_visits is number of visits.
    self.Terminal= []     #Terminal nodes (a list of keys of self.Tree)
    self.BwdOrder= []     #Order of backward computation (a list of keys of self.Tree)

  def PrintStructure(self):
    for key,node in self.Tree.items():
      print(node.Parent,'-->',key,'-->',node.Next)
    print('Start:',self.Start)
    print('Terminal:',self.Terminal)
    print('BwdOrder:',self.BwdOrder)

'''Obtain a TTree from graph with a specific start.
  Loops are unrolled.
  This is a breadth-first search algorithm.'''
def GraphToTree(graph, start, max_visits=2):
  #ng_*: key of node on Graph
  #nt_*: key of node on Tree
  ng_start= start
  tree= TTree()
  tree.Start= TStrInt(ng_start,0)
  num_visits= {key:0 for key in graph.Graph.keys()}
  queue= [(None,ng_start)]  #Stack of (nt_parent,ng_curr)
  while len(queue)>0:
    nt_parent,ng_curr= queue.pop(0)
    #print ng_curr, num_visits[ng_curr]
    if num_visits[ng_curr]<max_visits:
      #Add ng_curr to the tree:
      nt_curr= TStrInt(ng_curr,num_visits[ng_curr])
      t_node= TGNode()
      t_node.Parent= nt_parent
      tree.Tree[nt_curr]= t_node
      num_visits[ng_curr]+= 1
      #Done.  Prepare for the next:
      for ng_next in graph.Graph[ng_curr].Next:
        queue.append((nt_curr,ng_next))
        #Add to the Next list; if num_visits exceeds the threshold, None is added to keep the size of Next.
        t_node.Next.append(TStrInt(ng_next,num_visits[ng_next]) if num_visits[ng_next]<max_visits else None)
  #Get terminal nodes:
  for key,t_node in tree.Tree.items():
    if len(t_node.Next)==0:
      tree.Terminal.append(key)
  #Get order of backward computation:
  tree.BwdOrder= []
  processed= [None]+[key for key in tree.Terminal]
  queue= [tree.Tree[key].Parent for key in tree.Terminal if tree.Tree[key].Parent is not None]
  while len(queue)>0:
    key= queue.pop(0)
    if key in processed:  continue
    if all([key_next in processed for key_next in tree.Tree[key].Next]):
      tree.BwdOrder.append(key)
      processed.append(key)
      queue.append(tree.Tree[key].Parent)
    else:
      queue.append(key)  #This node (key) is not ready to compute backward
  return tree

def ForwardTree(tree, start):
  queue= [start]
  while len(queue)>0:
    n_curr= queue.pop(0)
    #Doing something with current node:
    print('Processing:',n_curr,n_curr.S,'that contains:',tree[n_curr].__dict__)
    #Done.  Prepare for the next:
    for n_next in tree[n_curr].Next:
      if n_next is not None:  queue.append(n_next)
      #Doing something with current to next edge:
      print('Processing:',n_curr,'-->',n_next)
      #Done.

if __name__=='__main__':
  graph= TGraph()
  graph.Graph= {
    'n1': TGNode(None, ('n2','n3')),
    'n2': TGNode('n1', ('n1','n4')),
    'n3': TGNode('n1', ()),
    'n4': TGNode('n2', ()),
    }
  print('Graph:')
  graph.PrintStructure()
  print('Tree:')
  tree= GraphToTree(graph,'n1',2)
  tree.PrintStructure()
  print('ForwardTree:')
  ForwardTree(tree.Tree, tree.Start)

  print('\n====================\n')

  graph= TGraph()
  graph.Graph= {
    'n0': TGNode(None, ['n1']),
    'n1': TGNode('n1', ['n2']),
    'n2': TGNode('n1', ['n3']),
    'n3': TGNode('n2', []),
    }
  print('Graph:')
  graph.PrintStructure()
  print('Tree:')
  tree= GraphToTree(graph,'n0',2)
  tree.PrintStructure()
  print('ForwardTree:')
  ForwardTree(tree.Tree, tree.Start)
