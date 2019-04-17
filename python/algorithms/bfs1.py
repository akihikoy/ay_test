#!/usr/bin/python
#\file    bfs1.py
#\brief   Breadth-first search
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.25, 2017

#https://en.wikipedia.org/wiki/Breadth-first_search

def BFS1(state, selectable, transition, goal):
  visited= [state]
  tracker= {}  #succeeding:[parents]
  goals= []
  queue= [state]
  while len(queue)>0:
    print queue
    current= queue.pop(0)
    #if goal(current):
      #if current not in goals:  goals.append(current)
      #return goals,tracker
    for action in selectable(current):
      succeeding= transition(current,action)
      if succeeding in tracker:  tracker[succeeding].append(current)
      else:  tracker[succeeding]= [current]
      if goal(succeeding):
        if succeeding not in goals:  goals.append(succeeding)
        return goals,tracker
      if succeeding not in visited:
        visited.append(succeeding)
        #if succeeding in tracker:  tracker[succeeding].append(current)
        #else:  tracker[succeeding]= [current]
        queue.append(succeeding)
  return goals,tracker

def BackTrack(start,goals,tracker):
  print tracker
  paths= []
  for g in goals:
    queue= [[g]]
    visited= [g]
    while len(queue)>0:
      print queue
      path= queue.pop(0)
      for prev in tracker[path[0]]:
        if prev not in visited:
          visited.append(prev)
          if prev==start:
            paths.append([prev]+path)
          else:
            queue.append([prev]+path)
  return paths

def BFS2(state, selectable, transition, goal):
  visited= [state]
  queue= [([state],state)]  #path, current
  while len(queue)>0:
    print queue
    path,current= queue.pop(0)
    #if goal(current):
      #return path
    for action in selectable(current):
      succeeding= transition(current,action)
      if goal(succeeding):
        return path+[succeeding]
      if succeeding not in visited:
        visited.append(succeeding)
        queue.append((path+[succeeding],succeeding))

#def BFS3(state, selectable, transition, goal):
  #visited= [state]
  #paths= []
  #goals= []
  #queue= [([state],state)]  #path, current
  #while len(queue)>0:
    #print queue
    #path,current= queue.pop(0)
    #if goal(current):
      #goals.append(current)
      #paths.append(path)
      ##return goals,paths
    #for action in selectable(current):
      #succeeding= transition(current,action)
      #if succeeding not in visited:
        #visited.append(succeeding)
        ##if succeeding in tracker:  tracker[succeeding].append(current)
        ##else:  tracker[succeeding]= [current]
        #queue.append((path+[succeeding],succeeding))
  #return goals,paths

def BFS4(state, selectable, transition, goal):
  paths= []
  goals= []
  queue= [([state],state)]  #path, current
  while len(queue)>0:
    print queue
    path,current= queue.pop(0)
    if goal(current):
      if current not in goals:  goals.append(current)
      paths.append((path,'GOAL'))
      #return goals,paths
    for action in selectable(current):
      succeeding= transition(current,action)
      if succeeding not in path:
        queue.append((path+[succeeding],succeeding))
      else:
        paths.append((path+[succeeding],'LOOP'))
  return goals,paths

def BFS5(state, selectable, transition, goal):
  paths= []
  goals= []
  queue= [([state],state)]  #path, current
  while len(queue)>0:
    print queue
    path,current= queue.pop(0)
    if goal(current):
      if current not in goals:  goals.append(current)
      paths.append((path,'GOAL'))
      #return goals,paths
    for action in selectable(current):
      succeeding= transition(current,action)
      if succeeding not in path:
        queue.append((path+[action,succeeding],succeeding))
      else:
        paths.append((path+[action,succeeding],'LOOP'))
  return goals,paths

def Example1():
  system= {'a':['b','c','d'],'b':['a','c','b'],'c':['d'],'d':['a','e'],'e':[]}
  selectable= lambda s: {state: range(len(nexts)) for state,nexts in system.iteritems()}[s]
  transition= lambda s,a: system[s][a]
  goal= lambda s: s=='e'
  #print BackTrack('a',*BFS1('a', selectable, transition, goal))
  #print BFS2('a', selectable, transition, goal)
  #print BFS4('a', selectable, transition, goal)
  print BFS5('a', selectable, transition, goal)

if __name__=='__main__':
  Example1()
