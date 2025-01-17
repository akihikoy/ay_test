#!/usr/bin/python3

def Test1(tree):
  def sub_parse(sub_tree):
    name= sub_tree[0]
    if sub_tree[1]=='x':
      return [name, sub_parse(sub_tree[2][-1])]
    else:
      return name
  print('last names=', sub_parse(tree))

def Parser(tree_struct, op):
  name= tree_struct[0]
  kind= tree_struct[1]
  op(name,kind)
  if tree_struct[1]=='x':
    for sub_tree_struct in tree_struct[2]:
      Parser(sub_tree_struct, op)

if __name__=='__main__':

  #tree= ['c1','c']
  #tree= ['d1','d']
  #tree= ['x1','x',[['c1','c'],['c2','c']]]
  tree= ['x1','x',[ ['c1','c'], ['x2','x',[['c2','c'],['c3','c']]] ]]

  print('tree=',tree)

  #seeker= tree
  #while seeker<>None:
    #print 'name=',seeker[0],'type=',seeker[1]
    #if seeker[1]=='x':  seeker= seeker[2]
    #else:               seeker= None

  def sub_parse(sub_tree,indent=0):
    print('  '*indent+'name=',sub_tree[0],'type=',sub_tree[1])
    if sub_tree[1]=='x':
      indent+= 1
      for sub2 in sub_tree[2]:
        sub_parse(sub2,indent)
  sub_parse(tree)

  Test1(tree)

  class TOp1:
    def __init__(self):  self.indent= 0
    def op1(self,name,kind):
      print('> '*self.indent+'name=',name,'type=',kind)
      if kind=='x':  self.indent+= 1
  Parser(tree, TOp1().op1)

