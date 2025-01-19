#!/usr/bin/python3

def Test1(key,*args):
  print('------------')
  print('key:',key)
  print('len(args):',len(args))
  print('type(args):',type(args))
  print('args:',args)

def Test2(key,*args):
  print('------------')
  assert(len(args)>=1)
  last= args[-1]
  print('len(args):',len(args[0:-1]))
  print('args[0:-1]:',args[0:-1])
  print('%r[%r]= %r' % (key, args[0:-1], last))

#def Test3(key,*args,value):  #ERROR:syntax
  #pass

#def Test4(key,*args,value=None):  #ERROR:syntax
  #pass

Test1('aaa',1,0.3,'hoge',[1,2,3])  #len(args):4
Test1('aaa',*(1,0.3,'hoge',[1,2,3]))  #len(args):4, ditto
Test1('aaa',(1,0.3,'hoge',[1,2,3]))  #len(args):1
Test1('aaa',)
Test1('aaa')

Test2('aaa','bbb','hoge',[1,2,3])
Test2('aaa',3.14)
