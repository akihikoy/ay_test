#!/usr/bin/python

if __name__=="__main__":
  from cubic_hermite_spline import TCubicHermiteSpline
  import gen_data

  #data= gen_data.Gen3d_1()
  data= gen_data.Gen3d_2()
  #data= gen_data.Gen3d_3()
  #data= gen_data.Gen3d_4()
  #data= gen_data.Gen3d_5()

  splines= [TCubicHermiteSpline() for d in range(len(data[0])-1)]
  for d in range(len(splines)):
    data_d= [[x[0],x[d+1]] for x in data]
    splines[d].Initialize(data_d, tan_method=splines[d].CARDINAL, c=0.0, m=0.0)

  pf= file('/tmp/spline1.dat','w')
  t= data[0][0]
  while True:
    x= [splines[d].Evaluate(t) for d in range(len(splines))]
    pf.write('%f %s\n' % (t, ' '.join(map(str,x))))
    if t>data[-1][0]:  break
    t+= 0.02
    #t+= 0.001
  print 'Generated:','/tmp/spline1.dat'

  pf= file('/tmp/spline0.dat','w')
  for d in data:
    pf.write('%s\n' % ' '.join(map(str,d)))
  print 'Generated:','/tmp/spline0.dat'


  print 'Plot by:'
  print 'qplot -x -3d /tmp/spline1.dat u 2:3:4 w l /tmp/spline0.dat u 2:3:4 w p pt 5 ps 2'
