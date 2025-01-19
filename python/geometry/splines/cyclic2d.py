#!/usr/bin/python3

if __name__=="__main__":
  from cyclic import *
  from cubic_hermite_spline import TCubicHermiteSpline
  import gen_data

  #data= gen_data.Gen2d_1()
  #data= gen_data.Gen2d_2()
  #data= gen_data.Gen2d_3()
  data= gen_data.Gen2d_cyc1()
  #data= gen_data.Gen2d_cyc2()
  #data= gen_data.Gen2d_cyc3()

  #FINITE_DIFF, CARDINAL
  splines= [TCubicHermiteSpline() for d in range(len(data[0])-1)]
  for d in range(len(splines)):
    data_d= [[x[0],x[d+1]] for x in data]
    splines[d].Initialize(data_d, tan_method=splines[d].CARDINAL, end_tan=splines[d].CYCLIC, c=0.0, m=0.0)

  pf= open('/tmp/spline0.dat','w')
  for d in data:
    pf.write('%s\n' % ' '.join(map(str,d)))
  print('Generated:','/tmp/spline0.dat')

  #splines[1].KeyPts[1].X0= splines[1].KeyPts[1].X
  #splines[1].KeyPts[2].X0= splines[1].KeyPts[2].X
  pf= open('/tmp/spline1.dat','w')
  t= -5.0
  while t<5.0:
    #splines[1].KeyPts[1].X= splines[1].KeyPts[1].X0 + 0.1*t
    #splines[1].KeyPts[2].X= splines[1].KeyPts[2].X0 - 0.5*t
    #splines[1].Update()
    ##Draw spiral (use Gen2d_cyc3)
    #for d in range(len(splines)):
      #for k in range(len(splines[d].KeyPts)):
        #splines[d].KeyPts[k].X*= 0.9999
      #splines[d].Update()
    x= [splines[d].EvaluateC(t) for d in range(len(splines))]
    pf.write('%f %s\n' % (t, ' '.join(map(str,x))))
    t+= 0.001
  print('Generated:','/tmp/spline1.dat')


  print('Plot by:')
  print('t-x:')
  print('qplot -x /tmp/spline1.dat u 1:2 w l /tmp/spline0.dat u 1:2 w p pt 5 ps 2')
  print('t-y:')
  print('qplot -x /tmp/spline1.dat u 1:3 w l /tmp/spline0.dat u 1:3 w p pt 5 ps 2')
  print('x-y:')
  print('qplot -x /tmp/spline1.dat u 2:3 w l /tmp/spline0.dat u 2:3 w p pt 5 ps 2')

