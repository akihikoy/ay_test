#!/usr/bin/python3

if __name__=="__main__":
  from cubic_hermite_spline import TCubicHermiteSpline
  import gen_data

  spline= TCubicHermiteSpline()

  #data= gen_data.Gen1d_1()
  #data= gen_data.Gen1d_2()
  #data= gen_data.Gen1d_3()
  data= gen_data.Gen1d_cyc1()
  #data= gen_data.Gen1d_cyc2()
  #data= gen_data.Gen1d_cyc3()

  #FINITE_DIFF, CARDINAL
  spline.Initialize(data, tan_method=spline.FINITE_DIFF, end_tan=spline.CYCLIC, c=0.0)
  #spline.KeyPts[0].M= 5.0
  #spline.KeyPts[-1].M= -5.0

  pf= open('/tmp/spline0.dat','w')
  for d in data:
    pf.write('%f %f\n' % (d[0],d[1]))
  print('Generated:','/tmp/spline0.dat')

  pf= open('/tmp/spline1.dat','w')
  t= -5.0
  while t<5.0:
    pf.write('%f %f\n' % (t, spline.EvaluateC(t)))
    t+= 0.001
  print('Generated:','/tmp/spline1.dat')


  print('Plot by:')
  print('qplot -x /tmp/spline1.dat w l /tmp/spline0.dat w p pt 5 ps 2')

