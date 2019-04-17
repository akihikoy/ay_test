#!/usr/bin/python

if __name__=='__main__':
  def PrintEq(s):  print '%s= %r' % (s, eval(s))
  from cubic_hermite_spline import TCubicHermiteSpline
  from vel_ctrl import ModifyTrajVelocityV
  import gen_data
  import math

  #data= gen_data.Gen3d_1()
  data= gen_data.Gen3d_2()
  #data= gen_data.Gen3d_3()
  #data= gen_data.Gen3d_4()

  splines= [TCubicHermiteSpline() for d in range(len(data[0])-1)]
  for d in range(len(splines)):
    data_d= [[x[0],x[d+1]] for x in data]
    splines[d].Initialize(data_d, tan_method=splines[d].CARDINAL, c=0.0, m=0.0)
  dt= 0.005

  pf= file('/tmp/spline1.dat','w')
  t= data[0][0]
  while True:
    x= [splines[d].Evaluate(t) for d in range(len(splines))]
    pf.write('%f %s\n' % (t, ' '.join(map(str,x))))
    if t>data[-1][0]:  break
    t+= dt
  print 'Generated:','/tmp/spline1.dat'

  pf= file('/tmp/spline2.dat','w')
  t= data[0][0]
  ti= data[0][0]  #Internal time
  T= data[-1][0]  #Length
  v_trg= 1.0
  traj= lambda s: [splines[d].Evaluate(s) for d in range(len(splines))]
  while ti<data[-1][0]:
    #v_trg= (1.0+math.cos(t))
    #v_trg= 1.0*(1.0+2.0*math.cos(3.0*t)); v_trg= 0.0 if v_trg<0.0 else v_trg
    v_trg= 1.0*(1.0+2.0*math.cos(8.0*t)+2.0*math.sin(15.0*t)); v_trg= 0.0 if v_trg<0.0 else v_trg
    ti,x,v= ModifyTrajVelocityV(ti, v_trg, traj, dt, T)
    pf.write('%f %s %f %f\n' % (t, ' '.join(map(str,x)), v, v_trg))
    t+= dt
  print 'Generated:','/tmp/spline2.dat'

  pf= file('/tmp/spline0.dat','w')
  for d in data:
    pf.write('%s\n' % ' '.join(map(str,d)))
  print 'Generated:','/tmp/spline0.dat'


  print 'Plot by:'
  print '3d:'
  print "qplot -x -3d -s 'set view equal xyz;set ticslevel 0' /tmp/spline1.dat u 2:3:4 w l /tmp/spline0.dat u 2:3:4 w p pt 5 ps 2 /tmp/spline2.dat u 2:3:4 ev 20 w lp"
  print 'x+v:'
  print 'set f=2; qplot -x /tmp/spline1.dat u 1:$f w l /tmp/spline0.dat u 1:$f w p pt 5 ps 2 /tmp/spline2.dat u 1:$f w l /tmp/spline2.dat u 1:6 w l /tmp/spline2.dat u 1:5 w lp'

