#!/usr/bin/python3

#ref. http://stackoverflow.com/questions/11716268/point-in-polygon-algorithm
#Ray-casting algorithm (http://en.wikipedia.org/wiki/Point_in_polygon)
#with considering point on an edge of polygon.
def PointInPolygon2D2(points, point, include_on_edge=True):
  if PointOnPolygon2D(points, point):  return include_on_edge
  points= list(points)  #Convert numpy.array
  s= sum(1 for p1,p2 in zip(points,[points[-1]]+points[:-1])
         if ((p1[1]>point[1]) != (p2[1]>point[1]))
          and (point[0] < (p2[0]-p1[0]) * (point[1]-p1[1]) / (p2[1]-p1[1]) + p1[0]))
  return s%2==1

#Check if point is on an edge of polygon points.
def PointOnPolygon2D(points, point, tol=1.0e-10):
  def PointOnLine(p1,p2,p):
    if (p[0]==p1[0] and p[1]==p1[1]) or (p[0]==p2[0] and p[1]==p2[1]):  return True
    if p[0]==p1[0] or p[0]==p2[0]:
      if p[0]==p1[0] and p[0]==p2[0]:
        if (p1[1]<p[1] and p[1]<p2[1]) or (p2[1]<p[1] and p[1]<p1[1]):  return True
      return False
    if abs((p[1]-p1[1])/(p[0]-p1[0])-(p2[1]-p[1])/(p2[0]-p[0]))<tol:  return True
    return False
  return PointOnLine(points[-1],points[0],point) or any(PointOnLine(p1,p2,point) for p1,p2 in zip(points[:-1],points[1:]))

def FRange(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

def Main():
  def PrintEq(s):  print('%s= %r' % (s, eval(s)))

  import gen_data
  import time
  #points= gen_data.To2d(gen_data.Gen3d_01())
  #points= gen_data.To2d(gen_data.Gen3d_02())
  #points= gen_data.To2d(gen_data.Gen3d_11())
  points= gen_data.To2d(gen_data.Gen3d_12())
  #points= gen_data.To2d(gen_data.Gen3d_13())

  with open('/tmp/orig.dat','w') as fp:
    for p in points:
      fp.write(' '.join(map(str,p))+'\n')

  bb_min= [min(x for x,y in points), min(y for x,y in points)]
  bb_max= [max(x for x,y in points), max(y for x,y in points)]

  t_start= time.time()
  x_y_inout= [(x,y,PointInPolygon2D2(points,[x,y],include_on_edge=True))
              for x in FRange(bb_min[0],bb_max[0],50)
              for y in FRange(bb_min[1],bb_max[1],50)]
  t_end= time.time()
  print('Computation time: ',t_end-t_start)

  def write_in_out(fp1,fp2,x,y,inout=None):
    p= [x,y]
    if inout is None:  inout= PointInPolygon2D2(points,p,include_on_edge=True)
    if inout:
      fp1.write(' '.join(map(str,p))+'\n')
    else:
      fp2.write(' '.join(map(str,p))+'\n')
    #print p,inout
  with open('/tmp/points_in.dat','w') as fp1:
    with open('/tmp/points_out.dat','w') as fp2:
      for x,y,inout in x_y_inout:
        write_in_out(fp1,fp2,x,y,inout)
      fp1.write('\n')
      fp2.write('\n')
      for i,(x,y) in enumerate(points):
        write_in_out(fp1,fp2,x,y)
        if i>0:  write_in_out(fp1,fp2,0.5*(x+points[i-1][0]),0.5*(y+points[i-1][1]))

  print('Plot by')
  print("qplot -x /tmp/orig.dat w l /tmp/points_in.dat /tmp/points_out.dat")

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa
        /tmp/orig.dat w l /tmp/points_in.dat /tmp/points_out.dat
        &''',
        #/tmp/polygons.dat u 1:2:-1 lc var w l
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print('###',cmd)
      os.system(cmd)

  print('##########################')
  print('###Press enter to close###')
  print('##########################')
  input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()

  PlotGraphs()
  sys.exit(0)
