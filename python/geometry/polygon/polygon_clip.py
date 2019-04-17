#!/usr/bin/python
#\file    polygon_clip.py
#\brief   Test Sutherland-Hodgman-Algorithm
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.01, 2016


import math,random

#Clip subject_polygon by clip_polygon with Sutherland-Hodgman-Algorithm.
#subject_polygon is an arbitrary polygon, clip_polygon is a convex polygon.
#Output polygon is counterclockwise.
#Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping
def ClipPolygon(subject_polygon, clip_polygon):
  def is_left_of(edge_1, edge_2, test):
    tmp1 = [edge_2[0] - edge_1[0], edge_2[1] - edge_1[1]]
    tmp2 = [test[0] - edge_2[0], test[1] - edge_2[1]]
    x = (tmp1[0] * tmp2[1]) - (tmp1[1] * tmp2[0])
    if x < 0:  return False
    elif x > 0:  return True
    else:  return None  # Colinear points;

  def is_clockwise(polygon):
    for p in polygon:
      is_left = is_left_of(polygon[0], polygon[1], p);
      if is_left != None:  #some of the points may be colinear.  That's ok as long as the overall is a polygon
        return not is_left
    return None #All the points in the polygon are colinear

  def inside(p):
    return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

  def compute_intersection():
    dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
    dp = [ s[0] - e[0], s[1] - e[1] ]
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

  ic = is_clockwise(subject_polygon)
  if ic is None:  return []
  if ic:  subject_polygon.reverse()
  ic = is_clockwise(clip_polygon)
  #print 'is_clockwise(clip_polygon)=',ic
  if ic is None:  return []
  if ic:  clip_polygon.reverse()

  output_list = subject_polygon
  cp1 = clip_polygon[-1]
  #print 'clip_polygon=',clip_polygon

  for clip_vertex in clip_polygon:
    cp2 = clip_vertex
    input_list = output_list
    output_list = []
    #print 'input_list =',input_list
    if len(input_list)==0:
      return []
    s = input_list[-1]
    #print '  s=',s
    #print '  cp1=',cp1
    #print '  cp2=',cp2

    for subject_vertex in input_list:
      e = subject_vertex
      #print '  e=',e,inside(e),inside(s)
      if inside(e):
        if not inside(s):
          output_list.append(compute_intersection())
        output_list.append(e)
      elif inside(s):
        output_list.append(compute_intersection())
      s = e
    cp1 = cp2
  return(output_list)

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

#Generate a random number of uniform distribution of specified bound.
def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

def Main():
  #Generate polygon:
  #polygon= [[50, 150], [200, 50], [350, 150], [350, 300],
            #[250, 300], [200, 250], [150, 350], [100, 250],
            #[100, 200]]
  polygon= []
  N= 100
  for i in xrange(N):
    th= (float(i)+Rand(-0.4,0.4))*2.0*math.pi/float(N)
    r= Rand(1.0, 100.0)
    x= r*math.cos(th)
    y= r*math.sin(th)
    polygon.append([x,y])

  #rect= [[30.0,30.0],[30.0,-30.0],[-30.0,-30.0],[-30.0,30.0]]
  #rect= [[10*(p[0]+50),10*(p[1]+50)] for p in rect]
  #rect= [[30.0,30.0],[-30.0,30.0],[-30.0,-30.0],[30.0,-30.0]]
  #rect= [[100, 100], [300, 100], [300, 300], [100, 300]]
  #p1x= Rand(1.0, 100.0); p1y= Rand(1.0, 100.0); p2x= Rand(1.0, 100.0); p2y= Rand(1.0, 100.0)
  p1x= Rand(-100.0, 100.0); p1y= Rand(-100, 100.0); p2x= Rand(-100, 100.0); p2y= Rand(-100, 100.0)
  rect= [[p1x,p1y],[p2x,p1y],[p2x,p2y],[p1x,p2y]]
  #rect= [[p1x,p1y],[p2x,p2y],[p1x,p1y],[p2x,p2y]]

  #polygon= [[-1.69021, -87.6474],[-40.5151, -2.28697],[-25.867, 44.6599],[29.6493, -3.34294]]
  #rect= [[-88.5993, 13.5659],[57.7085, 13.5659],[57.7085, -55.5519],[-88.5993, -55.5519]]
  polygon= [[45.672222, -2.9928033],[1.6948826, 8.3420715],[-77.632523, 33.676598],[-29.458261, -45.078251]]
  rect= [[-61.265263, -16.213083],[-37.288738, -16.213083],[-37.288738, 72.973518],[-61.265263, 72.973518]]

  polygon2= ClipPolygon(polygon, rect)
  #try:
    #polygon2= ClipPolygon(polygon, rect)
  #except Exception as e:
    #print 'Error'
    #print '  e: ',e
    #print '  type: ',type(e)
    #print '  args: ',e.args
    #print '  message: ',e.message
    #polygon2= []

  def save_poly(poly,name):
    fp= open('/tmp/'+name,'w')
    for p in poly:
      fp.write('{x} {y}\n'.format(x=p[0],y=p[1]))
    if len(poly)>0:
      fp.write('{x} {y}\n'.format(x=poly[0][0],y=poly[0][1]))  #closing loop
    fp.close()
  save_poly(polygon,'polygon.dat')
  save_poly(rect,'clip_rect.dat')
  save_poly(polygon2,'polygon_clipped.dat')


def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa
      /tmp/polygon.dat w l lw 2 t '"Original"'
      /tmp/clip_rect.dat w l lw 2 t '"Rect"'
      /tmp/polygon_clipped.dat w l lw 3 t '"Clipped"'
      &''',
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print '###',cmd
      os.system(cmd)

  print '##########################'
  print '###Press enter to close###'
  print '##########################'
  raw_input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
