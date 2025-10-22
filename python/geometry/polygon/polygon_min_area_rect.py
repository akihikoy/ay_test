#!/usr/bin/python3
#\file    polygon_min_area_rect.py
#\brief   Wrap of minAreaRect of OpenCv.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.31, 2023

from cv2 import minAreaRect as cv2_minAreaRect
import numpy as np

#Convert angle (radian) to the range according to mode:
#  mode='positive': [0,pi]
#  mode='symmetric': [-pi/2,pi/2]
#  mode='' or None: No modification.
def AngleModHalf(q, mode='symmetric'):
  if mode=='symmetric':
    # Normalize to [-pi/2, pi/2)
    return (q + np.pi/2.0) % np.pi - np.pi/2.0
  elif mode=='positive':
    # Normalize to [0, π)
    return q % np.pi
  elif mode in (None, ''):
    return q
  else:
    raise Exception(f'AngleModHalf: Invalid mode={angle_mode}')

#Get a rectangle of minimum area.
#  angle_mode: Angle normalization mode ('positive': [0,pi], 'symmetric': [-pi/2,pi/2], None).
#Return: center,size,angle.
#  angle is in radian.  angle direction is always the longer side direction.
#  size[0] is always greater than size[1].
#ref. https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
def MinAreaRect(points, angle_mode='symmetric'):
  center,size,angle= cv2_minAreaRect(np.array(points,np.float32))
  angle*= np.pi/180.0
  if size[0]<size[1]:
    size= (size[1],size[0])
    angle+= np.pi/2.0
  angle= AngleModHalf(angle, angle_mode)
  return center,size,angle

def Main():
  def write_polygon(fp,polygon):
    if len(polygon)>0:
      for pt in polygon+[polygon[0]]:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  import gen_data
  polygons= gen_data.Gen2d_01(None, 8)

  polygon= polygons[np.random.choice(list(range(len(polygons))))]
  center,size,angle= MinAreaRect(polygon)
  print('##########################')
  print('MinAreaRect: center={}, size={}, angle={}'.format(center,size,angle))
  print('##########################')
  rot= np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  rect= [np.array(center)+rot.dot(p) for p in [[size[0]*0.5,size[1]*0.5],[size[0]*0.5,-size[1]*0.5],[-size[0]*0.5,-size[1]*0.5],[-size[0]*0.5,size[1]*0.5]]]

  with open('/tmp/polygons.dat','w') as fp:
    write_polygon(fp,polygon)
  with open('/tmp/rect.dat','w') as fp:
    write_polygon(fp,rect)

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa
        -s 'set size ratio 1;'
        /tmp/polygons.dat u 1:2:'(column(-1)+1)' w lp lc var pt 4
        /tmp/rect.dat w lp pt 4
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
