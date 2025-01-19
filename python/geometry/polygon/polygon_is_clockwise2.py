#!/usr/bin/python3
#\file    polygon_is_clockwise2.py
#\brief   Check if a polygon is clockwise (robust for non-convex polygons).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.30, 2023

#Check if a polygon is clockwise.
#ref. https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
def PolygonIsClockwise(polygon):
  if len(polygon)<3:  return None
  polygon= list(polygon)
  s= sum((p2[0]-p1[0])*(p2[1]+p1[1]) for p1,p2 in zip(polygon,polygon[1:]+[polygon[0]]))
  return s>0

def Main():
  import numpy as np
  polygons=[
    [[0.729,0.049],[0.723,0.082],[0.702,0.125],[0.682,0.124],[0.654,0.106],[0.656,0.101],[0.647,0.081],[0.652,0.078],[0.651,0.071],[0.655,0.071],[0.673,0.031]],
    [[0.722,0.219],[0.717,0.220],[0.712,0.229],[0.693,0.235],[0.681,0.227],[0.672,0.230],[0.649,0.211],[0.637,0.213],[0.629,0.208],[0.626,0.216],[0.620,0.202],[0.616,0.203],[0.617,0.207],[0.609,0.200],[0.603,0.201],[0.601,0.191],[0.587,0.181],[0.589,0.175],[0.580,0.166],[0.585,0.133],[0.593,0.121],[0.605,0.113],[0.626,0.113],[0.645,0.121],[0.644,0.127],[0.651,0.123],[0.661,0.135],[0.669,0.134],[0.675,0.140],[0.702,0.148],[0.715,0.159],[0.717,0.150],[0.720,0.149],[0.721,0.167],[0.727,0.167],[0.730,0.195],[0.724,0.204]],
    [[0.820,0.156],[0.793,0.154],[0.812,0.154],[0.812,0.150],[0.803,0.149],[0.806,0.134],[0.802,0.139],[0.796,0.133],[0.786,0.140],[0.779,0.139],[0.772,0.131],[0.774,0.126],[0.782,0.127],[0.779,0.134],[0.789,0.130],[0.788,0.115],[0.794,0.109],[0.773,0.111],[0.769,0.124],[0.755,0.143],[0.749,0.144],[0.753,0.150],[0.750,0.153],[0.737,0.147],[0.731,0.149],[0.738,0.141],[0.722,0.144],[0.722,0.124],[0.726,0.126],[0.729,0.123],[0.725,0.118],[0.733,0.107],[0.733,0.090],[0.738,0.086],[0.738,0.077],[0.740,0.082],[0.744,0.080],[0.749,0.041],[0.757,0.039],[0.758,0.032],[0.763,0.034],[0.762,0.040],[0.769,0.037],[0.769,0.008],[0.781,0.024],[0.778,0.034],[0.788,0.043],[0.828,0.144],[0.819,0.150]],
    #[[0.6,0.05],[0.65,0.05],[0.65,0.1],[0.6,0.1]],
    #[[0.6,0.15],[0.6,0.2],[0.65,0.15],[0.65,0.2]],
    #[[0.65,0.05],[0.7,0.05],[0.7,0.1]],
    #[[0.6,0.05],[0.6,0.1],[0.65,0.05],[0.65,0.1]],
    ]

  #Modify polygons
  for i in np.random.choice(list(range(len(polygons))), size=np.random.choice(list(range(len(polygons))))):
    polygons[i].reverse()
  for i in range(len(polygons)):
    i_split= np.random.choice(len(polygons[i]))
    polygons[i]= polygons[i][i_split:]+polygons[i][:i_split]

  def write_polygon(fp,polygon):
    if len(polygon)>0:
      d_cross= np.max(np.max(polygon,axis=0)-np.min(polygon,axis=0))*0.02
      p0= np.array(polygon[0])
      poly_cross= [p0, p0+[d_cross,d_cross], p0-[d_cross,d_cross], p0, p0+[-d_cross,d_cross], p0-[-d_cross,d_cross], p0]
      for pt in poly_cross+polygon:
        fp.write('%s\n'%' '.join(map(str,pt)))
    fp.write('\n')

  with open('/tmp/polygons1.dat','w') as fp1, open('/tmp/polygons2.dat','w') as fp2:
    for polygon in polygons:
      if PolygonIsClockwise(polygon):
        write_polygon(fp1,polygon)
      else:
        write_polygon(fp2,polygon)

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa
        /tmp/polygons1.dat u 1:2 lc 1 w l
        /tmp/polygons2.dat u 1:2 lc 2 w l
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
