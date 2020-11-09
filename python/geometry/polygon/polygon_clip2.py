#!/usr/bin/python
#\file    polygon_clip2.py
#\brief   Test ClipPolygon(2).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.02, 2020

from polygon_clip import ClipPolygon
from polygon_area import PolygonArea

def Main():
  polygon= [[0.36,  0.19], [0.36, -0.07], [0.65, -0.07], [0.65,  0.19]]
  #rect= [[0.3915922280999888,  0.16468070163072782], [0.3934531504087612, -0.06192327915589308], [0.4734504529096543, -0.06126632348665271], [0.4715895306008819,  0.1653376572999682 ]]
  rect= [[0.5707179144078721, -0.03216382557975851], [0.5957170714394012, -0.03195852693312089], [0.5946495185126174,  0.09803708963250936], [0.5696503614810883,  0.09783179098587173]]

  polygon2= ClipPolygon(polygon, rect)

  print '''Areas:
    polygon: {0}
    rect: {1}
    polygon2: {2}
    polygon2-rect: {3}
    '''.format(PolygonArea(polygon),PolygonArea(rect),PolygonArea(polygon2),PolygonArea(polygon2)-PolygonArea(rect))

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
