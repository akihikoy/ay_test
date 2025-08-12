#!/usr/bin/python3
#\file    rqt_graph_gml.py
#\brief   Generate a GML file of ROS node graph with rqt_graph and pydot.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.13, 2025

import subprocess
import shlex
import rosgraph.impl.graph
from rqt_graph.dotcode import RosGraphDotcodeGenerator
from qt_dotgraph.pydotfactory import PydotFactory

def gml_escape(s: str) -> str:
  # Escape quotes for GML strings
  return s.replace('"', '\\"')

def main():
  # Build graph snapshot from ROS master
  g = rosgraph.impl.graph.Graph()
  g.set_master_stale(5.0)
  g.set_node_stale(5.0)
  g.update()

  # Generate DOT (same as rqt_graph)
  gen = RosGraphDotcodeGenerator()
  dot = gen.generate_dotcode(
    rosgraphinst=g,
    ns_filter='/',                 # include all
    topic_filter='/',              # include all
    graph_mode='node_topic_all',   # 'node_topic', 'node_topic_all', 'node_node'
    dotcode_factory=PydotFactory(),
    # Group
    cluster_namespaces_level=5,
    group_image_nodes=True,
    group_tf_nodes=True,
    # Hide
    quiet=True,                    # Hide Debug (rviz, rosout, etc.)
    hide_tf_nodes=True,
    hide_single_connection_topics=False,
    hide_dead_end_topics=False,
    hide_dynamic_reconfigure=False,
    # Others keep default behavior of rqt_graph
    accumulate_actions=True,
    orientation='LR',
    rank='same',
    simplify=True,
    unreachable=False,
  )

  # Run Graphviz to compute geometry and resolved labels ('plain' format)
  try:
    proc = subprocess.run(
      ['dot', '-Tplain'],
      input=dot.encode('utf-8'),
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=True
    )
  except FileNotFoundError:
    print("Error: Graphviz 'dot' not found. Please install graphviz.")
    return
  plain = proc.stdout.decode('utf-8')

  # Parse 'plain' output
  scale_in = 72.0  # inches -> points (tweak if needed)
  graph_h_in = 0.0
  nodes = {}   # name -> {x,y,w,h,shape,label}
  edges = []   # (tail, head, [(x,y), ...])

  for raw in plain.splitlines():
    parts = shlex.split(raw)
    if not parts:
      continue
    tag = parts[0]
    if tag == 'graph':
      # graph <scale> <width> <height>
      if len(parts) >= 4:
        graph_h_in = float(parts[3])
      elif len(parts) == 3:
        graph_h_in = float(parts[2])
    elif tag == 'node' and len(parts) >= 11:
      # node <name> <x> <y> <w> <h> <label> <style> <shape> <color> <fillcolor>
      name = parts[1]
      x = float(parts[2]) * scale_in
      y = float(parts[3]) * scale_in
      w = float(parts[4]) * scale_in
      h = float(parts[5]) * scale_in
      label = parts[6]  # <-- resolved label from Graphviz (e.g., "/ctrl_panel")
      shape_gv = parts[8].lower()
      if shape_gv in ('box', 'rectangle', 'square'):
        gml_shape = 'rectangle'
      elif shape_gv in ('ellipse', 'circle', 'oval'):
        gml_shape = 'ellipse'
      else:
        gml_shape = 'roundrectangle'
      nodes[name] = {'x': x, 'y': y, 'w': w, 'h': h, 'shape': gml_shape, 'label': label}
    elif tag == 'edge' and len(parts) >= 5:
      # edge <tail> <head> <n> x1 y1 ... xn yn ...
      tail, head = parts[1], parts[2]
      n = int(parts[3])
      pts = []
      for i in range(n):
        xi = float(parts[4 + 2*i]) * scale_in
        yi = float(parts[5 + 2*i]) * scale_in
        pts.append((xi, yi))
      edges.append((tail, head, pts))

  # Flip Y (make coordinates natural for yEd)
  max_y = graph_h_in * scale_in if graph_h_in > 0 else max((nd['y'] for nd in nodes.values()), default=0.0)
  for nd in nodes.values():
    nd['y'] = max_y - nd['y']
  edges = [(t, h, [(x, max_y - y) for (x, y) in pts]) for (t, h, pts) in edges]

  # Assign integer IDs for yFiles GML
  name_to_id = {name: idx for idx, name in enumerate(nodes.keys())}

  # Build yFiles GML text
  include_edge_bends = False  # keep False to allow dynamic routing in yEd

  out = []
  out.append('Creator "rqt_graph_to_gml"')
  out.append('Version 2.2')
  out.append('graph')
  out.append('[')
  out.append('  directed 1')

  # Nodes with fixed positions and correct labels
  for name, nid in name_to_id.items():
    nd = nodes[name]
    out.append('  node')
    out.append('  [')
    out.append(f'    id {nid}')
    out.append(f'    label "{gml_escape(nd["label"])}"')
    out.append('    graphics')
    out.append('    [')
    out.append(f'      x {nd["x"]}')
    out.append(f'      y {nd["y"]}')
    out.append(f'      w {nd["w"]}')
    out.append(f'      h {nd["h"]}')
    out.append(f'      type "{nd["shape"]}"')
    out.append('    ]')
    out.append('    LabelGraphics')
    out.append('    [')
    out.append(f'      text "{gml_escape(nd["label"])}"')
    out.append('    ]')
    out.append('  ]')

  # Edges
  for tail, head, pts in edges:
    if tail not in name_to_id or head not in name_to_id:
      continue
    out.append('  edge')
    out.append('  [')
    out.append(f'    source {name_to_id[tail]}')
    out.append(f'    target {name_to_id[head]}')
    out.append('    graphics')
    out.append('    [')
    if include_edge_bends and pts:
      out.append('      Line')
      out.append('      [')
      for (x, y) in pts:
        out.append('        point')
        out.append('        [')
        out.append(f'          x {x}')
        out.append(f'          y {y}')
        out.append('        ]')
      out.append('      ]')
    out.append('      sourceArrow "none"')
    out.append('      targetArrow "standard"')
    out.append('    ]')
    out.append('  ]')

  out.append(']')

  with open('rosgraph.gml', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
  print('Wrote rosgraph.gml')

if __name__ == '__main__':
  main()
