#!/usr/bin/python3
#\file    dot2graphml.py
#\brief   Convert a DOT graph string/file to a yEd-compatible GraphML file.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.21, 2026
#\version 0.2
#\date    Feb.23, 2026

import sys
import os
import argparse
import subprocess
import shlex

def xml_escape(s: str) -> str:
  # Escape characters for XML attributes
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')

def DotToGraphML(dot_content: str) -> str:
  # Run Graphviz to compute geometry and resolved labels ('plain' format)
  try:
    proc = subprocess.run(
      ['dot', '-Tplain'],
      input=dot_content.encode('utf-8'),
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=True
    )
  except FileNotFoundError:
    print("Error: Graphviz 'dot' not found. Please install graphviz.")
    return ""
  except subprocess.CalledProcessError as e:
    print(f"Graphviz error:\n{e.stderr.decode('utf-8')}")
    return ""

  plain = proc.stdout.decode('utf-8')

  # Parse 'plain' output
  scale_in = 72.0  # inches -> points
  graph_h_in = 0.0
  nodes = {}   # name -> {...}
  edges = []   # (tail, head, [(x,y), ...])

  for raw in plain.splitlines():
    parts = shlex.split(raw)
    if not parts:
      continue
    tag = parts[0]
    if tag == 'graph':
      if len(parts) >= 4:
        graph_h_in = float(parts[3])
      elif len(parts) == 3:
        graph_h_in = float(parts[2])
    elif tag == 'node' and len(parts) >= 11:
      name = parts[1]
      x = float(parts[2]) * scale_in
      y = float(parts[3]) * scale_in
      w = float(parts[4]) * scale_in
      h = float(parts[5]) * scale_in
      label = parts[6]
      shape_gv = parts[8].lower()

      is_service = name.startswith('srv:')

      # Determine styles based on node type
      if is_service:
        gml_shape = 'roundrectangle'
        ros_type = 'service'
        fill_color = "#eeeeff"
        font_size = "12"
        font_style = "plain"
        target_h = 18.0
        border_style = "dashed"
      elif shape_gv in ('ellipse', 'circle', 'oval'):
        gml_shape = 'ellipse'
        ros_type = 'node'
        fill_color = "#ffff99"
        font_size = "16"
        font_style = "bold"
        target_h = 36.0
        border_style = "line"
      elif shape_gv in ('box', 'rectangle', 'square', 'box3d'):
        gml_shape = 'rectangle'
        ros_type = 'topic'
        fill_color = "#ccccff"
        font_size = "12"
        font_style = "plain"
        target_h = 18.0
        border_style = "line"
      else:
        gml_shape = 'rectangle'
        ros_type = 'unknown'
        fill_color = "#EEEEEE"
        font_size = "12"
        font_style = "plain"
        target_h = h
        border_style = "line"

      informative_desc = f"{ros_type}:{label}"
      nodes[name] = {
        'x': x, 'y': y, 'w': w, 'h': target_h, 'shape': gml_shape,
        'label': label, 'description': informative_desc,
        'fill': fill_color, 'fsize': font_size, 'fstyle': font_style, 'border': border_style
      }
    elif tag == 'edge' and len(parts) >= 5:
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

  name_to_id = {name: idx for idx, name in enumerate(nodes.keys())}
  include_edge_bends = False

  out = []
  out.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
  out.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"')
  out.append('         xmlns:y="http://www.yworks.com/xml/graphml"')
  out.append('         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
  out.append('         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">')
  out.append('  <key id="d_desc" for="node" attr.name="description" attr.type="string"/>')
  out.append('  <key id="d_nodegraphics" for="node" yfiles.type="nodegraphics"/>')
  out.append('  <key id="d_edgegraphics" for="edge" yfiles.type="edgegraphics"/>')
  out.append('  <graph id="G" edgedefault="directed">')

  # Nodes
  for name, nid in name_to_id.items():
    nd = nodes[name]
    x_topleft = nd["x"] - nd["w"] / 2.0
    y_topleft = nd["y"] - nd["h"] / 2.0

    out.append(f'    <node id="n{nid}">')
    out.append(f'      <data key="d_desc"><![CDATA[{nd["description"]}]]></data>')
    out.append('      <data key="d_nodegraphics">')
    out.append('        <y:ShapeNode>')
    out.append(f'          <y:Geometry x="{x_topleft}" y="{y_topleft}" width="{nd["w"]}" height="{nd["h"]}"/>')
    out.append(f'          <y:Fill color="{nd["fill"]}" transparent="false"/>')
    out.append(f'          <y:BorderStyle type="{nd["border"]}" width="1.0" color="#000000"/>')
    out.append(f'          <y:NodeLabel textColor="#000000" fontSize="{nd["fsize"]}" fontStyle="{nd["fstyle"]}"><![CDATA[{nd["label"]}]]></y:NodeLabel>')
    out.append(f'          <y:Shape type="{nd["shape"]}"/>')
    out.append('        </y:ShapeNode>')
    out.append('      </data>')
    out.append('    </node>')

  # Edges
  edge_id = 0
  for tail, head, pts in edges:
    if tail not in name_to_id or head not in name_to_id:
      continue

    # Make edge dashed if it connects to a service
    is_srv_edge = tail.startswith('srv:') or head.startswith('srv:')
    edge_style = "dashed" if is_srv_edge else "line"

    out.append(f'    <edge id="e{edge_id}" source="n{name_to_id[tail]}" target="n{name_to_id[head]}">')
    out.append('      <data key="d_edgegraphics">')
    out.append('        <y:PolyLineEdge>')
    out.append('          <y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0">')
    if include_edge_bends and pts:
      for (x, y) in pts:
        out.append(f'            <y:Point x="{x}" y="{y}"/>')
    out.append('          </y:Path>')
    out.append(f'          <y:LineStyle type="{edge_style}" width="1.0" color="#000000"/>')
    out.append('          <y:Arrows source="none" target="standard"/>')
    out.append('        </y:PolyLineEdge>')
    out.append('      </data>')
    out.append('    </edge>')
    edge_id += 1

  out.append('  </graph>')
  out.append('</graphml>')

  return '\n'.join(out)

def main():
  parser = argparse.ArgumentParser(description='Convert a DOT file to a GraphML file.')
  parser.add_argument('input_dot', type=str, help='Input DOT file')
  args = parser.parse_args()

  input_dot = args.input_dot
  output_graphml = os.path.splitext(input_dot)[0] + ".graphml"

  try:
    with open(input_dot, 'r', encoding='utf-8') as f:
      dot_content = f.read()
  except Exception as e:
    print(f"Error reading {input_dot}: {e}")
    sys.exit(1)

  graphml_str = DotToGraphML(dot_content)
  if graphml_str:
    try:
      with open(output_graphml, 'w', encoding='utf-8') as f:
        f.write(graphml_str)
      print(f'Success! Wrote to {output_graphml}')
    except Exception as e:
      print(f"Error writing to {output_graphml}: {e}")

if __name__ == '__main__':
  main()
