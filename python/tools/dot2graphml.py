#!/usr/bin/python3
#\file    dot2graphml.py
#\brief   Convert a DOT file to a yEd-compatible GraphML file using Graphviz plain text output.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.21, 2026

import sys
import os
import subprocess
import shlex

def xml_escape(s: str) -> str:
  # Escape characters for XML attributes
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')

def main():
  creator = "dot2yed_graphml"

  if len(sys.argv) < 2:
    print("Usage: python3 dot2yed_graphml.py <input_file.dot>")
    sys.exit(1)

  input_dot = sys.argv[1]
  output_graphml = os.path.splitext(input_dot)[0] + ".graphml"

  # 1. Read the DOT file
  try:
    with open(input_dot, 'r', encoding='utf-8') as f:
      dot_content = f.read()
  except Exception as e:
    print(f"Error reading {input_dot}: {e}")
    return

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
    return
  except subprocess.CalledProcessError as e:
    print(f"Graphviz error:\n{e.stderr.decode('utf-8')}")
    return

  plain = proc.stdout.decode('utf-8')

  # Parse 'plain' output
  scale_in = 72.0  # inches -> points (tweak if needed)
  graph_h_in = 0.0
  nodes = {}   # name -> {x,y,w,h,shape,label,description}
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
        ros_type = 'topic'
      elif shape_gv in ('ellipse', 'circle', 'oval'):
        gml_shape = 'ellipse'
        ros_type = 'node'
      else:
        gml_shape = 'roundrectangle'
        ros_type = 'unknown'
      informative_description = f"{ros_type}:{label}"
      nodes[name] = {'x': x, 'y': y, 'w': w, 'h': h, 'shape': gml_shape, 'label': label, 'description': informative_description}
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

  # Assign integer IDs for yFiles GraphML
  name_to_id = {name: idx for idx, name in enumerate(nodes.keys())}

  # Build yFiles GraphML text
  include_edge_bends = False  # keep False to allow dynamic routing in yEd

  out = []
  out.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
  out.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"')
  out.append('         xmlns:y="http://www.yworks.com/xml/graphml"')
  out.append('         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
  out.append('         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">')

  # Define keys for yEd properties
  out.append('  <key id="d_desc" for="node" attr.name="description" attr.type="string"/>')
  out.append('  <key id="d_nodegraphics" for="node" yfiles.type="nodegraphics"/>')
  out.append('  <key id="d_edgegraphics" for="edge" yfiles.type="edgegraphics"/>')

  out.append('  <graph id="G" edgedefault="directed">')

  # Nodes with fixed positions and correct labels
  for name, nid in name_to_id.items():
    nd = nodes[name]

    # Determine styles and target height based on shape type
    if nd["shape"] == 'ellipse':
      fill_color = "#ffff99"
      font_size = "16"
      font_style = "bold"
      target_h = 36.0
    elif nd["shape"] == 'rectangle':
      fill_color = "#ccccff"
      font_size = "12"
      font_style = "plain"
      target_h = 18.0
    else:
      # default / unknown
      fill_color = "#EEEEEE"
      font_size = "12"
      font_style = "plain"
      target_h = nd["h"]

    # yEd GraphML Geometry x,y represent the top-left corner, while Graphviz plain outputs the center.
    x_topleft = nd["x"] - nd["w"] / 2.0
    y_topleft = nd["y"] - target_h / 2.0

    out.append(f'    <node id="n{nid}">')
    # Use CDATA for description to safely handle arbitrary text
    out.append(f'      <data key="d_desc"><![CDATA[{nd["description"]}]]></data>')
    out.append('      <data key="d_nodegraphics">')
    out.append('        <y:ShapeNode>')
    # Apply the target height
    out.append(f'          <y:Geometry x="{x_topleft}" y="{y_topleft}" width="{nd["w"]}" height="{target_h}"/>')
    out.append(f'          <y:Fill color="{fill_color}" transparent="false"/>')
    out.append('          <y:BorderStyle type="line" width="1.0" color="#000000"/>')
    # Use CDATA for label, and add font styling attributes
    out.append(f'          <y:NodeLabel textColor="#000000" fontSize="{font_size}" fontStyle="{font_style}"><![CDATA[{nd["label"]}]]></y:NodeLabel>')
    out.append(f'          <y:Shape type="{nd["shape"]}"/>')
    out.append('        </y:ShapeNode>')
    out.append('      </data>')
    out.append('    </node>')

  # Edges
  edge_id = 0
  for tail, head, pts in edges:
    if tail not in name_to_id or head not in name_to_id:
      continue
    out.append(f'    <edge id="e{edge_id}" source="n{name_to_id[tail]}" target="n{name_to_id[head]}">')
    out.append('      <data key="d_edgegraphics">')
    out.append('        <y:PolyLineEdge>')
    out.append('          <y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0">')
    if include_edge_bends and pts:
      for (x, y) in pts:
        out.append(f'            <y:Point x="{x}" y="{y}"/>')
    out.append('          </y:Path>')
    out.append('          <y:LineStyle type="line" width="1.0" color="#000000"/>')
    out.append('          <y:Arrows source="none" target="standard"/>')
    out.append('        </y:PolyLineEdge>')
    out.append('      </data>')
    out.append('    </edge>')
    edge_id += 1

  out.append('  </graph>')
  out.append('</graphml>')

  # Write to output file
  try:
    with open(output_graphml, 'w', encoding='utf-8') as f:
      f.write('\n'.join(out))
    print(f'Success! Wrote to {output_graphml}')
  except Exception as e:
    print(f"Error writing to {output_graphml}: {e}")

if __name__ == '__main__':
  main()
