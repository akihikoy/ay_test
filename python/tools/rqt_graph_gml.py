#!/usr/bin/python3
#\file    rqt_graph_gml.py
#\brief   Generate DOT and GraphML files of ROS node graph.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.13, 2025
#\version 0.2
#\date    Feb.23, 2026

import argparse
from rqt_graph_dot import GenerateRosDot
from dot2graphml import DotToGraphML

def main():
  parser = argparse.ArgumentParser(description='Generate DOT and GraphML files of ROS node graph.')
  parser.add_argument('--graph_mode', type=str, default='node_topic_all', choices=['node_topic', 'node_topic_all', 'node_node'], help='Graph mode')
  parser.add_argument('--no-services', dest='services', action='store_false', help='Exclude ROS services (Included by default)')
  args = parser.parse_args()

  print(f"Generating graph (Mode: {args.graph_mode}, Services: {args.services})...")

  # 1. Generate DOT
  dot_content = GenerateRosDot(graph_mode=args.graph_mode, include_services=args.services)
  if not dot_content:
    print("Failed to generate DOT graph.")
    return

  base_name = f'rosgraph-{args.graph_mode}' + ('-srv' if args.services else '')
  out_dot = f'{base_name}.dot'
  out_graphml = f'{base_name}.graphml'

  # Write DOT
  with open(out_dot, 'w', encoding='utf-8') as f:
    f.write(dot_content)
  print(f'Wrote: {out_dot}')

  # 2. Convert to GraphML
  graphml_content = DotToGraphML(dot_content)
  if graphml_content:
    with open(out_graphml, 'w', encoding='utf-8') as f:
      f.write(graphml_content)
    print(f'Wrote: {out_graphml}')

if __name__ == '__main__':
  main()
