#!/usr/bin/python3
#\file    rqt_graph_dot.py
#\brief   Generate a DOT file of ROS node graph with rqt_graph.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.12, 2025

import rosgraph.impl.graph
from rqt_graph.dotcode import RosGraphDotcodeGenerator
from qt_dotgraph.pydotfactory import PydotFactory

def main():
  # Build graph snapshot from ROS master
  g = rosgraph.impl.graph.Graph()
  g.set_master_stale(5.0)
  g.set_node_stale(5.0)
  g.update()

  # Use the same generator as rqt_graph
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

  with open('rosgraph.dot', 'w', encoding='utf-8') as f:
    f.write(dot)
  print('Wrote rosgraph.dot')

if __name__ == '__main__':
  main()
