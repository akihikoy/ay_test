# DEPRECATED

#!/usr/bin/python3
#\file    rqt_graph_srv_gml.py
#\brief   Generate a GML file of ROS node graph including services with rqt_graph.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.13, 2025

from rqt_graph_srv_dot import *
from rqt_graph_gml import *

def main():
  # Build graph snapshot from ROS master
  g = rosgraph.impl.graph.Graph()
  g.set_master_stale(5.0)
  g.set_node_stale(5.0)
  g.update()

  # Use the augmented generator
  gen = ServiceAugmentedDotcodeGenerator()
  dot = gen.generate_dotcode(
    rosgraphinst=g,
    ns_filter='/',                 # include all
    topic_filter='/',              # include all
    graph_mode='node_topic_service_all',  # NEW: include services
    dotcode_factory=PydotFactory(),
    # Group (kept same as rqt_graph defaults)
    cluster_namespaces_level=5,
    group_image_nodes=True,
    group_tf_nodes=True,
    # Hide
    quiet=True,                    # Hide Debug (rviz, rosout, etc.)
    hide_tf_nodes=True,
    hide_single_connection_topics=False,
    hide_dead_end_topics=False,
    hide_dynamic_reconfigure=True,
    hide_service_type=True,
    # Others keep default behavior of rqt_graph
    accumulate_actions=True,
    orientation='LR',
    rank='same',
    simplify=True,
    unreachable=False,
  )

  out = DotToGML(dot)

  out_path = 'rosgraph_with_services.gml'
  with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
  print('Wrote', out_path)


if __name__ == '__main__':
  main()

