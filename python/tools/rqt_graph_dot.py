#!/usr/bin/python3
#\file    rqt_graph_dot.py
#\brief   Generate a DOT file of ROS node graph with optional services using rqt_graph.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.12, 2025
#\version 0.2
#\date    Feb.23, 2026

import sys
import argparse
import rosgraph.impl.graph
from rqt_graph.dotcode import RosGraphDotcodeGenerator
from qt_dotgraph.pydotfactory import PydotFactory

import pydot
import rosservice
from rosgraph.masterapi import Master

class ServiceAugmentedDotcodeGenerator(RosGraphDotcodeGenerator):
  """
  Wrapper around RosGraphDotcodeGenerator that augments the DOT
  with ROS service nodes and edges (provider_node -> service).
  """
  def generate_dotcode(self,
                       rosgraphinst=None,
                       ns_filter='/',
                       topic_filter='/',
                       graph_mode='node_topic_all',
                       dotcode_factory=None,
                       include_services=True,
                       hide_service_type=False,
                       **kwargs):

    # Generate the base DOT (nodes + topics or nodes only)
    base_dot = super(ServiceAugmentedDotcodeGenerator, self).generate_dotcode(
      rosgraphinst=rosgraphinst,
      ns_filter=ns_filter,
      topic_filter=topic_filter,
      graph_mode=graph_mode,
      dotcode_factory=dotcode_factory or PydotFactory(),
      **kwargs
    )

    if not include_services:
      return base_dot

    # Parse to pydot graph to allow appending service nodes/edges
    graphs = pydot.graph_from_dot_data(base_dot)
    if not graphs:
      return base_dot
    dot_graph = graphs[0]

    # Build a mapping from ROS node label to DOT node id for edge connection.
    label_to_id = {}

    def _canon(s):
      if s is None:
        return None
      s = str(s)
      if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == '<' and s[-1] == '>')):
        return s[1:-1]
      return s

    def _index_nodes(container):
      for n in container.get_nodes():
        name = _canon(n.get_name())
        label = _canon(n.get('label'))
        if name in ('graph', 'node', 'edge'):
          continue
        if name:
          label_to_id[name] = n.get_name()
        if label:
          label_to_id[label] = n.get_name()
      for sg in container.get_subgraphs():
        _index_nodes(sg)

    _index_nodes(dot_graph)

    def _find_node_id(ros_node_name):
      nid = None
      if ros_node_name in label_to_id:
        nid = label_to_id[ros_node_name]
      if ros_node_name and not ros_node_name.startswith('/'):
        cand = '/' + ros_node_name
        if cand in label_to_id:
          nid = label_to_id[cand]
      if ros_node_name and ros_node_name.startswith('/'):
        cand = ros_node_name.lstrip('/')
        if cand in label_to_id:
          nid = label_to_id[cand]
      if nid in ('graph', 'node', 'edge'):
        return None
      return nid

    # Query ROS Master for services and their providers
    try:
      master = Master('/rqt_graph_services')
      pubs, subs, srvs = master.getSystemState()
    except Exception as e:
      print('Failed to query ROS master for services:', e, file=sys.stderr)
      return base_dot

    hide_dyn = bool(kwargs.get('hide_dynamic_reconfigure', False))
    dyn_suffixes = (
      '/set_parameters', '/get_parameters', '/describe_parameters',
      '/get_parameter_types', '/get_loggers', '/set_logger_level'
    )

    ns_prefix = None
    if isinstance(ns_filter, str) and ns_filter not in ('', '/'):
      ns_prefix = ns_filter if ns_filter.startswith('/') else '/' + ns_filter

    for service_name, providers in srvs:
      if ns_prefix and not service_name.startswith(ns_prefix + '/'):
        continue
      if hide_dyn and any(service_name.endswith(suf) for suf in dyn_suffixes):
        continue

      try:
        if not hide_service_type:
          srv_type = rosservice.get_service_type(service_name)
        else:
          srv_type = None
      except Exception:
        srv_type = None

      provider_ids = []
      for prov in providers:
        pid = _find_node_id(prov)
        if pid:
          provider_ids.append(pid)

      if not provider_ids:
        continue

      srv_node_id = '"srv:{0}"'.format(service_name)
      srv_label = service_name if not srv_type else '{0}\\n[{1}]'.format(service_name, srv_type)

      srv_node = pydot.Node(
        srv_node_id,
        label=srv_label,
        shape='box',
        style='"rounded"',
        fontsize='10',
        penwidth='1.2'
      )
      dot_graph.add_node(srv_node)

      for pid in provider_ids:
        dot_graph.add_edge(pydot.Edge(pid, srv_node_id, label='srv'))

    return dot_graph.to_string()

def GenerateRosDot(graph_mode='node_topic_all', include_services=True):
  # Build graph snapshot from ROS master
  g = rosgraph.impl.graph.Graph()
  g.set_master_stale(5.0)
  g.set_node_stale(5.0)
  g.update()

  gen = ServiceAugmentedDotcodeGenerator()
  dot = gen.generate_dotcode(
    rosgraphinst=g,
    ns_filter='/',
    topic_filter='/',
    graph_mode=graph_mode,
    include_services=include_services,
    dotcode_factory=PydotFactory(),
    cluster_namespaces_level=5,
    group_image_nodes=True,
    group_tf_nodes=True,
    quiet=True,
    hide_tf_nodes=True,
    hide_single_connection_topics=False,
    hide_dead_end_topics=False,
    hide_dynamic_reconfigure=True,
    hide_service_type=True,
    accumulate_actions=True,
    orientation='LR',
    rank='same',
    simplify=True,
    unreachable=False,
  )
  return dot

def main():
  parser = argparse.ArgumentParser(description='Generate a DOT file of ROS node graph.')
  parser.add_argument('--graph_mode', type=str, default='node_topic_all', choices=['node_topic', 'node_topic_all', 'node_node'], help='Graph mode')
  parser.add_argument('--no-services', dest='services', action='store_false', help='Exclude ROS services (Included by default)')
  args = parser.parse_args()

  dot = GenerateRosDot(graph_mode=args.graph_mode, include_services=args.services)
  out_name = f'rosgraph-{args.graph_mode}' + ('-srv' if args.services else '') + '.dot'

  with open(out_name, 'w', encoding='utf-8') as f:
    f.write(dot)
  print(f'Wrote: {out_name}')

if __name__ == '__main__':
  main()
