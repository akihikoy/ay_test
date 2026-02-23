# DEPRECATED


#!/usr/bin/python3
#\file    rqt_graph_srv_dot.py
#\brief   Generate a DOT file of ROS node graph including services with rqt_graph.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.13, 2025

import sys
import rosgraph.impl.graph
from rqt_graph.dotcode import RosGraphDotcodeGenerator
from qt_dotgraph.pydotfactory import PydotFactory

# pydot is used to post-process the DOT graph produced by rqt_graph
import pydot

# rosservice helps fetch service types (optional but useful for labels)
import rosservice
from rosgraph.masterapi import Master


class ServiceAugmentedDotcodeGenerator(RosGraphDotcodeGenerator):
  """
  Wrapper around RosGraphDotcodeGenerator that augments the DOT
  with ROS service nodes and edges (provider_node -> service).
  New graph_mode values:
    - 'node_topic_service'       -> base as 'node_topic' then add services
    - 'node_topic_service_all'   -> base as 'node_topic_all' then add services
    - 'node_node_service'        -> base as 'node_node' then add services
  """

  def generate_dotcode(self,
                       rosgraphinst=None,
                       ns_filter='/',
                       topic_filter='/',
                       graph_mode='node_topic_service_all',
                       dotcode_factory=None,
                       hide_service_type=False,
                       **kwargs):
    # Map extended modes to the original modes
    include_services = False
    base_graph_mode = graph_mode
    if graph_mode == 'node_topic_service':
      include_services = True
      base_graph_mode = 'node_topic'
    elif graph_mode == 'node_topic_service_all':
      include_services = True
      base_graph_mode = 'node_topic_all'
    elif graph_mode == 'node_node_service':
      include_services = True
      base_graph_mode = 'node_node'

    # Generate the base DOT (nodes + topics or nodes only)
    base_dot = super(ServiceAugmentedDotcodeGenerator, self).generate_dotcode(
      rosgraphinst=rosgraphinst,
      ns_filter=ns_filter,
      topic_filter=topic_filter,
      graph_mode=base_graph_mode,
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
    # We search recursively across subgraphs to index every existing node.
    label_to_id = {}

    def _canon(s):
      if s is None:
        return None
      s = str(s)
      # Strip surrounding quotes if present
      if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == '<' and s[-1] == '>')):
        return s[1:-1]
      return s

    # B) Ignore pseudo-nodes and guard edges against them.
    def _index_nodes(container):
      # container can be the main graph or a subgraph
      for n in container.get_nodes():
        name = _canon(n.get_name())
        label = _canon(n.get('label'))
        # Skip Graphviz default attribute pseudo-nodes
        if name in ('graph', 'node', 'edge'):
          continue
        if name:
          label_to_id[name] = n.get_name()
        if label:
          label_to_id[label] = n.get_name()
      for sg in container.get_subgraphs():
        _index_nodes(sg)

    _index_nodes(dot_graph)

    # Helper to find an existing DOT node id by ROS node name.
    def _find_node_id(ros_node_name):
      # try as-is
      nid = None
      if ros_node_name in label_to_id:
        nid = label_to_id[ros_node_name]
      # try with leading slash normalization
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
      pubs, subs, srvs = master.getSystemState()  # srvs: List[(service_name, [providers])]
    except Exception as e:
      print('Failed to query ROS master for services:', e, file=sys.stderr)
      return base_dot

    # Optionally hide some very chatty service names if requested
    hide_dyn = bool(kwargs.get('hide_dynamic_reconfigure', False))
    dyn_suffixes = (
      '/set_parameters', '/get_parameters', '/describe_parameters',
      '/get_parameter_types', '/get_loggers', '/set_logger_level'
    )

    # Normalize ns_filter to a prefix check
    ns_prefix = None
    if isinstance(ns_filter, str) and ns_filter not in ('', '/'):
      ns_prefix = ns_filter if ns_filter.startswith('/') else '/' + ns_filter

    # Collect and append service nodes + edges
    # We only add a service node if it connects to at least one visible provider node.
    for service_name, providers in srvs:
      if ns_prefix and not service_name.startswith(ns_prefix + '/'):
        continue
      if hide_dyn and any(service_name.endswith(suf) for suf in dyn_suffixes):
        continue

      # Resolve service type for label (best-effort)
      try:
        if not hide_service_type:
          srv_type = rosservice.get_service_type(service_name)
        else:
          srv_type = None
      except Exception:
        srv_type = None

      # Find providers that are present in the current DOT (respecting rqt_graph filters)
      provider_ids = []
      for prov in providers:
        pid = _find_node_id(prov)
        if pid:
          provider_ids.append(pid)

      # Skip if no provider is currently shown (prevents creating floating service nodes)
      if not provider_ids:
        continue

      # Construct a new DOT node for the service
      # Use a unique internal id (quoted) to avoid collisions with existing nodes
      srv_node_id = '"srv:{0}"'.format(service_name)
      srv_label = service_name if not srv_type else '{0}\\n[{1}]'.format(service_name, srv_type)

      srv_node = pydot.Node(
        srv_node_id,
        label=srv_label,
        shape='box',
        style='"rounded,dashed"',  # Quote the multi-style value to satisfy Graphviz
        fontsize='10',
        penwidth='1.2'
      )
      dot_graph.add_node(srv_node)

      # Connect provider_node -> service with a dashed edge
      for pid in provider_ids:
        dot_graph.add_edge(pydot.Edge(pid, srv_node_id, style='dashed', label='srv'))

    # Return the augmented DOT
    return dot_graph.to_string()


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
    hide_service_type=False,
    # Others keep default behavior of rqt_graph
    accumulate_actions=True,
    orientation='LR',
    rank='same',
    simplify=True,
    unreachable=False,
  )

  out_path = 'rosgraph_with_services.dot'
  with open(out_path, 'w', encoding='utf-8') as f:
    f.write(dot)
  print('Wrote', out_path)


if __name__ == '__main__':
  main()
