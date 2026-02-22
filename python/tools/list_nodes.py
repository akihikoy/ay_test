#!/usr/bin/python3
#\file    list_nodes.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.20, 2026

import rospy
import rosnode
import rosgraph
import rospkg
import csv
import subprocess
import os
import xmlrpc.client

OUTPUT_FILE = "ros_system_report.csv"

def get_repo_url(package_path):
  """
  Try to retrieve the git repository URL from the package path.
  """
  try:
    url = subprocess.check_output(
      ["git", "config", "--get", "remote.origin.url"],
      cwd=package_path, stderr=subprocess.DEVNULL
    ).decode('utf-8').strip()
    return url
  except Exception:
    return "-"

def get_package_from_path(file_path, packages):
  """
  Determine which ROS package contains the given file path.
  """
  best_match = "-"
  max_len = 0
  file_path = os.path.abspath(file_path)

  for pkg, pkg_path in packages.items():
    if file_path.startswith(pkg_path):
      # Find the most specific match (longest path)
      if len(pkg_path) > max_len:
        max_len = len(pkg_path)
        best_match = pkg

  return best_match

def generate_report():
  # Initialize node to communicate with ROS Master
  rospy.init_node('system_reporter', anonymous=True)
  master = rosgraph.Master('/system_reporter')

  node_names = rosnode.get_node_names()
  rp = rospkg.RosPack()

  # Build a dictionary of all packages and their absolute paths
  packages = {pkg: rp.get_path(pkg) for pkg in rp.list()}

  # Get System State (Pub/Sub/Srv lists)
  try:
    state = master.getSystemState()
    pub_state, sub_state, srv_state = state
  except Exception as e:
    print(f"Failed to get system state: {e}")
    return

  print(f"Generating report for {len(node_names)} nodes...")

  with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow([
      "Node Name", "Package", "Repository URL",
      "Publications", "Subscriptions", "Services Offered"
    ])

    for node in node_names:
      package_name = "-"
      repo_url = "-"

      try:
        # 1. Identify package by querying the node's PID and inspecting /proc
        uri = master.lookupNode(node)
        proxy = xmlrpc.client.ServerProxy(uri)
        code, msg, pid = proxy.getPid('/system_reporter')

        # Read the command line arguments used to launch the node
        try:
          with open(f'/proc/{pid}/cmdline', 'rb') as cmd_f:
            # cmdline items are null-byte separated
            cmdline = cmd_f.read().split(b'\x00')[:-1]
            cmd_args = [arg.decode('utf-8') for arg in cmdline]

            # Heuristic: Find the first absolute path in arguments
            for arg in cmd_args:
              if arg.startswith('/'):
                pkg = get_package_from_path(arg, packages)
                if pkg != "-":
                  package_name = pkg
                  repo_url = get_repo_url(packages[pkg])
                  break

                # Fallback for compiled nodes in /opt/ros/.../lib/<pkg>/<node>
                if "/lib/" in arg:
                  parts = arg.split("/lib/")
                  if len(parts) == 2:
                    potential_pkg = parts[1].split("/")[0]
                    if potential_pkg in packages:
                      package_name = potential_pkg
                      repo_url = get_repo_url(packages[potential_pkg])
                      break
        except Exception:
          pass # /proc not accessible or parsing failed
      except Exception:
        pass # XMLRPC call failed

      # 2. Extract Pub/Sub/Srv correctly
      # Ensure exact match by handling leading slashes
      clean_node = node if node.startswith('/') else '/' + node

      pubs = [topic for topic, nodes in pub_state if clean_node in nodes]
      subs = [topic for topic, nodes in sub_state if clean_node in nodes]
      srvs = [srv for srv, nodes in srv_state if clean_node in nodes]

      # Use comma + space for joining to avoid CSV formatting issues
      writer.writerow([
        node,
        package_name,
        repo_url,
        ", ".join(pubs),
        ", ".join(subs),
        ", ".join(srvs)
      ])

  print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == '__main__':
  generate_report()


