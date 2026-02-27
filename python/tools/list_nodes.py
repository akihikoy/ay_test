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
import urllib.parse

OUTPUT_FILE = "ros_system_report.csv"

# Mapping of remote IPs to SSH usernames
REMOTE_USERS = {
  '10.10.6.205': 'jetson',
  # Add other IPs if necessary
}


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
  Tolerates different workspace base paths (e.g., local vs remote over SSH).
  """
  parts = file_path.split('/')

  # 1. Fallback for compiled nodes in /devel/lib/<pkg>/ or /opt/ros/*/lib/<pkg>/
  if 'lib' in parts:
    idx = parts.index('lib')
    if idx + 1 < len(parts):
      pkg = parts[idx + 1]
      if pkg in packages:
        return pkg

  # 2. Check for script files inside packages (e.g., /src/my_pkg/scripts/...)
  # Search from the deepest directory upwards to find the package name
  for part in reversed(parts[:-1]):
    if part in packages:
      return part

  return "-"

def generate_report():
  # Initialize node to communicate with ROS Master
  rospy.init_node('system_reporter', anonymous=True)
  master = rosgraph.Master('/system_reporter')

  node_names = rosnode.get_node_names()

  # Sort node names alphabetically to ensure consistent output for diffs
  node_names.sort()

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
      "Node Name", "Package", "Repository URL", "Executable",
      "Publications", "Subscriptions", "Services Offered"
    ])

    for node in node_names:
      package_name = "-"
      repo_url = "-"
      executable_name = "-"

      try:
        # 1. Identify package by querying the node's PID and inspecting /proc
        uri = master.lookupNode(node)
        proxy = xmlrpc.client.ServerProxy(uri)
        code, msg, pid = proxy.getPid('/system_reporter')

        # Read the command line arguments used to launch the node
        try:
          hostname = urllib.parse.urlparse(uri).hostname
          cmdline_bytes = b''

          if hostname in REMOTE_USERS:
            # Fetch process info via SSH for remote nodes
            ssh_user = REMOTE_USERS[hostname]
            cmd = ["ssh", f"{ssh_user}@{hostname}", f"cat /proc/{pid}/cmdline"]
            cmdline_bytes = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
          else:
            # Fetch process info locally
            if os.path.exists(f'/proc/{pid}/cmdline'):
              with open(f'/proc/{pid}/cmdline', 'rb') as cmd_f:
                cmdline_bytes = cmd_f.read()

          if cmdline_bytes:
            # cmdline items are null-byte separated
            cmdline = cmdline_bytes.split(b'\x00')[:-1]
            cmd_args = [arg.decode('utf-8') for arg in cmdline]

            # Extract executable name
            if cmd_args:
              exec_path = cmd_args[0]
              if 'python' in os.path.basename(exec_path).lower() and len(cmd_args) > 1:
                exec_path = cmd_args[1]
              executable_name = os.path.basename(exec_path)

            # Find the package name using the execution path
            for arg in cmd_args:
              if arg.startswith('/'):
                pkg = get_package_from_path(arg, packages)
                if pkg != "-":
                  package_name = pkg
                  repo_url = get_repo_url(packages[pkg])
                  break
        except Exception:
          pass # /proc not accessible, SSH failed, or parsing failed
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
        executable_name,
        ", ".join(pubs),
        ", ".join(subs),
        ", ".join(srvs)
      ])

  print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == '__main__':
  generate_report()
