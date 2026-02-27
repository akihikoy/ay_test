#!/usr/bin/python3
#\file    extract_executables.py
#\brief   Scan catkin_ws for ROS C++ and Python executables and extract their descriptions.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.23, 2026

import os
import re
import csv
import sys
import subprocess
import argparse

OUTPUT_FILE = "executables_descriptions.csv"

def get_repo_info(filepath):
  """
  Find the Git repository containing the file.
  Returns the remote URL or the repository directory name as a fallback.
  """
  current = os.path.dirname(os.path.abspath(filepath))
  while current != '/':
    if os.path.isdir(os.path.join(current, '.git')):
      try:
        url = subprocess.check_output(
          ['git', 'config', '--get', 'remote.origin.url'],
          cwd=current, stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        if url:
          return url
      except Exception:
        pass
      return os.path.basename(current)
    current = os.path.dirname(current)
  return "Unknown"

def get_package_info(filepath):
  """
  Find the ROS package containing the file by looking for package.xml.
  Returns the package directory name.
  """
  current = os.path.dirname(os.path.abspath(filepath))
  while current != '/':
    if os.path.isfile(os.path.join(current, 'package.xml')):
      return os.path.basename(current)
    current = os.path.dirname(current)
  return "Unknown"

def extract_descriptions(content, is_python):
  """
  Extract descriptions from the source code content.
  Looks for \\brief, @brief, docstrings, and top-level block comments.
  """
  descriptions = []

  # 1. Extract \brief or @brief (works for both C++ and Python comments)
  briefs = re.findall(r'(?:\\|@)brief\s+(.*)', content, re.IGNORECASE)
  descriptions.extend([b.strip() for b in briefs if b.strip()])

  ## 2. Extract \file or @file for additional context
  #files = re.findall(r'(?:\\|@)file\s+(.*)', content, re.IGNORECASE)
  #descriptions.extend([f"File: {f.strip()}" for f in files if f.strip()])

  # 3. Python specific fallbacks (Module Docstrings)
  if is_python:
    pass
    #doc_match = re.search(r'(?:^#!.*?\n)?\s*([\'"]{3})([\s\S]*?)\1', content)
    #if doc_match:
      ## Clean up newlines and extra spaces
      #doc_text = " ".join(doc_match.group(2).split())
      #descriptions.append(f"Docstring: {doc_text}")

  # 4. C++ specific fallbacks (Top-level block comments /* ... */)
  else:
    pass
    #if not briefs:
      #block_match = re.search(r'^\s*/\*\*?([\s\S]*?)\*/', content)
      #if block_match:
        #block_text = block_match.group(1)
        ## Remove leading asterisks from multiline comments
        #block_text = re.sub(r'^[ \t]*\*[ \t]*', '', block_text, flags=re.MULTILINE)
        #block_text = " ".join(block_text.split())
        #if block_text:
          #descriptions.append(f"Top Comment: {block_text}")

  # Remove duplicates but preserve the extraction order
  unique_desc = []
  for d in descriptions:
    if d not in unique_desc:
      unique_desc.append(d)

  if not unique_desc:
    return "No description found."

  return " | ".join(unique_desc)

def main():
  parser = argparse.ArgumentParser(description="Scan catkin_ws for ROS executables and extract descriptions.")
  parser.add_argument("ws_path", nargs="?", default=".", help="Path to catkin_ws (default: current directory)")
  parser.add_argument("-o", "--output", default=OUTPUT_FILE, help="Output CSV filename")
  args = parser.parse_args()

  ws_path = os.path.abspath(args.ws_path)
  print(f"Scanning workspace: {ws_path}")

  results = []

  for root, dirs, files in os.walk(ws_path):
    # Skip hidden directories like .git or .catkin_tools to speed up scanning
    dirs[:] = [d for d in dirs if not d.startswith('.')]

    for file in files:
      filepath = os.path.join(root, file)

      # Skip broken symlinks or inaccessible files
      if not os.path.exists(filepath):
        continue

      # Determine file type
      is_python = file.endswith('.py') or file.endswith('.pyw')
      is_cpp = file.endswith('.cpp') or file.endswith('.cc') or file.endswith('.cxx')

      # Fallback: Check shebang for Python scripts without a .py extension
      if not (is_python or is_cpp):
        try:
          with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line.startswith('#!') and 'python' in first_line:
              is_python = True
        except Exception:
          pass

      if not (is_python or is_cpp):
        continue

      try:
        with open(filepath, 'r', encoding='utf-8') as f:
          content = f.read()
      except (UnicodeDecodeError, OSError):
        continue  # Skip binary files, broken symlinks, or permission errors

      is_executable = False

      # Check for executable signatures (main function / __main__ block)
      if is_cpp:
        if re.search(r'\bint\s+main\s*\(', content):
          is_executable = True
      elif is_python:
        if re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', content):
          is_executable = True

      if is_executable:
        desc = extract_descriptions(content, is_python)
        repo = get_repo_info(filepath)
        pkg = get_package_info(filepath)
        rel_dir = os.path.relpath(root, ws_path)

        results.append({
          'Directory': rel_dir,
          'Repository': repo,
          'Package': pkg,
          'Executable': file,
          'Description': desc
        })

  # Write output to CSV
  print(f"Found {len(results)} executables. Writing to {args.output}...")
  with open(args.output, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['Directory', 'Repository', 'Package', 'Executable', 'Description']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
      writer.writerow(row)

  print("Done.")

if __name__ == "__main__":
  main()
