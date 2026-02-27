#!/usr/bin/python3
#\file    extract_help_descriptions.py
#\brief   Scan Python scripts for a Help() function and extract the description before 'Usage:'.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.24, 2026

import os
import re
import csv
import argparse

OUTPUT_FILE = "help_descriptions.csv"

def get_fvm_relative_path(abs_dir):
  """
  Extract the path starting from 'fvm'.
  If 'fvm' is not in the path, returns the original normalized path.
  """
  # Normalize path separators for cross-platform compatibility
  normalized_path = abs_dir.replace('\\', '/')
  parts = normalized_path.split('/')

  if 'fvm' in parts:
    idx = parts.index('fvm')
    return '/'.join(parts[idx:])

  return normalized_path

def extract_description(filepath):
  """
  Extract the brief description from the Help() function.
  Looks for text before 'Usage:' and cleans up formatting (newlines, indents).
  """
  try:
    with open(filepath, 'r', encoding='utf-8') as f:
      content = f.read()
  except (UnicodeDecodeError, OSError):
    return None

  # Find the Help function callable with no arguments
  # It captures everything until the next function definition, the main block, or end of file
  help_match = re.search(
    r'def\s+Help\s*\(\s*\)\s*:([\s\S]*?)(?:def\s+[a-zA-Z_]\w*\s*\(|if\s+__name__\s*==\s*[\'"]__main__[\'"]|\Z)',
    content
  )

  if not help_match:
    return None

  help_body = help_match.group(1)

  # Find everything before "Usage.*:" (case-insensitive) to catch Usage-1:, etc.
  usage_match = re.search(r'([\s\S]*?)\bUsage.*:', help_body, re.IGNORECASE)

  if usage_match:
    raw_desc = usage_match.group(1)
  else:
    # Fallback: If no "Usage" is found, use the entire Help function body
    raw_desc = help_body

  # Clean up the raw description
  # 1. Remove common python syntax like return, print(), sys.stdout.write(), or variable assignments
  cleaned = re.sub(r'\b(?:return|print|sys\.stdout\.write)\b\s*\(?', ' ', raw_desc)
  cleaned = re.sub(r'^[a-zA-Z_]\w*\s*=\s*', ' ', cleaned, flags=re.MULTILINE)

  # 2. Remove all types of quotes
  cleaned = re.sub(r'[\'"]', ' ', cleaned)

  # 3. Replace literal '\n' strings (e.g. from print("...\n")) with space
  cleaned = re.sub(r'\\n', ' ', cleaned)

  # 4. Collapse all whitespace (actual newlines, tabs, indents, multiple spaces) into a single space
  cleaned = " ".join(cleaned.split())

  # 5. Remove rogue 'f' or 'r' prefixes from f-strings/raw strings that might be left behind
  cleaned = re.sub(r'^[frFR]\s+', '', cleaned)

  return cleaned.strip()

def main():
  parser = argparse.ArgumentParser(description="Scan Python scripts for Help() descriptions.")
  parser.add_argument(
    "start_dir",
    nargs="?",
    default="~/catkin_ws/src_noros/fvm/ppx2_donburi",
    help="Directory to start scanning (default: ~/catkin_ws/src_noros/fvm/ppx2_donburi)"
  )
  parser.add_argument("-o", "--output", default=OUTPUT_FILE, help="Output CSV filename")
  args = parser.parse_args()

  # Expand '~' to the user's home directory, then get the absolute path
  start_dir = os.path.abspath(os.path.expanduser(args.start_dir))
  print(f"Scanning directory: {start_dir}")

  results = []

  for root, dirs, files in os.walk(start_dir):
    # Skip hidden directories like .git to speed up scanning
    dirs[:] = [d for d in dirs if not d.startswith('.')]

    for file in files:
      filepath = os.path.join(root, file)

      # Skip broken symlinks or inaccessible files
      if not os.path.exists(filepath):
        continue

      # Determine if the file is a Python script (either by extension or shebang)
      is_python = file.endswith('.py')
      if not is_python:
        try:
          with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line.startswith('#!') and 'python' in first_line:
              is_python = True
        except Exception:
          pass

      if not is_python:
        continue

      desc = extract_description(filepath)

      # If a description was successfully extracted, add it to results
      if desc:
        rel_dir = get_fvm_relative_path(root)
        results.append({
          'Directory': rel_dir,
          'Script': file,
          'Description': desc
        })

  print(f"Found {len(results)} scripts with Help() descriptions.")

  # Sort the results by Directory, then by Script name
  results.sort(key=lambda x: (x['Directory'], x['Script']))

  # Write output to CSV
  with open(args.output, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['Directory', 'Script', 'Description']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
      writer.writerow(row)

  print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
  main()
