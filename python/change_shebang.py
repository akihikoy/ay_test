#!/usr/bin/python3
#\file    change_shebang.py
#\brief   Change Shebang of Python scripts.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.17, 2025
import os
import sys

def ChangeShebangOfScript(file_path):
  if os.path.isfile(file_path):
    try:
      with open(file_path, 'r', encoding='utf-8') as f:
        lines= f.readlines()
      modified= False
      if len(lines)>0:
        firstline= lines[0].strip()
        if lines and firstline.startswith('#!') and 'python2' in firstline:
          lines[0] = lines[0].replace('python2', 'python3')
          modified= True
        elif lines and firstline.startswith('#!') and firstline.endswith('python'):
          lines[0] = lines[0].replace('python', 'python3')
          modified= True
      if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
          f.writelines(lines)
        print(f'Changed shebang in: {file_path}')
    except Exception as e:
      print(f'Error processing {file_path}: {e}')
  else:
    print(f'Error: {file_path} is not a Python script.')

def ChangeShebangInDir(root_dir):
  for subdir, _, files in os.walk(root_dir):
    for file in files:
      if file.endswith('.py'):
        file_path= os.path.join(subdir, file)
        ChangeShebangOfScript(file_path)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: ./change_shebang.py </path/to/python-script>")
    print("Usage: ./change_shebang.py </path/to/project>")
    sys.exit(1)
  input_file= sys.argv[1]
  if os.path.isdir(input_file):
    ChangeShebangInDir(input_file)
  else:
    ChangeShebangOfScript(input_file)

