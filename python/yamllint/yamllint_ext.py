#!/usr/bin/python3
#\file    yamllint_ext.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.07, 2025
from yamllint import linter
from yamllint.config import YamlLintConfig
import yaml
import re
import sys

# Speedup YAML using CLoader/CDumper
try:
  from yaml import CLoader as YLoader
except ImportError:
  from yaml import Loader as YLoader


class LineList(list):
  def __init__(self, *args, line_num=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.__line__ = line_num


def is_mixed_types_invalid(types):
  numeric_types = {int, float, type(None)}
  str_types = {str, type(None)}

  if types.issubset(numeric_types) or types.issubset(str_types):
    return False

  if int in types and str in types:
    return True

  if float in types and str in types:
    return True

  return False


def check_list_uniformity(lst, path='root', line_num=None):
  types = {type(x) for x in lst}
  line_info = f": L.{line_num}" if line_num else ""

  if is_mixed_types_invalid(types):
    print(f"WARNING{line_info}: List '{path}' has irregular mixed types: {types}")

  for i, val in enumerate(lst):
    if isinstance(val, str) and re.match(r'^-?\d+(\.\d+)?\s+-?\d+(\.\d+)?$', val):
      print(f"WARNING{line_info}: Possible missing comma in '{path}[{i}]': '{val}'")


def custom_yaml_check(yaml_content, config_file):
  config = YamlLintConfig(file=config_file)

  # Run yamllint
  problems = list(linter.run(yaml_content, config))
  has_syntax_error = False
  for problem in problems:
    print(f"{problem.level.upper()}: L.{problem}")
    if problem.rule == 'syntax':
      has_syntax_error = True

  if has_syntax_error:
    print("Error: YAML syntax error detected. Stopping further checks.")
    return

  # Load YAML with PyYAML and track line numbers
  class LineLoader(YLoader):
    pass

  def construct_mapping(loader, node, deep=False):
    mapping = loader.construct_mapping(node, deep=deep)
    mapping['__line__'] = node.start_mark.line + 1
    return mapping

  def construct_sequence(loader, node, deep=False):
    seq = loader.construct_sequence(node, deep=deep)
    return LineList(seq, line_num=node.start_mark.line + 1)

  LineLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
  LineLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, construct_sequence)

  try:
    data = yaml.load(yaml_content, Loader=LineLoader)
  except yaml.YAMLError as exc:
    print(f"YAML parsing error: {exc}")
    return

  # Recursive check for lists with line numbers
  def recursive_check(data, path='root', parent_line=None):
    line_num = getattr(data, '__line__', parent_line)
    if isinstance(data, list):
      check_list_uniformity(data, path, line_num)
    elif isinstance(data, dict):
      for key, value in data.items():
        if key != '__line__':
          recursive_check(value, f"{path}.{key}", line_num)

  recursive_check(data)


if __name__ == '__main__':
  input_file = sys.argv[1]
  config_file = 'yamllint.yaml'

  wrong_directive = '%YAML:1.0'
  with open(input_file) as fp:
    text = fp.read()
    if text.startswith(wrong_directive):
      text = text[len(wrong_directive):]

    custom_yaml_check(text, config_file)
