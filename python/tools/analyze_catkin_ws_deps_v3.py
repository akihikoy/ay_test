#!/usr/bin/python3
#\file    analyze_catkin_ws_deps.py
#\brief   Analyze catkin_ws dependencies.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.06, 2026

import os
import xml.etree.ElementTree as ET
import csv
import sys
import re

def find_package_xmls(root_dir):
    """指定ディレクトリ以下のすべてのpackage.xmlとそのパスを探す"""
    package_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'package.xml' in filenames:
            package_files.append(os.path.join(dirpath, 'package.xml'))
    return package_files

def parse_package_xml(file_path):
    """package.xmlをパースしてパッケージ名と静的な依存リストを返す"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        name = root.find('name').text.strip()

        dep_tags = [
            'depend', 'build_depend', 'build_export_depend',
            'run_depend', 'exec_depend', 'test_depend'
        ]

        dependencies = set()
        for tag in dep_tags:
            for node in root.findall(tag):
                if node.text:
                    dependencies.add(node.text.strip())

        return {
            'name': name,
            'path': file_path,
            'deps': dependencies
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        return None

def get_repo_name(file_path, root_dir):
    """ファイルパスからリポジトリ名を推定"""
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)
    if len(parts) > 1:
        return os.path.dirname(os.path.dirname(rel_path))
    return "unknown"

def get_readme_summary(package_xml_path, limit=200):
    """READMEの要約を取得"""
    package_dir = os.path.dirname(package_xml_path)
    readme_path = os.path.join(package_dir, 'README.md')
    if not os.path.exists(readme_path):
        readme_path = os.path.join(package_dir, 'readme.md')
        if not os.path.exists(readme_path):
            return ""

    try:
        with open(readme_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(limit * 2)
            content = " ".join(content.split())
            if len(content) > limit:
                return content[:limit] + "..."
            return content
    except Exception:
        return ""

def scan_for_load_manifest(package_xml_path):
    """
    パッケージディレクトリ内のPythonファイルをスキャンし、
    roslib.load_manifest() で参照されているパッケージを抽出する
    """
    package_dir = os.path.dirname(package_xml_path)
    dynamic_deps = set()

    # 正規表現: roslib.load_manifest('package_name')
    pattern = re.compile(r"roslib\.load_manifest\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")

    for root, dirs, files in os.walk(package_dir):
        # .git などの隠しディレクトリはスキップ
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in files:
            file_path = os.path.join(root, filename)

            # 対象ファイルの判定
            is_python_script = False
            if filename.endswith('.py'):
                is_python_script = True
            elif os.access(file_path, os.X_OK) or '.' not in filename:
                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        first_line = f.readline()
                        if 'python' in first_line:
                            is_python_script = True
                except:
                    pass

            if is_python_script:
                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        matches = pattern.findall(content)
                        for m in matches:
                            dynamic_deps.add(m)
                except Exception as e:
                    pass

    return dynamic_deps

def calculate_levels(packages_data):
    """移行レベルの計算（package.xmlの依存関係を基準とする）"""
    levels = {}
    pkg_names = set(packages_data.keys())

    for _ in range(100):
        updated = False
        for name, data in packages_data.items():
            if name in levels:
                continue

            # 内部依存（XML記述分）のみを順序決定に使用
            internal_deps = data['internal_xml']

            if not internal_deps:
                levels[name] = 0
                updated = True
                continue

            dep_levels = [levels.get(d) for d in internal_deps if d in pkg_names]

            if None in dep_levels:
                continue

            levels[name] = max(dep_levels) + 1
            updated = True

        if not updated:
            break

    for name in packages_data:
        if name not in levels:
            levels[name] = 999

    return levels

def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Scanning directory: {os.path.abspath(root_dir)}", file=sys.stderr)

    xml_files = find_package_xmls(root_dir)
    print(f"Found {len(xml_files)} packages.", file=sys.stderr)

    all_packages = {}
    for f in xml_files:
        info = parse_package_xml(f)
        if info:
            # 動的依存関係のスキャンを実行
            info['dynamic_deps'] = scan_for_load_manifest(f)
            all_packages[info['name']] = info

    internal_pkg_names = set(all_packages.keys())

    # 依存関係の分類（4象限）
    for name, info in all_packages.items():
        xml_deps = info['deps']
        dyn_deps = info['dynamic_deps']

        # 1. Internal & XML (社内パケ, package.xml記載あり)
        info['internal_xml'] = [d for d in xml_deps if d in internal_pkg_names]
        # 2. Internal & Dynamic (社内パケ, load_manifest記載あり)
        info['internal_dyn'] = [d for d in dyn_deps if d in internal_pkg_names]

        # 3. External & XML (社外パケ, package.xml記載あり)
        info['external_xml'] = [d for d in xml_deps if d not in internal_pkg_names]
        # 4. External & Dynamic (社外パケ, load_manifest記載あり)
        info['external_dyn'] = [d for d in dyn_deps if d not in internal_pkg_names]

        info['repo'] = get_repo_name(info['path'], root_dir)
        info['readme'] = get_readme_summary(info['path'])

    levels = calculate_levels(all_packages)

    writer = csv.writer(sys.stdout)
    header = [
        "Level", "Repository", "Package",
        "Internal Deps (XML)", "Internal Deps (Dynamic)",
        "External Deps (XML)", "External Deps (Dynamic)",
        "README Summary", "Path"
    ]
    writer.writerow(header)

    sorted_packages = sorted(all_packages.values(), key=lambda x: (levels[x['name']], x['repo'], x['name']))

    for pkg in sorted_packages:
        writer.writerow([
            levels[pkg['name']],
            pkg['repo'],
            pkg['name'],
            " ".join(sorted(pkg['internal_xml'])),
            " ".join(sorted(pkg['internal_dyn'])),
            " ".join(sorted(pkg['external_xml'])),
            " ".join(sorted(pkg['external_dyn'])),
            pkg['readme'],
            pkg['path']
        ])

if __name__ == "__main__":
    main()
