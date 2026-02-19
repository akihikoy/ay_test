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
import collections

def find_package_xmls(root_dir):
    """指定ディレクトリ以下のすべてのpackage.xmlとそのパスを探す"""
    package_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'package.xml' in filenames:
            package_files.append(os.path.join(dirpath, 'package.xml'))
    return package_files

def parse_package_xml(file_path):
    """package.xmlをパースしてパッケージ名と依存リストを返す"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        name = root.find('name').text.strip()

        # 依存関係を表すタグ一覧
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
    """ファイルパスからリポジトリ名（親ディレクトリ構造）を推定する"""
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)
    if len(parts) > 1:
        # src/RepoName/PkgName -> RepoName を返す想定
        return os.path.dirname(os.path.dirname(rel_path))
    return "unknown"

def get_readme_summary(package_xml_path, limit=200):
    """package.xmlと同じ階層にあるREADME.mdを探し、冒頭を抽出する"""
    package_dir = os.path.dirname(package_xml_path)
    readme_path = os.path.join(package_dir, 'README.md')

    # ファイルがない場合は小文字も試す
    if not os.path.exists(readme_path):
        readme_path = os.path.join(package_dir, 'readme.md')
        if not os.path.exists(readme_path):
            return ""

    try:
        with open(readme_path, 'r', encoding='utf-8', errors='replace') as f:
            # 少し多めに読み込む
            content = f.read(limit * 2)

            # Markdownの改行や余分な空白をスペース1つに置換して1行にする
            # CSVでの視認性を高めるため
            content = " ".join(content.split())

            # 文字数制限でカット
            if len(content) > limit:
                return content[:limit] + "..."
            return content
    except Exception:
        return ""

def calculate_levels(packages_data):
    """トポロジカルソート的なロジックで移行レベルを決定する"""
    levels = {}
    pkg_names = set(packages_data.keys())

    for _ in range(100):
        updated = False
        for name, data in packages_data.items():
            if name in levels:
                continue

            internal_deps = data['internal_deps']

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
            levels[name] = 999 # 循環参照の可能性

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
            all_packages[info['name']] = info

    internal_pkg_names = set(all_packages.keys())

    # 情報の拡充
    for name, info in all_packages.items():
        info['internal_deps'] = [d for d in info['deps'] if d in internal_pkg_names]
        info['external_deps'] = [d for d in info['deps'] if d not in internal_pkg_names]
        info['repo'] = get_repo_name(info['path'], root_dir)
        # ここでREADMEを取得
        info['readme'] = get_readme_summary(info['path'])

    levels = calculate_levels(all_packages)

    writer = csv.writer(sys.stdout)
    # ヘッダーに 'README Summary' を追加
    header = ["Level", "Repository", "Package", "Internal Dependencies", "External Dependencies", "README Summary", "Path"]
    writer.writerow(header)

    sorted_packages = sorted(all_packages.values(), key=lambda x: (levels[x['name']], x['repo'], x['name']))

    for pkg in sorted_packages:
        writer.writerow([
            levels[pkg['name']],
            pkg['repo'],
            pkg['name'],
            " ".join(sorted(pkg['internal_deps'])),
            " ".join(sorted(pkg['external_deps'])),
            pkg['readme'], # 追加
            pkg['path']
        ])

if __name__ == "__main__":
    main()

