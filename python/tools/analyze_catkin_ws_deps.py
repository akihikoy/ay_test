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

        # 依存関係を表すタグ一覧 (ROS1/ROS2両対応)
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
    # root_dir からの相対パスを取得
    rel_path = os.path.relpath(file_path, root_dir)
    # ディレクトリ構成に応じて調整（例: src/RepoName/PkgName -> RepoName）
    parts = rel_path.split(os.sep)
    if len(parts) > 1:
        # src/ay_tools/ay_common/package.xml のような構造を想定し、
        # ay_tools/ay_common のようなリポジトリ識別子を返す
        return os.path.dirname(os.path.dirname(rel_path))
    return "unknown"

def calculate_levels(packages_data):
    """トポロジカルソートに似たロジックで移行レベルを決定する"""
    levels = {}
    pkg_names = set(packages_data.keys())

    # 依存解決ループ（最大100回でループ検出として停止）
    for _ in range(100):
        updated = False
        for name, data in packages_data.items():
            if name in levels:
                continue

            internal_deps = data['internal_deps']

            # 社内依存がない場合はレベル0
            if not internal_deps:
                levels[name] = 0
                updated = True
                continue

            # 社内依存がすべてレベル確定済みか確認
            dep_levels = [levels.get(d) for d in internal_deps if d in pkg_names]

            # まだレベルが決まっていない依存先がある場合はスキップ
            if None in dep_levels:
                continue

            # すべての依存先の最大レベル + 1
            levels[name] = max(dep_levels) + 1
            updated = True

        if not updated:
            break

    # 循環参照などで解決できなかったものは Level 999 とする
    for name in packages_data:
        if name not in levels:
            levels[name] = 999

    return levels

def main():
    # 探索するルートディレクトリ（カレントディレクトリまたは引数）
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Scanning directory: {os.path.abspath(root_dir)}", file=sys.stderr)

    xml_files = find_package_xmls(root_dir)
    print(f"Found {len(xml_files)} packages.", file=sys.stderr)

    # 全パッケージ情報の収集
    all_packages = {}
    for f in xml_files:
        info = parse_package_xml(f)
        if info:
            all_packages[info['name']] = info

    # 社内パッケージ名のセット（これに含まれる依存のみInternalとみなす）
    internal_pkg_names = set(all_packages.keys())

    # Internal/Externalの分類
    for name, info in all_packages.items():
        info['internal_deps'] = [d for d in info['deps'] if d in internal_pkg_names]
        info['external_deps'] = [d for d in info['deps'] if d not in internal_pkg_names]
        # リポジトリ名の推定
        info['repo'] = get_repo_name(info['path'], root_dir)

    # 移行レベルの計算
    levels = calculate_levels(all_packages)

    # CSV出力
    writer = csv.writer(sys.stdout)
    header = ["Level", "Repository", "Package", "Internal Dependencies", "External Dependencies", "Path"]
    writer.writerow(header)

    # レベル順、次にリポジトリ順でソートして出力
    sorted_packages = sorted(all_packages.values(), key=lambda x: (levels[x['name']], x['repo'], x['name']))

    for pkg in sorted_packages:
        writer.writerow([
            levels[pkg['name']],
            pkg['repo'],
            pkg['name'],
            " ".join(sorted(pkg['internal_deps'])), # スペース区切り
            " ".join(sorted(pkg['external_deps'])), # スペース区切り
            pkg['path']
        ])

if __name__ == "__main__":
    main()
