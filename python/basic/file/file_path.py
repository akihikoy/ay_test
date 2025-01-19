#!/usr/bin/python3

print(__file__)
import sys,os
print(os.path.abspath(__file__))
print(os.path.basename(os.path.abspath(__file__)))
print(os.path.dirname(os.path.abspath(__file__)))
print(os.path.split(os.path.abspath(__file__)))
print(os.path.dirname(os.path.abspath('../../')))
print(os.path.abspath('../../'))
print(os.path.splitext('hoge/hoge/hehe.txt'))



