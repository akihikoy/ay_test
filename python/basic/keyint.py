#!/usr/bin/python3
# -*- coding: utf-8 -*-
import time

while True:
  try:
    print("a")
    time.sleep(0.1)
  except KeyboardInterrupt:
    print("b")
    break

print("c")
