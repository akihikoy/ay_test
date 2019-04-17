#!/usr/bin/python
import random

print random.random(),random.random(),random.random()

random.seed()
print random.random(),random.random(),random.random()

random.seed(3)
print random.random(),random.random(),random.random()

random.seed(3)
print random.random(),random.random(),random.random()
