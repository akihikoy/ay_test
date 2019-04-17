#!/usr/bin/python
from numpy import array,transpose,sin,double

features=array([range(0,100)],dtype=double)
features.resize(100,1)
labels=sin(features).flatten()

print features

print labels.transpose()

