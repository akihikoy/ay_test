#!/usr/bin/python3

# Compute the convex hull of a set of 2D points
# A Python implementation of the qhull algorithm
#
# Tested with Python 2.6.5 on Ubuntu 10.04.4

# Copyright (c) 2008 Dave (www.literateprograms.org)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


from __future__ import print_function
from numpy import *
import sys

link = lambda a,b: concatenate((a,b[1:]))
edge = lambda a,b: concatenate(([a],[b]))
#link = lambda a, b: concatenate((a, b[1:]), axis=0)
#edge = lambda a, b: array([a, b])

def qhull2D(sample):
    EPS = 1.0e-14
    sample = asarray(sample)
    def dome(sample,base):
        h, t = base
        dists = dot(sample-h, dot(((0,-1),(1,0)),(t-h)))
        #outer = repeat(sample, dists>0, 0)
        #outer = sample[dists > 0]
        outer = sample[dists > EPS]
        if len(outer):
            pivot = sample[argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                    dome(outer, edge(pivot, t)))
        else:
            return base
    if len(sample) > 2:
        axis = sample[:,0]
        base = take(sample, [argmin(axis), argmax(axis)], 0)
        return link(dome(sample, base), dome(sample, base[::-1]))
    else:
        return sample
