#!/bin/bash
#\file    compile.sh
#\brief   certain bash script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.22, 2016

g++ -Wall -O2 $@ cma_es/cmaes.c cma_es/boundary_transformation.c -lm
