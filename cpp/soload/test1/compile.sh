#!/bin/bash -x

g++ -rdynamic -g -Wall libmain.cpp -o libmain.o -c && \
g++ -rdynamic -g -Wall -shared -o libmain.so libmain.o && \

g++ -rdynamic -g -Wall -o test.out test.cpp -ldl -Wl,-rpath ./
