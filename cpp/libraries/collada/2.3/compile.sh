#!/bin/bash -x

CDOM=/home/akihiko/prg/src/collada/collada-dom-2.3
g++ -Wall -g $@ -I${CDOM}/dom/include -I${CDOM}/dom/include/1.4 ${CDOM}/build/dom/src/1.4/libcollada14dom.so -Wl,-rpath ${CDOM}/build/dom/src/1.4 -lxml2

# g++ -Wall -g $@ -I/usr/include/collada-dom2.4 -lcollada-dom2.4-dp

# g++ -Wall -g $@ -I/usr/local/include/collada-dom2.4 -L/usr/loca/lib -lcollada-dom2.4-dp -lboost_system

# g++ -Wall -g $@ `pkg-config collada-dom-141 --cflags` `pkg-config collada-dom-141 --libs` -lboost_system
