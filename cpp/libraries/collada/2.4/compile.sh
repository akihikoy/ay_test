#!/bin/bash -x

CDOM=/home/akihiko/prg/src/collada/collada-dom-2.4.0
g++ -Wall -g daeTinyXMLPlugin.cpp -c -DDOM_INCLUDE_TINYXML -I${CDOM}/dom/include -I${CDOM}/dom/include/1.4
g++ -Wall -g $@ daeTinyXMLPlugin.o -DDOM_INCLUDE_TINYXML -DCOLLADA_DOM_SUPPORT141 -DCOLLADA_DOM_SUPPORT150 -DCOLLADA_DOM_DAEFLOAT_IS64 -DCOLLADA_DOM_USING_141 -I${CDOM}/dom/include -I${CDOM}/dom/include/1.4 ${CDOM}/build/dom/libcollada-dom2.4-dp.so -Wl,-rpath ${CDOM}/build/dom -lboost_system -ltinyxml

# g++ -Wall -g $@ -I/usr/include/collada-dom2.4 -lcollada-dom2.4-dp

# g++ -Wall -g $@ -I/usr/local/include/collada-dom2.4 -L/usr/loca/lib -lcollada-dom2.4-dp -lboost_system

# g++ -Wall -g $@ `pkg-config collada-dom-141 --cflags` `pkg-config collada-dom-141 --libs` -lboost_system
