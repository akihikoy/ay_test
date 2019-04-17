#!/bin/bash -x

# sudo apt-get -f install libcoin60-dev libcoin60
# sudo apt-get -f install libsoqt4-dev libsoqt4-20

x++ $@ -- -I/usr/include/qt4 -I/usr/include/qt4/Qt -lCoin -lSoQt -lboost_thread
