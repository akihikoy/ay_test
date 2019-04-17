#!/usr/bin/python
# -*- coding: utf-8 -*-
from sg import sg
from pylab import figure,plot,show
from numpy import array,transpose,sin,double

sg('new_svm','LIBSVR')
features=array([range(0,100)],dtype=double)
features.resize(1,100)
labels=sin(features).flatten()
print features
print labels
print len(labels)
sg('set_features','TRAIN', features)
sg('set_labels','TRAIN', labels)
sg('set_kernel','GAUSSIAN','REAL',20,10.0)
sg('init_kernel','TRAIN')
#sg('c',1.0)
sg('c',10.0)
sg('svm_train')
sv=sg('get_svm');
features2=array([range(0,1000)],dtype=double)/10.0
sg('set_features','TEST', features2)
sg('init_kernel','TEST')
out=sg('svm_classify')

print out
features.resize(100,1)
features2.resize(1000,1)
print len(features)
print len(labels)
figure()
plot(features,labels,'b-')
plot(features,labels,'bo')
plot(features2,out,'r-')
plot(features2,out,'ro')
show()
