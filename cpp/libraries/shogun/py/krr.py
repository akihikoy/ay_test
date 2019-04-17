#!/usr/bin/python
# -*- coding: utf-8 -*-
from pylab import figure,plot,show
from numpy import array,transpose,sin,double

# In this example a kernelized version of ridge regression (KRR) is trained on a
# real-valued data set. The KRR is trained with regularization parameter tau=1e-6
# and a gaussian kernel with width=0.8.

def krr ():
        print 'KRR'

        size_cache=10
        width=2.1
        C=1.2
        tau=1e-6

        from sg import sg
        sg('set_features', 'TRAIN', fm_train)
        sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

        sg('set_labels', 'TRAIN', label_train)

        sg('new_regression', 'KRR')
        sg('krr_tau', tau)
        sg('c', C)
        sg('train_regression')

        sg('set_features', 'TEST', fm_test)
        result=sg('classify')
        return result

fm_train=array([range(0,100)],dtype=double)
fm_train.resize(1,100)
label_train=sin(fm_train).flatten()

fm_test=array([range(0,1000)],dtype=double)/10.0

result=krr()

fm_train.resize(100,1)
fm_test.resize(1000,1)
figure()
plot(fm_train,label_train,'b-')
plot(fm_train,label_train,'bo')
plot(fm_test,result,'r-')
#plot(fm_test,result,'ro')
show()
