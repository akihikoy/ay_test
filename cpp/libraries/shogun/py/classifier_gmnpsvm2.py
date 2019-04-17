#!/usr/bin/python
# -*- coding: utf-8 -*-

# In this example a multi-class support vector machine is trained on a toy data
# set and the trained classifier is used to predict labels of test examples.
# The training algorithm is based on BSVM formulation (L2-soft margin
# and the bias added to the objective function) which is solved by the Improved
# Mitchell-Demyanov-Malozemov algorithm. The training algorithm uses the Gaussian
# kernel of width 2.1 and the regularization constant C=1.2. The bias term of the
# classification rule is not used. The solver stops if the relative duality gap
# falls below 1e-5 and it uses 10MB for kernel cache.
#
# For more details on the used SVM solver see
#  V.Franc: Optimization Algorithms for Kernel Methods. Research report.
#  CTU-CMP-2005-22. CTU FEL Prague. 2005.
#  ftp://cmp.felk.cvut.cz/pub/cmp/articles/franc/Franc-PhD.pdf .
#

def gmnpsvm ():
        print 'GMNPSVM'

        size_cache=10
        width=2.1
        C=1.2
        epsilon=1e-5
        use_bias=False

        from sg import sg
        sg('set_features', 'TRAIN', fm_train_real)
        sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

        sg('set_labels', 'TRAIN', label_train_multiclass)
        sg('new_classifier', 'GMNPSVM')
        sg('svm_epsilon', epsilon)
        sg('c', C)
        sg('svm_use_bias', use_bias)
        sg('train_classifier')

        sg('set_features', 'TEST', fm_test_real)
        result=sg('classify')
        return result

from tools.load import LoadMatrix
from pylab import figure,pcolor,scatter,contour,colorbar,show,imshow
from numpy import meshgrid,reshape,linspace,ones,min,max,concatenate,transpose
from numpy import ravel,array
from numpy.random import randn

lm=LoadMatrix()
fm_train_real=lm.load_numbers('../data/toy/fm_train_real.dat')
label_train_multiclass=lm.load_labels('../data/toy/label_train_multiclass.dat')

x1=linspace(1.2*min(fm_train_real),1.2*max(fm_train_real), 50)
x2=linspace(1.2*min(fm_train_real),1.2*max(fm_train_real), 50)
x,y=meshgrid(x1,x2);
#fm_test_real=transpose(array((ravel(x), ravel(y))))
fm_test_real=array((ravel(x), ravel(y)))

result=gmnpsvm()
print 'fm_train_real=',fm_train_real
print 'label_train_multiclass=',label_train_multiclass
print 'fm_test_real=',fm_test_real
print 'result=',result


figure()

z=result
features=fm_train_real
labels=label_train_multiclass
z.resize((50,50))
i=imshow((z),  origin='lower', extent=(1.2*min(features),1.2*max(features),1.2*min(features),1.2*max(features))) #for smooth visualization
scatter(features[0,:],features[1,:], s=20, marker='o', c=labels, hold=True)
contour(x, y, (z), linewidths=1, colors='black', hold=True)
colorbar(i)
show()

