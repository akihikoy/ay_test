#!/usr/bin/python
# -*- coding: utf-8 -*-

# In this example a support vector regression algorithm is trained on a
# real-valued toy data set. The underlying library used for the SVR training is
# LIBSVM. The SVR is trained with regularization parameter C=1 and a gaussian
# kernel with width=2.1.
#
# For more details on LIBSVM solver see http://www.csie.ntu.edu.tw/~cjlin/libsvm/ .

def libsvr ():
	print 'LibSVR'

	size_cache=10
	width=2.1
	C=1.2
	epsilon=1e-5
	tube_epsilon=1e-2

	from sg import sg
	sg('set_features', 'TRAIN', fm_train)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

	sg('set_labels', 'TRAIN', label_train)
	sg('new_regression', 'LIBSVR')
	sg('svr_tube_epsilon', tube_epsilon)
	sg('c', C)
	sg('train_regression')

	sg('set_features', 'TEST', fm_test)
	result=sg('classify')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train=lm.load_numbers('../data/toy/fm_train_real.dat')
	fm_test=lm.load_numbers('../data/toy/fm_test_real.dat')
	label_train=lm.load_labels('../data/toy/label_train_twoclass.dat')
	libsvr()
