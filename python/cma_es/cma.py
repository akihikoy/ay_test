#!/usr/bin/env python3
"""Module cma implements the CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

CMA-ES is a stochastic optimizer for robust non-linear non-convex
derivative- and function-value-free numerical optimization.

This implementation can be used with Python versions 2.6, 2.7, 3.x.

CMA-ES searches for a minimizer (a solution x in :math:`R^n`) of an
objective function f (cost function), such that f(x) is minimal.
Regarding f, only a passably reliable ranking of the candidate
solutions in each iteration is necessary. Neither the function values
itself, nor the gradient of f need to be available or do matter (like
in the downhill simplex Nelder-Mead algorithm). Some termination
criteria however depend on actual f-values.

Two interfaces are provided:

  - function `fmin(func, x0, sigma0,...)`
        runs a complete minimization
        of the objective function func with CMA-ES.

  - class `CMAEvolutionStrategy`
      allows for minimization such that the
      control of the iteration loop remains with the user.


Used packages:

    - unavoidable: `numpy` (see `barecmaes2.py` if `numpy` is not
      available),
    - avoidable with small changes: `time`, `sys`
    - optional: `matplotlib.pyplot` (for `plot` etc., highly
      recommended), `pprint` (pretty print), `pickle` (in class
      `Sections`), `doctest`, `inspect`, `pygsl` (never by default)

Install
-------
The file ``cma.py`` only needs to be visible in the python path (e.g. in
the current working directory), but can also be installed in the
terminal command line by::

    python cma.py --install

which solely calls the ``setup`` function from the standard
``distutils.core`` package for installation. If the ``setup.py``
file is been provided with ``cma.py``, the standard call is

    python setup.py cma

Both calls need to see ``cma.py`` in the current working directory and
might need to be preceded with ``sudo``.

We can install or upgrade the currently installed version also with::

    pip install --upgrade cma

Testing
-------
From the system shell::

    python cma.py --test

or from the Python shell ``ipython -pylab``::

    run cma.py --test

or from any python shell

    import cma
    cma.main('--test')

runs ``doctest.testmod(cma)`` showing only exceptions (and not the
tests that fail due to small differences in the output) and should
run without complaints in about between 20 and 100 seconds.

Example
-------
From a python shell::

    import cma
    help(cma)  # "this" help message, use cma? in ipython
    help(cma.fmin)
    help(cma.CMAEvolutionStrategy)
    help(cma.CMAOptions)
    cma.CMAOptions('tol')  # display 'tolerance' termination options
    cma.CMAOptions('verb') # display verbosity options
    res = cma.fmin(cma.Fcts.tablet, 15 * [1], 1)
    res[0]  # best evaluated solution
    res[5]  # mean solution, presumably better with noise

:See: `fmin()`, `CMAOptions`, `CMAEvolutionStrategy`

:Author: Nikolaus Hansen, 2008-2014

:License: MIT, see below.

"""

# The MIT License (MIT)
# Copyright (c) 2014 Inria
# Author: Nikolaus Hansen, 2008-2014
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright and authorship notice and this permission notice
# shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# (note to self) for testing:
#   pyflakes cma.py   # finds bugs by static analysis
#   pychecker --limit 60 cma.py  # also executes, all 60 warnings checked
#   or python ~/Downloads/pychecker-0.8.19/pychecker/checker.py cma.py
#   python cma.py -t -quiet # executes implemented tests based on doctest
#   python -3 cma.py --test  2> out2to3warnings.txt #

# to create a html documentation file:
#    pydoc -w cma  # edit the header (remove local pointers)
#    epydoc cma.py  # comes close to javadoc but does not find the
#                   # links of function references etc
#    doxygen needs @package cma as first line in the module docstring
#       some things like class attributes are not interpreted correctly
#    sphinx: doc style of doc.python.org, could not make it work (yet)

# TODO: should optimize return self? Disadvantage: getting a hand on
#       the logger reference becomes obscure in a one-line call.
# TODO: separate display and logging options, those CMAEvolutionStrategy
#       instances don't use themselves (probably all?)
# TODO: disp method is implemented in CMAEvolutionStrategy and in
#       CMADataLogger separately, OOOptimizer.disp_str should return a str
#       which can be used uniformly?
# TODO: check scitools.easyviz and how big the adaptation would be
# TODO: split tell into a variable transformation part and the "pure"
#       functionality
#       usecase: es.tell_geno(X, [func(es.pheno(x)) for x in X])
#       genotypic repair is not part of tell_geno
# TODO: copy_always optional parameter does not make much sense,
#       as one can always copy the input argument first,
#       however some calls are simpler
# TODO: generalize input logger in optimize() as after_iteration_handler
#       (which is logger.add by default)? One difficulty is that
#       the logger object is returned (not anymore when return of optimize
#       is change). Another difficulty is the obscure usage of modulo
#       for writing a final data line in optimize.
# TODO: separate initialize==reset_state from __init__
# TODO: introduce Ypos == diffC which makes the code more consistent and
#       the active update "exact"?
# TODO: dynamically read "signals" from a file, see import ConfigParser
#       or myproperties.py (to be called after tell())
#
# typical parameters in scipy.optimize: disp, xtol, ftol, maxiter, maxfun,
#         callback=None
#         maxfev, diag (A sequency of N positive entries that serve as
#                 scale factors for the variables.)
#           full_output -- non-zero to return all optional outputs.
#   If xtol < 0.0, xtol is set to sqrt(machine_precision)
#    'infot -- a dictionary of optional outputs with the keys:
#                      'nfev': the number of function calls...
#
#    see eg fmin_powell
# typical returns
#        x, f, dictionary d
#        (xopt, {fopt, gopt, Hopt, func_calls, grad_calls, warnflag},
#         <allvecs>)
#
# TODO: keep best ten solutions
# TODO: implement constraints handling
# TODO: extend function unitdoctest, or use unittest?
# TODO: apply style guide
# TODO: eigh(): thorough testing would not hurt

# changes:
# 14/05/11: global rcParams['font.size'] not permanently changed anymore,
#           a little nicer annotations for the plots
# 14/05/07: added method result_pretty to pretty print optimization result
# 14/05/06: associated show() everywhere with ion() which should solve the
#           blocked terminal problem
# 14/05/05: all instances of "unicode" removed (was incompatible to 3.x)
# 14/05/05: replaced type(x) == y with isinstance(x, y), reorganized the
#           comments before the code starts
# 14/05/xx: change the order of kwargs of OOOptimizer.optimize,
#           remove prepare method in AdaptSigma classes, various changes/cleaning
# 14/03/01: bug fix BoundaryHandlerBase.has_bounds didn't check lower bounds correctly
#           bug fix in BoundPenalty.repair len(bounds[0]) was used instead of len(bounds[1])
#           bug fix in GenoPheno.pheno, where x was not copied when only boundary-repair was applied
# 14/02/27: bug fixed when BoundPenalty was combined with fixed variables.
# 13/xx/xx: step-size adaptation becomes a class derived from CMAAdaptSigmaBase,
#           to make testing different adaptation rules (much) easier
# 12/12/14: separated CMAOptions and arguments to fmin
# 12/10/25: removed useless check_points from fmin interface
# 12/10/17: bug fix printing number of infeasible samples, moved not-in-use methods
#           timesCroot and divCroot to the right class
# 12/10/16 (0.92.00): various changes commit: bug bound[0] -> bounds[0], more_to_write fixed,
#   sigma_vec introduced, restart from elitist, trace normalization, max(mu,popsize/2)
#   is used for weight calculation.
# 12/07/23: (bug:) BoundPenalty.update respects now genotype-phenotype transformation
# 12/07/21: convert value True for noisehandling into 1 making the output compatible
# 12/01/30: class Solution and more old stuff removed r3101
# 12/01/29: class Solution is depreciated, GenoPheno and SolutionDict do the job (v0.91.00, r3100)
# 12/01/06: CMA_eigenmethod option now takes a function (integer still works)
# 11/09/30: flat fitness termination checks also history length
# 11/09/30: elitist option (using method clip_or_fit_solutions)
# 11/09/xx: method clip_or_fit_solutions for check_points option for all sorts of
#           injected or modified solutions and even reliable adaptive encoding
# 11/08/19: fixed: scaling and typical_x type clashes 1 vs array(1) vs ones(dim) vs dim * [1]
# 11/07/25: fixed: fmin wrote first and last line even with verb_log==0
#           fixed: method settableOptionsList, also renamed to versatileOptions
#           default seed depends on time now
# 11/07/xx (0.9.92): added: active CMA, selective mirrored sampling, noise/uncertainty handling
#           fixed: output argument ordering in fmin, print now only used as function
#           removed: parallel option in fmin
# 11/07/01: another try to get rid of the memory leak by replacing self.unrepaired = self[:]
# 11/07/01: major clean-up and reworking of abstract base classes and of the documentation,
#           also the return value of fmin changed and attribute stop is now a method.
# 11/04/22: bug-fix: option fixed_variables in combination with scaling
# 11/04/21: stopdict is not a copy anymore
# 11/04/15: option fixed_variables implemented
# 11/03/23: bug-fix boundary update was computed even without boundaries
# 11/03/12: bug-fix of variable annotation in plots
# 11/02/05: work around a memory leak in numpy
# 11/02/05: plotting routines improved
# 10/10/17: cleaning up, now version 0.9.30
# 10/10/17: bug-fix: return values of fmin now use phenotyp (relevant
#           if input scaling_of_variables is given)
# 08/10/01: option evalparallel introduced,
#           bug-fix for scaling being a vector
# 08/09/26: option CMAseparable becomes CMA_diagonal
# 08/10/18: some names change, test functions go into a class
# 08/10/24: more refactorizing
# 10/03/09: upper bound exp(min(1,...)) for step-size control


import sys
import time  # not really essential
import collections
import numpy as np
# arange, cos, size, eye, inf, dot, floor, outer, zeros, linalg.eigh,
# sort, argsort, random, ones,...
from numpy import inf, array, dot, exp, log, sqrt, sum
# to access the built-in sum fct:  ``__builtins__.sum`` or ``del sum``
# removes the imported sum and recovers the shadowed build-in
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as pyplot  # also: use ipython -pyplot
    savefig = pyplot.savefig  # now we can use cma.savefig() etc
    closefig = pyplot.close
    def show():
        # is_interactive = matplotlib.is_interactive()
        pyplot.ion()
        pyplot.show()
        # if we call now matplotlib.interactive(True), the console is
        # blocked
    pyplot.ion()  # prevents that execution stops after plotting
except:
    pyplot = None
    savefig = None
    closefig = None
    def show():
        print('pyplot.show() is not available')
    print('Could not import matplotlib.pyplot, therefore ``cma.plot()``" +'
          ' etc. is not available')

__author__ = 'Nikolaus Hansen'
__version__ = "1.0.09  $Revision: 3809 $ $Date: 2014-05-08 02:37:39 +0200 (Thu, 08 May 2014) $"
# $Source$  # according to PEP 8 style guides, but what is it good for?
# $Id: cma.py 3809 2014-05-08 00:37:39Z hansen $
# bash $: svn propset svn:keywords 'Date Revision Id' cma.py

__docformat__ = "reStructuredText"  # this hides some comments entirely?
__all__ = (
        'main',
        'fmin',
        'fcts',
        'Fcts',
        'felli',
        'rotate',
        'pprint',
        'plot',
        'disp',
        'show',
        'savefig',
        'closefig',
        'use_archives',
        'is_feasible',
        'unitdoctest',
        'DerivedDictBase',
        'SolutionDict',
        'CMASolutionDict',
        'BestSolution',
        'BoundaryHandlerBase',
        'BoundNone',
        'BoundTransform',
        'BoundPenalty',
        'BoxConstraintsTransformationBase',
        'BoxConstraintsLinQuadTransformation',
        'GenoPheno',
        'OOOptimizer',
        'CMAEvolutionStrategy',
        'CMAOptions',
        'CMASolutionDict',
        'CMAAdaptSigmaBase',
        'CMAAdaptSigmaNone',
        'CMAAdaptSigmaDistanceProportional',
        'CMAAdaptSigmaCSA',
        'CMAAdaptSigmaTPA',
        'CMAAdaptSigmaMedianImprovement',
        'BaseDataLogger',
        'CMADataLogger',
        'DEAPCMADataLogger',
        'NoiseHandler',
        'Sections',
        'Misc',
        'Mh',
        'Rotation',
        'FitnessFunctions'
        )
use_archives = True
# speed up for very large population size, prevents the need for an
# inverse gp-transformation, relies on collections module
# not sure what happens if set to False


# emptysets = ('', (), [], {})
# array([]) does not work but np.size(.) == 0
# here is the problem:
# bool(array([0])) is False
# bool(list(array([0]))) is True
# bool(list(array([0, 1]))) is True
# bool(array([0, 1])) raises ValueError
#
# "x in emptysets" cannot be well replaced by "not x"
# which is also True for array([]) and None, but also for 0 and False,
# and False for NaN, and an exception for array([0,1]), see also
# http://google-styleguide.googlecode.com/svn/trunk/pyguide.html#True/False_evaluations

# ____________________________________________________________
# ____________________________________________________________
#
def rglen(ar):
    """shortcut for the iterator ``range(len(ar))``"""
    return range(len(ar))

def is_feasible(x, f):
    """default to check feasibility, see also ``cma_default_options``"""
    return f is not None and f is not np.NaN

def _print_warning(msg, method_name=None, class_name=None, iteration=None,
                  verbose=1):
    if verbose > 0:
        print('WARNING (module=' + __name__ +
              (', class=' + str(class_name) if class_name else '') +
              (', method=' + str(method_name) if method_name else '') +
              (', iteration=' + str(iteration) if iteration else '') +
              '): ' + msg)

# ____________________________________________________________
# ____________________________________________________________
#
def unitdoctest():
    """is used to describe test cases and might in future become helpful
    as an experimental tutorial as well. The main testing feature at the
    moment is by doctest with ``cma._test()`` or conveniently by
    ``python cma.py --test``. With the ``--verbose`` option added, the
    results will always slightly differ and many "failed" test cases
    might be reported.

    A simple first overall test:
        >>> import cma
        >>> res = cma.fmin(cma.fcts.elli, 3*[1], 1,
        ...                {'CMA_diagonal':2, 'seed':1, 'verb_time':0})
        (3_w,7)-CMA-ES (mu_w=2.3,w_1=58%) in dimension 3 (seed=1)
           Covariance matrix is diagonal for 2 iterations (1/ccov=7.0)
        Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
            1       7 1.453161670768570e+04 1.2e+00 1.08e+00  1e+00  1e+00
            2      14 3.281197961927601e+04 1.3e+00 1.22e+00  1e+00  2e+00
            3      21 1.082851071704020e+04 1.3e+00 1.24e+00  1e+00  2e+00
          100     700 8.544042012075362e+00 1.4e+02 3.18e-01  1e-03  2e-01
          200    1400 5.691152415221861e-12 1.0e+03 3.82e-05  1e-09  1e-06
          220    1540 3.890107746209078e-15 9.5e+02 4.56e-06  8e-11  7e-08
        termination on tolfun : 1e-11
        final/bestever f-value = 3.89010774621e-15 2.52273602735e-15
        mean solution:  [ -4.63614606e-08  -3.42761465e-10   1.59957987e-11]
        std deviation: [  6.96066282e-08   2.28704425e-09   7.63875911e-11]

    Test on the Rosenbrock function with 3 restarts. The first trial only
    finds the local optimum, which happens in about 20% of the cases.

        >>> import cma
        >>> res = cma.fmin(cma.fcts.rosen, 4*[-1], 1,
        ...                {'ftarget':1e-6, 'restarts':3,
        ...                 'verb_time':0, 'verb_disp':500, 'seed':3})
        (4_w,8)-CMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=3)
        Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
            1       8 4.875315645656848e+01 1.0e+00 8.43e-01  8e-01  8e-01
            2      16 1.662319948123120e+02 1.1e+00 7.67e-01  7e-01  8e-01
            3      24 6.747063604799602e+01 1.2e+00 7.08e-01  6e-01  7e-01
          184    1472 3.701428610430019e+00 4.3e+01 9.41e-07  3e-08  5e-08
        termination on tolfun : 1e-11
        final/bestever f-value = 3.70142861043 3.70142861043
        mean solution:  [-0.77565922  0.61309336  0.38206284  0.14597202]
        std deviation: [  2.54211502e-08   3.88803698e-08   4.74481641e-08   3.64398108e-08]
        (8_w,16)-CMA-ES (mu_w=4.8,w_1=32%) in dimension 4 (seed=4)
        Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
            1    1489 2.011376859371495e+02 1.0e+00 8.90e-01  8e-01  9e-01
            2    1505 4.157106647905128e+01 1.1e+00 8.02e-01  7e-01  7e-01
            3    1521 3.548184889359060e+01 1.1e+00 1.02e+00  8e-01  1e+00
          111    3249 6.831867555502181e-07 5.1e+01 2.62e-02  2e-04  2e-03
        termination on ftarget : 1e-06
        final/bestever f-value = 6.8318675555e-07 1.18576673231e-07
        mean solution:  [ 0.99997004  0.99993938  0.99984868  0.99969505]
        std deviation: [ 0.00018973  0.00038006  0.00076479  0.00151402]
        >>> assert res[1] <= 1e-6

    Notice the different termination conditions. Termination on the target
    function value ftarget prevents further restarts.

    Test of scaling_of_variables option

        >>> import cma
        >>> opts = cma.CMAOptions()
        >>> opts['seed'] = 456
        >>> opts['verb_disp'] = 0
        >>> opts['CMA_active'] = 1
        >>> # rescaling of third variable: for searching in  roughly
        >>> #   x0 plus/minus 1e3*sigma0 (instead of plus/minus sigma0)
        >>> opts.scaling_of_variables = [1, 1, 1e3, 1]
        >>> res = cma.fmin(cma.fcts.rosen, 4 * [0.1], 0.1, opts)
        termination on tolfun : 1e-11
        final/bestever f-value = 2.68096173031e-14 1.09714829146e-14
        mean solution:  [ 1.00000001  1.00000002  1.00000004  1.00000007]
        std deviation: [  3.00466854e-08   5.88400826e-08   1.18482371e-07   2.34837383e-07]

    The printed std deviations reflect the actual true value (not the one
    in the internal representation which would be different).

        >>> import cma
        >>> r = cma.fmin(cma.fcts.diffpow, 15 * [1], 1,
        ...              {'CMA_dampsvec_fac':0.5, 'ftarget':1e-9})
        >>> assert(r[1] < 1e-9)
        >>> assert(r[2] < 13000)  # only passed with CMA_dampsvec_fac


    :See: cma.main(), cma._test()

    """

    pass

# ____________________________________________________________
# ____________________________________________________________
#
class _BlancClass(object):
    """blanc container class for having a collection of attributes"""

# _____________________________________________________________________
# _____________________________________________________________________
#
class DerivedDictBase(collections.MutableMapping):
    """for conveniently adding "features" to a dictionary. The actual
    dictionary is in ``self.data``. Copy-paste
    and modify setitem, getitem, and delitem, if necessary"""
    def __init__(self, *args, **kwargs):
        # collections.MutableMapping.__init__(self)
        super(DerivedDictBase, self).__init__()
        # super(SolutionDict, self).__init__()  # the same
        self.data = dict(*args, **kwargs)
    def __len__(self):
        return len(self.data)
    def __contains__(self, value):
        return value in self.data
    def __iter__(self):
        return iter(self.data)
    def __setitem__(self, key, value):
        """defines self[key] = value"""
        self.data[key] = value
    def __getitem__(self, key):
        """defines self[key]"""
        return self.data[key]
    def __delitem__(self, key):
        del self.data[key]

class SolutionDict(DerivedDictBase):
    """dictionary with computation of an hash key.

    The hash key is generated from the inserted solution and a stack of
    previously inserted same solutions is provided. Each entry is meant
    to store additional information related to the solution.

        >>> import cma, numpy as np
        >>> d = cma.SolutionDict()
        >>> x = np.array([1,2,4])
        >>> d[x] = {'f': sum(x**2), 'iteration': 1}
        >>> assert d[x]['iteration'] == 1
        >>> assert d.get(x) == (d[x] if d.key(x) in d.keys() else None)

    TODO: data_with_same_key behaves like a stack (see setitem and
    delitem), but rather should behave like a queue?! A queue is less
    consistent with the operation self[key] = ..., if
    self.data_with_same_key[key] is not empty.

    TODO: iteration key is used to clean up without error management

    """
    def __init__(self, *args, **kwargs):
        DerivedDictBase.__init__(self, *args, **kwargs)
        self.data_with_same_key = {}
        self.last_iteration = 0
    def key(self, x):
        try:
            return tuple(x)
            # using sum(x) is slower, using x[0] is slightly faster
        except TypeError:
            return x
    def __setitem__(self, key, value):
        """defines self[key] = value"""
        key = self.key(key)
        if key in self.data_with_same_key:
            self.data_with_same_key[key] += [self.data[key]]
        elif key in self.data:
            self.data_with_same_key[key] = [self.data[key]]
        self.data[key] = value
    def __getitem__(self, key):  # 50% of time of
        """defines self[key]"""
        return self.data[self.key(key)]
    def __delitem__(self, key):
        """remove only most current key-entry"""
        key = self.key(key)
        if key in self.data_with_same_key:
            if len(self.data_with_same_key[key]) == 1:
                self.data[key] = self.data_with_same_key.pop(key)[0]
            else:
                self.data[key] = self.data_with_same_key[key].pop(-1)
        else:
            del self.data[key]
    def truncate(self, max_len, min_iter):
        if len(self) > max_len:
            for k in list(self.keys()):
                if self[k]['iteration'] < min_iter:
                    del self[k]
                    # deletes one item with k as key, better delete all?

class CMASolutionDict(SolutionDict):
    def __init__(self, *args, **kwargs):
        SolutionDict.__init__(self, *args, **kwargs)
        self.last_solution_index = 0

    # TODO: insert takes 30% of the overall CPU time, mostly in def key()
    #       with about 15% of the overall CPU time
    def insert(self, key, geno=None, iteration=None, fitness=None, value=None):
        """insert an entry with key ``key`` and value
        ``value if value is not None else {'geno':key}`` and
        ``self[key]['kwarg'] = kwarg if kwarg is not None`` for the further kwargs.

        """
        # archive returned solutions, first clean up archive
        if iteration is not None and iteration > self.last_iteration and (iteration % 10) < 1:
            self.truncate(300, iteration - 3)
        elif value is not None and value.get('iteration'):
            iteration = value['iteration']
            if (iteration % 10) < 1:
                self.truncate(300, iteration - 3)

        self.last_solution_index += 1
        if value is not None:
            try:
                iteration = value['iteration']
            except:
                pass
        if iteration is not None:
            if iteration > self.last_iteration:
                self.last_solution_index = 0
            self.last_iteration = iteration
        else:
            iteration = self.last_iteration + 0.5  # a hack to get a somewhat reasonable value
        if value is not None:
            self[key] = value
        else:
            self[key] = {'pheno': key}
        if geno is not None:
            self[key]['geno'] = geno
        if iteration is not None:
            self[key]['iteration'] = iteration
        if fitness is not None:
            self[key]['fitness'] = fitness
        return self[key]

if not use_archives:
    class CMASolutionDict(SolutionDict):
        def insert(self, *args, **kwargs):
            pass

class BestSolution(object):
    """container to keep track of the best solution seen"""
    def __init__(self, x=None, f=np.inf, evals=None):
        """initialize the best solution with `x`, `f`, and `evals`.
        Better solutions have smaller `f`-values.

        """
        self.x = x
        self.x_geno = None
        self.f = f if f is not None and f is not np.nan else np.inf
        self.evals = evals
        self.evalsall = evals
        self.last = _BlancClass()
        self.last.x = x
        self.last.f = f
    def update(self, arx, xarchive=None, arf=None, evals=None):
        """checks for better solutions in list `arx`.

        Based on the smallest corresponding value in `arf`,
        alternatively, `update` may be called with a `BestSolution`
        instance like ``update(another_best_solution)`` in which case
        the better solution becomes the current best.

        `xarchive` is used to retrieve the genotype of a solution.

        """
        if isinstance(arx, BestSolution):
            if self.evalsall is None:
                self.evalsall = arx.evalsall
            elif arx.evalsall is not None:
                self.evalsall = max((self.evalsall, arx.evalsall))
            if arx.f is not None and arx.f < np.inf:
                self.update([arx.x], xarchive, [arx.f], arx.evals)
            return self
        assert arf is not None
        # find failsave minimum
        minidx = np.nanargmin(arf)
        if minidx is np.nan:
            return
        minarf = arf[minidx]
        # minarf = reduce(lambda x, y: y if y and y is not np.nan
        #                   and y < x else x, arf, np.inf)
        if minarf < np.inf and (minarf < self.f or self.f is None):
            self.x, self.f = arx[minidx], arf[minidx]
            if xarchive is not None and xarchive.get(self.x) is not None:
                self.x_geno = xarchive[self.x].get('geno')
            else:
                self.x_geno = None
            self.evals = None if not evals else evals - len(arf) + minidx + 1
            self.evalsall = evals
        elif evals:
            self.evalsall = evals
        self.last.x = arx[minidx]
        self.last.f = minarf
    def get(self):
        """return ``(x, f, evals)`` """
        return self.x, self.f, self.evals  # , self.x_geno


# ____________________________________________________________
# ____________________________________________________________
#
class BoundaryHandlerBase(object):
    """hacked base class """
    def __init__(self, bounds):
        """bounds are not copied, but possibly modified and
        put into a normalized form: ``bounds`` can be ``None``
        or ``[lb, ub]`` where ``lb`` and ``ub`` are
        either None or a vector (which can have ``None`` entries).

        Generally, the last entry is recycled to compute bounds
        for any dimension.

        """
        if not bounds:
            self.bounds = None
        else:
            l = [None, None]  # figure out lenths
            for i in [0, 1]:
                try:
                    l[i] = len(bounds[i])
                except TypeError:
                    bounds[i] = [bounds[i]]
                    l[i] = 1
                if all([bounds[i][j] is None or not np.isfinite(bounds[i][j])
                        for j in rglen(bounds[i])]):
                    bounds[i] = None
                if bounds[i] is not None and any([bounds[i][j] == (-1)**i * np.inf
                                                  for j in rglen(bounds[i])]):
                    raise ValueError('lower/upper is +inf/-inf and ' +
                                     'therefore no finite feasible solution is available')
            self.bounds = bounds

    def __call__(self, solutions, *args, **kwargs):
        """return penalty or list of penalties, by default zero(s).

        This interface seems too specifically tailored to the derived
        BoundPenalty class, it should maybe change.

        """
        if np.isscalar(solutions[0]):
            return 0.0
        else:
            return len(solutions) * [0.0]

    def update(self, *args, **kwargs):
        return self

    def repair(self, x, copy_if_changed=True, copy_always=False):
        """projects infeasible values on the domain bound, might be
        overwritten by derived class """
        if copy_always:
            x = array(x, copy=True)
            copy = False
        else:
            copy = copy_if_changed
        if self.bounds is None:
            return x
        for ib in [0, 1]:
            if self.bounds[ib] is None:
                continue
            for i in rglen(x):
                idx = min([i, len(self.bounds[ib]) - 1])
                if self.bounds[ib][idx] is not None and \
                        (-1)**ib * x[i] < (-1)**ib * self.bounds[ib][idx]:
                    if copy:
                        x = array(x, copy=True)
                        copy = False
                    x[i] = self.bounds[ib][idx]

    def inverse(self, y, copy_if_changed=True, copy_always=False):
        return y if not copy_always else array(y, copy=True)

    def get_bounds(self, which, dimension):
        """``get_bounds('lower', 8)`` returns the lower bounds in 8-D"""
        if which == 'lower' or which == 0:
            return self._get_bounds(0, dimension)
        elif which == 'upper' or which == 1:
            return self._get_bounds(1, dimension)
        else:
            raise ValueError("argument which must be 'lower' or 'upper'")

    def _get_bounds(self, ib, dimension):
        """ib == 0/1 means lower/upper bound, return a vector of length
        `dimension` """
        sign_ = 2 * ib - 1
        assert sign_**2 == 1
        if self.bounds is None or self.bounds[ib] is None:
            return array(dimension * [sign_ * np.Inf])
        res = []
        for i in range(dimension):
            res.append(self.bounds[ib][min([i, len(self.bounds[ib]) - 1])])
            if res[-1] is None:
                res[-1] = sign_ * np.Inf
        return array(res)

    def has_bounds(self):
        """return True, if any variable is bounded"""
        bounds = self.bounds
        if bounds in (None, [None, None]):
            return False
        for ib, bound in enumerate(bounds):
            if bound is not None:
                sign_ = 2 * ib - 1
                for bound_i in bound:
                    if bound_i is not None and sign_ * bound_i < np.inf:
                        return True
        return False

    def is_in_bounds(self, x):
        """not yet tested"""
        if self.bounds is None:
            return True
        for ib in [0, 1]:
            if self.bounds[ib] is None:
                continue
            for i in rglen(x):
                idx = min([i, len(self.bounds[ib]) - 1])
                if self.bounds[ib][idx] is not None and \
                        (-1)**ib * x[i] < (-1)**ib * self.bounds[ib][idx]:
                    return False
        return True

    def to_dim_times_two(self, bounds):
        """return boundaries in format ``[[lb0, ub0], [lb1, ub1], ...]``,
        as used by ``BoxConstraints...`` class.

        """
        if not bounds:
            b = [[None, None]]
        else:
            l = [None, None]  # figure out lenths
            for i in [0, 1]:
                try:
                    l[i] = len(bounds[i])
                except TypeError:
                    bounds[i] = [bounds[i]]
                    l[i] = 1
            b = []  # bounds in different format
            try:
                for i in range(max(l)):
                    b.append([bounds[0][i] if i < l[0] else None,
                              bounds[1][i] if i < l[1] else None])
            except (TypeError, IndexError):
                print("boundaries must be provided in the form " +
                      "[scalar_of_vector, scalar_or_vector]")
                raise
        return b

# ____________________________________________________________
# ____________________________________________________________
#
class BoundNone(BoundaryHandlerBase):
    def __init__(self, bounds=None):
        if bounds is not None:
            raise ValueError()
        BoundaryHandlerBase.__init__(self, None)
    def is_in_bounds(self, x):
        return True

# ____________________________________________________________
# ____________________________________________________________
#
class BoundTransform(BoundaryHandlerBase):
    """Handles boundary by a smooth, piecewise linear and quadratic
    transformation into the feasible domain.

    >>> import cma
    >>> b = cma.BoundTransform([None, 1])
    >>> assert b.bounds == [[None], [1]]
    >>> assert cma.Mh.vequals_approximately(b.repair([0, 1, 1.2]),
    ...                                     array([ 0., 0.975, 0.975]))
    >>> assert b.is_in_bounds([0, 0.5, 1])
    >>> assert cma.Mh.vequals_approximately(b.transform([0, 1, 2]),
    ...                                     [ 0.   ,  0.975,  0.2  ])
    >>> o=cma.fmin(cma.fcts.sphere, 6 * [-2], 0.5, options={
    ...    'boundary_handling': 'BoundTransform ',
    ...    'bounds': [[], 5 * [-1] + [inf]] })
    >>> assert o[1] < 5 + 1e-8

    Details: this class uses ``class BoxConstraintsLinQuadTransformation``

    """
    def __init__(self, bounds=None):
        """Argument bounds can be `None` or ``bounds[0]`` and ``bounds[1]``
        are lower and upper domain boundaries, each is either `None` or
        a scalar or a list or array of appropriate size.

        """
        BoundaryHandlerBase.__init__(self, bounds)
        self.bounds_tf = BoxConstraintsLinQuadTransformation(self.to_dim_times_two(bounds))

    def repair(self, x, copy_if_changed=True, copy_always=False):
        """transforms ``x`` into the bounded domain.

        ``copy_always`` option might disappear.

        """
        copy = copy_if_changed
        if copy_always:
            x = array(x, copy=True)
            copy = False
        if self.bounds is None or (self.bounds[0] is None and
                                   self.bounds[1] is None):
            return x
        return self.bounds_tf(x, copy)

    def transform(self, x):
        return self.repair(x)

    def inverse(self, x, copy_if_changed=True, copy_always=False):
        """inverse transform of ``x`` from the bounded domain.

        """
        copy = copy_if_changed
        if copy_always:
            x = array(x, copy=True)
            copy = False
        if self.bounds is None or (self.bounds[0] is None and
                                   self.bounds[1] is None):
            return x
        return self.bounds_tf.inverse(x, copy)  # this doesn't exist

# ____________________________________________________________
# ____________________________________________________________
#
class BoundPenalty(BoundaryHandlerBase):
    """Computes the boundary penalty. Must be updated each iteration,
    using the `update` method.

    Details
    -------
    The penalty computes like ``sum(w[i] * (x[i]-xfeas[i])**2)``,
    where `xfeas` is the closest feasible (in-bounds) solution from `x`.
    The weight `w[i]` should be updated during each iteration using
    the update method.

    Example:

    >>> import cma
    >>> cma.fmin(cma.felli, 6 * [1], 1,
    ...          {
    ...              'boundary_handling': 'BoundPenalty',
    ...              'bounds': [-1, 1],
    ...              'fixed_variables': {0: 0.012, 2:0.234}
    ...          })

    Reference: Hansen et al 2009, A Method for Handling Uncertainty...
    IEEE TEC, with addendum, see
    http://www.lri.fr/~hansen/TEC2009online.pdf

    """
    def __init__(self, bounds=None):
        """Argument bounds can be `None` or ``bounds[0]`` and ``bounds[1]``
        are lower  and upper domain boundaries, each is either `None` or
        a scalar or a list or array of appropriate size.
        """
        # #
        # bounds attribute reminds the domain boundary values
        BoundaryHandlerBase.__init__(self, bounds)

        self.gamma = 1  # a very crude assumption
        self.weights_initialized = False  # gamma becomes a vector after initialization
        self.hist = []  # delta-f history

    def repair(self, x, copy_if_changed=True, copy_always=False):
        """sets out-of-bounds components of ``x`` on the bounds.

        """
        # TODO (old data): CPU(N,lam,iter=20,200,100): 3.3s of 8s for two bounds, 1.8s of 6.5s for one bound
        # remark: np.max([bounds[0], x]) is about 40 times slower than max((bounds[0], x))
        copy = copy_if_changed
        if copy_always:
            x = array(x, copy=True)
        bounds = self.bounds
        if bounds not in (None, [None, None], (None, None)):  # solely for effiency
            x = array(x, copy=True) if copy and not copy_always else x
            if bounds[0] is not None:
                if np.isscalar(bounds[0]):
                    for i in rglen(x):
                        x[i] = max((bounds[0], x[i]))
                else:
                    for i in rglen(x):
                        j = min([i, len(bounds[0]) - 1])
                        if bounds[0][j] is not None:
                            x[i] = max((bounds[0][j], x[i]))
            if bounds[1] is not None:
                if np.isscalar(bounds[1]):
                    for i in rglen(x):
                        x[i] = min((bounds[1], x[i]))
                else:
                    for i in rglen(x):
                        j = min((i, len(bounds[1]) - 1))
                        if bounds[1][j] is not None:
                            x[i] = min((bounds[1][j], x[i]))
        return x

    # ____________________________________________________________
    #
    def __call__(self, x, archive, gp):
        """returns the boundary violation penalty for `x` ,where `x` is a
        single solution or a list or array of solutions.

        """
        if x in (None, (), []):
            return x
        if self.bounds in (None, [None, None], (None, None)):
            return 0.0 if np.isscalar(x[0]) else [0.0] * len(x)  # no penalty

        x_is_single_vector = np.isscalar(x[0])
        x = [x] if x_is_single_vector else x

        # add fixed variables to self.gamma
        try:
            gamma = list(self.gamma)  # fails if self.gamma is a scalar
            for i in sorted(gp.fixed_values):  # fails if fixed_values is None
                gamma.insert(i, 0.0)
            gamma = array(gamma, copy=False)
        except TypeError:
            gamma = self.gamma
        pen = []
        for xi in x:
            # CAVE: this does not work with already repaired values!!
            # CPU(N,lam,iter=20,200,100)?: 3s of 10s, array(xi): 1s
            # remark: one deep copy can be prevented by xold = xi first
            xpheno = gp.pheno(archive[xi]['geno'])
            # necessary, because xi was repaired to be in bounds
            xinbounds = self.repair(xpheno)
            # could be omitted (with unpredictable effect in case of external repair)
            fac = 1  # exp(0.1 * (log(self.scal) - np.mean(self.scal)))
            pen.append(sum(gamma * ((xinbounds - xpheno) / fac)**2) / len(xi))
        return pen[0] if x_is_single_vector else pen

    # ____________________________________________________________
    #
    def feasible_ratio(self, solutions):
        """counts for each coordinate the number of feasible values in
        ``solutions`` and returns an array of length ``len(solutions[0])``
        with the ratios.

        `solutions` is a list or array of repaired `Solution` instances

        """
        count = np.zeros(len(solutions[0]))
        for x in solutions:
            count += x.unrepaired == x
        return count / float(len(solutions))

    # ____________________________________________________________
    #
    def update(self, function_values, es):
        """updates the weights for computing a boundary penalty.

        Arguments
        ---------
        `function_values`
            all function values of recent population of solutions
        `es`
            `CMAEvolutionStrategy` object instance, in particular
            mean and variances and the methods from the attribute
            `gp` of type `GenoPheno` are used.

        """
        if self.bounds is None or (self.bounds[0] is None and
                                   self.bounds[1] is None):
            return self

        N = es.N
        # ## prepare
        # compute varis = sigma**2 * C_ii
        varis = es.sigma**2 * array(N * [es.C] if np.isscalar(es.C) else (# scalar case
                                es.C if np.isscalar(es.C[0]) else  # diagonal matrix case
                                [es.C[i][i] for i in range(N)]))  # full matrix case

        # relative violation in geno-space
        dmean = (es.mean - es.gp.geno(self.repair(es.gp.pheno(es.mean)))) / varis**0.5

        # ## Store/update a history of delta fitness value
        fvals = sorted(function_values)
        l = 1 + len(fvals)
        val = fvals[3 * l // 4] - fvals[l // 4]  # exact interquartile range apart interpolation
        val = val / np.mean(varis)  # new: val is normalized with sigma of the same iteration
        # insert val in history
        if np.isfinite(val) and val > 0:
            self.hist.insert(0, val)
        elif val == inf and len(self.hist) > 1:
            self.hist.insert(0, max(self.hist))
        else:
            pass  # ignore 0 or nan values
        if len(self.hist) > 20 + (3 * N) / es.popsize:
            self.hist.pop()

        # ## prepare
        dfit = np.median(self.hist)  # median interquartile range
        damp = min(1, es.sp.mueff / 10. / N)

        # ## set/update weights
        # Throw initialization error
        if len(self.hist) == 0:
            raise _Error('wrongful initialization, no feasible solution sampled. ' +
                'Reasons can be mistakenly set bounds (lower bound not smaller than upper bound) or a too large initial sigma0 or... ' +
                'See description of argument func in help(cma.fmin) or an example handling infeasible solutions in help(cma.CMAEvolutionStrategy). ')
        # initialize weights
        if (dmean.any() and (not self.weights_initialized or es.countiter == 2)):  # TODO
            self.gamma = array(N * [2 * dfit])  ## BUGBUGzzzz: N should be phenotypic (bounds are in phenotype), but is genotypic
            self.weights_initialized = True
        # update weights gamma
        if self.weights_initialized:
            edist = array(abs(dmean) - 3 * max(1, N**0.5 / es.sp.mueff))
            if 1 < 3:  # this is better, around a factor of two
                # increase single weights possibly with a faster rate than they can decrease
                #     value unit of edst is std dev, 3==random walk of 9 steps
                self.gamma *= exp((edist > 0) * np.tanh(edist / 3) / 2.)**damp
                # decrease all weights up to the same level to avoid single extremely small weights
                #    use a constant factor for pseudo-keeping invariance
                self.gamma[self.gamma > 5 * dfit] *= exp(-1. / 3)**damp
                #     self.gamma[idx] *= exp(5*dfit/self.gamma[idx] - 1)**(damp/3)
        es.more_to_write += list(self.gamma) if self.weights_initialized else N * [1.0]
        # ## return penalty
        # es.more_to_write = self.gamma if not np.isscalar(self.gamma) else N*[1]
        return self  # bound penalty values

# ____________________________________________________________
# ____________________________________________________________
#
class BoxConstraintsTransformationBase(object):
    """Implements a transformation into boundaries and is used for
    boundary handling::

        tf = BoxConstraintsTransformationAnyDerivedClass([[1, 4]])
        x = [3, 2, 4.4]
        y = tf(x)  # "repaired" solution
        print(tf([2.5]))  # middle value is never changed
        [2.5]

    :See: ``BoundaryHandler``

    """
    def __init__(self, bounds):
        try:
            if len(bounds[0]) != 2:
                raise ValueError
        except:
            raise ValueError(' bounds must be either [[lb0, ub0]] or [[lb0, ub0], [lb1, ub1],...], \n where in both cases the last entry is reused for all remaining dimensions')
        self.bounds = bounds
        self.initialize()

    def initialize(self):
        """initialize in base class"""
        self._lb = [b[0] for b in self.bounds]  # can be done more efficiently?
        self._ub = [b[1] for b in self.bounds]

    def _lowerupperval(self, a, b, c):
        return np.max([np.max(a), np.min([np.min(b), c])])
    def bounds_i(self, i):
        """return ``[ith_lower_bound, ith_upper_bound]``"""
        return self.bounds[self._index(i)]
    def __call__(self, solution_in_genotype):
        res = [self._transform_i(x, i) for i, x in enumerate(solution_in_genotype)]
        return res
    transform = __call__
    def inverse(self, solution_in_phenotype, copy_if_changed=True, copy_always=True):
        return [self._inverse_i(y, i) for i, y in enumerate(solution_in_phenotype)]
    def _index(self, i):
        return min((i, len(self.bounds) - 1))
    def _transform_i(self, x, i):
        raise NotImplementedError('this is an abstract method that should be implemented in the derived class')
    def _inverse_i(self, y, i):
        raise NotImplementedError('this is an abstract method that should be implemented in the derived class')
    def shift_or_mirror_into_invertible_domain(self, solution_genotype):
        """return the reference solution that has the same ``box_constraints_transformation(solution)``
        value, i.e. ``tf.shift_or_mirror_into_invertible_domain(x) = tf.inverse(tf.transform(x))``.
        This is an idempotent mapping (leading to the same result independent how often it is
        repeatedly applied).

        """
        return self.inverse(self(solution_genotype))
        raise NotImplementedError('this is an abstract method that should be implemented in the derived class')

class _BoxConstraintsTransformationTemplate(BoxConstraintsTransformationBase):
    """copy/paste this template to implement a new boundary handling transformation"""
    def __init__(self, bounds):
        BoxConstraintsTransformationBase.__init__(self, bounds)
    def initialize(self):
        BoxConstraintsTransformationBase.initialize(self)  # likely to be removed
    def _transform_i(self, x, i):
        raise NotImplementedError('this is an abstract method that should be implemented in the derived class')
    def _inverse_i(self, y, i):
        raise NotImplementedError('this is an abstract method that should be implemented in the derived class')
    __doc__ = BoxConstraintsTransformationBase.__doc__ + __doc__

class BoxConstraintsLinQuadTransformation(BoxConstraintsTransformationBase):
    """implements a bijective, monotonous transformation between [lb - al, ub + au]
    and [lb, ub] which is the identity (and therefore linear) in [lb + al, ub - au]
    (typically about 90% of the interval) and quadratic in [lb - 3*al, lb + al]
    and in [ub - au, ub + 3*au]. The transformation is periodically
    expanded beyond the limits (somewhat resembling the shape sin(x-pi/2))
    with a period of ``2 * (ub - lb + al + au)``.

    Details
    =======
    Partly due to numerical considerations depend the values ``al`` and ``au``
    on ``abs(lb)`` and ``abs(ub)`` which makes the transformation non-translation
    invariant. In contrast to sin(.), the transformation is robust to "arbitrary"
    values for boundaries, e.g. a lower bound of ``-1e99`` or ``np.Inf`` or
    ``None``.

    Examples
    ========
    Example to use with cma:

    >>> import cma
    >>> # only the first variable has an upper bound
    >>> tf = cma.BoxConstraintsLinQuadTransformation([[1,2], [1,None]]) # second==last pair is re-cycled
    >>> cma.fmin(cma.felli, 9 * [2], 1, {'transformation': [tf.transform, tf.inverse], 'verb_disp': 0})
    >>> # ...or...
    >>> es = cma.CMAEvolutionStrategy(9 * [2], 1)
    >>> while not es.stop():
    ...     X = es.ask()
    ...     f = [cma.felli(tf(x)) for x in X]  # tf(x) == tf.transform(x)
    ...     es.tell(X, f)

    Example of the internal workings:

    >>> import cma
    >>> tf = cma.BoxConstraintsLinQuadTransformation([[1,2], [1,11], [1,11]])
    >>> tf.bounds
    [[1, 2], [1, 11], [1, 11]]
    >>> tf([1.5, 1.5, 1.5])
    [1.5, 1.5, 1.5]
    >>> tf([1.52, -2.2, -0.2, 2, 4, 10.4])
    [1.52, 4.0, 2.0, 2.0, 4.0, 10.4]
    >>> res = np.round(tf._au, 2)
    >>> assert list(res[:4]) == [ 0.15, 0.6, 0.6, 0.6]
    >>> res = [round(x, 2) for x in tf.shift_or_mirror_into_invertible_domain([1.52, -12.2, -0.2, 2, 4, 10.4])]
    >>> assert res == [1.52, 9.2, 2.0, 2.0, 4.0, 10.4]
    >>> tmp = tf([1])  # call with lower dimension

    """
    def __init__(self, bounds):
        """``x`` is defined in ``[lb - 3*al, ub + au + r - 2*al]`` with ``r = ub - lb + al + au``,
        and ``x == transformation(x)`` in ``[lb + al, ub - au]``.
        ``beta*x - alphal = beta*x - alphau`` is then defined in ``[lb, ub]``,

        ``alphal`` and ``alphau`` represent the same value, but respectively numerically
        better suited for values close to lb and ub.

        """
        # super().__init__(bounds) # only in Python 3.x available
        BoxConstraintsTransformationBase.__init__(self, bounds)
        # super(BB, self).__init__(bounds) # is supposed to call initialize
        # ## super(BoxConstraintsTransformationBase, self).__init__(bounds) # is probably invalid

    def initialize(self, length=None):
        """see ``__init__``"""
        if length is None:
            length = len(self.bounds)
        max_i = min((len(self.bounds) - 1, length - 1))
        self._lb = array([self.bounds[min((i, max_i))][0]
                          if self.bounds[min((i, max_i))][0] is not None else -np.Inf
                          for i in range(length)], copy=False)
        self._ub = array([self.bounds[min((i, max_i))][1]
                          if self.bounds[min((i, max_i))][1] is not None else np.Inf
                          for i in range(length)], copy=False)
        lb = self._lb
        ub = self._ub
        # define added values for lower and upper bound
        self._al = array([min([(ub[i] - lb[i]) / 2, (1 + np.abs(lb[i])) / 20])
                             if np.isfinite(lb[i]) else 1 for i in rglen(lb)], copy=False)
        self._au = array([min([(ub[i] - lb[i]) / 2, (1 + np.abs(ub[i])) / 20])
                             if np.isfinite(ub[i]) else 1 for i in rglen(ub)], copy=False)

    def __call__(self, solution_genotype, copy_if_changed=True, copy_always=False):
        # about four times faster version of array([self._transform_i(x, i) for i, x in enumerate(solution_genotype)])
        # still, this makes a typical run on a test function two times slower, but there might be one too many copies
        # during the transformations in gp
        if len(self._lb) != len(solution_genotype):
            self.initialize(len(solution_genotype))
        lb = self._lb
        ub = self._ub
        al = self._al
        au = self._au

        if copy_always or not isinstance(solution_genotype[0], float):
            # transformed value is likely to be a float
            y = np.array(solution_genotype, copy=True, dtype=float)
            # if solution_genotype is not a float, copy value is disregarded
            copy = False
        else:
            y = solution_genotype
            copy = copy_if_changed
        idx = (y < lb - 2 * al - (ub - lb) / 2.0) | (y > ub + 2 * au + (ub - lb) / 2.0)
        if idx.any():
            r = 2 * (ub[idx] - lb[idx] + al[idx] + au[idx])  # period
            s = lb[idx] - 2 * al[idx] - (ub[idx] - lb[idx]) / 2.0  # start
            if copy:
                y = np.array(y, copy=True)
                copy = False
            y[idx] -= r * ((y[idx] - s) // r)  # shift
        idx = y > ub + au
        if idx.any():
            if copy:
                y = np.array(y, copy=True)
                copy = False
            y[idx] -= 2 * (y[idx] - ub[idx] - au[idx])
        idx = y < lb - al
        if idx.any():
            if copy:
                y = np.array(y, copy=True)
                copy = False
            y[idx] += 2 * (lb[idx] - al[idx] - y[idx])
        idx = y < lb + al
        if idx.any():
            if copy:
                y = np.array(y, copy=True)
                copy = False
            y[idx] = lb[idx] + (y[idx] - (lb[idx] - al[idx]))**2 / 4 / al[idx]
        idx = y > ub - au
        if idx.any():
            if copy:
                y = np.array(y, copy=True)
                copy = False
            y[idx] = ub[idx] - (y[idx] - (ub[idx] + au[idx]))**2 / 4 / au[idx]
        # assert Mh.vequals_approximately(y, BoxConstraintsTransformationBase.__call__(self, solution_genotype))
        return y
    __call__.doc = BoxConstraintsTransformationBase.__doc__
    transform = __call__
    def idx_infeasible(self, solution_genotype):
        """return indices of "infeasible" variables, that is,
        variables that do not directly map into the feasible domain such that
        ``tf.inverse(tf(x)) == x``.

        """
        res = [i for i, x in enumerate(solution_genotype) if not self.is_feasible_i(x, i)]
        return res
    def is_feasible_i(self, x, i):
        """return True if value ``x`` is in the invertible domain of
        variable ``i``

        """
        lb = self._lb[self._index(i)]
        ub = self._ub[self._index(i)]
        al = self._al[self._index(i)]
        au = self._au[self._index(i)]
        return lb - al < x < ub + au
    def is_loosely_feasible_i(self, x, i):
        """never used"""
        lb = self._lb[self._index(i)]
        ub = self._ub[self._index(i)]
        al = self._al[self._index(i)]
        au = self._au[self._index(i)]
        return lb - 2 * al - (ub - lb) / 2.0 <= x <= ub + 2 * au + (ub - lb) / 2.0

    def shift_or_mirror_into_invertible_domain(self, solution_genotype, copy=False):
        """Details: input ``solution_genotype`` is changed. The domain is
        [lb - al, ub + au] and in [lb - 2*al - (ub - lb) / 2, lb - al]
        mirroring is applied.

        """
        assert solution_genotype is not None
        if copy:
            y = [val for val in solution_genotype]
        else:
            y = solution_genotype
        if isinstance(y, np.ndarray) and not isinstance(y[0], float):
            y = array(y, dtype=float)
        for i in rglen(y):
            lb = self._lb[self._index(i)]
            ub = self._ub[self._index(i)]
            al = self._al[self._index(i)]
            au = self._au[self._index(i)]
            # x is far from the boundary, compared to ub - lb
            if y[i] < lb - 2 * al - (ub - lb) / 2.0 or y[i] > ub + 2 * au + (ub - lb) / 2.0:
                r = 2 * (ub - lb + al + au)  # period
                s = lb - 2 * al - (ub - lb) / 2.0  # start
                y[i] -= r * ((y[i] - s) // r)  # shift
            if y[i] > ub + au:
                y[i] -= 2 * (y[i] - ub - au)
            if y[i] < lb - al:
                y[i] += 2 * (lb - al - y[i])
        return y
    shift_or_mirror_into_invertible_domain.__doc__ = BoxConstraintsTransformationBase.shift_or_mirror_into_invertible_domain.__doc__ + shift_or_mirror_into_invertible_domain.__doc__

    def _shift_or_mirror_into_invertible_i(self, x, i):
        """shift into the invertible domain [lb - ab, ub + au], mirror close to
        boundaries in order to get a smooth transformation everywhere

        """
        assert x is not None
        lb = self._lb[self._index(i)]
        ub = self._ub[self._index(i)]
        al = self._al[self._index(i)]
        au = self._au[self._index(i)]
        # x is far from the boundary, compared to ub - lb
        if x < lb - 2 * al - (ub - lb) / 2.0 or x > ub + 2 * au + (ub - lb) / 2.0:
            r = 2 * (ub - lb + al + au)  # period
            s = lb - 2 * al - (ub - lb) / 2.0  # start
            x -= r * ((x - s) // r)  # shift
        if x > ub + au:
            x -= 2 * (x - ub - au)
        if x < lb - al:
            x += 2 * (lb - al - x)
        return x
    def _transform_i(self, x, i):
        """return transform of x in component i"""
        x = self._shift_or_mirror_into_invertible_i(x, i)
        lb = self._lb[self._index(i)]
        ub = self._ub[self._index(i)]
        al = self._al[self._index(i)]
        au = self._au[self._index(i)]
        if x < lb + al:
            return lb + (x - (lb - al))**2 / 4 / al
        elif x < ub - au:
            return x
        elif x < ub + 3 * au:
            return ub - (x - (ub + au))**2 / 4 / au
        else:
            assert False  # shift removes this case
            return ub + au - (x - (ub + au))
    def _inverse_i(self, y, i):
        """return inverse of y in component i"""
        lb = self._lb[self._index(i)]
        ub = self._ub[self._index(i)]
        al = self._al[self._index(i)]
        au = self._au[self._index(i)]
        if 1 < 3:
            if not lb <= y <= ub:
                raise ValueError('argument of inverse must be within the given bounds')
        if y < lb + al:
            return (lb - al) + 2 * (al * (y - lb))**0.5
        elif y < ub - au:
            return y
        else:
            return (ub + au) - 2 * (au * (ub - y))**0.5

# ____________________________________________________________
# ____________________________________________________________
#
class GenoPheno(object):
    """Genotype-phenotype transformation.

    Method `pheno` provides the transformation from geno- to phenotype,
    that is from the internal representation to the representation used
    in the objective function. Method `geno` provides the "inverse" pheno-
    to genotype transformation. The geno-phenotype transformation comprises,
    in this order:

       - insert fixed variables (with the phenotypic and therefore quite
         possibly "wrong" values)
       - affine linear transformation (scaling and shift)
       - user-defined transformation
       - repair (e.g. into feasible domain due to boundaries)
       - assign fixed variables their original phenotypic value

    By default all transformations are the identity. The repair is only applied,
    if the transformation is given as argument to the method `pheno`.

    ``geno`` is only necessary, if solutions have been injected.

    """
    def __init__(self, dim, scaling=None, typical_x=None, fixed_values=None, tf=None):
        """return `GenoPheno` instance with fixed phenotypic dimension `dim`.

        Keyword Arguments
        -----------------
            `scaling`
                the diagonal of a scaling transformation matrix, multipliers
                in the genotyp-phenotyp transformation, see `typical_x`
            `typical_x`
                ``pheno = scaling*geno + typical_x``
            `fixed_values`
                a dictionary of variable indices and values, like ``{0:2.0, 2:1.1}``,
                that are not subject to change, negative indices are ignored
                (they act like incommenting the index), values are phenotypic
                values.
            `tf`
                list of two user-defined transformation functions, or `None`.

                ``tf[0]`` is a function that transforms the internal representation
                as used by the optimizer into a solution as used by the
                objective function. ``tf[1]`` does the back-transformation.
                For example::

                    tf_0 = lambda x: [xi**2 for xi in x]
                    tf_1 = lambda x: [abs(xi)**0.5 fox xi in x]

                or "equivalently" without the `lambda` construct::

                    def tf_0(x):
                        return [xi**2 for xi in x]
                    def tf_1(x):
                        return [abs(xi)**0.5 fox xi in x]

                ``tf=[tf_0, tf_1]`` is a reasonable way to guaranty that only positive
                values are used in the objective function.

        Details
        -------
        If ``tf_1`` is ommitted, the initial x-value must be given as genotype (because
        the phenotype-genotype transformation is unknown in this case) and "injection" of
        solutions might lead to unexpected results.

        """
        self.N = dim
        self.fixed_values = fixed_values
        if tf is not None:
            self.tf_pheno = tf[0]
            self.tf_geno = tf[1]  # TODO: should not necessarily be needed
            # r = np.random.randn(dim)
            # assert all(tf[0](tf[1](r)) - r < 1e-7)
            # r = np.random.randn(dim)
            # assert all(tf[0](tf[1](r)) - r > -1e-7)
            print("WARNING in class GenoPheno: user defined transformations have not been tested thoroughly")
        else:
            self.tf_geno = None
            self.tf_pheno = None

        if fixed_values:
            if not isinstance(fixed_values, dict):
                raise _Error("fixed_values must be a dictionary {index:value,...}")
            if max(fixed_values.keys()) >= dim:
                raise _Error("max(fixed_values.keys()) = " + str(max(fixed_values.keys())) +
                    " >= dim=N=" + str(dim) + " is not a feasible index")
            # convenience commenting functionality: drop negative keys
            for k in list(fixed_values.keys()):
                if k < 0:
                    fixed_values.pop(k)

        def vec_is_default(vec, default_val=0):
            """return True if `vec` has the value `default_val`,
            None or [None] are also recognized as default

            """
            # TODO: rather let default_val be a list of default values, cave comparison of arrays
            try:
                if len(vec) == 1:
                    vec = vec[0]  # [None] becomes None and is always default
            except TypeError:
                pass  # vec is a scalar

            if vec is None or (vec == default_val).all():
                return True
            try:
                if (vec == array(None)).all():
                    return True
            except NotImplementedError:
                pass
            return False

        self.scales = array(scaling) if scaling is not None else None
        if vec_is_default(self.scales, 1):
            self.scales = 1  # CAVE: 1 is not array(1)
        elif self.scales.shape is not () and len(self.scales) != self.N:
            raise _Error('len(scales) == ' + str(len(self.scales)) +
                         ' does not match dimension N == ' + str(self.N))

        self.typical_x = array(typical_x) if typical_x is not None else None
        if vec_is_default(self.typical_x, 0):
            self.typical_x = 0
        elif self.typical_x.shape is not () and len(self.typical_x) != self.N:
            raise _Error('len(typical_x) == ' + str(len(self.typical_x)) +
                         ' does not match dimension N == ' + str(self.N))

        if (self.scales is 1 and
                self.typical_x is 0 and
                self.fixed_values is None and
                self.tf_pheno is None):
            self.isidentity = True
        else:
            self.isidentity = False

    def pheno(self, x, into_bounds=(lambda x, copy=False: x if not copy else array(x, copy=copy)),
              copy=True, copy_always=False,
              archive=None, iteration=None):
        """maps the genotypic input argument into the phenotypic space, see
        help for class `GenoPheno`

        Details
        -------
        If ``copy``, values from ``x`` are copied if changed under the transformation.

        """
        # TODO: copy_always seems superfluous, as it could be done in the calling code
        if copy_always and not copy:
            raise ValueError('arguments copy_always=' + str(copy_always) +
                             ' and copy=' + str(copy) + ' have inconsistent values')
        if copy_always:
            x = array(x, copy=True)
            copy = False

        if self.isidentity:
            y = into_bounds(x) # was into_bounds(x, False) before (bug before v0.96.22)
        else:
            if self.fixed_values is None:
                y = array(x, copy=copy)  # make a copy, in case
            else:  # expand with fixed values
                y = list(x)  # is a copy
                for i in sorted(self.fixed_values.keys()):
                    y.insert(i, self.fixed_values[i])
                y = array(y, copy=False)
            copy = False

            if self.scales is not 1:  # just for efficiency
                y *= self.scales

            if self.typical_x is not 0:
                y += self.typical_x

            if self.tf_pheno is not None:
                y = array(self.tf_pheno(y), copy=False)

            y = into_bounds(y, copy)  # copy is False

            if self.fixed_values is not None:
                for i, k in list(self.fixed_values.items()):
                    y[i] = k

        if archive is not None:
            archive.insert(y, geno=x, iteration=iteration)
        return y

    def geno(self, y, from_bounds=lambda x: x,
             copy_if_changed=True, copy_always=False,
             repair=None, archive=None):
        """maps the phenotypic input argument into the genotypic space,
        that is, computes essentially the inverse of ``pheno``.

        By default a copy is made only to prevent to modify ``y``.

        The inverse of the user-defined transformation (if any)
        is only needed if external solutions are injected, it is not
        applied to the initial solution x0.

        Details
        =======
        ``geno`` searches first in ``archive`` for the genotype of
        ``y`` and returns the found value, typically unrepaired.
        Otherwise, first ``from_bounds`` is applied, to revert a
        projection into the bound domain, (if necessary) and ``pheno``
        is reverted. ``repair`` is applied last, and is usually the
        method ``CMAEvolutionStrategy.repair_genotype`` that limits the
        Mahalanobis norm of ``geno(y) - mean``.

        """
        if archive is not None:
            try:
                x = archive[y]['geno']
            except KeyError:
                x = None
            if x is not None:
                if archive[y]['iteration'] < archive.last_iteration and repair is not None:
                    x = repair(x, copy_if_changed=copy_always)
                return x
        x = y
        if copy_always:
            x = array(y, copy=True)
            copy = False
        else:
            copy = copy_if_changed

        x = from_bounds(x)  # TODO should also take copy?

        if self.isidentity:
            if repair is not None:
                x = repair(x, copy)
            return x

        if copy:  # could be improved?
            x = array(x, copy=True)
            copy = False

        # user-defined transformation
        if self.tf_geno is not None:
            x = array(self.tf_geno(x), copy=False)
        elif self.tf_pheno is not None:
            raise ValueError('t1 of options transformation was not defined but is needed as being the inverse of t0')

        # affine-linear transformation: shift and scaling
        if self.typical_x is not 0:
            x -= self.typical_x
        if self.scales is not 1:  # just for efficiency
            x /= self.scales

        # kick out fixed_values
        if self.fixed_values is not None:
            # keeping the transformed values does not help much
            # therefore it is omitted
            if 1 < 3:
                keys = sorted(self.fixed_values.keys())
                x = array([x[i] for i in range(len(x)) if i not in keys], copy=False)
        # repair injected solutions
        if repair is not None:
            x = repair(x, copy)
        return x

# ____________________________________________________________
# ____________________________________________________________
# check out built-in package abc: class ABCMeta, abstractmethod, abstractproperty...
# see http://docs.python.org/whatsnew/2.6.html PEP 3119 abstract base classes
#
class OOOptimizer(object):
    """"abstract" base class for an OO optimizer interface.

     Relevant methods are `__init__`, `ask`, `tell`, `stop`, `result`,
     and `optimize`. Only `optimize` is fully implemented in this base
     class.

    Examples
    --------
    All examples minimize the function `elli`, the output is not shown.
    (A preferred environment to execute all examples is ``ipython -pylab``.)
    First we need::

        from cma import CMAEvolutionStrategy, CMADataLogger
        # CMAEvolutionStrategy derives from the OOOptimizer class
        felli = lambda x: sum(1e3**((i-1.)/(len(x)-1.)*x[i])**2 for i in range(len(x)))

    The shortest example uses the inherited method `OOOptimizer.optimize()`::

        res = CMAEvolutionStrategy(8 * [0.1], 0.5).optimize(felli)

    The input parameters to `CMAEvolutionStrategy` are specific to this
    inherited class. The remaining functionality is based on interface
    defined by `OOOptimizer`. We might have a look at the result::

        print(res[0])  # best solution and
        print(res[1])  # its function value

    `res` is the return value from method `CMAEvolutionStrategy.result()`
    appended with CMAEvolutionStrategy.logger.
    In order to display more exciting output we do::

        res[-1].plot()  # if matplotlib is available

    Virtually the same example can be written with an explicit loop
    instead of using `optimize()`. This gives the necessary insight into
    the `OOOptimizer` class interface and gives entire control over the
    iteration loop::

        optim = CMAEvolutionStrategy(9 * [0.5], 0.3)
        # a new CMAEvolutionStrategy instance calling CMAEvolutionStrategy.__init__()
        logger = CMADataLogger(optim)
        # a logger instance, we could also use the instance optim.logger

        # this loop resembles optimize()
        while not optim.stop(): # iterate
            X = optim.ask()     # get candidate solutions
            f = [felli(x) for x in X]  # evaluate solutions
            #  in case do something else that needs to be done
            optim.tell(X, f)    # do all the real work: prepare for next iteration
            optim.disp(20)      # display info every 20th iteration
            logger.add()        # log another "data line"

        # final output
        print('termination by', optim.stop())
        print('best f-value =', optim.result()[1])
        print('best solution =', optim.result()[0])
        logger.plot()  # if matplotlib is available
        input('press enter to continue')  # prevents exiting and closing figures

    Details
    -------
    Most of the work is done in the method `tell(...)`. The method
    `result()` returns more useful output.

    """
    def __init__(self, xstart, **more_args):
        """``xstart`` is a mandatory argument"""
        self.xstart = xstart
        self.more_args = more_args
        self.initialize()
    def initialize(self):
        """(re-)set to the initial state"""
        self.countiter = 0
        self.xcurrent = self.xstart[:]
        raise NotImplementedError('method initialize() must be implemented in derived class')
    def ask(self):
        """abstract method, AKA "get" or "sample_distribution", deliver
        new candidate solution(s), a list of "vectors"
        """
        raise NotImplementedError('method ask() must be implemented in derived class')
    def tell(self, solutions, function_values):
        """abstract method, AKA "update", prepare for next iteration"""
        self.countiter += 1
        raise NotImplementedError('method tell() must be implemented in derived class')
    def stop(self):
        """abstract method, return satisfied termination conditions in
        a dictionary like ``{'termination reason': value, ...}``,
        for example ``{'tolfun': 1e-12}``, or the empty dictionary ``{}``.
        The implementation of `stop()` should prevent an infinite
        loop.
        """
        raise NotImplementedError('method stop() is not implemented')
    def disp(self, modulo=None):
        """abstract method, display some iteration infos if
        ``self.iteration_counter % modulo == 0`` """
        pass  # raise NotImplementedError('method disp() is not implemented')
    def result(self):
        """abstract method, return ``(x, f(x), ...)``, that is, the
        minimizer, its function value, ..."""
        raise NotImplementedError('method result() is not implemented')

    # previous ordering:
    #    def optimize(self, objectivefct,
    #                 logger=None, verb_disp=20,
    #                 iterations=None, min_iterations=1,
    #                 call_back=None):
    def optimize(self, objective_fct,
                 iterations=None, min_iterations=1, args=(),
                 verb_disp=None, logger=None,
                 call_back=None):
        """find minimizer of `objective_fct`.

        CAVEAT: the return value for `optimize` is versatile and might
        change in near future to ``self``.

        Arguments
        ---------

            `objective_fct`
                function be to minimized
            `iterations`
                number of (maximal) iterations, while ``not self.stop()``
            `min_iterations`
                minimal number of iterations, even if ``not self.stop()``
            `args`
                arguments passed to `objective_fct`
            `verb_disp`
                print to screen every `verb_disp` iteration, if ``None``
                the value from ``self.logger`` is "inherited", if
                available.
            ``logger``
                a `BaseDataLogger` instance, which must be compatible
                with the type of ``self``.
            ``call_back``
                call back function called like ``call_back(self)`` or
                a list of call back functions.

        ``return self.result() + (self.stop(), self, logger)`` which
        might change in near future.

        Example
        -------
        >>> import cma
        >>> res = cma.CMAEvolutionStrategy(7 * [0.1], 0.5).optimize(cma.fcts.rosen, verb_disp=100)
        (4_w,9)-CMA-ES (mu_w=2.8,w_1=49%) in dimension 7 (seed=630721393)
        Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
            1       9 3.163954777181882e+01 1.0e+00 4.12e-01  4e-01  4e-01 0:0.0
            2      18 3.299006223906629e+01 1.0e+00 3.60e-01  3e-01  4e-01 0:0.0
            3      27 1.389129389866704e+01 1.1e+00 3.18e-01  3e-01  3e-01 0:0.0
          100     900 2.494847340045985e+00 8.6e+00 5.03e-02  2e-02  5e-02 0:0.3
          200    1800 3.428234862999135e-01 1.7e+01 3.77e-02  6e-03  3e-02 0:0.5
          300    2700 3.216640032470860e-04 5.6e+01 6.62e-03  4e-04  9e-03 0:0.8
          400    3600 6.155215286199821e-12 6.6e+01 7.44e-06  1e-07  4e-06 0:1.1
          438    3942 1.187372505161762e-14 6.0e+01 3.27e-07  4e-09  9e-08 0:1.2
          438    3942 1.187372505161762e-14 6.0e+01 3.27e-07  4e-09  9e-08 0:1.2
        ('termination by', {'tolfun': 1e-11})
        ('best f-value =', 1.1189867885201275e-14)
        ('solution =', array([ 1.        ,  1.        ,  1.        ,  0.99999999,  0.99999998,
                0.99999996,  0.99999992]))
        >>> print(res[0])
        [ 1.          1.          1.          0.99999999  0.99999998  0.99999996
          0.99999992]

        """
        assert iterations is None or min_iterations <= iterations
        if not hasattr(self, 'logger'):
            self.logger = logger
        if logger is None:
            logger = self.logger
        self.logger_from_optimize_method_call = logger
        if not isinstance(call_back, list):
            call_back = [call_back]

        citer = 0
        while not self.stop() or citer < min_iterations:
            if iterations is not None and citer >= iterations:
                return self.result()
            citer += 1

            X = self.ask()  # deliver candidate solutions
            fitvals = [objective_fct(x, *args) for x in X]
            self.tell(X, fitvals)  # all the work is done here
            self.disp(verb_disp)
            for f in call_back:
                if f is not None:
                    f(self)
            logger.add(self) if logger else None

        # signal logger that we left the loop
        # TODO: this is very ugly, because it assumes modulo keyword
        #       argument *and* modulo attribute to be available
        try:
            logger.add(self, modulo=bool(logger.modulo)) if logger else None
        except TypeError:
            print('  suppressing the final call of the logger in OOOptimizer.optimize (modulo keyword parameter not available)')
        except AttributeError:
            print('  suppressing the final call of the logger in OOOptimizer.optimize (modulo attribute not available)')
        if verb_disp:
            self.disp(1)
        if verb_disp in (1, True):
            print('termination by', self.stop())
            print('best f-value =', self.result()[1])
            print('solution =', self.result()[0])

        return self.result() + (self.stop(), self, logger)

class CMAAdaptSigmaBase(object):
    """step-size adaptation base class, implementing hsig functionality
    via an isotropic evolution path.

    """
    def __init__(self, *args, **kwargs):
        self.is_initialized_base = False
        self._ps_updated_iteration = -1
    def initialize_base(self, es):
        """set parameters and state variable based on dimension,
        mueff and possibly further options.

        """
        self.cs = (es.sp.mueff + 2) / (es.N + es.sp.mueff + 3)
        self.ps = np.zeros(es.N)
        self.is_initialized_base = True
    def _update_ps(self, es):
        """update the isotropic evolution path

        :type es: CMAEvolutionStrategy
        """
        if not self.is_initialized_base:
            self.initialize_base(es)
        if self._ps_updated_iteration == es.countiter:
            return
        if es.countiter <= es.itereigenupdated:
            # es.B and es.D must/should be those from the last iteration
            assert es.countiter >= es.itereigenupdated
            _print_warning('distribution transformation (B and D) have been updated before ps could be computed',
                          '_update_ps', 'CMAAdaptSigmaBase')
        z = dot(es.B, (1. / es.D) * dot(es.B.T, (es.mean - es.mean_old) / es.sigma_vec))
        z *= es.sp.mueff**0.5 / es.sigma / es.sp.cmean
        # self.cs or es.sp.cs could be used here
        self.ps = (1 - self.cs) * self.ps + sqrt(self.cs * (2 - self.cs)) * z
        self._ps_updated_iteration = es.countiter
    def hsig(self, es):
        """return "OK-signal" for rank-one update, `True` (OK) or `False`
        (stall rank-one update), based on the length of an evolution path

        """
        self._update_ps(es)
        if self.ps is None:
            return True
        squared_sum = sum(self.ps**2) / (1 - (1 - self.cs)**(2 * es.countiter))
        # correction with self.countiter seems not necessary,
        # as pc also starts with zero
        return squared_sum / es.N - 1 < 1 + 4. / (es.N + 1)
    def update(self, es, **kwargs):
        """update ``es.sigma``"""
        self._update_ps(es)
        raise NotImplementedError('must be implemented in a derived class')
class CMAAdaptSigmaNone(CMAAdaptSigmaBase):
    def update(self, es, **kwargs):
        """no update, ``es.sigma`` remains constant.

        :param es: ``CMAEvolutionStrategy`` class instance
        :param kwargs: whatever else is needed to update ``es.sigma``

        """
        pass
class CMAAdaptSigmaDistanceProportional(CMAAdaptSigmaBase):
    """artificial setting of ``sigma`` for test purposes, e.g.
    to simulate optimal progress rates.

    """
    def __init__(self, coefficient=1.2):
        self.coefficient = coefficient
    def update(self, es, **kwargs):
        # optimal step-size is
        es.sigma = self.coefficient * self.sp.mueff * sum(self.mean**2)**0.5 / self.N
class CMAAdaptSigmaCSA(CMAAdaptSigmaBase):
    def __init__(self):
        """postpone initialization to a method call where dimension and mueff should be known.

        """
        self.is_initialized = False
    def initialize(self, es):
        """set parameters and state variable based on dimension,
        mueff and possibly further options.

        """
        self.disregard_length_setting = True if es.opts['CSA_disregard_length'] else False
        if es.opts['CSA_clip_length_value'] is not None:
            try:
                if len(es.opts['CSA_clip_length_value']) == 0:
                    es.opts['CSA_clip_length_value'] = [-np.Inf, np.Inf]
                elif len(es.opts['CSA_clip_length_value']) == 1:
                    es.opts['CSA_clip_length_value'] = [-np.Inf, es.opts['CSA_clip_length_value'][0]]
                elif len(es.opts['CSA_clip_length_value']) == 2:
                    es.opts['CSA_clip_length_value'] = np.sort(es.opts['CSA_clip_length_value'])
                else:
                    raise ValueError('option CSA_clip_length_value should be a number of len(.) in [1,2]')
            except TypeError:  # len(...) failed
                es.opts['CSA_clip_length_value'] = [-np.Inf, es.opts['CSA_clip_length_value']]
            es.opts['CSA_clip_length_value'] = list(np.sort(es.opts['CSA_clip_length_value']))
            if es.opts['CSA_clip_length_value'][0] > 0 or es.opts['CSA_clip_length_value'][1] < 0:
                raise ValueError('option CSA_clip_length_value must be a single positive or a negative and a positive number')
        self.cs = (es.sp.mueff + 2) / (es.N + es.sp.mueff + 3)
        self.damps = es.opts['CSA_dampfac'] * (0.5 +
                                          0.5 * min([1, (es.sp.lam_mirr / (0.159 * es.sp.popsize) - 1)**2])**1 +
                                          2 * max([0, ((es.sp.mueff - 1) / (es.N + 1))**es.opts['CSA_damp_mueff_exponent'] - 1]) +
                                          self.cs
                                          )
        self.max_delta_log_sigma = 1  # in symmetric use (strict lower bound is -cs/damps anyway)

        if self.disregard_length_setting:
            es.opts['CSA_clip_length_value'] = [0, 0]
            self.cs = (es.sp.mueff + 1)**0.5 / (es.N**0.5 + 2 * es.sp.mueff**0.5)
            self.damps = es.opts['CSA_dampfac'] * 1  # * (1.1 - 1/(es.N+1)**0.5)
        if es.opts['verbose'] > 1 or self.disregard_length_setting or 11 < 3:
            print('SigmaCSA Parameters')
            for k, v in self.__dict__.items():
                print('  ', k, ':', v)
        self.ps = np.zeros(es.N)
        self._ps_updated_iteration = -1
        self.is_initialized = True

    def _update_ps(self, es):
        if not self.is_initialized:
            self.initialize(es)
        if self._ps_updated_iteration == es.countiter:
            return
        z = dot(es.B, (1. / es.D) * dot(es.B.T, (es.mean - es.mean_old) / es.sigma_vec))
        z *= es.sp.mueff**0.5 / es.sigma / es.sp.cmean
        # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        if es.opts['CSA_clip_length_value'] is not None:
            vals = es.opts['CSA_clip_length_value']
            min_len = es.N**0.5 + vals[0] * es.N / (es.N + 2)
            max_len = es.N**0.5 + vals[1] * es.N / (es.N + 2)
            act_len = sum(z**2)**0.5
            new_len = Mh.minmax(act_len, min_len, max_len)
            if new_len != act_len:
                z *= new_len / act_len
                # z *= (es.N / sum(z**2))**0.5  # ==> sum(z**2) == es.N
                # z *= es.const.chiN / sum(z**2)**0.5
        self.ps = (1 - self.cs) * self.ps + sqrt(self.cs * (2 - self.cs)) * z
        self._ps_updated_iteration = es.countiter
    def update(self, es, **kwargs):
        self._update_ps(es)  # caveat: if es.B or es.D are already updated and ps is not, this goes wrong!
        if es.opts['CSA_squared']:
            s = (sum(self.ps**2) / es.N - 1) / 2
            # sum(self.ps**2) / es.N has mean 1 and std sqrt(2/N) and is skewed
            # divided by 2 to have the derivative d/dx (x**2 / N - 1) for x**2=N equal to 1
        else:
            s = sum(self.ps**2)**0.5 / es.const.chiN - 1
        s *= self.cs / self.damps
        s_clipped = Mh.minmax(s, -self.max_delta_log_sigma, self.max_delta_log_sigma)
        es.sigma *= np.exp(s_clipped)
        # "error" handling
        if s_clipped != s:
            _print_warning('sigma change exp(' + str(s) + ') = ' + str(np.exp(s)) +
                          ' clipped to exp(+-' + str(self.max_delta_log_sigma) + ')',
                          'update',
                          'CMAAdaptSigmaCSA',
                          es.countiter, es.opts['verbose'])
class CMAAdaptSigmaMedianImprovement(CMAAdaptSigmaBase):
    """Compares median fitness against a fitness percentile of the previous iteration,
    see Ait ElHara et al, GECCO 2013.

    """
    def __init__(self):
        CMAAdaptSigmaBase.__init__(self)
    def initialize(self, es):
        r = es.sp.mueff / es.popsize
        self.index_to_compare = 0.5 * (r**0.5 + 2.0 * (1 - r**0.5) / log(es.N + 9)**2) * (es.popsize)  # TODO
        self.index_to_compare = (0.30 if not es.opts['vv'] else es.opts['vv']) * es.popsize  # TODO
        self.damp = 2 - 2 / es.N  # sign-rule: 2
        self.c = 0.3  # sign-rule needs <= 0.3
        self.s = 0  # averaged statistics, usually between -1 and +1
    def update(self, es, **kwargs):
        if es.countiter < 2:
            self.initialize(es)
            self.fit = es.fit.fit
        else:
            ft1, ft2 = self.fit[int(self.index_to_compare)], self.fit[int(np.ceil(self.index_to_compare))]
            ftt1, ftt2 = es.fit.fit[(es.popsize - 1) // 2], es.fit.fit[int(np.ceil((es.popsize - 1) / 2))]
            pt2 = self.index_to_compare - int(self.index_to_compare)
            # ptt2 = (es.popsize - 1) / 2 - (es.popsize - 1) // 2  # not in use
            s = 0
            if 1 < 3:
                s += (1 - pt2) * sum(es.fit.fit <= self.fit[int(np.ceil(self.index_to_compare))])
                s += pt2 * sum(es.fit.fit < self.fit[int(self.index_to_compare)])
                s -= (es.popsize + 1) / 2
                s *= 2 / es.popsize  # the range was popsize, is 2
            self.s = (1 - self.c) * self.s + self.c * s
            es.sigma *= exp(self.s / self.damp)
        # es.more_to_write.append(10**(self.s))

        #es.more_to_write.append(10**((2 / es.popsize) * (sum(es.fit.fit < self.fit[int(self.index_to_compare)]) - (es.popsize + 1) / 2)))
        # # es.more_to_write.append(10**(self.index_to_compare - sum(self.fit <= es.fit.fit[es.popsize // 2])))
        # # es.more_to_write.append(10**(np.sign(self.fit[int(self.index_to_compare)] - es.fit.fit[es.popsize // 2])))
        self.fit = es.fit.fit
class CMAAdaptSigmaTPA(CMAAdaptSigmaBase):
    """two point adaptation for step-size sigma. Relies on a specific
    sampling of the first two offspring, whose objective function
    value ranks are used to decide on the step-size change.

    Example
    =======

    >>> import cma
    >>> cma.CMAOptions('adapt').pprint()
    >>> es = cma.CMAEvolutionStrategy(10 * [0.2], 0.1, {'AdaptSigma': cma.CMAAdaptSigmaTPA, 'ftarget': 1e-8})
    >>> es.optimize(cma.fcts.rosen)
    >>> assert 'ftarget' in es.stop()
    >>> assert es.result()[1] <= 1e-8
    >>> assert es.result()[2] < 6500  # typically < 5500

    References: loosely based on Hansen 2008, CMA-ES with Two-Point
    Step-Size Adaptation, more tightly based on an upcoming paper by
    Hansen et al.

    """
    def __init__(self, dimension=None, opts=None):
        CMAAdaptSigmaBase.__init__(self) # base class provides method hsig()
        self.initialized = False
    def initialize(self, N):
        self.sp = _BlancClass()
        self.sp.damp = eval('N**0.5')  # why do we need 10 <-> exp(1/10) == 1.1? 2 should be fine!?
        self.sp.dampup = 1.0 * self.sp.damp  # 0.5 fails to converge on the Rastrigin function
        self.sp.dampdown = 1.0 * self.sp.damp
        self.sp.c = 1.0  # rank difference is asymetric and therefore the switch from increase to decrease takes too long
        self.sp.z_exponent = 0.5  # sign(z) * abs(z)**z_exponent, 0.5 seems better with larger popsize
        self.sp.sigma_fac = 1.0  # (obsolete) 0.5 feels better, but no evidence whether it is
        self.sp.relative_to_delta_mean = True  # (obsolete)
        self.s = 0  # the state variable
        self.last = None
        self.initialized = True
        return self
    def update(self, es, function_values, **kwargs):
        """the first and second value in ``function_values``
        must reflect two mirrored solutions sampled
        in direction / in opposite direction of
        the previous mean shift, respectively.

        """
        # TODO: on the linear function, the two mirrored samples lead
        # to a sharp increase of condition of the covariance matrix.
        # They should not be used to update the covariance matrix,
        # if the step-size inreases quickly.
        if not self.initialized:
            self.initialize(es.N)
        if 1 < 3:
            # use the ranking difference of the mirrors for adaptation
            # damp = 5 should be fine
            z = np.where(es.fit.idx == 1)[0][0] - np.where(es.fit.idx == 0)[0][0]
            z /= es.popsize - 1
        self.s = (1 - self.sp.c) * self.s + self.sp.c * np.sign(z) * np.abs(z)**self.sp.z_exponent
        if self.s > 0:
            es.sigma *= exp(self.s / self.sp.dampup)
        else:
            es.sigma *= exp(self.s / self.sp.dampdown)
        #es.more_to_write.append(10**z)


# ____________________________________________________________
# ____________________________________________________________
#
class CMAEvolutionStrategy(OOOptimizer):
    """CMA-ES stochastic optimizer class with ask-and-tell interface.

    Calling Sequences
    =================

        ``es = CMAEvolutionStrategy(x0, sigma0)``

        ``es = CMAEvolutionStrategy(x0, sigma0, opts)``

        ``res = CMAEvolutionStrategy(x0, sigma0).optimize(objective_fct)``

    CAVEAT: return value of `optimize` might become ``optim`` in near
    future.

    Arguments
    =========
        `x0`
            initial solution, starting point. `x0` is given as "genotype"
            which means, if::

                opts={'transformation':[transform, inverse]}

            is given (``inverse`` can be ``None``), then ``transform(x0)``
            is the "phenotypic" initial solution and
            ``objective_function(transform(x0))`` is the objective
            function value of ``x0``.

        `sigma0`
            initial standard deviation.  The problem variables should
            have been scaled, such that a single standard deviation
            on all variables is useful and the optimum is expected to
            lie within about `x0` +- ``3*sigma0``. See also options
            `scaling_of_variables`. Often one wants to check for
            solutions close to the initial point. This allows,
            for example, for an easier check of consistency of the
            objective function and its interfacing with the optimizer.
            In this case, a much smaller `sigma0` is advisable.
        `opts`
            options, a dictionary with optional settings,
            see class `CMAOptions`.

    Main interface / usage
    ======================
    The interface is inherited from the generic `OOOptimizer`
    class (see also there). An object instance is generated from

        es = cma.CMAEvolutionStrategy(8 * [0.5], 0.2)

    The least verbose interface is via the optimize method::

        es.optimize(objective_func)
        res = es.result()

    More verbosely, the optimization is done using the
    methods ``stop``, ``ask``, and ``tell``::

        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [cma.fcts.rosen(s) for s in solutions])


    where ``ask`` delivers new candidate solutions and ``tell`` updates
    the ``optim`` instance by passing the respective function values
    (the objective function ``cma.fcts.rosen`` can be replaced by any
    properly defined objective function, see ``cma.fcts`` for more
    examples).

    The class `CMAEvolutionStrategy` also provides::

        (solutions, func_values) = es.ask_and_eval(objective_func)

    and an entire optimization can also be written like::

        while not es.stop():
            es.tell(*es.ask_and_eval(objective_func))

    Besides for termination criteria, in CMA-ES only the ranks of the
    `func_values` are relevant.

    Attributes and Properties
    =========================
        - `inputargs` -- passed input arguments
        - `inopts` -- passed options
        - `opts` -- actually used options, some of them can be changed any
          time, see class `CMAOptions`
        - `popsize` -- population size lambda, number of candidate solutions
          returned by `ask()`
        - `logger` -- a `CMADataLogger` instance utilized by `optimize`

    Examples
    ========
    Super-short example, with output shown:

    >>> import cma
    >>> # construct an object instance in 4-D, sigma0=1:
    >>> es = cma.CMAEvolutionStrategy(4 * [1], 1, {'seed':234})
    (4_w,8)-CMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=234)
    >>>
    >>> # optimize the ellipsoid function
    >>> es.optimize(cma.fcts.elli, verb_disp=1)
    Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
        1       8 2.093015112685775e+04 1.0e+00 9.27e-01  9e-01  9e-01 0:0.0
        2      16 4.964814235917688e+04 1.1e+00 9.54e-01  9e-01  1e+00 0:0.0
        3      24 2.876682459926845e+05 1.2e+00 1.02e+00  9e-01  1e+00 0:0.0
      100     800 6.809045875281943e-01 1.3e+02 1.41e-02  1e-04  1e-02 0:0.2
      200    1600 2.473662150861846e-10 8.0e+02 3.08e-05  1e-08  8e-06 0:0.5
      233    1864 2.766344961865341e-14 8.6e+02 7.99e-07  8e-11  7e-08 0:0.6
    >>>
    >>> cma.pprint(es.result())
    (array([ -1.98546755e-09,  -1.10214235e-09,   6.43822409e-11,
            -1.68621326e-11]),
     4.5119610261406537e-16,
     1666,
     1672,
     209,
     array([ -9.13545269e-09,  -1.45520541e-09,  -6.47755631e-11,
            -1.00643523e-11]),
     array([  3.20258681e-08,   3.15614974e-09,   2.75282215e-10,
             3.27482983e-11]))
    >>> assert es.result()[1] < 1e-9
    >>> help(es.result)
    Help on method result in module cma:

    result(self) method of cma.CMAEvolutionStrategy instance
        return ``(xbest, f(xbest), evaluations_xbest, evaluations, iterations, pheno(xmean), effective_stds)``


    The optimization loop can also be written explicitly.

    >>> import cma
    >>> es = cma.CMAEvolutionStrategy(4 * [1], 1)
    >>> while not es.stop():
    ...    X = es.ask()
    ...    es.tell(X, [cma.fcts.elli(x) for x in X])
    ...    es.disp()
    <output omitted>

    achieving the same result as above.

    An example with lower bounds (at zero) and handling infeasible
    solutions:

    >>> import cma
    >>> import numpy as np
    >>> es = cma.CMAEvolutionStrategy(10 * [0.2], 0.5, {'bounds': [0, np.inf]})
    >>> while not es.stop():
    ...     fit, X = [], []
    ...     while len(X) < es.popsize:
    ...         curr_fit = None
    ...         while curr_fit in (None, np.NaN):
    ...             x = es.ask(1)[0]
    ...             curr_fit = cma.fcts.somenan(x, cma.fcts.elli) # might return np.NaN
    ...         X.append(x)
    ...         fit.append(curr_fit)
    ...     es.tell(X, fit)
    ...     es.logger.add()
    ...     es.disp()
    <output omitted>
    >>>
    >>> assert es.result()[1] < 1e-9
    >>> assert es.result()[2] < 9000  # by internal termination
    >>> # es.logger.plot()  # will plot data
    >>> # cma.show()  # display plot window

    An example with user-defined transformation, in this case to realize
    a lowwer bound of 2.

    >>> es = cma.CMAEvolutionStrategy(5 * [3], 1,
    ...                 {"transformation": [lambda x: x**2+2, None]})
    >>> es.optimize(cma.fcts.rosen)
    <output omitted>
    >>> assert cma.fcts.rosen(es.result()[0]) < 1e-6 + 5.530760944396627e+02
    >>> assert es.result()[2] < 3300

    The inverse transformation is (only) necessary if the `BoundPenalty`
    boundary handler is used at the same time.

    The ``CMAEvolutionStrategy`` class also provides a default logger
    (cave: files are overwritten when the logger is used with the same
    filename prefix):

    >>> import cma
    >>> es = cma.CMAEvolutionStrategy(4 * [0.2], 0.5, {'verb_disp': 0})
    >>> es.logger.disp_header()  # to understand the print of disp
    Iterat Nfevals  function value    axis ratio maxstd   minstd
    >>> while not es.stop():
    ...     X = es.ask()
    ...     es.tell(X, [cma.fcts.sphere(x) for x in X])
    ...     es.logger.add()  # log current iteration
    ...     es.logger.disp([-1])  # display info for last iteration
    1      8 2.72769793021748e+03 1.0e+00 4.05e-01 3.99e-01
    2     16 6.58755537926063e+03 1.1e+00 4.00e-01 3.39e-01
    <output ommitted>
    193   1544 3.15195320957214e-15 1.2e+03 3.70e-08 3.45e-11
    >>> es.logger.disp_header()
    Iterat Nfevals  function value    axis ratio maxstd   minstd
    >>> # es.logger.plot() # will make a plot

    Example implementing restarts with increasing popsize (IPOP), output
    is not displayed:

    >>> import cma, numpy as np
    >>>
    >>> # restart with increasing population size (IPOP)
    >>> bestever = cma.BestSolution()
    >>> for lam in 10 * 2**np.arange(8):  # 10, 20, 40, 80, ..., 10 * 2**7
    ...     es = cma.CMAEvolutionStrategy('6 - 8 * np.random.rand(9)',  # 9-D
    ...                                   5,  # initial std sigma0
    ...                                   {'popsize': lam,  # options
    ...                                    'verb_append': bestever.evalsall})
    ...     logger = cma.CMADataLogger().register(es, append=bestever.evalsall)
    ...     while not es.stop():
    ...         X = es.ask()    # get list of new solutions
    ...         fit = [cma.fcts.rastrigin(x) for x in X]  # evaluate each solution
    ...         es.tell(X, fit) # besides for termination only the ranking in fit is used
    ...
    ...         # display some output
    ...         logger.add()  # add a "data point" to the log, writing in files
    ...         es.disp()  # uses option verb_disp with default 100
    ...
    ...     print('termination:', es.stop())
    ...     cma.pprint(es.best.__dict__)
    ...
    ...     bestever.update(es.best)
    ...
    ...     # show a plot
    ...     # logger.plot();
    ...     if bestever.f < 1e-8:  # global optimum was hit
    ...         break
    <output omitted>
    >>> assert es.result()[1] < 1e-8

    On the Rastrigin function, usually after five restarts the global optimum
    is located.

    Using the ``multiprocessing`` module, we can evaluate the function in parallel with a simple
    modification of the example (however multiprocessing seems not always reliable)::

        try:
            import multiprocessing as mp
            import cma
            es = cma.CMAEvolutionStrategy(22 * [0.0], 1.0, {'maxiter':10})
            pool = mp.Pool(es.popsize)
            while not es.stop():
                X = es.ask()
                es.tell(X, pool.map_async(cma.felli, X).get()) # use chunksize parameter as popsize/len(pool)?
                es.logger.add()
        except ImportError:
            pass

    The final example shows how to resume:

    >>> import cma, pickle
    >>>
    >>> es = cma.CMAEvolutionStrategy(12 * [0.1],  # a new instance, 12-D
    ...                               0.5)         # initial std sigma0
    >>> es.optimize(cma.fcts.rosen, iterations=100)
    >>> pickle.dump(es, open('saved-cma-object.pkl', 'wb'))
    >>> print('saved')
    >>> del es  # let's start fresh
    >>>
    >>> es = pickle.load(open('saved-cma-object.pkl', 'rb'))
    >>> print('resumed')
    >>> es.optimize(cma.fcts.rosen, verb_disp=200)
    >>> assert es.result()[2] < 15000
    >>> cma.pprint(es.result())

    Details
    =======
    The following two enhancements are implemented, the latter is turned
    on by default only for very small population size.

    *Active CMA* is implemented with option ``CMA_active`` and
    conducts an update of the covariance matrix with negative weights.
    The negative update is implemented, such that positive definiteness
    is guarantied. The update is applied after the default update and
    only before the covariance matrix is decomposed, which limits the
    additional computational burden to be at most a factor of three
    (typically smaller). A typical speed up factor (number of
    f-evaluations) is between 1.1 and two.

    References: Jastrebski and Arnold, CEC 2006, Glasmachers et al, GECCO 2010.

    *Selective mirroring* is implemented with option ``CMA_mirrors``
    in the method ``get_mirror()``. Only the method `ask_and_eval()`
    (used by `fmin`) will then sample selectively mirrored vectors. In
    selective mirroring, only the worst solutions are mirrored. With
    the default small number of mirrors, *pairwise selection* (where at
    most one of the two mirrors contribute to the update of the
    distribution mean) is implicitly guarantied under selective
    mirroring and therefore not explicitly implemented.

    References: Brockhoff et al, PPSN 2010, Auger et al, GECCO 2011.

    :See: `fmin()`, `CMAOptions`, `plot()`, `ask()`, `tell()`, `ask_and_eval()`

    """
    # ____________________________________________________________
    @property  # read only attribute decorator for a method
    def popsize(self):
        """number of samples by default returned by` ask()`
        """
        return self.sp.popsize

    # this is not compatible with python2.5:
    #     @popsize.setter
    #     def popsize(self, p):
    #         """popsize cannot be set (this might change in future)
    #         """
    #         raise _Error("popsize cannot be changed (this might change in future)")

    # ____________________________________________________________
    # ____________________________________________________________
    def stop(self, check=True):
        """return a dictionary with the termination status.
        With ``check==False``, the termination conditions are not checked and
        the status might not reflect the current situation.

        """
        if (check and self.countiter > 0 and self.opts['termination_callback'] and
                self.opts['termination_callback'] != str(self.opts['termination_callback'])):
            self.callbackstop = self.opts['termination_callback'](self)

        return self.stopdict(self if check else None)  # update the stopdict and return a Dict

    # ____________________________________________________________
    # ____________________________________________________________
    def __init__(self, x0, sigma0, inopts={}):
        """see class `CMAEvolutionStrategy`

        """
        self.inputargs = dict(locals())  # for the record
        del self.inputargs['self']  # otherwise the instance self has a cyclic reference
        self.inopts = inopts
        opts = CMAOptions(inopts).complement()  # CMAOptions() == fmin([],[]) == defaultOptions()

        if 'noise_handling' in opts and opts.eval('noise_handling'):
            raise ValueError('noise_handling not available with class CMAEvolutionStrategy, use function fmin')
        if 'restarts' in opts and opts.eval('restarts'):
            raise ValueError('restarts not available with class CMAEvolutionStrategy, use function fmin')

        self.set_x0(x0)  # manage weird shapes, set self.x0
        self.N_pheno = len(self.x0)

        self.sigma0 = sigma0
        if isinstance(sigma0, str):  # TODO: no real need here (do rather in fmin)
            self.sigma0 = eval(sigma0)  # like '1./N' or 'np.random.rand(1)[0]+1e-2'
        if np.size(self.sigma0) != 1 or np.shape(self.sigma0):
            raise _Error('input argument sigma0 must be (or evaluate to) a scalar')
        self.sigma = self.sigma0  # goes to inialize

        # extract/expand options
        N = self.N_pheno
        assert isinstance(opts['fixed_variables'], (str, dict)) \
            or opts['fixed_variables'] is None
        # TODO: in case of a string we need to eval the fixed_variables
        if isinstance(opts['fixed_variables'], dict):
            N = self.N_pheno - len(opts['fixed_variables'])
        opts.evalall(locals())  # using only N
        self.opts = opts

        self.randn = opts['randn']
        self.gp = GenoPheno(self.N_pheno, opts['scaling_of_variables'], opts['typical_x'],
            opts['fixed_variables'], opts['transformation'])
        self.boundary_handler = opts.eval('boundary_handling')(opts.eval('bounds'))
        print('initial solution x0=',self.gp.pheno(self.x0)) #DEBUG
        if not self.boundary_handler.has_bounds():
            self.boundary_handler = BoundNone()  # just a little faster and well defined
        elif not self.boundary_handler.is_in_bounds(self.gp.pheno(self.x0)):
            if opts['verbose'] >= 0:
                print('WARNING: initial solution is out of the domain boundaries:')
                print('  x0   = ' + str(self.gp.pheno(self.x0)))
                print('  ldom = ' + str(self.boundary_handler.bounds[0]))
                print('  udom = ' + str(self.boundary_handler.bounds[1]))

        # self.mean = array(self.x0, copy=True)
        tmp,  self.gp.tf_geno = self.gp.tf_geno, lambda x: x  # a hack to avoid an exception if tf_geno is None
        self.mean = self.gp.geno(self.x0, copy_always=True)
        self.gp.tf_geno = tmp
        # without copy_always interface:
        # self.mean = self.gp.geno(array(self.x0, copy=True), copy_if_changed=False)
        self.N = len(self.mean)
        assert N == self.N
        self.fmean = np.NaN  # TODO name should change? prints nan in output files (OK with matlab&octave)
        self.fmean_noise_free = 0.  # for output only

        self.adapt_sigma = opts['AdaptSigma']
        if self.adapt_sigma is False:
            self.adapt_sigma = CMAAdaptSigmaNone
        self.adapt_sigma = self.adapt_sigma()  # class instance

        self.sp = _CMAParameters(N, opts)
        self.sp0 = self.sp  # looks useless, as it is not a copy

        # initialization of state variables
        self.countiter = 0
        self.countevals = max((0, opts['verb_append'])) \
            if not isinstance(opts['verb_append'], bool) else 0
        self.pc = np.zeros(N)

        self.sigma_vec = np.ones(N) if np.isfinite(self.sp.dampsvec) else 1
        stds = np.ones(N)
        if self.opts['CMA_teststds'] is not None and np.all(self.opts['CMA_teststds']):  # also 0 would not make sense
            stds = array(self.opts['CMA_teststds'])
            if np.size(stds) != N:
                raise _Error('CMA_teststds option must have dimension = ' + str(N))
        if self.opts['CMA_diagonal']:  # is True or > 0
            # linear time and space complexity
            self.B = array(1)  # works fine with np.dot(self.B, anything) and self.B.T
            self.C = stds**2  # TODO: remove this!?
            self.dC = self.C
        else:
            self.B = np.eye(N)  # identity(N), do not from matlib import *, as eye is a matrix there
            # prevent equal eigenvals, a hack for np.linalg:
            self.C = np.diag(stds**2 * exp(1e-6 * (np.random.rand(N) - 0.5)))
            self.dC = np.diag(self.C).copy()
            self.Yneg = np.zeros((N, N))
        self.D = stds

        # self.gp.pheno adds fixed variables
        relative_stds = ((self.gp.pheno(self.mean + self.sigma * self.sigma_vec * self.D)
                          - self.gp.pheno(self.mean - self.sigma * self.sigma_vec * self.D)) / 2.0
                         / (self.boundary_handler.get_bounds('upper', self.N_pheno)
                            - self.boundary_handler.get_bounds('lower', self.N_pheno)))
        if np.any(relative_stds > 1):
            raise ValueError('initial standard deviations larger than the bounded domain size in variables '
                         + str(np.where(relative_stds > 1)[0]))
        self.flgtelldone = True
        self.itereigenupdated = self.countiter
        self.noiseS = 0  # noise "signal"
        self.hsiglist = []

        if not opts['seed']:
            np.random.seed()
            six_decimals = (time.time() - 1e6 * (time.time() // 1e6))
            opts['seed'] = 1e5 * np.random.rand() + six_decimals + 1e5 * (time.time() % 1)
        opts['seed'] = int(opts['seed'])
        np.random.seed(opts['seed'])

        self.sent_solutions = CMASolutionDict()
        self.archive = CMASolutionDict()
        self.best = BestSolution()

        out = {}  # TODO: obsolete, replaced by method results()?
        out['best'] = self.best
        # out['hsigcount'] = 0
        out['termination'] = {}
        self.out = out

        self.const = _BlancClass()
        self.const.chiN = N**0.5 * (1 - 1. / (4.*N) + 1. / (21.*N**2))  # expectation of norm(randn(N,1))

        self.logger = CMADataLogger(opts['verb_filenameprefix'], modulo=opts['verb_log']).register(self)

        # attribute for stopping criteria in function stop
        self.stopdict = CMAStopDict()
        self.callbackstop = 0

        self.fit = _BlancClass()
        self.fit.fit = []  # not really necessary
        self.fit.hist = []  # short history of best
        self.fit.histbest = []  # long history of best
        self.fit.histmedian = []  # long history of median

        self.more_to_write = []  # [1, 1, 1, 1]  #  N*[1]  # needed when writing takes place before setting

        # say hello
        if opts['verb_disp'] > 0 and opts['verbose'] >= 0:
            sweighted = '_w' if self.sp.mu > 1 else ''
            smirr = 'mirr%d' % (self.sp.lam_mirr) if self.sp.lam_mirr else ''
            print('(%d' % (self.sp.mu) + sweighted + ',%d' % (self.sp.popsize) + smirr +
                  ')-' + ('a' if opts['CMA_active'] else '') + 'CMA-ES' +
                  ' (mu_w=%2.1f,w_1=%d%%)' % (self.sp.mueff, int(100 * self.sp.weights[0])) +
                  ' in dimension %d (seed=%d, %s)' % (N, opts['seed'], time.asctime()))  # + func.__name__
            if opts['CMA_diagonal'] and self.sp.CMA_on:
                s = ''
                if opts['CMA_diagonal'] is not True:
                    s = ' for '
                    if opts['CMA_diagonal'] < np.inf:
                        s += str(int(opts['CMA_diagonal']))
                    else:
                        s += str(np.floor(opts['CMA_diagonal']))
                    s += ' iterations'
                    s += ' (1/ccov=' + str(round(1. / (self.sp.c1 + self.sp.cmu))) + ')'
                print('   Covariance matrix is diagonal' + s)

    def set_x0(self, x0):
        if x0 == str(x0):
            x0 = eval(x0)
        self.x0 = array(x0)  # should not have column or row, is just 1-D
        if self.x0.ndim == 2:
            if self.opts.eval('verbose') >= 0:
                print('WARNING: input x0 should be a list or 1-D array, trying to flatten ' +
                        str(self.x0.shape) + '-array')
            if self.x0.shape[0] == 1:
                self.x0 = self.x0[0]
            elif self.x0.shape[1] == 1:
                self.x0 = array([x[0] for x in self.x0])
        if self.x0.ndim != 1:
            raise _Error('x0 must be 1-D array')
        #if len(self.x0) <= 1:
            #raise _Error('optimization in 1-D is not supported (code was never tested)')
        if len(self.x0) <= 0:
            raise _Error('x0 is empty')
        if len(self.x0) == 1:
            print('WARNING: optimization in 1-D is not supported (code was never tested)')
        self.x0.resize(self.x0.shape[0])  # 1-D array, not really necessary?!

    # ____________________________________________________________
    # ____________________________________________________________
    def ask(self, number=None, xmean=None, sigma_fac=1):
        """get new candidate solutions, sampled from a multi-variate
        normal distribution and transformed to f-representation
        (phenotype) to be evaluated.

        Arguments
        ---------
            `number`
                number of returned solutions, by default the
                population size ``popsize`` (AKA ``lambda``).
            `xmean`
                distribution mean
            `sigma_fac`
                multiplier for internal sample width (standard
                deviation)

        Return
        ------
        A list of N-dimensional candidate solutions to be evaluated

        Example
        -------
        >>> import cma
        >>> es = cma.CMAEvolutionStrategy([0,0,0,0], 0.3)
        >>> while not es.stop() and es.best.f > 1e-6:  # my_desired_target_f_value
        ...     X = es.ask()  # get list of new solutions
        ...     fit = [cma.fcts.rosen(x) for x in X]  # call function rosen with each solution
        ...     es.tell(X, fit)  # feed values

        :See: `ask_and_eval`, `ask_geno`, `tell`

        """
        pop_geno = self.ask_geno(number, xmean, sigma_fac)


        # N,lambda=20,200: overall CPU 7s vs 5s == 40% overhead, even without bounds!
        #                  new data: 11.5s vs 9.5s == 20%
        # TODO: check here, whether this is necessary?
        # return [self.gp.pheno(x, copy=False, into_bounds=self.boundary_handler.repair) for x in pop]  # probably fine
        # return [Solution(self.gp.pheno(x, copy=False), copy=False) for x in pop]  # here comes the memory leak, now solved
        # pop_pheno = [Solution(self.gp.pheno(x, copy=False), copy=False).repair(self.gp.bounds) for x in pop_geno]
        pop_pheno = [self.gp.pheno(x, copy=True, into_bounds=self.boundary_handler.repair) for x in pop_geno]

        # insert solutions, this could also (better?) be done in self.gp.pheno
        for i in rglen((pop_geno)):
            self.sent_solutions.insert(pop_pheno[i], geno=pop_geno[i], iteration=self.countiter)
        return pop_pheno

    # ____________________________________________________________
    # ____________________________________________________________
    def ask_geno(self, number=None, xmean=None, sigma_fac=1):
        """get new candidate solutions in genotyp, sampled from a
        multi-variate normal distribution.

        Arguments are
            `number`
                number of returned solutions, by default the
                population size `popsize` (AKA lambda).
            `xmean`
                distribution mean
            `sigma_fac`
                multiplier for internal sample width (standard
                deviation)

        `ask_geno` returns a list of N-dimensional candidate solutions
        in genotyp representation and is called by `ask`.

        :See: `ask`, `ask_and_eval`

        """

        if number is None or number < 1:
            number = self.sp.popsize
        if xmean is None:
            xmean = self.mean
        else:
            try:
                xmean = self.archive[xmean]['geno']
                # noise handling after call of tell
            except KeyError:
                try:
                    xmean = self.sent_solutions[xmean]['geno']
                    # noise handling before calling tell
                except KeyError:
                    pass

        if self.countiter == 0:
            self.tic = time.clock()  # backward compatible
            self.elapsed_time = ElapsedTime()

        sigma = sigma_fac * self.sigma

        # update parameters for sampling the distribution
        #        fac  0      1      10
        # 150-D cigar:
        #           50749  50464   50787
        # 200-D elli:               == 6.9
        #                  99900   101160
        #                 100995   103275 == 2% loss
        # 100-D elli:               == 6.9
        #                 363052   369325  < 2% loss
        #                 365075   365755

        # update distribution
        if self.sp.CMA_on and (
                (self.opts['updatecovwait'] is None and
                 self.countiter >=
                     self.itereigenupdated + 1. / (self.sp.c1 + self.sp.cmu) / self.N / 10
                 ) or
                (self.opts['updatecovwait'] is not None and
                 self.countiter > self.itereigenupdated + self.opts['updatecovwait']
                 ) or
                (self.sp.neg.cmuexp * (self.countiter - self.itereigenupdated) > 0.5
                )  # TODO (minor): not sure whether this is "the right" criterion
            ):
            self.updateBD()
        # sample distribution
        if self.flgtelldone:  # could be done in tell()!?
            self.flgtelldone = False
            self.ary = []

        # each row is a solution
        arz = self.randn((number, self.N))
        # zzzzzzzzzzzzzzzzzzzzzzzzzzz
        if self.opts['CMA_sample_on_sphere_surface']:  # normalize the length to chiN
            for i in rglen((arz)):
                ss = sum(arz[i]**2)
                if 1 < 3 or ss > self.N + 10.1:
                    arz[i] *= (self.N**0.5 if self.opts['CSA_squared'] else self.const.chiN) / ss**0.5
            # or to average
            # arz *= 1 * self.const.chiN / np.mean([sum(z**2)**0.5 for z in arz])

        # fac = np.mean(sum(arz**2, 1)**0.5)
        # print fac
        # arz *= self.const.chiN / fac
        ary = self.sigma_vec * np.dot(self.B, (self.D * arz).T).T
        if number > 2 and self.countiter > 2:
            if (isinstance(self.adapt_sigma, CMAAdaptSigmaTPA) or
                self.opts['mean_shift_line_samples'] or
                self.opts['pc_line_samples']):
                ys = []
                if self.opts['pc_line_samples']:
                    ys.append(self.pc[:])  # now TPA is with pc_line_samples
                if self.opts['mean_shift_line_samples']:
                    ys.append(self.mean - self.mean_old)
                if not len(ys):
                    ys.append(self.mean - self.mean_old)
                # assign a mirrored pair from each element of ys into ary
                for i, y in enumerate(ys):
                    if len(arz) > 2 * i + 1:  # at least two more samples
                        assert y is not self.pc
                        y *= sum(self.randn(self.N)**2)**0.5 / self.mahalanobisNorm(y)
                        # TODO: rescale y depending on some parameter?
                        ary[2*i] = y / self.sigma
                        ary[2*i + 1] = y / -self.sigma
                    else:
                        _print_warning('line samples omitted due to small popsize',
                            method_name='ask_geno', iteration=self.countiter)

        pop = xmean + sigma * ary
        self.evaluations_per_f_value = 1
        self.ary = ary  # ask_geno is called recursively in CMAAdaptSigmaTPA
        if number == self.sp.popsize:
            self.arz = arz  # is never used
        return pop

    def get_mirror(self, x, preserve_length=False):
        """return ``pheno(self.mean - (geno(x) - self.mean))``.

        >>> import cma
        >>> es = cma.CMAEvolutionStrategy(cma.np.random.randn(3), 1)
        >>> x = cma.np.random.randn(3)
        >>> assert cma.Mh.vequals_approximately(es.mean - (x - es.mean), es.get_mirror(x, preserve_length=True))
        >>> x = es.ask(1)[0]
        >>> vals = (es.get_mirror(x) - es.mean) / (x - es.mean)
        >>> assert cma.Mh.equals_approximately(sum(vals), len(vals) * vals[0])

        TODO: this implementation is yet experimental.

        Selectively mirrored sampling improves to a moderate extend but
        overadditively with active CMA for quite understandable reasons.

        Optimal number of mirrors are suprisingly small: 1,2,3 for maxlam=7,13,20
        however note that 3,6,10 are the respective maximal possible mirrors that
        must be clearly suboptimal.

        """
        try:
            dx = self.sent_solutions[x]['geno'] - self.mean
        except:  # can only happen with injected solutions?!
            dx = self.gp.geno(x, from_bounds=self.boundary_handler.inverse, copy_if_changed=True) - self.mean

        if not preserve_length:
            dx *= sum(self.randn(self.N)**2)**0.5 / self.mahalanobisNorm(dx)
        x = self.mean - dx
        y = self.gp.pheno(x, into_bounds=self.boundary_handler.repair)
        # old measure: costs 25% in CPU performance with N,lambda=20,200
        self.sent_solutions.insert(y, geno=x, iteration=self.countiter)
        return y

    def mirror_penalized(self, f_values, idx):
        """obsolete and subject to removal (TODO),
        return modified f-values such that for each mirror one becomes worst.

        This function is useless when selective mirroring is applied with no
        more than (lambda-mu)/2 solutions.

        Mirrors are leading and trailing values in ``f_values``.

        """
        assert len(f_values) >= 2 * len(idx)
        m = np.max(np.abs(f_values))
        for i in len(idx):
            if f_values[idx[i]] > f_values[-1 - i]:
                f_values[idx[i]] += m
            else:
                f_values[-1 - i] += m
        return f_values

    def mirror_idx_cov(self, f_values, idx1):  # will most likely be removed
        """obsolete and subject to removal (TODO),
        return indices for negative ("active") update of the covariance matrix
        assuming that ``f_values[idx1[i]]`` and ``f_values[-1-i]`` are
        the corresponding mirrored values

        computes the index of the worse solution sorted by the f-value of the
        better solution.

        TODO: when the actual mirror was rejected, it is better
        to return idx1 instead of idx2.

        Remark: this function might not be necessary at all: if the worst solution
        is the best mirrored, the covariance matrix updates cancel (cave: weights
        and learning rates), which seems what is desirable. If the mirror is bad,
        as strong negative update is made, again what is desirable.
        And the fitness--step-length correlation is in part addressed by
        using flat weights.

        """
        idx2 = np.arange(len(f_values) - 1, len(f_values) - 1 - len(idx1), -1)
        f = []
        for i in rglen((idx1)):
            f.append(min((f_values[idx1[i]], f_values[idx2[i]])))
            # idx.append(idx1[i] if f_values[idx1[i]] > f_values[idx2[i]] else idx2[i])
        return idx2[np.argsort(f)][-1::-1]

    def eval_mean(self, func, args=()):
        """evaluate the distribution mean, this is not (yet) effective
        in terms of termination or display"""
        self.fmean = func(self.mean, *args)

    # ____________________________________________________________
    # ____________________________________________________________
    #
    def ask_and_eval(self, func, args=(), number=None, xmean=None, sigma_fac=1,
                     evaluations=1, aggregation=np.median, kappa=1):
        """samples `number` solutions and evaluates them on `func`, where
        each solution `s` is resampled until ``self.is_feasible(s, func(s)) is True``.

        Arguments
        ---------
            `func`
                objective function
            `args`
                additional parameters for `func`
            `number`
                number of solutions to be sampled, by default
                population size ``popsize`` (AKA lambda)
            `xmean`
                mean for sampling the solutions, by default ``self.mean``.
            `sigma_fac`
                multiplier for sampling width, standard deviation, for example
                to get a small perturbation of solution `xmean`
            `evaluations`
                number of evaluations for each sampled solution
            `aggregation`
                function that aggregates `evaluations` values to
                as single value.
            `kappa`
                multiplier used for the evaluation of the solutions, in
                that ``func(m + kappa*(x - m))`` is the f-value for x.

        Return
        ------
        ``(X, fit)``, where
            X -- list of solutions
            fit -- list of respective function values

        Details
        -------
        While ``not self.is_feasible(x, func(x))``new solutions are sampled. By
        default ``self.is_feasible == cma.feasible == lambda x, f: f not in (None, np.NaN)``.
        The argument to `func` can be freely modified within `func`.

        Depending on the ``CMA_mirrors`` option, some solutions are not sampled
        independently but as mirrors of other bad solutions. This is a simple
        derandomization that can save 10-30% of the evaluations in particular
        with small populations, for example on the cigar function.

        Example
        -------
        >>> import cma
        >>> x0, sigma0 = 8*[10], 1  # 8-D
        >>> es = cma.CMAEvolutionStrategy(x0, sigma0)
        >>> while not es.stop():
        ...     X, fit = es.ask_and_eval(cma.fcts.elli)  # handles NaN with resampling
        ...     es.tell(X, fit)  # pass on fitness values
        ...     es.disp(20) # print every 20-th iteration
        >>> print('terminated on ' + str(es.stop()))
        <output omitted>

        A single iteration step can be expressed in one line, such that
        an entire optimization after initialization becomes
        ::

            while not es.stop():
                es.tell(*es.ask_and_eval(cma.fcts.elli))

        """
        # initialize
        popsize = self.sp.popsize
        if number is not None:
            popsize = number
        selective_mirroring = True
        nmirrors = self.sp.lam_mirr
        if popsize != self.sp.popsize:
            nmirrors = Mh.sround(popsize * self.sp.lam_mirr / self.sp.popsize)
            # TODO: now selective mirroring might be impaired
        assert nmirrors <= popsize // 2
        self.mirrors_idx = np.arange(nmirrors)  # might never be used
        self.mirrors_rejected_idx = []  # might never be used
        if xmean is None:
            xmean = self.mean
        is_feasible = self.opts['is_feasible']

        # do the work
        fit = []  # or np.NaN * np.empty(number)
        X_first = self.ask(popsize)
        X = []
        for k in range(int(popsize)):
            x, f = X_first.pop(0), None
            nreject = -1
            while nreject < 0 or not is_feasible(x, f):  # rejection sampling
                nreject += 1
                if nreject:  # resample
                    x = self.ask(1, xmean, sigma_fac)[0]
                elif k >= popsize - nmirrors:  # mirrored sample
                    if k == popsize - nmirrors and selective_mirroring:
                        self.mirrors_idx = np.argsort(fit)[-1:-1 - nmirrors:-1]
                    x = self.get_mirror(X[self.mirrors_idx[popsize - 1 - k]])
                if nreject == 1 and k >= popsize - nmirrors:
                    self.mirrors_rejected_idx.append(k)

                # contraints handling test hardwired ccccccccccc
                length_normalizer = 1
                # zzzzzzzzzzzzzzzzzzzzzzzzz
                f = func(x, *args) if kappa == 1 else func(xmean + kappa * length_normalizer * (x - xmean), *args)
                if is_feasible(x, f) and evaluations > 1:
                    f = aggregation([f] + [(func(x, *args) if kappa == 1 else func(xmean + kappa * length_normalizer * (x - xmean), *args)) for _i in range(int(evaluations - 1))])
                if nreject + 1 % 1000 == 0:
                    print('  %d solutions rejected (f-value NaN or None) at iteration %d' %
                          (nreject, self.countiter))
            fit.append(f)
            X.append(x)
        self.evaluations_per_f_value = int(evaluations)
        return X, fit


    # ____________________________________________________________
    def tell(self, solutions, function_values, check_points=None, copy=False):
        """pass objective function values to prepare for next
        iteration. This core procedure of the CMA-ES algorithm updates
        all state variables, in particular the two evolution paths, the
        distribution mean, the covariance matrix and a step-size.

        Arguments
        ---------
            `solutions`
                list or array of candidate solution points (of
                type `numpy.ndarray`), most presumably before
                delivered by method `ask()` or `ask_and_eval()`.
            `function_values`
                list or array of objective function values
                corresponding to the respective points. Beside for termination
                decisions, only the ranking of values in `function_values`
                is used.
            `check_points`
                If ``check_points is None``, only solutions that are not generated
                by `ask()` are possibly clipped (recommended). ``False`` does not clip
                any solution (not recommended).
                If ``True``, clips solutions that realize long steps (i.e. also
                those that are unlikely to be generated with `ask()`). `check_points`
                can be a list of indices to be checked in solutions.
            `copy`
                ``solutions`` can be modified in this routine, if ``copy is False``
            `AdaptSigma`:
                sigma adaptation class like ``CMAAdaptSigmaCSA``, with an adhoc interface
                very specific to the ``CMAEvolutionStrategy.tell`` method
                (this interface might change in future). Overwrites `self.AdaptSigma`.

        Details
        -------
        `tell()` updates the parameters of the multivariate
        normal search distribution, namely covariance matrix and
        step-size and updates also the attributes `countiter` and
        `countevals`. To check the points for consistency is quadratic
        in the dimension (like sampling points).

        Bugs
        ----
        The effect of changing the solutions delivered by `ask()` depends on whether
        boundary handling is applied. With boundary handling, modifications are
        disregarded. This is necessary to apply the default boundary handling that
        uses unrepaired solutions but might change in future.

        Example
        -------
        ::

            import cma
            func = cma.fcts.elli  # choose objective function
            es = cma.CMAEvolutionStrategy(cma.np.random.rand(10), 1)
            while not es.stop():
               X = es.ask()
               es.tell(X, [func(x) for x in X])
            es.result()  # where the result can be found

        :See: class `CMAEvolutionStrategy`, `ask()`, `ask_and_eval()`, `fmin()`

        """
        if self.flgtelldone:
            raise _Error('tell should only be called once per iteration')

        lam = len(solutions)
        if lam != array(function_values).shape[0]:
            raise _Error('for each candidate solution '
                        + 'a function value must be provided')
        if lam + self.sp.lam_mirr < 3:
            raise _Error('population size ' + str(lam) + ' is too small when option CMA_mirrors * popsize < 0.5')

        if not np.isscalar(function_values[0]):
            if np.isscalar(function_values[0][0]):
                if self.countiter <= 1:
                    print('WARNING: function values are not a list of scalars (further warnings are suppressed)')
                function_values = [val[0] for val in function_values]
            else:
                raise _Error('objective function values must be a list of scalars')


        # ## prepare
        N = self.N
        sp = self.sp
        if lam < sp.mu:  # rather decrease cmean instead of having mu > lambda//2
            raise _Error('not enough solutions passed to function tell (mu>lambda)')

        self.countiter += 1  # >= 1 now
        self.countevals += sp.popsize * self.evaluations_per_f_value
        self.best.update(solutions, self.sent_solutions, function_values, self.countevals)

        flgseparable = self.opts['CMA_diagonal'] is True \
                       or self.countiter <= self.opts['CMA_diagonal']
        if not flgseparable and len(self.C.shape) == 1:  # C was diagonal ie 1-D
            # enter non-separable phase (no easy return from here)
            self.B = np.eye(N)  # identity(N)
            self.C = np.diag(self.C)
            idx = np.argsort(self.D)
            self.D = self.D[idx]
            self.B = self.B[:, idx]
            self.Yneg = np.zeros((N, N))

        # ## manage fitness
        fit = self.fit  # make short cut

        # CPU for N,lam=20,200: this takes 10s vs 7s
        fit.bndpen = self.boundary_handler.update(function_values, self)(solutions, self.sent_solutions, self.gp)
        # for testing:
        # fit.bndpen = self.boundary_handler.update(function_values, self)([s.unrepaired for s in solutions])
        fit.idx = np.argsort(array(fit.bndpen) + array(function_values))
        fit.fit = array(function_values, copy=False)[fit.idx]

        # update output data TODO: this is obsolete!? However: need communicate current best x-value?
        # old: out['recent_x'] = self.gp.pheno(pop[0])
        self.out['recent_x'] = array(solutions[fit.idx[0]])  # TODO: change in a data structure(?) and use current as identify
        self.out['recent_f'] = fit.fit[0]

        # fitness histories
        fit.hist.insert(0, fit.fit[0])
        # if len(self.fit.histbest) < 120+30*N/sp.popsize or  # does not help, as tablet in the beginning is the critical counter-case
        if ((self.countiter % 5) == 0):  # 20 percent of 1e5 gen.
            fit.histbest.insert(0, fit.fit[0])
            fit.histmedian.insert(0, np.median(fit.fit) if len(fit.fit) < 21
                                    else fit.fit[self.popsize // 2])
        if len(fit.histbest) > 2e4:  # 10 + 30*N/sp.popsize:
            fit.histbest.pop()
            fit.histmedian.pop()
        if len(fit.hist) > 10 + 30 * N / sp.popsize:
            fit.hist.pop()

        # TODO: clean up inconsistency when an unrepaired solution is available and used
        # now get the genotypes
        pop = []  # create pop from input argument solutions
        for k, s in enumerate(solutions):  # use phenotype before Solution.repair()
            if 1 < 3:
                pop += [self.gp.geno(s,
                                     from_bounds=self.boundary_handler.inverse,
                                     repair=(self.repair_genotype if check_points not in (False, 0, [], ()) else None),
                                     archive=self.sent_solutions)]  # takes genotype from sent_solutions, if available
                try:
                    self.archive.insert(s, value=self.sent_solutions.pop(s), fitness=function_values[k])
                    # self.sent_solutions.pop(s)
                except KeyError:
                    pass
        try:
            moldold = self.mean_old
        except:
            pass
        self.mean_old = self.mean
        mold = self.mean_old  # just an alias

        # check and normalize each x - m
        # check_points is a flag (None is default: check non-known solutions) or an index list
        # should also a number possible (first check_points points)?
        if check_points not in (None, False, 0, [], ()):  # useful in case of injected solutions and/or adaptive encoding, however is automatic with use_sent_solutions
            try:
                if len(check_points):
                    idx = check_points
            except:
                idx = range(sp.popsize)

            for k in idx:
                self.repair_genotype(pop[k])

        # only arrays can be multiple indexed
        pop = array(pop, copy=False)

        # sort pop
        pop = pop[fit.idx]

        if self.opts['CMA_elitist'] and self.best.f < fit.fit[0]:
            if self.best.x_geno is not None:
                xp = [self.best.x_geno]
                # xp = [self.best.xdict['geno']]
                # xp = [self.gp.geno(self.best.x[:])]  # TODO: remove
                # print self.mahalanobisNorm(xp[0]-self.mean)
            else:
                xp = [self.gp.geno(array(self.best.x, copy=True), self.boundary_handler.inverse, copy_if_changed=False)]
                print('genotype for elitist not found')
            self.clip_or_fit_solutions(xp, [0])
            pop = array([xp[0]] + list(pop))

        # compute new mean
        self.mean = mold + self.sp.cmean * \
                    (sum(sp.weights * pop[0:sp.mu].T, 1) - mold)


        # check Delta m (this is not default, but could become at some point)
        # CAVE: upper_length=sqrt(2)+2 is too restrictive, test upper_length = sqrt(2*N) thoroughly.
        # replaced by repair_geno?
        # simple test case injecting self.mean:
        # self.mean = 1e-4 * self.sigma * np.random.randn(N)
        if 1 < 3:
            cmean = self.sp.cmean

        # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        # get learning rate constants
        cc, c1, cmu = sp.cc, sp.c1, sp.cmu
        if flgseparable:
            cc, c1, cmu = sp.cc_sep, sp.c1_sep, sp.cmu_sep

        # now the real work can start

        hsig = self.adapt_sigma.hsig(self) # ps update must be done here in separable case

        # hsig = sum(self.ps**2) / self.N < 2 + 4./(N+1)
        # adjust missing variance due to hsig, in 4-D with damps=1e99 and sig0 small
        #       hsig leads to premature convergence of C otherwise
        # hsiga = (1-hsig**2) * c1 * cc * (2-cc)  # to be removed in future
        c1a = c1 - (1 - hsig**2) * c1 * cc * (2 - cc)  # adjust for variance loss

        self.pc = (1 - cc) * self.pc + \
                  hsig * (sqrt(cc * (2 - cc) * sp.mueff) / self.sigma / cmean) * \
                  (self.mean - mold) / self.sigma_vec

        # covariance matrix adaptation/udpate
        if sp.CMA_on:
            # assert sp.c1 + sp.cmu < sp.mueff / N  # ??
            assert c1 + cmu <= 1

            # default full matrix case
            if not flgseparable:
                Z = (pop[0:sp.mu] - mold) / (self.sigma * self.sigma_vec)
                Z = dot((cmu * sp.weights) * Z.T, Z)  # learning rate integrated
                if self.sp.neg.cmuexp:
                    tmp = (pop[-sp.neg.mu:] - mold) / (self.sigma * self.sigma_vec)
                    if 1 < 3:  # normalize to constant length (seems preferable in several aspects)
                        # print(tmp.shape)
                        for i in range(tmp.shape[0]):
                            tmp[i, :] *= N**0.5 / self.mahalanobisNorm(tmp[i, :]) / (self.sigma * self.sigma_vec)
                        # print(tmp.shape)
                    self.Yneg *= 1 - self.sp.neg.cmuexp  # for some reason necessary?
                    self.Yneg += dot(sp.neg.weights * tmp.T, tmp) - self.C
                    # self.update_exponential(dot(sp.neg.weights * tmp.T, tmp) - 1 * self.C, -1*self.sp.neg.cmuexp)

                self.C *= 1 - c1a - cmu
                self.C += np.outer(c1 * self.pc, self.pc) + Z
                self.dC = np.diag(self.C).copy()  # for output and termination checking

            else:  # separable/diagonal linear case
                assert(c1 + cmu <= 1)
                Z = np.zeros(N)
                for k in range(sp.mu):
                    z = (pop[k] - mold) / (self.sigma * self.sigma_vec)  # TODO see above
                    Z += sp.weights[k] * z * z  # is 1-D
                self.C = (1 - c1a - cmu) * self.C + c1 * self.pc * self.pc + cmu * Z
                # TODO: self.C *= exp(cmuneg * (N - dot(sp.neg.weights,  **2)
                self.dC = self.C
                self.D = sqrt(self.C)  # C is a 1-D array, this is why adapt_sigma needs to prepare before
                self.itereigenupdated = self.countiter

                # idx = self.mirror_idx_cov()  # take half of mirrored vectors for negative update

        # step-size adaptation, adapt sigma
        # in case of TPA, function_values[0] and [1] must reflect samples colinear to xmean - xmean_old
        self.adapt_sigma.update(self, function_values=function_values)

        if self.sigma * min(self.dC)**0.5 < self.opts['minstd']:
            self.sigma = self.opts['minstd'] / min(self.dC)**0.5
        # g = self.countiter
        # N = self.N
        mindx = eval(self.opts['mindx']) if isinstance(self.opts['mindx'], str) else self.opts['mindx']
        if self.sigma * min(self.D) < mindx:  # TODO: sigma_vec is missing here
            self.sigma = mindx / min(self.D)

        if self.sigma > 1e9 * self.sigma0:
            alpha = self.sigma / max(self.D)
            self.multiplyC(alpha)
            self.sigma /= alpha**0.5
            self.opts['tolupsigma'] /= alpha**0.5  # to be compared with sigma

        # TODO increase sigma in case of a plateau?

        # Uncertainty noise measurement is done on an upper level

        self.flgtelldone = True
    # end tell()

    def result(self):
        """return ``(xbest, f(xbest), evaluations_xbest, evaluations, iterations, pheno(xmean), effective_stds)``"""
        # TODO: how about xcurrent?
        return self.best.get() + (
            self.countevals, self.countiter, self.gp.pheno(self.mean), self.gp.scales * self.sigma * self.sigma_vec * self.dC**0.5)


    def result_pretty(self, number_of_runs=0, time_str=None):
        """pretty print result.

        Returns ``self.result()``

        """
        s = (' after %i restart' + ('s' if number_of_runs > 1 else '')) \
            % number_of_runs if number_of_runs else ''
        for k, v in self.stop().items():
            print('termination on %s=%s%s' % (k, str(v), s) +
                  (' (%s)' % time_str if time_str else ''))

        print('final/bestever f-value = %e %e' % (self.best.last.f, self.best.f))
        if self.N < 9:
            print('mean solution: ' + str(self.gp.pheno(self.mean)))
            print('std deviation: ' + str(self.sigma * sqrt(self.dC) * self.gp.scales))
        else:
            print('mean solution: %s ...]' % (str(self.gp.pheno(self.mean)[:8])[:-1]))
            print('std deviations: %s ...]' % (str((self.sigma * sqrt(self.dC) * self.gp.scales)[:8])[:-1]))
        return self.result()


    def clip_or_fit_solutions(self, pop, idx):
        """make sure that solutions fit to sample distribution, this interface will probably change.

        In particular the frequency of long vectors appearing in pop[idx] - self.mean is limited.

        """
        for k in idx:
            self.repair_genotype(pop[k])

    def repair_genotype(self, x, copy_if_changed=False):
        """make sure that solutions fit to the sample distribution, this interface will probably change.

        In particular the frequency of x - self.mean being long is limited.

        """
        mold = self.mean
        if 1 < 3:  # hard clip at upper_length
            upper_length = self.N**0.5 + 2 * self.N / (self.N + 2)  # should become an Option, but how? e.g. [0, 2, 2]
            fac = self.mahalanobisNorm(x - mold) / upper_length

            if fac > 1:
                if copy_if_changed:
                    x = (x - mold) / fac + mold
                else:  # should be 25% faster:
                    x -= mold
                    x /= fac
                    x += mold
                # print self.countiter, k, fac, self.mahalanobisNorm(pop[k] - mold)
                # adapt also sigma: which are the trust-worthy/injected solutions?
        else:
            if 'checktail' not in self.__dict__:  # hasattr(self, 'checktail')
                raise NotImplementedError
                # from check_tail_smooth import CheckTail  # for the time being
                # self.checktail = CheckTail()
                # print('untested feature checktail is on')
            fac = self.checktail.addchin(self.mahalanobisNorm(x - mold))

            if fac < 1:
                x = fac * (x - mold) + mold

        return x

    # ____________________________________________________________
    # ____________________________________________________________
    #
    def updateBD(self):
        """update internal variables for sampling the distribution with the
        current covariance matrix C. This method is O(N^3), if C is not diagonal.

        """
        # itereigenupdated is always up-to-date in the diagonal case
        # just double check here
        if self.itereigenupdated == self.countiter:
            return

        # C has already positive updates, here come the additional negative updates
        if self.sp.neg.cmuexp:  # cave:
            if (self.countiter - self.itereigenupdated) * self.sp.neg.cmuexp * self.N < 0.5: # pos.def. guarantied
                self.C -= self.sp.neg.cmuexp * self.Yneg
            else: # guaranties pos.def. unconditionally
                # print('exponential update for negative weights (internally more expensive) in iteration', self.countiter)
                self.update_exponential(self.Yneg, -self.sp.neg.cmuexp)
                # self.C = self.Ypos + Cs * Mh.expms(-self.sp.neg.cmuexp*Csi*self.Yneg*Csi) * Cs
            self.Yneg = np.zeros((self.N, self.N))

        if self.sigma_vec is not 1 and not np.all(self.sigma_vec == 1):
            self.C = dot(dot(np.diag(self.sigma_vec), self.C), np.diag(self.sigma_vec))
            self.sigma_vec[:] = 1

        if self.opts['CMA_const_trace'] in (True, 1, 2):  # normalize trace of C
            if self.opts['CMA_const_trace'] == 2:
                s = np.exp(np.mean(np.log(self.dC)))
            else:
                s = np.mean(self.dC)
            self.C /= s
            self.dC /= s
        self.C = (self.C + self.C.T) / 2
        # self.C = np.triu(self.C) + np.triu(self.C,1).T  # should work as well
        # self.D, self.B = eigh(self.C) # hermitian, ie symmetric C is assumed

        if isinstance(self.opts['CMA_eigenmethod'], type(1)):
            print('WARNING: option CMA_eigenmethod should be a function, not an integer')
            if self.opts['CMA_eigenmethod'] == -1:
                # pygsl
                # easy to install (well, in Windows install gsl binaries first,
                # set system path to respective libgsl-0.dll (or cp the dll to
                # python\DLLS ?), in unzipped pygsl edit
                # gsl_dist/gsl_site_example.py into gsl_dist/gsl_site.py
                # and run "python setup.py build" and "python setup.py install"
                # in MINGW32)
                if 1 < 3:  # import pygsl on the fly
                    try:
                        import pygsl.eigen.eigenvectors  # TODO efficient enough?
                    except ImportError:
                        print('WARNING: could not find pygsl.eigen module, either install pygsl \n' +
                              '  or set option CMA_eigenmethod=1 (is much slower), option set to 1')
                        self.opts['CMA_eigenmethod'] = 0  # use 0 if 1 is too slow

                    self.D, self.B = pygsl.eigen.eigenvectors(self.C)

            elif self.opts['CMA_eigenmethod'] == 0:
                # TODO: thoroughly test np.linalg.eigh
                #       numpy.linalg.eig crashes in 200-D
                #       and EVecs with same EVals are not orthogonal
                self.D, self.B = np.linalg.eigh(self.C)  # self.B[i] is a row and not an eigenvector
            else:  # is overall two;ten times slower in 10;20-D
                self.D, self.B = Misc.eig(self.C)  # def eig, see below
        else:
            self.D, self.B = self.opts['CMA_eigenmethod'](self.C)


        # assert(sum(self.D-DD) < 1e-6)
        # assert(sum(sum(np.dot(BB, BB.T)-np.eye(self.N))) < 1e-6)
        # assert(sum(sum(np.dot(BB * DD, BB.T) - self.C)) < 1e-6)
        idx = np.argsort(self.D)
        self.D = self.D[idx]
        self.B = self.B[:, idx]  # self.B[i] is a row, columns self.B[:,i] are eigenvectors
        # assert(all(self.B[self.countiter % self.N] == self.B[self.countiter % self.N,:]))

        # qqqqqqqqqq
        # is O(N^3)
        # assert(sum(abs(self.C - np.dot(self.D * self.B,  self.B.T))) < N**2*1e-11)

        self.D **= 0.5
        self.itereigenupdated = self.countiter

    def multiplyC(self, alpha):
        """multiply C with a scalar and update all related internal variables (dC, D,...)"""
        self.C *= alpha
        if self.dC is not self.C:
            self.dC *= alpha
        self.D *= alpha**0.5
    def update_exponential(self, Z, eta, BDpair=None):
        """exponential update of C that guarantees positive definiteness, that is,
        instead of the assignment ``C = C + eta * Z``,
        we have ``C = C**.5 * exp(eta * C**-.5 * Z * C**-.5) * C**.5``.

        Parameter `Z` should have expectation zero, e.g. sum(w[i] * z[i] * z[i].T) - C
        if E z z.T = C.

        Parameter `eta` is the learning rate, for ``eta == 0`` nothing is updated.

        This function conducts two eigendecompositions, assuming that
        B and D are not up to date, unless `BDpair` is given. Given BDpair,
        B is the eigensystem and D is the vector of sqrt(eigenvalues), one
        eigendecomposition is omitted.

        Reference: Glasmachers et al 2010, Exponential Natural Evolution Strategies

        """
        if eta == 0:
            return
        if BDpair:
            B, D = BDpair
        else:
            D, B = self.opts['CMA_eigenmethod'](self.C)
            D **= 0.5
        Cs = dot(B, (B * D).T)   # square root of C
        Csi = dot(B, (B / D).T)  # square root of inverse of C
        self.C = dot(Cs, dot(Mh.expms(eta * dot(Csi, dot(Z, Csi)), self.opts['CMA_eigenmethod']), Cs))

    # ____________________________________________________________
    # ____________________________________________________________
    def feedForResume(self, X, function_values):
        """Given all "previous" candidate solutions and their respective
        function values, the state of a `CMAEvolutionStrategy` object
        can be reconstructed from this history. This is the purpose of
        function `feedForResume`.

        Arguments
        ---------
            `X`
              (all) solution points in chronological order, phenotypic
              representation. The number of points must be a multiple
              of popsize.
            `function_values`
              respective objective function values

        Details
        -------
        `feedForResume` can be called repeatedly with only parts of
        the history. The part must have the length of a multiple
        of the population size.
        `feedForResume` feeds the history in popsize-chunks into `tell`.
        The state of the random number generator might not be
        reconstructed, but this would be only relevant for the future.

        Example
        -------
        ::

            import cma

            # prepare
            (x0, sigma0) = ... # initial values from previous trial
            X = ... # list of generated solutions from a previous trial
            f = ... # respective list of f-values

            # resume
            es = cma.CMAEvolutionStrategy(x0, sigma0)
            es.feedForResume(X, f)

            # continue with func as objective function
            while not es.stop():
               X = es.ask()
               es.tell(X, [func(x) for x in X])

        Credits to Dirk Bueche and Fabrice Marchal for the feeding idea.

        :See: class `CMAEvolutionStrategy` for a simple dump/load to resume

        """
        if self.countiter > 0:
            print('WARNING: feed should generally be used with a new object instance')
        if len(X) != len(function_values):
            raise _Error('number of solutions ' + str(len(X)) +
                ' and number function values ' +
                str(len(function_values)) + ' must not differ')
        popsize = self.sp.popsize
        if (len(X) % popsize) != 0:
            raise _Error('number of solutions ' + str(len(X)) +
                    ' must be a multiple of popsize (lambda) ' +
                    str(popsize))
        for i in rglen((X) / popsize):
            # feed in chunks of size popsize
            self.ask()  # a fake ask, mainly for a conditioned calling of updateBD
                        # and secondary to get possibly the same random state
            self.tell(X[i * popsize:(i + 1) * popsize], function_values[i * popsize:(i + 1) * popsize])

    # ____________________________________________________________
    # ____________________________________________________________
    def readProperties(self):
        """reads dynamic parameters from property file (not implemented)
        """
        print('not yet implemented')

    # ____________________________________________________________
    # ____________________________________________________________
    def mahalanobisNorm(self, dx):
        """compute the Mahalanobis norm that is induced by the adapted sample
        distribution, covariance matrix C times sigma**2. The expected
        Mahalanobis distance to the sample mean is about sqrt(dimension).

        Argument
        --------
        A *genotype* difference `dx`.

        Example
        -------
        >>> import cma, numpy
        >>> es = cma.CMAEvolutionStrategy(numpy.ones(10), 1)
        >>> xx = numpy.random.randn(2, 10)
        >>> d = es.mahalanobisNorm(es.gp.geno(xx[0]-xx[1]))

        `d` is the distance "in" the true sample distribution,
        sampled points have a typical distance of ``sqrt(2*es.N)``,
        where `N` is the dimension, and an expected distance of
        close to ``sqrt(N)`` to the sample mean. In the example,
        `d` is the Euclidean distance, because C = I and sigma = 1.

        """
        return sqrt(sum((self.D**-1 * np.dot(self.B.T, dx / self.sigma_vec))**2)) / self.sigma

    # ____________________________________________________________
    # ____________________________________________________________
    #
    def timesCroot(self, mat):
        """return C**0.5 times mat, where mat can be a vector or matrix.
        Not functional, because _Croot=C**0.5 is never computed (should be in updateBD)
        """
        print("WARNING: timesCroot is not yet tested")
        if self.opts['CMA_diagonal'] is True \
                       or self.countiter <= self.opts['CMA_diagonal']:
            res = (self._Croot * mat.T).T
        else:
            res = np.dot(self._Croot, mat)
        return res
    def divCroot(self, mat):
        """return C**-1/2 times mat, where mat can be a vector or matrix.
        Not functional, because _Crootinv is never computed. """
        print("WARNING: divCroot is not yet tested")
        if self.opts['CMA_diagonal'] is True \
                       or self.countiter <= self.opts['CMA_diagonal']:
            res = (self._Crootinv * mat.T).T
        else:
            res = np.dot(self._Crootinv, mat)
        return res

    # ____________________________________________________________
    # ____________________________________________________________
    def disp_annotation(self):
        """print annotation for `disp()`"""
        print('Iterat #Fevals   function value    axis ratio  sigma  minstd maxstd min:sec')
        sys.stdout.flush()

    # ____________________________________________________________
    # ____________________________________________________________
    def disp(self, modulo=None):  # TODO: rather assign opt['verb_disp'] as default?
        """prints some infos according to `disp_annotation()`, if
        ``iteration_counter % modulo == 0``

        """
        if modulo is None:
            modulo = self.opts['verb_disp']

        # console display
        if modulo:
            if (self.countiter - 1) % (10 * modulo) < 1:
                self.disp_annotation()
            if self.countiter > 0 and (self.stop() or self.countiter < 4
                              or self.countiter % modulo < 1):
                if self.opts['verb_time']:
                    toc = self.elapsed_time()
                    stime = str(int(toc // 60)) + ':' + str(round(toc % 60, 1))
                else:
                    stime = ''
                print(' '.join((repr(self.countiter).rjust(5),
                                repr(self.countevals).rjust(7),
                                '%.15e' % (min(self.fit.fit)),
                                '%4.1e' % (self.D.max() / self.D.min()),
                                '%6.2e' % self.sigma,
                                '%6.0e' % (self.sigma * sqrt(min(self.dC))),
                                '%6.0e' % (self.sigma * sqrt(max(self.dC))),
                                stime)))
                # if self.countiter < 4:
                sys.stdout.flush()

cma_default_options = {
    # the follow string arguments are evaluated, besides the verb_filenameprefix
    'AdaptSigma': 'CMAAdaptSigmaCSA  # or any other CMAAdaptSigmaBase class e.g. CMAAdaptSigmaTPA',
    'CMA_active': 'True  # negative update, conducted after the original update',
    'CMA_activefac': '1  # learning rate multiplier for active update',
    'CMA_cmean': '1  # learning rate for the mean value',
    'CMA_const_trace': 'False  # normalize trace, value CMA_const_trace=2 normalizes sum log eigenvalues to zero',
    'CMA_diagonal': '0*100*N/sqrt(popsize)  # nb of iterations with diagonal covariance matrix, True for always',  # TODO 4/ccov_separable?
    'CMA_eigenmethod': 'np.linalg.eigh  # 0=numpy-s eigh, -1=pygsl, otherwise cma.Misc.eig (slower)',
    'CMA_elitist': 'False # elitism likely impairs global search performance',
    'CMA_mirrors': 'popsize < 6  # values <0.5 are interpreted as fraction, values >1 as numbers (rounded), otherwise about 0.16 is used',
    'CMA_mu': 'None  # parents selection parameter, default is popsize // 2',
    'CMA_on': 'True  # False or 0 for no adaptation of the covariance matrix',
    'CMA_sample_on_sphere_surface': 'False  #v all mutation vectors have the same length',
    'CMA_rankmu': 'True  # False or 0 for omitting rank-mu update of covariance matrix',
    'CMA_rankmualpha': '0.3  # factor of rank-mu update if mu=1, subject to removal, default might change to 0.0',
    'CMA_dampsvec_fac': 'np.Inf  # tentative and subject to changes, 0.5 would be a "default" damping for sigma vector update',
    'CMA_dampsvec_fade': '0.1  # tentative fading out parameter for sigma vector update',
    'CMA_teststds': 'None  # factors for non-isotropic initial distr. mainly for test purpose, see scaling_...',
    # 'CMA_AII': 'False  # not yet tested',
    'CSA_dampfac': '1  #v positive multiplier for step-size damping, 0.3 is close to optimal on the sphere',
    'CSA_damp_mueff_exponent': '0.5  # zero would mean no dependency of damping on mueff, useful with CSA_disregard_length option',
    'CSA_disregard_length': 'False  #v True is untested',
    'CSA_clip_length_value': 'None  #v untested, [0, 0] means disregarding length completely',
    'CSA_squared': 'False  #v use squared length for sigma-adaptation ',
    'boundary_handling': 'BoundTransform  # or BoundPenalty, unused when ``bounds in (None, [None, None])``',
    'bounds': '[None, None]  # lower (=bounds[0]) and upper domain boundaries, each a scalar or a list/vector',
     # , eval_parallel2': 'not in use {"processes": None, "timeout": 12, "is_feasible": lambda x: True} # distributes function calls to processes processes'
    'fixed_variables': 'None  # dictionary with index-value pairs like {0:1.1, 2:0.1} that are not optimized',
    'ftarget': '-inf  #v target function value, minimization',
    'is_feasible': 'is_feasible  #v a function that computes feasibility, by default lambda x, f: f not in (None, np.NaN)',
    'maxfevals': 'inf  #v maximum number of function evaluations',
    'maxiter': '100 + 50 * (N+3)**2 // popsize**0.5  #v maximum number of iterations',
    'mean_shift_line_samples': 'False #v sample two new solutions colinear to previous mean shift',
    'mindx': '0  #v minimal std in any direction, cave interference with tol*',
    'minstd': '0  #v minimal std in any coordinate direction, cave interference with tol*',
    'pc_line_samples': 'False #v two line samples along the evolution path pc',
    'popsize': '4+int(3*log(N))  # population size, AKA lambda, number of new solution per iteration',
    'randn': 'np.random.standard_normal  #v randn((lam, N)) must return an np.array of shape (lam, N)',
    'scaling_of_variables': 'None  # scale for each variable, sigma0 is interpreted w.r.t. this scale, in that effective_sigma0 = sigma0*scaling. Internally the variables are divided by scaling_of_variables and sigma is unchanged, default is ones(N)',
    'seed': 'None  # random number seed',
    'termination_callback': 'None  #v a function returning True for termination, called after each iteration step and could be abused for side effects',
    'tolfacupx': '1e3  #v termination when step-size increases by tolfacupx (diverges). That is, the initial step-size was chosen far too small and better solutions were found far away from the initial solution x0',
    'tolupsigma': '1e20  #v sigma/sigma0 > tolupsigma * max(sqrt(eivenvals(C))) indicates "creeping behavior" with usually minor improvements',
    'tolfun': '1e-11  #v termination criterion: tolerance in function value, quite useful',
    'tolfunhist': '1e-12  #v termination criterion: tolerance in function value history',
    'tolstagnation': 'int(100 + 100 * N**1.5 / popsize)  #v termination if no improvement over tolstagnation iterations',
    'tolx': '1e-11  #v termination criterion: tolerance in x-changes',
    'transformation': 'None  # [t0, t1] are two mappings, t0 transforms solutions from CMA-representation to f-representation (tf_pheno), t1 is the (optional) back transformation, see class GenoPheno',
    'typical_x': 'None  # used with scaling_of_variables',
    'updatecovwait': 'None  #v number of iterations without distribution update, name is subject to future changes',  # TODO: rename: iterwaitupdatedistribution?
    'verbose': '1  #v verbosity e.v. of initial/final message, -1 is very quiet, not yet fully implemented',
    'verb_append': '0  # initial evaluation counter, if append, do not overwrite output files',
    'verb_disp': '100  #v verbosity: display console output every verb_disp iteration',
    'verb_filenameprefix': 'outcmaes  # output filenames prefix',
    'verb_log': '1  #v verbosity: write data to files every verb_log iteration, writing can be time critical on fast to evaluate functions',
    'verb_plot': '0  #v in fmin(): plot() is called every verb_plot iteration',
    'verb_time': 'True  #v output timings on console',
    'vv': '0  #? versatile variable for hacking purposes, value found in self.opts["vv"]'
}
class CMAOptions(dict):
    """``CMAOptions()`` returns a dictionary with the available options
    and their default values for class ``CMAEvolutionStrategy``.

    ``CMAOptions('pop')`` returns a subset of recognized options that
    contain 'pop' in there keyword name or (default) value or description.

    ``CMAOptions(opts)`` returns the subset of recognized options in
    ``dict(opts)``.

    Option values can be "written" in a string and, when passed to fmin
    or CMAEvolutionStrategy, are evaluated using "N" and "popsize" as
    known values for dimension and population size (sample size, number
    of new solutions per iteration). All default option values are such
    a string.

    Details
    -------
    ``CMAOptions`` entries starting with ``tol`` are termination
    "tolerances".

    For `tolstagnation`, the median over the first and the second half
    of at least `tolstagnation` iterations are compared for both, the
    per-iteration best and per-iteration median function value.

    Example
    -------
    ::

        import cma
        cma.CMAOptions('tol')

    is a shortcut for cma.CMAOptions().match('tol') that returns all options
    that contain 'tol' in their name or description.

    :See: `fmin`(), `CMAEvolutionStrategy`, `_CMAParameters`

    """

    # @classmethod # self is the class, not the instance
    # @property
    # def default(self):
    #     """returns all options with defaults"""
    #     return fmin([],[])

    @staticmethod
    def defaults():
        """return a dictionary with default option values and description"""
        return dict(cma_default_options)

    @staticmethod
    def versatileOptions():
        """return list of options that can be changed at any time (not
        only be initialized), however the list might not be entirely up
        to date.

        The string ' #v ' in the default value indicates a 'versatile'
        option that can be changed any time.

        """
        return tuple(sorted(i[0] for i in list(CMAOptions.defaults().items()) if i[1].find(' #v ') > 0))

    def __init__(self, s=None, unchecked=False):
        """return an `CMAOptions` instance, either with the default
        options, if ``s is None``, or with all options whose name or
        description contains `s`, if `s` is a string (case is
        disregarded), or with entries from dictionary `s` as options,
        not complemented with default options or settings

        Returns: see above.

        """
        # if not CMAOptions.defaults:  # this is different from self.defaults!!!
        #     CMAOptions.defaults = fmin([],[])
        if s is None:
            super(CMAOptions, self).__init__(CMAOptions.defaults())  # dict.__init__(self, CMAOptions.defaults()) should be the same
            # self = CMAOptions.defaults()
        elif isinstance(s, str):
            super(CMAOptions, self).__init__(CMAOptions().match(s))
            # we could return here
        else:
            super(CMAOptions, self).__init__(s)

        if not unchecked:
            for key in list(self.keys()):
                if key not in CMAOptions.defaults():
                    print('Warning in cma.CMAOptions.__init__(): invalid key ``' + str(key) + '`` removed')
                    self.pop(key)
        # self.evaluated = False  # would become an option entry

    def init(self, dict_or_str, val=None, warn=True):
        """initialize one or several options.

        Arguments
        ---------
            `dict_or_str`
                a dictionary if ``val is None``, otherwise a key.
                If `val` is provided `dict_or_str` must be a valid key.
            `val`
                value for key

        Details
        -------
        Only known keys are accepted. Known keys are in `CMAOptions.defaults()`

        """
        # dic = dict_or_key if val is None else {dict_or_key:val}
        dic = dict_or_str
        if val is not None:
            dic = {dict_or_str:val}

        for key, val in dic.items():
            if key not in CMAOptions.defaults():
                # TODO: find a better solution?
                if warn:
                    print('Warning in cma.CMAOptions.init(): key ' +
                        str(key) + ' ignored')
            else:
                self[key] = val

        return self

    def set(self, dic, val=None, warn=True):
        """set can assign versatile options from `CMAOptions.versatileOptions()`
        with a new value, use `init()` for the others.

        Arguments
        ---------
            `dic`
                either a dictionary or a key. In the latter
                case, val must be provided
            `val`
                value for key
            `warn`
                bool, print a warning if the option cannot be changed
                and is therefore omitted

        This method will be most probably used with the ``opts`` attribute of
        a `CMAEvolutionStrategy` instance.

        """
        if val is not None:  # dic is a key in this case
            dic = {dic:val}  # compose a dictionary
        for key, val in dic.items():
            if key in CMAOptions.versatileOptions():
                self[key] = val
            elif warn:
                print('Warning in cma.CMAOptions.set(): key ' + str(key) + ' ignored')
        return self  # to allow o = CMAOptions(o).set(new)

    def complement(self):
        """add all missing options with their default values"""

        for key in CMAOptions.defaults():
            if key not in self:
                self[key] = CMAOptions.defaults()[key]
        return self

    def settable(self):
        """return the subset of those options that are settable at any
        time.

        Settable options are in `versatileOptions()`, but the
        list might be incomplete.

        """
        return CMAOptions([i for i in list(self.items())
                                if i[0] in CMAOptions.versatileOptions()])

    def __call__(self, key, default=None, loc=None):
        """evaluate and return the value of option `key` on the fly, or
        returns those options whose name or description contains `key`,
        case disregarded.

        Details
        -------
        Keys that contain `filename` are not evaluated.
        For ``loc==None``, `self` is used as environment
        but this does not define `N`.

        :See: `eval()`, `evalall()`

        """
        try:
            val = self[key]
        except:
            return self.match(key)

        if loc is None:
            loc = self  # TODO: this hack is not so useful: popsize could be there, but N is missing
        try:
            if isinstance(val, str):
                val = val.split('#')[0].strip()  # remove comments
                if isinstance(val, str) and key.find('filename') < 0 and key.find('mindx') < 0:
                    val = eval(val, globals(), loc)
            # invoke default
            # TODO: val in ... fails with array type, because it is applied element wise!
            # elif val in (None,(),[],{}) and default is not None:
            elif val is None and default is not None:
                val = eval(str(default), globals(), loc)
        except:
            pass  # slighly optimistic: the previous is bug-free
        return val

    def eval(self, key, default=None, loc=None):
        """Evaluates and sets the specified option value in
        environment `loc`. Many options need `N` to be defined in
        `loc`, some need `popsize`.

        Details
        -------
        Keys that contain 'filename' are not evaluated.
        For `loc` is None, the self-dict is used as environment

        :See: `evalall()`, `__call__`

        """
        # TODO: try: loc['dim'] = loc['N'] etc
        self[key] = self(key, default, loc)
        return self[key]

    def evalall(self, loc=None, defaults=None):
        """Evaluates all option values in environment `loc`.

        :See: `eval()`

        """
        if defaults is None:
            defaults = cma_default_options
        # TODO: this needs rather the parameter N instead of loc
        if 'N' in loc:  # TODO: __init__ of CMA can be simplified
            popsize = self('popsize', defaults['popsize'], loc)
            for k in list(self.keys()):
                self.eval(k, defaults[k],
                          {'N':loc['N'], 'popsize':popsize})
        return self

    def match(self, s=''):
        """return all options that match, in the name or the description,
        with string `s`, case is disregarded.

        Example: ``cma.CMAOptions().match('verb')`` returns the verbosity options.

        """
        match = s.lower()
        res = {}
        for k in sorted(self):
            s = str(k) + '=\'' + str(self[k]) + '\''
            if match in s.lower():
                res[k] = self[k]
        return CMAOptions(res)

    def pp(self):
        pprint(self)

    def pprint(self, linebreak=80):
        for i in sorted(self.items()):
            s = str(i[0]) + "='" + str(i[1]) + "'"
            a = s.split(' ')

            # print s in chunks
            l = ''  # start entire to the left
            while a:
                while a and len(l) + len(a[0]) < linebreak:
                    l += ' ' + a.pop(0)
                print(l)
                l = '        '  # tab for subsequent lines
    print_ = pprint  # Python style to prevent clash with keywords
    printme = pprint

# ____________________________________________________________
# ____________________________________________________________
class CMAStopDict(dict):
    """keep and update a termination condition dictionary, which is
    "usually" empty and returned by `CMAEvolutionStrategy.stop()`.
    The class methods entirely depend on `CMAEvolutionStrategy` class
    attributes.

    Details
    -------
    This class is not relevant for the end-user and could be a nested
    class, but nested classes cannot be serialized.

    Example
    -------
    >>> import cma
    >>> sd = cma.CMAStopDict()
    >>> es = cma.CMAEvolutionStrategy(4 * [1], 1, {'verbose':-1})
    >>> print(sd(es))
    {}
    >>> es.optimize(cma.fcts.sphere, verb_disp=0)
    >>> print(sd(es))
    {'tolfun': 1e-11}
    >>> assert sd(es) == es.stop()

    :See: `OOOptimizer.stop()`, `CMAEvolutionStrategy.stop()`

    """
    def __init__(self, d={}):
        update = isinstance(d, CMAEvolutionStrategy)
        inherit = isinstance(d, CMAStopDict)
        super(CMAStopDict, self).__init__({} if update else d)
        self._stoplist = d._stoplist if inherit else []  # multiple entries
        self.lastiter = d.lastiter if inherit else 0  # probably not necessary
        if update:
            self._update(d)

    def __call__(self, es=None):
        """update and return the termination conditions dictionary

        """
        if es is None and self.es is None:
            raise ValueError('termination conditions need an optimizer to act upon')
        self._update(es)
        return self

    def _update(self, es):
        """Test termination criteria and update dictionary

        """
        if es is None:
            es = self.es
        assert es is not None
        if es.countiter == self.lastiter:
            if es.countiter == 0:
                self.__init__()
                return self
            try:
                if es == self.es:
                    return self
            except:  # self.es not yet assigned
                pass

        self.lastiter = es.countiter
        self.es = es

        self.stoplist = []

        N = es.N
        opts = es.opts
        self.opts = opts  # a hack to get _addstop going

        # fitness: generic criterion, user defined w/o default
        self._addstop('ftarget',
                      es.best.f < opts['ftarget'])
        # maxiter, maxfevals: generic criteria
        self._addstop('maxfevals',
                      es.countevals - 1 >= opts['maxfevals'])
        self._addstop('maxiter',
                      es.countiter >= opts['maxiter'])
        # tolx, tolfacupx: generic criteria
        # tolfun, tolfunhist (CEC:tolfun includes hist)
        self._addstop('tolx',
                      all([es.sigma * xi < opts['tolx'] for xi in es.pc]) and
                      all([es.sigma * xi < opts['tolx'] for xi in sqrt(es.dC)]))
        self._addstop('tolfacupx',
                      any([es.sigma * sig > es.sigma0 * opts['tolfacupx']
                           for sig in sqrt(es.dC)]))
        self._addstop('tolfun',
                      es.fit.fit[-1] - es.fit.fit[0] < opts['tolfun'] and
                      max(es.fit.hist) - min(es.fit.hist) < opts['tolfun'])
        self._addstop('tolfunhist',
                      len(es.fit.hist) > 9 and
                      max(es.fit.hist) - min(es.fit.hist) < opts['tolfunhist'])

        # worst seen false positive: table N=80,lam=80, getting worse for fevals=35e3 \approx 50 * N**1.5
        # but the median is not so much getting worse
        # / 5 reflects the sparsity of histbest/median
        # / 2 reflects the left and right part to be compared
        l = int(max((opts['tolstagnation'] / 5. / 2, len(es.fit.histbest) / 10)))
        # TODO: why max(..., len(histbest)/10) ???
        # TODO: the problem in the beginning is only with best ==> ???
        # equality should handle flat fitness
        self._addstop('tolstagnation',  # leads sometimes early stop on ftablet, fcigtab, N>=50?
                      1 < 3 and opts['tolstagnation'] and es.countiter > N * (5 + 100 / es.popsize) and
                      len(es.fit.histbest) > 100 and 2 * l < len(es.fit.histbest) and
                      np.median(es.fit.histmedian[:l]) >= np.median(es.fit.histmedian[l:2 * l]) and
                      np.median(es.fit.histbest[:l]) >= np.median(es.fit.histbest[l:2 * l]))
        # iiinteger: stagnation termination can prevent to find the optimum

        self._addstop('tolupsigma', opts['tolupsigma'] and
                      es.sigma / es.sigma0 / np.max(es.D) > opts['tolupsigma'])

        if 1 < 3:
            # non-user defined, method specific
            # noeffectaxis (CEC: 0.1sigma), noeffectcoord (CEC:0.2sigma), conditioncov
            self._addstop('noeffectcoord',
                         any([es.mean[i] == es.mean[i] + 0.2 * es.sigma * sqrt(es.dC[i])
                              for i in range(N)]))
            if opts['CMA_diagonal'] is not True and es.countiter > opts['CMA_diagonal']:
                i = es.countiter % N
                self._addstop('noeffectaxis',
                             sum(es.mean == es.mean + 0.1 * es.sigma * es.D[i] * es.B[:, i]) == N)
            self._addstop('conditioncov',
                         es.D[-1] > 1e7 * es.D[0], 1e14)  # TODO

            self._addstop('callback', es.callbackstop)  # termination_callback
        if len(self):
            self._addstop('flat fitness: please (re)consider how to compute the fitness more elaborate',
                         len(es.fit.hist) > 9 and
                         max(es.fit.hist) == min(es.fit.hist))
        return self

    def _addstop(self, key, cond, val=None):
        if cond:
            self.stoplist.append(key)  # can have the same key twice
            self[key] = self.opts.get(key, None)

    def clear(self):
        for k in list(self):
            self.pop(k)
        self.stoplist = []

# ____________________________________________________________
# ____________________________________________________________
class _CMAParameters(object):
    """strategy parameters like population size and learning rates.

    Note:
        contrary to `CMAOptions`, `_CMAParameters` is not (yet) part of the
        "user-interface" and subject to future changes (it might become
        a `collections.namedtuple`)

    Example
    -------
    >>> import cma
    >>> es = cma.CMAEvolutionStrategy(20 * [0.1], 1)
    (6_w,12)-CMA-ES (mu_w=3.7,w_1=40%) in dimension 20 (seed=504519190)  # the seed is "random" by default
    >>>
    >>> type(es.sp)  # sp contains the strategy parameters
    <class 'cma._CMAParameters'>
    >>>
    >>> es.sp.disp()
    {'CMA_on': True,
     'N': 20,
     'c1': 0.004181139918745593,
     'c1_sep': 0.034327992810300939,
     'cc': 0.17176721127681213,
     'cc_sep': 0.25259494835857677,
     'cmean': 1.0,
     'cmu': 0.0085149624979034746,
     'cmu_sep': 0.057796356229390715,
     'cs': 0.21434997799189287,
     'damps': 1.2143499779918929,
     'mu': 6,
     'mu_f': 6.0,
     'mueff': 3.7294589343030671,
     'popsize': 12,
     'rankmualpha': 0.3,
     'weights': array([ 0.40240294,  0.25338908,  0.16622156,  0.10437523,  0.05640348,
            0.01720771])}
    >>>
    >> es.sp == cma._CMAParameters(20, 12, cma.CMAOptions().evalall({'N': 20}))
    True

    :See: `CMAOptions`, `CMAEvolutionStrategy`

    """
    def __init__(self, N, opts, ccovfac=1, verbose=True):
        """Compute strategy parameters, mainly depending on
        dimension and population size, by calling `set`

        """
        self.N = N
        if ccovfac == 1:
            ccovfac = opts['CMA_on']  # that's a hack
        self.popsize = None  # declaring the attribute, not necessary though
        self.set(opts, ccovfac=ccovfac, verbose=verbose)

    def set(self, opts, popsize=None, ccovfac=1, verbose=True):
        """Compute strategy parameters as a function
        of dimension and population size """

        alpha_cc = 1.0  # cc-correction for mueff, was zero before

        def cone(df, mu, N, alphacov=2.0):
            """rank one update learning rate, ``df`` is disregarded and obsolete, reduce alphacov on noisy problems, say to 0.5"""
            return alphacov / ((N + 1.3)**2 + mu)

        def cmu(df, mu, alphamu=0.0, alphacov=2.0):
            """rank mu learning rate, disregarding the constrant cmu <= 1 - cone"""
            c = alphacov * (alphamu + mu - 2 + 1 / mu) / ((N + 2)**2 + alphacov * mu / 2)
            # c = alphacov * (alphamu + mu - 2 + 1/mu) / (2 * (N + 2)**1.5 + alphacov * mu / 2)
            # print 'cmu =', c
            return c

        def conedf(df, mu, N):
            """used for computing separable learning rate"""
            return 1. / (df + 2.*sqrt(df) + float(mu) / N)

        def cmudf(df, mu, alphamu):
            """used for computing separable learning rate"""
            return (alphamu + mu - 2. + 1. / mu) / (df + 4.*sqrt(df) + mu / 2.)

        sp = self
        N = sp.N
        if popsize:
            opts.evalall({'N':N, 'popsize':popsize})
        else:
            popsize = opts.evalall({'N':N})['popsize']  # the default popsize is computed in CMAOptions()
        sp.popsize = popsize
        if opts['CMA_mirrors'] < 0.5:
            sp.lam_mirr = int(0.5 + opts['CMA_mirrors'] * popsize)
        elif opts['CMA_mirrors'] > 1:
            sp.lam_mirr = int(0.5 + opts['CMA_mirrors'])
        else:
            sp.lam_mirr = int(0.5 + 0.16 * min((popsize, 2 * N + 2)) + 0.29)  # 0.158650... * popsize is optimal
            # lam = arange(2,22)
            # mirr = 0.16 + 0.29/lam
            # print(lam); print([int(0.5 + l) for l in mirr*lam])
            # [ 2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21]
            # [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4]

        sp.mu_f = sp.popsize / 2.0  # float value of mu
        if opts['CMA_mu'] is not None:
            sp.mu_f = opts['CMA_mu']
        sp.mu = int(sp.mu_f + 0.499999)  # round down for x.5
        # in principle we have mu_opt = popsize/2 + lam_mirr/2,
        # which means in particular weights should only be negative for q > 0.5+mirr_frac/2
        if sp.mu > sp.popsize - 2 * sp.lam_mirr + 1:
            print("WARNING: pairwise selection is not implemented, therefore " +
                  " mu = %d > %d = %d - 2*%d + 1 = popsize - 2*mirr + 1 can produce a bias" % (
                    sp.mu, sp.popsize - 2 * sp.lam_mirr + 1, sp.popsize, sp.lam_mirr))
        if sp.lam_mirr > sp.popsize // 2:
            raise _Error("fraction of mirrors in the population as read from option CMA_mirrors cannot be larger 0.5, " +
                         "theoretically optimal is 0.159")
        sp.weights = log(max([sp.mu, sp.popsize / 2.0]) + 0.5) - log(1 + np.arange(sp.mu))
        sp.weights /= sum(sp.weights)
        sp.mueff = 1 / sum(sp.weights**2)
        sp.cs = (sp.mueff + 2) / (N + sp.mueff + 3)
        # TODO: clean up (here the cumulation constant is shorter if sigma_vec is used)
        sp.dampsvec = opts['CMA_dampsvec_fac'] * (N + 2) if opts['CMA_dampsvec_fac'] else np.Inf
        sp.dampsvec_fading = opts['CMA_dampsvec_fade']
        if np.isfinite(sp.dampsvec):
            sp.cs = ((sp.mueff + 2) / (N + sp.mueff + 3))**0.5
        # sp.cs = (sp.mueff + 2) / (N + 1.5*sp.mueff + 1)
        sp.cc = (4 + alpha_cc * sp.mueff / N) / (N + 4 + alpha_cc * 2 * sp.mueff / N)
        sp.cc_sep = (1 + 1 / N + alpha_cc * sp.mueff / N) / (N**0.5 + 1 / N + alpha_cc * 2 * sp.mueff / N)  # \not\gg\cc
        sp.rankmualpha = opts['CMA_rankmualpha']
        # sp.rankmualpha = _evalOption(opts['CMA_rankmualpha'], 0.3)
        sp.c1 = ccovfac * min(1, sp.popsize / 6) * cone((N**2 + N) / 2, sp.mueff, N)  # 2. / ((N+1.3)**2 + sp.mucov)
        sp.c1_sep = ccovfac * conedf(N, sp.mueff, N)
        if opts['CMA_rankmu'] != 0:  # also empty
            sp.cmu = min(1 - sp.c1, ccovfac * cmu((N**2 + N) / 2, sp.mueff, sp.rankmualpha))
            sp.cmu_sep = min(1 - sp.c1_sep, ccovfac * cmudf(N, sp.mueff, sp.rankmualpha))
        else:
            sp.cmu = sp.cmu_sep = 0

        sp.neg = _BlancClass()
        if opts['CMA_active']:
            # in principle we have mu_opt = popsize/2 + lam_mirr/2,
            # which means in particular weights should only be negative for q > 0.5+mirr_frac/2
            if 1 < 3: # seems most natural: continuation of log(lambda/2) - log(k) qqqqqqqqqqqqqqqqqqqqqqqqqq
                sp.neg.mu_f = popsize // 2  # not sure anymore what this is good for
                sp.neg.weights = array([log(k) - log(popsize/2 + 1/2) for k in np.arange(np.ceil(popsize/2 + 1.1/2), popsize + .1)])
            sp.neg.mu = len(sp.neg.weights)
            sp.neg.weights /= sum(sp.neg.weights)
            sp.neg.mueff = 1 / sum(sp.neg.weights**2)
            sp.neg.cmuexp = opts['CMA_activefac'] * 0.5 * sp.neg.mueff / ((N + 2)**1.5 + 1.0 * sp.neg.mueff)
            # reasoning on learning rate cmuexp: with sum |w| == 1 and
            #   length-normalized vectors in the update, the residual
            #   variance in any direction exceeds exp(-N*cmuexp)
            assert sp.neg.mu >= sp.lam_mirr  # not really necessary
            # sp.neg.minresidualvariance = 0.66  # not it use, keep at least 0.66 in all directions, small popsize is most critical
        else:
            sp.neg.cmuexp = 0

        sp.CMA_on = sp.c1 + sp.cmu > 0
        # print(sp.c1_sep / sp.cc_sep)

        if not opts['CMA_on'] and opts['CMA_on'] not in (None, [], (), ''):
            sp.CMA_on = False
            # sp.c1 = sp.cmu = sp.c1_sep = sp.cmu_sep = 0
        mueff_exponent = 0.5
        if 1 < 3:
            mueff_exponent = opts['CSA_damp_mueff_exponent']
        # TODO: this will disappear, as it is done in class CMAAdaptSigmaCSA
        sp.damps = opts['CSA_dampfac'] * (0.5 +
                                          0.5 * min([1, (sp.lam_mirr / (0.159 * sp.popsize) - 1)**2])**1 +
                                          2 * max([0, ((sp.mueff - 1) / (N + 1))**mueff_exponent - 1]) + sp.cs
                                          )
        sp.cmean = float(opts['CMA_cmean'])
        # sp.kappa = 1  # 4-D, lam=16, rank1, kappa < 4 does not influence convergence rate
                        # in larger dim it does, 15-D with defaults, kappa=8 factor 2
        if sp.cmean != 1:
            print('  cmean = %f' % (sp.cmean))

        if verbose:
            if not sp.CMA_on:
                print('covariance matrix adaptation turned off')
            if opts['CMA_mu'] != None:
                print('mu = %f' % (sp.mu_f))

        # return self  # the constructor returns itself

    def disp(self):
        pprint(self.__dict__)

def fmin(objective_function, x0, sigma0,
         options=None,
         args=(),
         restarts=0,
         restart_from_best='False',
         incpopsize=2,
         eval_initial_x=False,
         noise_handler=None,
         noise_change_sigma_exponent=1,
         noise_kappa_exponent=0  # TODO: add max kappa value as parameter
        ):
    """functional interface to the stochastic optimizer CMA-ES
    for non-convex function minimization.

    Calling Sequences
    =================
        ``fmin(objective_function, x0, sigma0)``
            minimizes `objective_function` starting at `x0` and with standard deviation
            `sigma0` (step-size)
        ``fmin(objective_function, x0, sigma0, options={'ftarget': 1e-5})``
            minimizes `objective_function` up to target function value 1e-5, which
            is typically useful for benchmarking.
        ``fmin(objective_function, x0, sigma0, args=('f',))``
            minimizes `objective_function` called with an additional argument ``'f'``.
        ``fmin(objective_function, x0, sigma0, options={'ftarget':1e-5, 'popsize':40})``
            uses additional options ``ftarget`` and ``popsize``
        ``fmin(objective_function, esobj, None, options={'maxfevals': 1e5})``
            uses the `CMAEvolutionStrategy` object instance `esobj` to optimize
            `objective_function`, similar to `esobj.optimize()`.

    Arguments
    =========
        `objective_function`
            function to be minimized. Called as ``objective_function(x,*args)``.
            `x` is a one-dimensional `numpy.ndarray`. `objective_function`
            can return `numpy.NaN`,
            which is interpreted as outright rejection of solution `x`
            and invokes an immediate resampling and (re-)evaluation
            of a new solution not counting as function evaluation.
        `x0`
            list or `numpy.ndarray`, initial guess of minimum solution
            before the application of the geno-phenotype transformation
            according to the ``transformation`` option. Otherwise
            `x0` can also be a `cma.CMAEvolutionStrategy` object instance.
            In the latter case `sigma0` can be ``None``.
        `sigma0`
            scalar, initial standard deviation in each coordinate.
            `sigma0` should be about 1/4th of the search domain width (where the
            optimum is to be expected). The variables in `objective_function`
            should be scaled such that they presumably have similar sensitivity.
            See also option `scaling_of_variables`.
        `options`
            a dictionary with additional options passed to the constructor
            of class ``CMAEvolutionStrategy``, see ``cma.CMAOptions()`` for
            a list of available options.
        ``args=()``
            arguments to be used to call the `objective_function`
        ``restarts=0``
            number of restarts
        ``restart_from_best=False``
            which point to restart from
        ``incpopsize=2``
            multiplier for increasing the population size `popsize` before each restart
        ``eval_initial_x=False``
            evaluate initial solution
        ``noise_handler=None``
            a ``class NoiseHandler`` object or ``None``
        ``noise_change_sigma_exponent=1``
            exponent for sigma increment for additional noise treatment
        ``noise_evaluations_as_kappa``
            instead of applying reevaluations, the "number of evaluations"
            is (ab)used as scaling factor kappa (experimental).

    Optional Arguments
    ==================
    All values in the `options` dictionary are evaluated if they are of
    type `str`, besides `verb_filenameprefix`, see class `CMAOptions` for details.
    The full list is available in ``cma.default_options``.

    >>> import cma
    >>> cma.CMAOptions()

    Subsets of options can be displayed, for example like ``cma.CMAOptions('tol')``,
    or ``cma.CMAOptions('bound')``, see also class `CMAOptions`.

    Return
    ======
    Similar to `OOOptimizer.optimize()` and/or `CMAEvolutionStrategy.optimize()`, return the
    list provided by `CMAEvolutionStrategy.result()` appended with an `OOOptimizer` and an
    `BaseDataLogger`::

        res = es.result() + (es.stop(), es, logger)

    where
        - ``res[0]`` (``xopt``) -- best evaluated solution
        - ``res[1]`` (``fopt``) -- respective function value
        - ``res[2]`` (``evalsopt``) -- respective number of function evaluations
        - ``res[3]`` (``evals``) -- number of overall conducted objective function evaluations
        - ``res[4]`` (``iterations``) -- number of overall conducted iterations
        - ``res[5]`` (``xmean``) -- mean of the final sample distribution
        - ``res[6]`` (``stds``) -- effective stds of the final sample distribution
        - ``res[-3]`` (``stop``) -- termination condition(s) in a dictionary
        - ``res[-2]`` (``cmaes``) -- class `CMAEvolutionStrategy` instance
        - ``res[-1]`` (``logger``) -- class `CMADataLogger` instance

    Details
    =======
    This function is an interface to the class `CMAEvolutionStrategy`. The
    latter class should be used when full control over the iteration loop
    of the optimizer is desired.

    The noise handling follows closely [Hansen et al 2009, A Method for Handling
    Uncertainty in Evolutionary Optimization...] in the measurement part, but the
    implemented treatment is slightly different: for ``noiseS > 0``, ``evaluations``
    (time) and sigma are increased by ``alpha``. For ``noiseS < 0``, ``evaluations``
    (time) is decreased by ``alpha**(1/4)``. The option ``noise_handling`` switches
    the noise handling on/off, the given value defines the maximal number
    of evaluations for a single fitness computation. If ``noise_handling`` is a list,
    the smallest element defines the minimal number and if the list has three elements,
    the median value is the start value for ``evaluations``. See also class
    `NoiseHandler`.

    Examples
    ========
    The following example calls `fmin` optimizing the Rosenbrock function
    in 10-D with initial solution 0.1 and initial step-size 0.5. The
    options are specified for the usage with the `doctest` module.

    >>> import cma
    >>> # cma.CMAOptions()  # returns all possible options
    >>> options = {'CMA_diagonal':100, 'seed':1234, 'verb_time':0}
    >>>
    >>> res = cma.fmin(cma.fcts.rosen, [0.1] * 10, 0.5, options)
    (5_w,10)-CMA-ES (mu_w=3.2,w_1=45%) in dimension 10 (seed=1234)
       Covariance matrix is diagonal for 10 iterations (1/ccov=29.0)
    Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
        1      10 1.264232686260072e+02 1.1e+00 4.40e-01  4e-01  4e-01
        2      20 1.023929748193649e+02 1.1e+00 4.00e-01  4e-01  4e-01
        3      30 1.214724267489674e+02 1.2e+00 3.70e-01  3e-01  4e-01
      100    1000 6.366683525319511e+00 6.2e+00 2.49e-02  9e-03  3e-02
      200    2000 3.347312410388666e+00 1.2e+01 4.52e-02  8e-03  4e-02
      300    3000 1.027509686232270e+00 1.3e+01 2.85e-02  5e-03  2e-02
      400    4000 1.279649321170636e-01 2.3e+01 3.53e-02  3e-03  3e-02
      500    5000 4.302636076186532e-04 4.6e+01 4.78e-03  3e-04  5e-03
      600    6000 6.943669235595049e-11 5.1e+01 5.41e-06  1e-07  4e-06
      650    6500 5.557961334063003e-14 5.4e+01 1.88e-07  4e-09  1e-07
    termination on tolfun : 1e-11
    final/bestever f-value = 5.55796133406e-14 2.62435631419e-14
    mean solution:  [ 1.          1.00000001  1.          1.
        1.          1.00000001  1.00000002  1.00000003 ...]
    std deviation: [ 3.9193387e-09  3.7792732e-09  4.0062285e-09  4.6605925e-09
        5.4966188e-09   7.4377745e-09   1.3797207e-08   2.6020765e-08 ...]
    >>>
    >>> print('best solutions fitness = %f' % (res[1]))
    best solutions fitness = 2.62435631419e-14
    >>> assert res[1] < 1e-12

    The above call is pretty much equivalent with the slightly more
    verbose call::

        res = cma.CMAEvolutionStrategy([0.1] * 10, 0.5,
                    options=options).optimize(cma.fcts.rosen)

    In either case, the method::

        cma.plot();

    (based on `matplotlib.pyplot`) produces a plot of the run and, if
    necessary::

        cma.show()

    shows the plot in a window. To continue you might need to
    close the pop-up window. This behavior seems to disappear in
    subsequent calls of `cma.plot()` and is avoided by using
    `ipython` with `-pylab` option. Finally::

        cma.savefig('myfirstrun')  # savefig from matplotlib.pyplot

    will save the figure in a png.

    :See: `CMAEvolutionStrategy`, `OOOptimizer.optimize(), `plot()`,
        `CMAOptions`, `scipy.optimize.fmin()`

    """  # style guides say there should be the above empty line
    if 1 < 3:  # try: # pass on KeyboardInterrupt
        if not objective_function:  # return available options in a dictionary
            return CMAOptions()  # these opts are by definition valid

        fmin_options = locals().copy()  # archive original options
        del fmin_options['objective_function']
        del fmin_options['x0']
        del fmin_options['sigma0']
        del fmin_options['options']
        del fmin_options['args']

        if options is None:
            options = cma_default_options
        opts = CMAOptions(options.copy()).complement()

        irun = 0
        best = BestSolution()
        while True:  # restart loop
            # recover from a CMA object
            if irun == 0 and isinstance(x0, CMAEvolutionStrategy):
                es = x0
                x0 = es.inputargs['x0']  # for the next restarts
                if sigma0 is None or not np.isscalar(array(sigma0)):
                    sigma0 = es.inputargs['sigma0']  # for the next restarts
                # ignore further input args and keep original options
            else:  # default case
                if irun and eval(str(fmin_options['restart_from_best'])):
                    print('CAVE: restart_from_best is often not useful')
                    es = CMAEvolutionStrategy(best.x, sigma0, opts)
                else:
                    es = CMAEvolutionStrategy(x0, sigma0, opts)
                if eval_initial_x:
                    x = es.gp.pheno(es.mean, into_bounds=es.boundary_handler.repair, archive=es.sent_solutions)
                    es.best.update([x], es.sent_solutions, [objective_function(x, *args)], 1)
                    es.countevals += 1

            opts = es.opts  # processed options, unambiguous
            # a hack:
            fmin_opts = CMAOptions(fmin_options.copy(), unchecked=True)
            for k in fmin_opts:
                # locals() cannot be modified directly, exec won't work in 3.x, therefore
                fmin_opts.eval(k, loc={'N': es.N, 'popsize': opts['popsize']})

            append = opts['verb_append'] or es.countiter > 0 or irun > 0
            # es.logger is "the same" logger, because the "identity" is only determined by the `filenameprefix`
            logger = CMADataLogger(opts['verb_filenameprefix'], opts['verb_log'])
            logger.register(es, append).add()  # initial values, not fitness values

            # if es.countiter == 0 and es.opts['verb_log'] > 0 and not es.opts['verb_append']:
            #    logger = CMADataLogger(es.opts['verb_filenameprefix']).register(es)
            #    logger.add()
            # es.writeOutput()  # initial values for sigma etc

            if 1 < 3:
                if noise_handler:
                    noisehandler = noise_handler
                    noise_handling = True
                else:
                    noisehandler = NoiseHandler(es.N, 0)
                    noise_handling = False
                es.noise_handler = noisehandler

            # the problem: this assumes that good solutions cannot take longer than bad ones:
            # with EvalInParallel(objective_function, 2, is_feasible=opts['is_feasible']) as eval_in_parallel:
            if 1 < 3:
                while not es.stop():  # iteration loop
                    # X, fit = eval_in_parallel(lambda: es.ask(1)[0], es.popsize, args, repetitions=noisehandler.evaluations-1)
                    X, fit = es.ask_and_eval(objective_function, args,
                                             evaluations=noisehandler.evaluations,
                                             aggregation=np.median)  # treats NaN with resampling
                    # TODO: check args and in case use args=(noisehandler.evaluations, )

                    es.tell(X, fit)  # prepare for next iteration
                    if noise_handling:  # it would be better to also use these f-evaluations in tell
                        es.sigma *= noisehandler(X, fit, objective_function, es.ask,
                                                 args=args)**fmin_opts['noise_change_sigma_exponent']
                        es.countevals += noisehandler.evaluations_just_done  # TODO: this is a hack, not important though
                        if 1 < 3:
                            es.sp.cmean *= exp(-noise_kappa_exponent * np.tanh(noisehandler.noiseS))
                            if es.sp.cmean > 1:
                                es.sp.cmean = 1

                    es.disp()
                    logger.add(more_data=[noisehandler.evaluations, 10**noisehandler.noiseS] if noise_handling else [],
                               modulo=1 if es.stop() and logger.modulo else None)
                    if (opts['verb_log'] and opts['verb_plot'] and
                          (es.countiter % max(opts['verb_plot'], opts['verb_log']) == 0 or es.stop())):
                        logger.plot(324)

            # end while not es.stop
            mean_pheno = es.gp.pheno(es.mean, into_bounds=es.boundary_handler.repair, archive=es.sent_solutions)
            fmean = objective_function(mean_pheno, *args)
            es.countevals += 1

            es.best.update([mean_pheno], es.sent_solutions, [fmean], es.countevals)
            best.update(es.best, es.sent_solutions)  # in restarted case

            # final message
            if opts['verb_disp']:
                es.result_pretty(irun, time.asctime(time.localtime()))

            irun += 1
            if irun > fmin_opts['restarts'] or 'ftarget' in es.stopdict or 'maxfevals' in es.stopdict:
                break
            opts['verb_append'] = es.countevals
            opts['popsize'] = fmin_opts['incpopsize'] * es.sp.popsize  # TODO: use rather options?
            opts['seed'] += 1

        # while irun

        es.out['best'] = best  # TODO: this is a rather suboptimal type for inspection in the shell
        if 1 < 3:
            return es.result() + (es.stop(), es, logger)

        else:  # previously: to be removed
            return (best.x.copy(), best.f, es.countevals,
                    dict((('stopdict', CMAStopDict(es.stopdict))
                          , ('mean', es.gp.pheno(es.mean))
                          , ('std', es.sigma * sqrt(es.dC) * es.gp.scales)
                          , ('out', es.out)
                          , ('opts', es.opts)  # last state of options
                          , ('cma', es)
                          , ('inputargs', es.inputargs)
                          ))
                   )
        # TODO refine output, can #args be flexible?
        # is this well usable as it is now?
    else:  # except KeyboardInterrupt:  # Exception, e:
        if eval(str(options['verb_disp'])) > 0:
            print(' in/outcomment ``raise`` in last line of cma.fmin to prevent/restore KeyboardInterrupt exception')
        raise  # cave: swallowing this exception can silently mess up experiments, if ctrl-C is hit

# _____________________________________________________________________
# _____________________________________________________________________
#
class BaseDataLogger(object):
    """"abstract" base class for a data logger that can be used with an `OOOptimizer`

    Details: attribute `modulo` is used in ``OOOptimizer.optimize``

    """
    def add(self, optim=None, more_data=[]):
        """abstract method, add a "data point" from the state of `optim` into the
        logger, the argument `optim` can be omitted if it was `register()`-ed before,
        acts like an event handler"""
        raise NotImplementedError()
    def register(self, optim):
        """abstract method, register an optimizer `optim`, only needed if `add()` is
        called without a value for the `optim` argument"""
        self.optim = optim
    def disp(self):
        """display some data trace (not implemented)"""
        print('method BaseDataLogger.disp() not implemented, to be done in subclass ' + str(type(self)))
    def plot(self):
        """plot data (not implemented)"""
        print('method BaseDataLogger.plot() is not implemented, to be done in subclass ' + str(type(self)))
    def data(self):
        """return logged data in a dictionary (not implemented)"""
        print('method BaseDataLogger.data() is not implemented, to be done in subclass ' + str(type(self)))

# _____________________________________________________________________
# _____________________________________________________________________
#
class CMADataLogger(BaseDataLogger):
    """data logger for class `CMAEvolutionStrategy`. The logger is
    identified by its name prefix and (over-)writes or reads according
    data files. Therefore, the logger must be considered as *global* variable
    with unpredictable side effects, if two loggers with the same name
    and on the same working folder are used at the same time.

    Examples
    ========
    ::

        import cma
        es = cma.CMAEvolutionStrategy(...)
        logger = cma.CMADataLogger().register(es)
        while not es.stop():
            ...
            logger.add()  # add can also take an argument

        logger.plot() # or a short cut can be used:
        cma.plot()  # plot data from logger with default name


        logger2 = cma.CMADataLogger('just_another_filename_prefix').load()
        logger2.plot()
        logger2.disp()

    ::

        import cma
        from matplotlib.pylab import *
        res = cma.fmin(cma.Fcts.sphere, rand(10), 1e-0)
        logger = res[-1]  # the CMADataLogger
        logger.load()  # by "default" data are on disk
        semilogy(logger.f[:,0], logger.f[:,5])  # plot f versus iteration, see file header
        show()

    Details
    =======
    After loading data, the logger has the attributes `xmean`, `xrecent`, `std`, `f`, and `D`,
    corresponding to ``xmean``, ``xrecentbest``, ``stddev``, ``fit``, and ``axlen`` filename
    trails.

    :See: `disp()`, `plot()`

    """
    default_prefix = 'outcmaes'
    # names = ('axlen','fit','stddev','xmean','xrecentbest')
    # key_names_with_annotation = ('std', 'xmean', 'xrecent')

    def __init__(self, name_prefix=default_prefix, modulo=1, append=False):
        """initialize logging of data from a `CMAEvolutionStrategy` instance,
        default ``modulo=1`` means logging with each call

        """
        # super(CMAData, self).__init__({'iter':[], 'stds':[], 'D':[], 'sig':[], 'fit':[], 'xm':[]})
        # class properties:
        self.file_names = ('axlen', 'fit', 'stddev', 'xmean', 'xrecentbest')  # used in load, however hard-coded in add
        self.key_names = ('D', 'f', 'std', 'xmean', 'xrecent')  # used in load, however hard-coded in plot
        self.key_names_with_annotation = ('std', 'xmean', 'xrecent')  # used in load
        self.modulo = modulo  # allows calling with None
        self.append = append
        self.counter = 0  # number of calls of add, should initial value depend on `append`?
        self.last_iteration = 0
        self.name_prefix = name_prefix if name_prefix else CMADataLogger.default_prefix
        if isinstance(self.name_prefix, CMAEvolutionStrategy):
            self.name_prefix = self.name_prefix.opts.eval('verb_filenameprefix')
        self.registered = False

    def register(self, es, append=None, modulo=None):
        """register a `CMAEvolutionStrategy` instance for logging,
        ``append=True`` appends to previous data logged under the same name,
        by default previous data are overwritten.

        """
        if not isinstance(es, CMAEvolutionStrategy):
            raise TypeError("only class CMAEvolutionStrategy can be registered for logging")
        self.es = es
        if append is not None:
            self.append = append
        if modulo is not None:
            self.modulo = modulo
        self.registered = True
        return self

    def initialize(self, modulo=None):
        """reset logger, overwrite original files, `modulo`: log only every modulo call"""
        if modulo is not None:
            self.modulo = modulo
        try:
            es = self.es  # must have been registered
        except AttributeError:
            pass  # TODO: revise usage of es... that this can pass
            raise _Error('call register() before initialize()')

        self.counter = 0  # number of calls of add
        self.last_iteration = 0  # some lines are only written if iteration>last_iteration

        # write headers for output
        fn = self.name_prefix + 'fit.dat'
        strseedtime = 'seed=%d, %s' % (es.opts['seed'], time.asctime())

        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, sigma, axis ratio, ' +
                        'bestever, best, median, worst objective function value, ' +
                        'further objective values of best", ' +
                        strseedtime +
                        # strftime("%Y/%m/%d %H:%M:%S", localtime()) + # just asctime() would do
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'axlen.dat'
        try:
            f = open(fn, 'w')
            f.write('%  columns="iteration, evaluation, sigma, max axis length, ' +
                    ' min axis length, all principle axes lengths ' +
                    ' (sorted square roots of eigenvalues of C)", ' +
                    strseedtime +
                    '\n')
            f.close()
        except (IOError, OSError):
            print('could not open file ' + fn)
        finally:
            f.close()
        fn = self.name_prefix + 'stddev.dat'
        try:
            f = open(fn, 'w')
            f.write('% # columns=["iteration, evaluation, sigma, void, void, ' +
                    ' stds==sigma*sqrt(diag(C))", ' +
                    strseedtime +
                    '\n')
            f.close()
        except (IOError, OSError):
            print('could not open file ' + fn)
        finally:
            f.close()

        fn = self.name_prefix + 'xmean.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, void, void, void, xmean", ' +
                        strseedtime)
                f.write(' # scaling_of_variables: ')
                if np.size(es.gp.scales) > 1:
                    f.write(' '.join(map(str, es.gp.scales)))
                else:
                    f.write(str(es.gp.scales))
                f.write(', typical_x: ')
                if np.size(es.gp.typical_x) > 1:
                    f.write(' '.join(map(str, es.gp.typical_x)))
                else:
                    f.write(str(es.gp.typical_x))
                f.write('\n')
                f.close()
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        fn = self.name_prefix + 'xrecentbest.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # iter+eval+sigma+0+fitness+xbest, ' +
                        strseedtime +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        return self
    # end def __init__

    def load(self, filenameprefix=None):
        """loads data from files written and return a data dictionary, *not*
        a prerequisite for using `plot()` or `disp()`.

        Argument `filenameprefix` is the filename prefix of data to be loaded (five files),
        by default ``'outcmaes'``.

        Return data dictionary with keys `xrecent`, `xmean`, `f`, `D`, `std`

        """
        if not filenameprefix:
            filenameprefix = self.name_prefix
        for i in rglen((self.file_names)):
            fn = filenameprefix + self.file_names[i] + '.dat'
            try:
                self.__dict__[self.key_names[i]] = _fileToMatrix(fn)
            except:
                print('WARNING: reading from file "' + fn + '" failed')
            if self.key_names[i] in self.key_names_with_annotation:
                self.__dict__[self.key_names[i]].append(self.__dict__[self.key_names[i]][-1])  # copy last row to later fill in annotation position for display
            self.__dict__[self.key_names[i]] = array(self.__dict__[self.key_names[i]], copy=False)
        return self

    def add(self, es=None, more_data=[], modulo=None):  # TODO: find a different way to communicate current x and f
        """append some logging data from `CMAEvolutionStrategy` class instance `es`,
        if ``number_of_times_called % modulo`` equals to zero, never if ``modulo==0``.

        The sequence ``more_data`` must always have the same length.

        When used for a different optimizer class, this function can be
        (easily?) adapted by changing the assignments under INTERFACE
        in the implemention.

        """
        mod = modulo if modulo is not None else self.modulo
        self.counter += 1
        if mod == 0 or (self.counter > 3 and (self.counter - 1) % mod):
            return
        if es is None:
            try:
                es = self.es  # must have been registered
            except AttributeError :
                raise _Error('call `add` with argument `es` or ``register(es)`` before ``add()``')
        elif not self.registered:
            self.register(es)

        if 1 < 3:
            if self.counter == 1 and not self.append and self.modulo != 0:
                self.initialize()  # write file headers
                self.counter = 1

        # --- INTERFACE, can be changed if necessary ---
        if not isinstance(es, CMAEvolutionStrategy):  # not necessary
            print('WARNING: <type \'CMAEvolutionStrategy\'> expected, found '
                            + str(type(es)) + ' in method CMADataLogger.add')
        evals = es.countevals
        iteration = es.countiter
        sigma = es.sigma
        axratio = es.D.max() / es.D.min()
        xmean = es.mean  # TODO: should be optionally phenotype?
        fmean_noise_free = es.fmean_noise_free
        fmean = es.fmean
        try:
            besteverf = es.best.f
            bestf = es.fit.fit[0]
            worstf = es.fit.fit[-1]
            medianf = es.fit.fit[es.sp.popsize // 2]
        except:
            if iteration > 0:  # first call without f-values is OK
                raise
        try:
            xrecent = es.best.last.x
        except:
            xrecent = None
        maxD = es.D.max()
        minD = es.D.min()
        diagD = es.D
        diagC = es.sigma * es.sigma_vec * sqrt(es.dC)
        more_to_write = es.more_to_write
        es.more_to_write = []
        # --- end interface ---

        try:
            # fit
            if iteration > self.last_iteration:
                fn = self.name_prefix + 'fit.dat'
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(sigma) + ' '
                            + str(axratio) + ' '
                            + str(besteverf) + ' '
                            + '%.16e' % bestf + ' '
                            + str(medianf) + ' '
                            + str(worstf) + ' '
                            # + str(es.sp.popsize) + ' '
                            # + str(10**es.noiseS) + ' '
                            # + str(es.sp.cmean) + ' '
                            + ' '.join(str(i) for i in more_to_write) + ' '
                            + ' '.join(str(i) for i in more_data) + ' '
                            + '\n')
            # axlen
            fn = self.name_prefix + 'axlen.dat'
            with open(fn, 'a') as f:  # does not rely on reference counting
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(sigma) + ' '
                        + str(maxD) + ' '
                        + str(minD) + ' '
                        + ' '.join(map(str, diagD))
                        + '\n')
            # stddev
            fn = self.name_prefix + 'stddev.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(sigma) + ' '
                        + '0 0 '
                        + ' '.join(map(str, diagC))
                        + '\n')
            # xmean
            fn = self.name_prefix + 'xmean.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        # + str(sigma) + ' '
                        + '0 '
                        + str(fmean_noise_free) + ' '
                        + str(fmean) + ' '  # TODO: this does not make sense
                        # TODO should be optional the phenotyp?
                        + ' '.join(map(str, xmean))
                        + '\n')
            # xrecent
            fn = self.name_prefix + 'xrecentbest.dat'
            if iteration > 0 and xrecent is not None:
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(sigma) + ' '
                            + '0 '
                            + str(bestf) + ' '
                            + ' '.join(map(str, xrecent))
                            + '\n')

        except (IOError, OSError):
            if iteration <= 1:
                print('could not open/write file')
        self.last_iteration = iteration

    def closefig(self):
        pyplot.close(self.fighandle)

    def save(self, nameprefix, switch=False):
        """saves logger data to a different set of files, for
        ``switch=True`` also the loggers name prefix is switched to
        the new value

        """
        if not nameprefix or not isinstance(nameprefix, str):
            raise _Error('filename prefix must be a nonempty string')

        if nameprefix == self.default_prefix:
            raise _Error('cannot save to default name "' + nameprefix + '...", chose another name')

        if nameprefix == self.name_prefix:
            return

        for name in CMADataLogger.names:
            open(nameprefix + name + '.dat', 'w').write(open(self.name_prefix + name + '.dat').read())

        if switch:
            self.name_prefix = nameprefix

    def plot(self, fig=None, iabscissa=1, iteridx=None,
             plot_mean=False,  # was: plot_mean=True
             foffset=1e-19, x_opt=None, fontsize=9):
        """
        plot data from a `CMADataLogger` (using the files written by the logger).

        Arguments
        ---------
            `fig`
                figure number, by default 325
            `iabscissa`
                ``0==plot`` versus iteration count,
                ``1==plot`` versus function evaluation number
            `iteridx`
                iteration indices to plot

        Return `CMADataLogger` itself.

        Examples
        --------
        ::

            import cma
            logger = cma.CMADataLogger()  # with default name
            # try to plot the "default logging" data (e.g.
            #   from previous fmin calls, which is essentially what
            #   also cma.plot() does)
            logger.plot()
            cma.savefig('fig325.png')  # save current figure
            logger.closefig()

        Dependencies: matlabplotlib/pyplot.

        """
        dat = self.load(self.name_prefix)
        try:
            # pyplot: prodedural interface for matplotlib
            import matplotlib
            matplotlib.use('TkAgg')
            from  matplotlib.pyplot import figure, subplot, semilogy, plot, grid, \
                 axis, title, text, xlabel, isinteractive, gcf

        except ImportError:
            raise ImportError('could not find matplotlib.pyplot module, function plot() is not available')

        if iabscissa not in (0, 1):
            iabscissa = 1

        # interactive_status = matplotlib.is_interactive()
        pyplot.ioff()  # prevents immediate drawing, much faster

        dat.x = dat.xmean  # this is the genotyp
        if not plot_mean:
            if len(dat.x) < 2:
                print('not enough data to plot recent x')
            else:
                dat.x = dat.xrecent

        if iteridx is not None:
            dat.f = dat.f[np.where([x in iteridx for x in dat.f[:, 0]])[0], :]
            dat.D = dat.D[np.where([x in iteridx for x in dat.D[:, 0]])[0], :]
            iteridx.append(dat.x[-1, 1])  # last entry is artificial
            dat.x = dat.x[np.where([x in iteridx for x in dat.x[:, 0]])[0], :]
            dat.std = dat.std[np.where([x in iteridx for x in dat.std[:, 0]])[0], :]

        if iabscissa == 0:
            xlab = 'iterations'
        elif iabscissa == 1:
            xlab = 'function evaluations'

        # use fake last entry in x and std for line extension-annotation
        if dat.x.shape[1] < 100:
            minxend = int(1.06 * dat.x[-2, iabscissa])
            # write y-values for individual annotation into dat.x
            dat.x[-1, iabscissa] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat.x[-2, 5:])
            idx2 = np.argsort(idx)
            if x_opt is None:
                dat.x[-1, 5 + idx] = np.linspace(np.min(dat.x[:, 5:]),
                            np.max(dat.x[:, 5:]), dat.x.shape[1] - 5)
            else:
                dat.x[-1, 5 + idx] = np.logspace(np.log10(np.min(abs(dat.x[:, 5:]))),
                            np.log10(np.max(abs(dat.x[:, 5:]))), dat.x.shape[1] - 5)
        else:
            minxend = 0

        if len(dat.f) == 0:
            print('nothing to plot')
            return

        # not in use anymore, see formatter above
        # xticklocs = np.arange(5) * np.round(minxend/4., -int(np.log10(minxend/4.)))

        # dfit(dfit<1e-98) = NaN;

        # TODO: if abscissa==0 plot in chunks, ie loop over subsets where dat.f[:,0]==countiter is monotonous

        original_fontsize = pyplot.rcParams['font.size']
        pyplot.rcParams['font.size'] = fontsize
        if fig:
            figure(fig)
        else:
            figure(325)
            # show()  # should not be necessary
        self.fighandle = gcf()  # fighandle.number

        subplot(2, 2, 1)
        self.plotdivers(dat, iabscissa, foffset)

        # TODO: modularize also the remaining subplots
        subplot(2, 2, 2)
        #hold(False)
        if x_opt is not None:  # TODO: differentate neg and pos?
            semilogy(dat.x[:, iabscissa], abs(dat.x[:, 5:]) - x_opt, '-')
        else:
            plot(dat.x[:, iabscissa], dat.x[:, 5:], '-')
        #hold(True)
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        ax[1] -= 1e-6
        if dat.x.shape[1] < 100:
            yy = np.linspace(ax[2] + 1e-6, ax[3] - 1e-6, dat.x.shape[1] - 5)
            # yyl = np.sort(dat.x[-1,5:])
            idx = np.argsort(dat.x[-1, 5:])
            idx2 = np.argsort(idx)
            if x_opt is not None:
                semilogy([dat.x[-1, iabscissa], ax[1]], [abs(dat.x[-1, 5:]), yy[idx2]], 'k-')  # line from last data point
                semilogy(np.dot(dat.x[-2, iabscissa], [1, 1]), array([ax[2] + 1e-6, ax[3] - 1e-6]), 'k-')
            else:
                # plot([dat.x[-1, iabscissa], ax[1]], [dat.x[-1,5:], yy[idx2]], 'k-') # line from last data point
                plot(np.dot(dat.x[-2, iabscissa], [1, 1]), array([ax[2] + 1e-6, ax[3] - 1e-6]), 'k-')
            # plot(array([dat.x[-1, iabscissa], ax[1]]),
            #      reshape(array([dat.x[-1,5:], yy[idx2]]).flatten(), (2,4)), '-k')
            for i in range(len(idx)):
                # TODOqqq: annotate phenotypic value!?
                # text(ax[1], yy[i], 'x(' + str(idx[i]) + ')=' + str(dat.x[-2,5+idx[i]]))
                text(dat.x[-1, iabscissa], dat.x[-1, 5 + i], 'x(' + str(i) + ')=' + str(dat.x[-2, 5 + i]))

        i = 2  # find smallest i where iteration count differs (in case the same row appears twice)
        while i < len(dat.f) and dat.f[-i][0] == dat.f[-1][0]:
            i += 1
        title('Object Variables (' + ('mean' if plot_mean else 'curr best') +
                ', ' + str(dat.x.shape[1] - 5) + '-D, popsize~' +
                (str(int((dat.f[-1][1] - dat.f[-i][1]) / (dat.f[-1][0] - dat.f[-i][0])))
                    if len(dat.f.T[0]) > 1 and dat.f[-1][0] > dat.f[-i][0] else 'NA')
                + ')')
        # pyplot.xticks(xticklocs)

        # Scaling
        subplot(2, 2, 3)
        #hold(False)
        semilogy(dat.D[:, iabscissa], dat.D[:, 5:], '-b')
        #hold(True)
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        title('Scaling (All Main Axes)')
        # pyplot.xticks(xticklocs)
        xlabel(xlab)

        # standard deviations
        subplot(2, 2, 4)
        #hold(False)
        # remove sigma from stds (graphs become much better readible)
        dat.std[:, 5:] = np.transpose(dat.std[:, 5:].T / dat.std[:, 2].T)
        # ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        # axis(ax)
        if 1 < 2 and dat.std.shape[1] < 100:
            # use fake last entry in x and std for line extension-annotation
            minxend = int(1.06 * dat.x[-2, iabscissa])
            dat.std[-1, iabscissa] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat.std[-2, 5:])
            idx2 = np.argsort(idx)
            dat.std[-1, 5 + idx] = np.logspace(np.log10(np.min(dat.std[:, 5:])),
                            np.log10(np.max(dat.std[:, 5:])), dat.std.shape[1] - 5)

            dat.std[-1, iabscissa] = minxend  # TODO: should be ax[1]
            yy = np.logspace(np.log10(ax[2]), np.log10(ax[3]), dat.std.shape[1] - 5)
            # yyl = np.sort(dat.std[-1,5:])
            idx = np.argsort(dat.std[-1, 5:])
            idx2 = np.argsort(idx)
            # plot(np.dot(dat.std[-2, iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-') # vertical separator
            # vertical separator
            plot(np.dot(dat.std[-2, iabscissa], [1, 1]), array([np.min(dat.std[-2, 5:]), np.max(dat.std[-2, 5:])]), 'k-')
            #hold(True)
            # plot([dat.std[-1, iabscissa], ax[1]], [dat.std[-1,5:], yy[idx2]], 'k-') # line from last data point
            for i in rglen((idx)):
                # text(ax[1], yy[i], ' '+str(idx[i]))
                text(dat.std[-1, iabscissa], dat.std[-1, 5 + i], ' ' + str(i))
        semilogy(dat.std[:, iabscissa], dat.std[:, 5:], '-')
        grid(True)
        title(r'Standard Deviations $\times$ $\sigma^{-1}$ in All Coordinates')
        # pyplot.xticks(xticklocs)
        xlabel(xlab)
        pyplot.ion()
        pyplot.draw()  # update "screen"
        pyplot.show()  # show figure
        # matplotlib.interactive(interactive_status)


        pyplot.rcParams['font.size'] = original_fontsize
        return self

    # ____________________________________________________________
    # ____________________________________________________________
    #
    @staticmethod
    def plotdivers(dat, iabscissa, foffset):
        """helper function for `plot()` that plots all what is
        in the upper left subplot like fitness, sigma, etc.

        Arguments
        ---------
            `iabscissa` in ``(0,1)``
                0==versus fevals, 1==versus iteration
            `foffset`
                offset to fitness for log-plot

         :See: `plot()`

        """
        from matplotlib.pyplot import semilogy, grid, \
            axis, title, text
        fontsize = pyplot.rcParams['font.size']

        # interactive_status = matplotlib.is_interactive()
        pyplot.ioff()  # prevents immediate drawing
        #hold(False)

        dfit = dat.f[:, 5] - min(dat.f[:, 5])
        dfit[dfit < 1e-98] = np.NaN

        if dat.f.shape[1] > 7:
            # semilogy(dat.f[:, iabscissa], abs(dat.f[:,[6, 7, 10, 12]])+foffset,'-k')
            semilogy(dat.f[:, iabscissa], abs(dat.f[:, [6, 7]]) + foffset, '-k')
            #hold(True)

        # (larger indices): additional fitness data, for example constraints values
        if dat.f.shape[1] > 8:
            # dd = abs(dat.f[:,7:]) + 10*foffset
            # dd = np.where(dat.f[:,7:]==0, np.NaN, dd) # cannot be
            semilogy(dat.f[:, iabscissa], np.abs(dat.f[:, 8:]) + 10 * foffset, 'm')
            #hold(True)

        idx = np.where(dat.f[:, 5] > 1e-98)[0]  # positive values
        semilogy(dat.f[idx, iabscissa], dat.f[idx, 5] + foffset, '.b')
        #hold(True)
        grid(True)

        idx = np.where(dat.f[:, 5] < -1e-98)[0]  # negative values
        semilogy(dat.f[idx, iabscissa], abs(dat.f[idx, 5]) + foffset, '.r')

        semilogy(dat.f[:, iabscissa], abs(dat.f[:, 5]) + foffset, '-b')
        idx = np.isfinite(dfit)
        semilogy(dat.f[:, iabscissa][idx], dfit[idx], '-c')
        text(dat.f[idx, iabscissa][-1], dfit[idx][-1], r'$f - \min(f)$',
             fontsize=fontsize)

        # overall minimum
        i = np.argmin(dat.f[:, 5])
        semilogy(dat.f[i, iabscissa] * np.ones(2), dat.f[i, 5] * np.ones(2), 'rd')
        # semilogy(dat.f[-1, iabscissa]*np.ones(2), dat.f[-1,4]*np.ones(2), 'rd')

        # AR and sigma
        semilogy(dat.f[:, iabscissa], dat.f[:, 3], '-r')  # AR
        semilogy(dat.f[:, iabscissa], dat.f[:, 2], '-g')  # sigma
        text(dat.f[-2, iabscissa], dat.f[-2, 3], r'axis ratio',
             fontsize=fontsize)
        text(dat.f[-2, iabscissa], dat.f[-2, 2]/1.5, r'$\sigma$',
             fontsize=fontsize+3)
        semilogy(dat.std[:-1, iabscissa], np.vstack([list(map(max, dat.std[:-1, 5:])), list(map(min, dat.std[:-1, 5:]))]).T,
                     '-m', linewidth=2)
        text(dat.std[-2, iabscissa], max(dat.std[-2, 5:]), 'max std',
             fontsize=fontsize)
        text(dat.std[-2, iabscissa], min(dat.std[-2, 5:]), 'min std',
             fontsize=fontsize)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        text(ax[0] + 0.01, ax[2],  # 10**(log10(ax[2])+0.05*(log10(ax[3])-log10(ax[2]))),
             '.f_recent=' + repr(dat.f[-1, 5]))

        # title('abs(f) (blue), f-min(f) (cyan), Sigma (green), Axis Ratio (red)')
        title(r'blue:$\mathrm{abs}(f)$, cyan:$f - \min(f)$, green:$\sigma$, red:axis ratio',
              fontsize=fontsize - 0.0)
        title(r'$\mathrm{abs}(f)$, $f - \min(f)$, ' +
              r'$\sigma$, axis ratio')
        # pyplot.xticks(xticklocs)
        pyplot.ion()
        if __name__ != 'cma':  # should be handled by the caller
            pyplot.draw()  # update "screen"
            pyplot.show()  # show figure
        # matplotlib.interactive(interactive_status)


    def downsampling(self, factor=10, first=3, switch=True, verbose=True):
        """
        rude downsampling of a `CMADataLogger` data file by `factor`, keeping
        also the first `first` entries. This function is a stump and subject
        to future changes. Return self.

        Arguments
        ---------
           - `factor` -- downsampling factor
           - `first` -- keep first `first` entries
           - `switch` -- switch the new logger to the downsampled logger original_name+'down'

        Details
        -------
        ``self.name_prefix+'down'`` files are written

        Example
        -------
        ::

            import cma
            cma.downsampling()  # takes outcmaes* files
            cma.plot('outcmaesdown')

        """
        newprefix = self.name_prefix + 'down'
        for name in self.file_names:
            f = open(newprefix + name + '.dat', 'w')
            iline = 0
            cwritten = 0
            for line in open(self.name_prefix + name + '.dat'):
                if iline < first or iline % factor == 0:
                    f.write(line)
                    cwritten += 1
                iline += 1
            f.close()
            if verbose and iline > first:
                print('%d' % (cwritten) + ' lines written in ' + newprefix + name + '.dat')
        if switch:
            self.name_prefix += 'down'
        return self

    # ____________________________________________________________
    # ____________________________________________________________
    #
    def disp(self, idx=100):  # r_[0:5,1e2:1e9:1e2,-10:0]):
        """displays selected data from (files written by) the class `CMADataLogger`.

        Arguments
        ---------
           `idx`
               indices corresponding to rows in the data file;
               if idx is a scalar (int), the first two, then every idx-th,
               and the last three rows are displayed. Too large index values are removed.

        Example
        -------
        >>> import cma, numpy as np
        >>> res = cma.fmin(cma.fcts.elli, 7 * [0.1], 1, {'verb_disp':1e9})  # generate data
        >>> assert res[1] < 1e-9
        >>> assert res[2] < 4400
        >>> l = cma.CMADataLogger()  # == res[-1], logger with default name, "points to" above data
        >>> l.disp([0,-1])  # first and last
        >>> l.disp(20)  # some first/last and every 20-th line
        >>> l.disp(np.r_[0:999999:100, -1]) # every 100-th and last
        >>> l.disp(np.r_[0, -10:0]) # first and ten last
        >>> cma.disp(l.name_prefix, np.r_[0::100, -10:])  # the same as l.disp(...)

        Details
        -------
        The data line with the best f-value is displayed as last line.

        :See: `disp()`

        """

        filenameprefix = self.name_prefix

        def printdatarow(dat, iteration):
            """print data of iteration i"""
            i = np.where(dat.f[:, 0] == iteration)[0][0]
            j = np.where(dat.std[:, 0] == iteration)[0][0]
            print('%5d' % (int(dat.f[i, 0])) + ' %6d' % (int(dat.f[i, 1])) + ' %.14e' % (dat.f[i, 5]) +
                  ' %5.1e' % (dat.f[i, 3]) +
                  ' %6.2e' % (max(dat.std[j, 5:])) + ' %6.2e' % min(dat.std[j, 5:]))

        dat = CMADataLogger(filenameprefix).load()
        ndata = dat.f.shape[0]

        # map index to iteration number, is difficult if not all iteration numbers exist
        # idx = idx[np.where(map(lambda x: x in dat.f[:,0], idx))[0]] # TODO: takes pretty long
        # otherwise:
        if idx is None:
            idx = 100
        if np.isscalar(idx):
            # idx = np.arange(0, ndata, idx)
            if idx:
                idx = np.r_[0, 1, idx:ndata - 3:idx, -3:0]
            else:
                idx = np.r_[0, 1, -3:0]

        idx = array(idx)
        idx = idx[idx < ndata]
        idx = idx[-idx <= ndata]
        iters = dat.f[idx, 0]
        idxbest = np.argmin(dat.f[:, 5])
        iterbest = dat.f[idxbest, 0]

        if len(iters) == 1:
            printdatarow(dat, iters[0])
        else:
            self.disp_header()
            for i in iters:
                printdatarow(dat, i)
            self.disp_header()
            printdatarow(dat, iterbest)
        sys.stdout.flush()
    def disp_header(self):
        heading = 'Iterat Nfevals  function value    axis ratio maxstd  minstd'
        print(heading)

# end class CMADataLogger

# ____________________________________________________________
# ____________________________________________________________
#
# _____________________________________________________________________
# _____________________________________________________________________
#
class DEAPCMADataLogger(BaseDataLogger):
    """data logger for class `deap.cma.Strategy` of the DEAP library.
    Documentation is not entirely adapted to the deap case.

    The logger is identified by its name prefix and writes or reads according
    data files.

    Examples
    ========
    ::

        import cma_logger
        es = deap.cma.Strategy(...)
        data = cma.DEAPCMADataLogger().register(es)
        while not es.stop():
            ...
            data.add(fitness_values)  # add can also take `es` as additional argument

        data.plot() # or a short cut can be used:
        cma.plot()  # plot data from logger with default name


        data2 = cma.DEAPCMADataLogger(another_filename_prefix).load()
        data2.plot()
        data2.disp()

    ::

        import cma
        from pyplot import *
        res = cma.fmin(cma.Fcts.sphere, rand(10), 1e-0)
        dat = res[-1]  # the CMADataLogger
        dat.load()  # by "default" data are on disk
        semilogy(dat.f[:,0], dat.f[:,5])  # plot f versus iteration, see file header
        show()

    Details
    =======
    After loading data, the logger has the attributes `xmean`, `xrecent`, `std`, `f`, and `D`,
    corresponding to xmean, xrecentbest, stddev, fit, and axlen filename trails.

    :See: `disp()`, `plot()`

    """
    default_prefix = 'outcmaes'
    names = ('axlen', 'fit', 'stddev', 'xmean')  # ,'xrecentbest')
    key_names_with_annotation = ('std', 'xmean')

    def __init__(self, name_prefix=default_prefix, modulo=1, append=False):
        """initialize logging of data from a `CMAEvolutionStrategy` instance,
        default modulo expands to 1 == log with each call

        """
        # super(CMAData, self).__init__({'iter':[], 'stds':[], 'D':[], 'sig':[], 'fit':[], 'xm':[]})
        # class properties:
        self.counter = 0  # number of calls of add
        self.best_fitness = np.inf
        self.modulo = modulo  # allows calling with None
        self.append = append
        self.name_prefix = name_prefix if name_prefix else CMADataLogger.default_prefix
        if isinstance(self.name_prefix, CMAEvolutionStrategy):
            self.name_prefix = self.name_prefix.opts.eval('verb_filenameprefix')
        self.registered = False

    def register(self, es, append=None, modulo=None):
        """register a `CMAEvolutionStrategy` instance for logging,
        ``append=True`` appends to previous data logged under the same name,
        by default previous data are overwritten.

        """
        self.es = es
        if append is not None:
            self.append = append
        if modulo is not None:
            self.modulo = modulo
        if not self.append and self.modulo != 0:
            self.initialize()  # write file headers
        self.registered = True
        return self

    def initialize(self, modulo=None):
        """reset logger, overwrite original files, `modulo`: log only every modulo call"""
        if modulo is not None:
            self.modulo = modulo
        try:
            es = self.es  # must have been registered
        except AttributeError:
            pass  # TODO: revise usage of es... that this can pass
            raise _Error('call register() before initialize()')

        # write headers for output
        fn = self.name_prefix + 'fit.dat'
        if 1 < 3:
            strseedtime = 'seed=unkown, %s' % (time.asctime())

        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, sigma, axis ratio, ' +
                        'bestever, best, median, worst objective function value, ' +
                        'further objective values of best", ' +
                        strseedtime +
                        # strftime("%Y/%m/%d %H:%M:%S", localtime()) + # just asctime() would do
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'axlen.dat'
        try:
            f = open(fn, 'w')
            f.write('%  columns="iteration, evaluation, sigma, max axis length, ' +
                    ' min axis length, all principle axes lengths ' +
                    ' (sorted square roots of eigenvalues of C)", ' +
                    strseedtime +
                    '\n')
            f.close()
        except (IOError, OSError):
            print('could not open file ' + fn)
        finally:
            f.close()
        fn = self.name_prefix + 'stddev.dat'
        try:
            f = open(fn, 'w')
            f.write('% # columns=["iteration, evaluation, sigma, void, void, ' +
                    ' stds==sigma*sqrt(diag(C))", ' +
                    strseedtime +
                    '\n')
            f.close()
        except (IOError, OSError):
            print('could not open file ' + fn)
        finally:
            f.close()

        fn = self.name_prefix + 'xmean.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, void, void, void, xmean", ' +
                        strseedtime)
                f.write('\n')
                f.close()
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        return self
    # end def __init__

    def load(self, filenameprefix=None):
        """loads data from files written and return a data dictionary, *not*
        a prerequisite for using `plot()` or `disp()`.

        Argument `filenameprefix` is the filename prefix of data to be loaded (five files),
        by default ``'outcmaes'``.

        Return data dictionary with keys `xrecent`, `xmean`, `f`, `D`, `std`

        """
        if not filenameprefix:
            filenameprefix = self.name_prefix
        dat = self  # historical
        # dat.xrecent = _fileToMatrix(filenameprefix + 'xrecentbest.dat')
        dat.xmean = _fileToMatrix(filenameprefix + 'xmean.dat')
        dat.std = _fileToMatrix(filenameprefix + 'stddev' + '.dat')
        # a hack to later write something into the last entry
        for key in ['xmean', 'std']:  # 'xrecent',
            dat.__dict__[key].append(dat.__dict__[key][-1])  # copy last row to later fill in annotation position for display
            dat.__dict__[key] = array(dat.__dict__[key], copy=False)
        dat.f = array(_fileToMatrix(filenameprefix + 'fit.dat'))
        dat.D = array(_fileToMatrix(filenameprefix + 'axlen' + '.dat'))
        return dat


    def add(self, fitness_values, es=None, more_data=[], modulo=None):  # TODO: find a different way to communicate current x and f
        """append some logging data from `CMAEvolutionStrategy` class instance `es`,
        if ``number_of_times_called % modulo`` equals to zero, never if ``modulo==0``.

        The sequence ``more_data`` must always have the same length.

        """
        self.counter += 1
        fitness_values = np.sort(fitness_values)
        if fitness_values[0] < self.best_fitness:
            self.best_fitness = fitness_values[0]
        mod = modulo if modulo is not None else self.modulo
        if mod == 0 or (self.counter > 3 and self.counter % mod):
            return
        if es is None:
            try:
                es = self.es  # must have been registered
            except AttributeError :
                raise _Error('call register() before add() or add(es)')
        elif not self.registered:
            self.register(es)

        try:
            # fit
            if es.update_count > 0:
                # fit = es.fit.fit[0]  # TODO: where do we get the fitness from?
                fn = self.name_prefix + 'fit.dat'
                with open(fn, 'a') as f:
                    f.write(str(es.update_count) + ' '
                            + str(es.update_count * es.lambda_) + ' '
                            + str(es.sigma) + ' '
                            + str(es.diagD[-1] / es.diagD[0]) + ' '
                            + str(self.best_fitness) + ' '
                            + '%.16e' % fitness_values[0] + ' '
                            + str(fitness_values[es.lambda_ // 2]) + ' '
                            + str(fitness_values[-1]) + ' '
                            # + str(es.sp.popsize) + ' '
                            # + str(10**es.noiseS) + ' '
                            # + str(es.sp.cmean) + ' '
                            # + ' '.join(str(i) for i in es.more_to_write)
                            + ' '.join(str(i) for i in more_data)
                            + '\n')
                    # es.more_to_write = []
            # axlen
            fn = self.name_prefix + 'axlen.dat'
            with open(fn, 'a') as f:  # does not rely on reference counting
                f.write(str(es.update_count) + ' '
                        + str(es.update_count * es.lambda_) + ' '
                        + str(es.sigma) + ' '
                        + str(es.diagD[-1]) + ' '
                        + str(es.diagD[0]) + ' '
                        + ' '.join(map(str, es.diagD))
                        + '\n')
            # stddev
            fn = self.name_prefix + 'stddev.dat'
            with open(fn, 'a') as f:
                f.write(str(es.update_count) + ' '
                        + str(es.update_count * es.lambda_) + ' '
                        + str(es.sigma) + ' '
                        + '0 0 '
                        + ' '.join(map(str, es.sigma * np.sqrt([es.C[i][i] for i in range(es.dim)])))
                        + '\n')
            # xmean
            fn = self.name_prefix + 'xmean.dat'
            with open(fn, 'a') as f:
                if es.update_count < 1:
                    f.write('0 0 0 0 0 '
                            + ' '.join(map(str,
                                              # TODO should be optional the phenotyp?
                                              # es.x0
                                              es.mean))
                            + '\n')
                else:
                    f.write(str(es.update_count) + ' '
                            + str(es.update_count * es.lambda_) + ' '
                            # + str(es.sigma) + ' '
                            + '0 0 0 '
                            # + str(es.fmean_noise_free) + ' '
                            # + str(es.fmean) + ' '  # TODO: this does not make sense
                            # TODO should be optional the phenotyp?
                            + ' '.join(map(str, es.centroid))
                            + '\n')
            # xrecent
        except (IOError, OSError):
            if es.countiter == 1:
                print('could not open/write file')

    def closefig(self):
        pyplot.close(self.fighandle)

    def save(self, nameprefix, switch=False):
        """saves logger data to a different set of files, for
        ``switch=True`` also the loggers name prefix is switched to
        the new value

        """
        if not nameprefix or not isinstance(nameprefix, str):
            raise _Error('filename prefix must be a nonempty string')

        if nameprefix == self.default_prefix:
            raise _Error('cannot save to default name "' + nameprefix + '...", chose another name')

        if nameprefix == self.name_prefix:
            return

        for name in CMADataLogger.names:
            open(nameprefix + name + '.dat', 'w').write(open(self.name_prefix + name + '.dat').read())

        if switch:
            self.name_prefix = nameprefix

    def plot(self, fig=None, iabscissa=1, iteridx=None,
             plot_mean=False, # TODO: plot_mean default should be False
             foffset=1e-19, x_opt=None, fontsize=9):
        """
        plot data from a `DEAPCMADataLogger` (using the files written by the logger).

        Arguments
        ---------
            `fig`
                figure number, by default 325
            `iabscissa`
                ``0==plot`` versus iteration count,
                ``1==plot`` versus function evaluation number
            `iteridx`
                iteration indices to plot

        Return `CMADataLogger` itself.

        Examples
        --------
        ::

            import cma
            logger = cma.CMADataLogger()  # with default name
            # try to plot the "default logging" data (e.g. from previous fmin calls)
            logger.plot() # to continue you might need to close the pop-up window
                          # once and call plot() again.
                          # This behavior seems to disappear in subsequent
                          # calls of plot(). Also using ipython with -pylab
                          # option might help.
            cma.savefig('fig325.png')  # save current figure
            logger.closefig()

        Dependencies: matlabplotlib/pyplot.

        """

        dat = self.load(self.name_prefix)

        try:
            # pyplot: prodedural interface for matplotlib
            from  matplotlib.pyplot import figure, ioff, ion, subplot, semilogy, plot, grid, \
                 axis, title, text, xlabel, isinteractive, draw, gcf

        except ImportError:
            raise ImportError('could not find matplotlib.pyplot module, function plot() is not available')

        if fontsize and pyplot.rcParams['font.size'] != fontsize:
            print('global variable pyplot.rcParams[\'font.size\'] set (from ' +
                  str(pyplot.rcParams['font.size']) + ') to ' + str(fontsize))
            pyplot.rcParams['font.size'] = fontsize  # subtracted in the end, but return can happen inbetween

        if fig:
            figure(fig)
        else:
            figure(325)
            # show()  # should not be necessary
        self.fighandle = gcf()  # fighandle.number

        if iabscissa not in (0, 1):
            iabscissa = 1
        interactive_status = isinteractive()
        ioff()  # prevents immediate drawing

        dat.x = dat.xmean  # this is the genotyp
        if not plot_mean:
            if len(dat.x) < 2:
                print('not enough data to plot recent x, using mean instead')
            else:
                dat.x = dat.xrecent
        if iteridx is not None:
            dat.f = dat.f[np.where([x in iteridx for x in dat.f[:, 0]])[0], :]
            dat.D = dat.D[np.where([x in iteridx for x in dat.D[:, 0]])[0], :]
            iteridx.append(dat.x[-1, 1])  # last entry is artificial
            dat.x = dat.x[np.where([x in iteridx for x in dat.x[:, 0]])[0], :]
            dat.std = dat.std[np.where([x in iteridx for x in dat.std[:, 0]])[0], :]

        if iabscissa == 0:
            xlab = 'iterations'
        elif iabscissa == 1:
            xlab = 'function evaluations'

        # use fake last entry in x and std for line extension-annotation
        if dat.x.shape[1] < 100:
            minxend = int(1.06 * dat.x[-2, iabscissa])
            # write y-values for individual annotation into dat.x
            dat.x[-1, iabscissa] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat.x[-2, 5:])
            idx2 = np.argsort(idx)
            if x_opt is None:
                dat.x[-1, 5 + idx] = np.linspace(np.min(dat.x[:, 5:]),
                            np.max(dat.x[:, 5:]), dat.x.shape[1] - 5)
            else:
                dat.x[-1, 5 + idx] = np.logspace(np.log10(np.min(abs(dat.x[:, 5:]))),
                            np.log10(np.max(abs(dat.x[:, 5:]))), dat.x.shape[1] - 5)
        else:
            minxend = 0

        if len(dat.f) == 0:
            print('nothing to plot')
            return

        # not in use anymore, see formatter above
        # xticklocs = np.arange(5) * np.round(minxend/4., -int(np.log10(minxend/4.)))

        # dfit(dfit<1e-98) = NaN;

        ioff()  # turns update off

        # TODO: if abscissa==0 plot in chunks, ie loop over subsets where dat.f[:,0]==countiter is monotonous

        subplot(2, 2, 1)
        self.plotdivers(dat, iabscissa, foffset)

        # TODO: modularize also the remaining subplots
        subplot(2, 2, 2)
        #hold(False)
        if x_opt is not None:  # TODO: differentate neg and pos?
            semilogy(dat.x[:, iabscissa], abs(dat.x[:, 5:]) - x_opt, '-')
        else:
            plot(dat.x[:, iabscissa], dat.x[:, 5:], '-')
        #hold(True)
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        ax[1] -= 1e-6
        if dat.x.shape[1] < 100:
            yy = np.linspace(ax[2] + 1e-6, ax[3] - 1e-6, dat.x.shape[1] - 5)
            # yyl = np.sort(dat.x[-1,5:])
            idx = np.argsort(dat.x[-1, 5:])
            idx2 = np.argsort(idx)
            if x_opt is not None:
                semilogy([dat.x[-1, iabscissa], ax[1]], [abs(dat.x[-1, 5:]), yy[idx2]], 'k-')  # line from last data point
                semilogy(np.dot(dat.x[-2, iabscissa], [1, 1]), array([ax[2] + 1e-6, ax[3] - 1e-6]), 'k-')
            else:
                # plot([dat.x[-1, iabscissa], ax[1]], [dat.x[-1,5:], yy[idx2]], 'k-') # line from last data point
                plot(np.dot(dat.x[-2, iabscissa], [1, 1]), array([ax[2] + 1e-6, ax[3] - 1e-6]), 'k-')
            # plot(array([dat.x[-1, iabscissa], ax[1]]),
            #      reshape(array([dat.x[-1,5:], yy[idx2]]).flatten(), (2,4)), '-k')
            for i in range(len(idx)):
                # TODOqqq: annotate phenotypic value!?
                # text(ax[1], yy[i], 'x(' + str(idx[i]) + ')=' + str(dat.x[-2,5+idx[i]]))
                text(dat.x[-1, iabscissa], dat.x[-1, 5 + i], 'x(' + str(i) + ')=' + str(dat.x[-2, 5 + i]))

        i = 2  # find smallest i where iteration count differs (in case the same row appears twice)
        while i < len(dat.f) and dat.f[-i][0] == dat.f[-1][0]:
            i += 1
        title('Object Variables (' + ('mean' if plot_mean else 'curr best') +
                ', ' + str(dat.x.shape[1] - 5) + '-D, popsize~' +
                (str(int((dat.f[-1][1] - dat.f[-i][1]) / (dat.f[-1][0] - dat.f[-i][0])))
                    if len(dat.f.T[0]) > 1 and dat.f[-1][0] > dat.f[-i][0] else 'NA')
                + ')')
        # pyplot.xticks(xticklocs)

        # Scaling
        subplot(2, 2, 3)
        #hold(False)
        semilogy(dat.D[:, iabscissa], dat.D[:, 5:], '-b')
        #hold(True)
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        title('Scaling (All Main Axes)')
        # pyplot.xticks(xticklocs)
        xlabel(xlab)

        # standard deviations
        subplot(2, 2, 4)
        #hold(False)
        # remove sigma from stds (graphs become much better readible)
        dat.std[:, 5:] = np.transpose(dat.std[:, 5:].T / dat.std[:, 2].T)
        # ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        # axis(ax)
        if 1 < 2 and dat.std.shape[1] < 100:
            # use fake last entry in x and std for line extension-annotation
            minxend = int(1.06 * dat.x[-2, iabscissa])
            dat.std[-1, iabscissa] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat.std[-2, 5:])
            idx2 = np.argsort(idx)
            dat.std[-1, 5 + idx] = np.logspace(np.log10(np.min(dat.std[:, 5:])),
                            np.log10(np.max(dat.std[:, 5:])), dat.std.shape[1] - 5)

            dat.std[-1, iabscissa] = minxend  # TODO: should be ax[1]
            yy = np.logspace(np.log10(ax[2]), np.log10(ax[3]), dat.std.shape[1] - 5)
            # yyl = np.sort(dat.std[-1,5:])
            idx = np.argsort(dat.std[-1, 5:])
            idx2 = np.argsort(idx)
            # plot(np.dot(dat.std[-2, iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-') # vertical separator
            # vertical separator
            plot(np.dot(dat.std[-2, iabscissa], [1, 1]), array([np.min(dat.std[-2, 5:]), np.max(dat.std[-2, 5:])]), 'k-')
            #hold(True)
            # plot([dat.std[-1, iabscissa], ax[1]], [dat.std[-1,5:], yy[idx2]], 'k-') # line from last data point
            for i in rglen((idx)):
                # text(ax[1], yy[i], ' '+str(idx[i]))
                text(dat.std[-1, iabscissa], dat.std[-1, 5 + i], ' ' + str(i))
        semilogy(dat.std[:, iabscissa], dat.std[:, 5:], '-')
        grid(True)
        title('Standard Deviations / $\sigma$ in All Coordinates')
        # pyplot.xticks(xticklocs)
        xlabel(xlab)
        ion()
        draw()  # does not suffice
        show()

        return self


    # ____________________________________________________________
    # ____________________________________________________________
    #
    @staticmethod
    def plotdivers(dat, iabscissa, foffset):
        """helper function for `plot()` that plots all what is
        in the upper left subplot like fitness, sigma, etc.

        Arguments
        ---------
            `iabscissa` in ``(0,1)``
                0==versus fevals, 1==versus iteration
            `foffset`
                offset to fitness for log-plot

         :See: `plot()`

        """
        from  matplotlib.pyplot import semilogy, grid, \
                 axis, title, text
        fontsize = pyplot.rcParams['font.size']

        #hold(False)

        dfit = dat.f[:, 5] - min(dat.f[:, 5])
        dfit[dfit < 1e-98] = np.NaN

        if dat.f.shape[1] > 7:
            # semilogy(dat.f[:, iabscissa], abs(dat.f[:,[6, 7, 10, 12]])+foffset,'-k')
            semilogy(dat.f[:, iabscissa], abs(dat.f[:, [6, 7]]) + foffset, '-k')
            #hold(True)

        # (larger indices): additional fitness data, for example constraints values
        if dat.f.shape[1] > 8:
            # dd = abs(dat.f[:,7:]) + 10*foffset
            # dd = np.where(dat.f[:,7:]==0, np.NaN, dd) # cannot be
            semilogy(dat.f[:, iabscissa], np.abs(dat.f[:, 8:]) + 10 * foffset, 'm')
            #hold(True)

        idx = np.where(dat.f[:, 5] > 1e-98)[0]  # positive values
        semilogy(dat.f[idx, iabscissa], dat.f[idx, 5] + foffset, '.b')
        #hold(True)
        grid(True)

        idx = np.where(dat.f[:, 5] < -1e-98)  # negative values
        semilogy(dat.f[idx, iabscissa], abs(dat.f[idx, 5]) + foffset, '.r')

        semilogy(dat.f[:, iabscissa], abs(dat.f[:, 5]) + foffset, '-b')
        semilogy(dat.f[:, iabscissa], dfit, '-c')

        # overall minimum
        i = np.argmin(dat.f[:, 5])
        semilogy(dat.f[i, iabscissa] * np.ones(2), dat.f[i, 5] * np.ones(2), 'rd')
        # semilogy(dat.f[-1, iabscissa]*np.ones(2), dat.f[-1,4]*np.ones(2), 'rd')

        # AR and sigma
        semilogy(dat.f[:, iabscissa], dat.f[:, 3], '-r')  # AR
        semilogy(dat.f[:, iabscissa], dat.f[:, 2], '-g')  # sigma
        text(dat.f[-2, iabscissa], dat.f[-2, 2], r'$\sigma$',
             fontsize=fontsize)
        semilogy(dat.std[:-1, iabscissa], np.vstack([list(map(max, dat.std[:-1, 5:])), list(map(min, dat.std[:-1, 5:]))]).T,
                     '-m', linewidth=2)
        text(dat.std[-2, iabscissa], max(dat.std[-2, 5:]), 'max std',
             fontsize=fontsize)
        text(dat.std[-2, iabscissa], min(dat.std[-2, 5:]), 'min std',
             fontsize=fontsize)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        text(ax[0] + 0.01, ax[2],  # 10**(log10(ax[2])+0.05*(log10(ax[3])-log10(ax[2]))),
             '.f_recent=' + repr(dat.f[-1, 5]))

        # title('abs(f) (blue), f-min(f) (cyan), Sigma (green), Axis Ratio (red)')
        title('blue:abs(f), cyan:f-min(f), green:sigma, red:axis ratio', fontsize=fontsize - 1)
        # pyplot.xticks(xticklocs)



    def downsampling(self, factor=10, first=3, switch=True):
        """
        rude downsampling of a `CMADataLogger` data file by `factor`, keeping
        also the first `first` entries. This function is a stump and subject
        to future changes.

        Arguments
        ---------
           - `factor` -- downsampling factor
           - `first` -- keep first `first` entries
           - `switch` -- switch the new logger name to oldname+'down'

        Details
        -------
        ``self.name_prefix+'down'`` files are written

        Example
        -------
        ::

            import cma
            cma.downsampling()  # takes outcmaes* files
            cma.plot('outcmaesdown')

        """
        newprefix = self.name_prefix + 'down'
        for name in CMADataLogger.names:
            f = open(newprefix + name + '.dat', 'w')
            iline = 0
            cwritten = 0
            for line in open(self.name_prefix + name + '.dat'):
                if iline < first or iline % factor == 0:
                    f.write(line)
                    cwritten += 1
                iline += 1
            f.close()
            print('%d' % (cwritten) + ' lines written in ' + newprefix + name + '.dat')
        if switch:
            self.name_prefix += 'down'
        return self

    # ____________________________________________________________
    # ____________________________________________________________
    #
    def disp_header(self):
        heading = 'Iterat Nfevals  function value    axis ratio maxstd   minstd'
        print(heading)

    def disp(self, idx=100):  # r_[0:5,1e2:1e9:1e2,-10:0]):
        """displays selected data from (files written by) the class `CMADataLogger`.

        Arguments
        ---------
           `idx`
               indices corresponding to rows in the data file;
               if idx is a scalar (int), the first two, then every idx-th,
               and the last three rows are displayed. Too large index values are removed.
               If ``len(idx) == 1``, only a single row is displayed, e.g. the last
               entry when ``idx == [-1]``.

        Example
        -------
        >>> import cma, numpy as np
        >>> res = cma.fmin(cma.fcts.elli, 7 * [0.1], 1, {'verb_disp':1e9})  # generate data
        >>> assert res[1] < 1e-9
        >>> assert res[2] < 4400
        >>> l = cma.CMADataLogger()  # == res[-1], logger with default name, "points to" above data
        >>> l.disp([0,-1])  # first and last
        >>> l.disp(20)  # some first/last and every 20-th line
        >>> l.disp(np.r_[0:999999:100, -1]) # every 100-th and last
        >>> l.disp(np.r_[0, -10:0]) # first and ten last
        >>> cma.disp(l.name_prefix, np.r_[0::100, -10:])  # the same as l.disp(...)

        Details
        -------
        The data line with the best f-value is displayed as last line.

        :See: `disp()`

        """

        filenameprefix = self.name_prefix

        def printdatarow(dat, iteration):
            """print data of iteration i"""
            i = np.where(dat.f[:, 0] == iteration)[0][0]
            j = np.where(dat.std[:, 0] == iteration)[0][0]
            print('%5d' % (int(dat.f[i, 0])) + ' %6d' % (int(dat.f[i, 1])) + ' %.14e' % (dat.f[i, 5]) +
                  ' %5.1e' % (dat.f[i, 3]) +
                  ' %6.2e' % (max(dat.std[j, 5:])) + ' %6.2e' % min(dat.std[j, 5:]))

        dat = CMADataLogger(filenameprefix).load()
        ndata = dat.f.shape[0]

        # map index to iteration number, is difficult if not all iteration numbers exist
        # idx = idx[np.where(map(lambda x: x in dat.f[:,0], idx))[0]] # TODO: takes pretty long
        # otherwise:
        if idx is None:
            idx = 100
        if np.isscalar(idx):
            # idx = np.arange(0, ndata, idx)
            if idx:
                idx = np.r_[0, 1, idx:ndata - 3:idx, -3:0]
            else:
                idx = np.r_[0, 1, -3:0]

        idx = array(idx)
        idx = idx[idx <= ndata]  # TODO: shouldn't this be "<"?
        idx = idx[-idx <= ndata]
        iters = dat.f[idx, 0]
        idxbest = np.argmin(dat.f[:, 5])
        iterbest = dat.f[idxbest, 0]
        if len(iters) == 1:
            printdatarow(dat, iters[0])
        else:
            self.disp_header()
            for i in iters:
                printdatarow(dat, i)
            self.disp_header()
            printdatarow(dat, iterbest)
        sys.stdout.flush()

last_figure_number = 324
def plot(name=None, fig=None, abscissa=1, iteridx=None,
         plot_mean=False,
         foffset=1e-19, x_opt=None, fontsize=9):
    """
    plot data from files written by a `CMADataLogger`,
    the call ``cma.plot(name, **argsdict)`` is a shortcut for
    ``cma.CMADataLogger(name).plot(**argsdict)``

    Arguments
    ---------
        `name`
            name of the logger, filename prefix, None evaluates to
            the default 'outcmaes'
        `fig`
            filename or figure number, or both as a tuple (any order)
        `abscissa`
            0==plot versus iteration count,
            1==plot versus function evaluation number
        `iteridx`
            iteration indices to plot

    Return `None`

    Examples
    --------
    ::

       cma.plot();  # the optimization might be still
                    # running in a different shell
       cma.savefig('fig325.png')
       cma.closefig()

       cdl = cma.CMADataLogger().downsampling().plot()
       # in case the file sizes are large

    Details
    -------
    Data from codes in other languages (C, Java, Matlab, Scilab) have the same
    format and can be plotted just the same.

    :See: `CMADataLogger`, `CMADataLogger.plot()`

    """
    global last_figure_number
    if not fig:
        last_figure_number += 1
        fig = last_figure_number
    last_figure_number = fig
    CMADataLogger(name).plot(fig, abscissa, iteridx, plot_mean, foffset,
                             x_opt, fontsize)

def disp(name=None, idx=None):
    """displays selected data from (files written by) the class `CMADataLogger`.

    The call ``cma.disp(name, idx)`` is a shortcut for ``cma.CMADataLogger(name).disp(idx)``.

    Arguments
    ---------
        `name`
            name of the logger, filename prefix, `None` evaluates to
            the default ``'outcmaes'``
        `idx`
            indices corresponding to rows in the data file; by
            default the first five, then every 100-th, and the last
            10 rows. Too large index values are removed.

    Examples
    --------
    ::

       import cma, numpy
       # assume some data are available from previous runs
       cma.disp(None,numpy.r_[0,-1])  # first and last
       cma.disp(None,numpy.r_[0:1e9:100,-1]) # every 100-th and last
       cma.disp(idx=numpy.r_[0,-10:0]) # first and ten last
       cma.disp(idx=numpy.r_[0:1e9:1e3,-10:0])

    :See: `CMADataLogger.disp()`

    """
    return CMADataLogger(name if name else 'outcmaes'
                         ).disp(idx)

# ____________________________________________________________
def _fileToMatrix(file_name):
    """rudimentary method to read in data from a file"""
    # TODO: np.loadtxt() might be an alternative
    #     try:
    if 1 < 3:
        lres = []
        for line in open(file_name, 'r').readlines():
            if len(line) > 0 and line[0] not in ('%', '#'):
                lres.append(list(map(float, line.split())))
        res = lres
    while res != [] and res[0] == []:  # remove further leading empty lines
        del res[0]
    return res
    #     except:
    print('could not read file ' + file_name)

# ____________________________________________________________
# ____________________________________________________________
class NoiseHandler(object):
    """Noise handling according to [Hansen et al 2009, A Method for
    Handling Uncertainty in Evolutionary Optimization...]

    The interface of this class is yet versatile and subject to changes.

    The attribute ``evaluations`` serves to control the noise via
    number of evaluations, for example in ``fmin`` or with
    `ask_and_eval()`. The parameter ``maxevals`` (second parameter)
    provides the upper bound, or lower and upper bound, or lower and
    upper bound and initial value, all 1 by default, compare also the
    second example.

    Examples
    --------
    Minimal example together with `fmin` on a non-noisy function:

    >>> import cma
    >>> cma.fmin(cma.felli, 7 * [1], 1, noise_handler=cma.NoiseHandler(7))  # dimension 7

    More verbose example in the optimization loop with a noisy function
    defined in ``func``:

    >>> import cma, numpy as np
    >>> func = lambda x: cma.fcts.sphere(x) * (1 + 4 * np.random.randn() / len(x))  # cma.Fcts.noisysphere
    >>> es = cma.CMAEvolutionStrategy(np.ones(10), 1)
    >>> nh = cma.NoiseHandler(es.N, maxevals=[1, 1, 30])
    >>> while not es.stop():
    ...     X, fit_vals = es.ask_and_eval(func, evaluations=nh.evaluations)
    ...     es.tell(X, fit_vals)  # prepare for next iteration
    ...     es.sigma *= nh(X, fit_vals, func, es.ask)  # see method __call__
    ...     es.countevals += nh.evaluations_just_done  # this is a hack, not important though
    ...     es.logger.add(more_data = [nh.evaluations, nh.noiseS])  # add a data point
    ...     es.disp()
    ...     # nh.maxevals = ...  it might be useful to start with smaller values and then increase
    >>> print(es.stop())
    >>> print(es.result()[-2])  # take mean value, the best solution is totally off
    >>> assert sum(es.result()[-2]**2) < 1e-9
    >>> print(X[np.argmin(fit_vals)])  # not bad, but probably worse than the mean
    >>> # es.logger.plot()


    The command ``logger.plot()`` will plot the logged data.

    The noise options of `fmin()` control a `NoiseHandler` instance similar to this
    example. The command ``cma.CMAOptions('noise')`` lists in effect the parameters of
    `__init__` apart from ``aggregate``.

    Details
    -------
    The parameters reevals, theta, c_s, and alpha_t are set differently
    than in the original publication, see method `__init__()`. For a
    very small population size, say popsize <= 5, the measurement
    technique based on rank changes is likely to fail.

    Missing Features
    ----------------
    In case no noise is found, ``self.lam_reeval`` should be adaptive
    and get at least as low as 1 (however the possible savings from this
    are rather limited). Another option might be to decide during the
    first call by a quantitative analysis of fitness values whether
    ``lam_reeval`` is set to zero. More generally, an automatic noise
    mode detection might also set the covariance matrix learning rates
    to smaller values.

    :See: `fmin()`, `ask_and_eval()`

    """
    # TODO: for const additive noise a better version might be with alphasigma also used for sigma-increment,
    # while all other variance changing sources are removed (because they are intrinsically biased). Then
    # using kappa to get convergence (with unit sphere samples): noiseS=0 leads to a certain kappa increasing rate?
    def __init__(self, N, maxevals=[1, 1, 1], aggregate=np.median, reevals=None, epsilon=1e-7, parallel=False):
        """parameters are

            `N`
                dimension, (only) necessary to adjust the internal "alpha"-parameters
            `maxevals`
                maximal value for ``self.evaluations``, where
                ``self.evaluations`` function calls are aggregated for
                noise treatment. With ``maxevals == 0`` the noise
                handler is (temporarily) "switched off". If `maxevals`
                is a list, min value and (for >2 elements) median are
                used to define minimal and initial value of
                ``self.evaluations``. Choosing ``maxevals > 1`` is only
                reasonable, if also the original ``fit`` values (that
                are passed to `__call__`) are computed by aggregation of
                ``self.evaluations`` values (otherwise the values are
                not comparable), as it is done within `fmin()`.
            `aggregate`
                function to aggregate single f-values to a 'fitness', e.g.
                ``np.median``.
            `reevals`
                number of solutions to be reevaluated for noise measurement,
                can be a float, by default set to ``2 + popsize/20``, where
                ``popsize = len(fit)`` in ``__call__``.
                zero switches noise handling off.
            `epsilon`
                multiplier for perturbation of the reevaluated solutions
            `parallel`
                a single f-call with all resampled solutions

            :See: `fmin()`, `CMAOptions`, `CMAEvolutionStrategy.ask_and_eval()`

        """
        self.lam_reeval = reevals  # 2 + popsize/20, see method indices(), originally 2 + popsize/10
        self.epsilon = epsilon
        self.parallel = parallel
        self.theta = 0.5  # originally 0.2
        self.cum = 0.3  # originally 1, 0.3 allows one disagreement of current point with resulting noiseS
        self.alphasigma = 1 + 2 / (N + 10) # unit sphere sampling: 1 + 1 / (N + 10)
        self.alphaevals = 1 + 2 / (N + 10)  # originally 1.5
        self.alphaevalsdown = self.alphaevals**-0.25  # originally 1/1.5
        # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        self.evaluations = 1  # to aggregate for a single f-evaluation
        self.minevals = 1
        self.maxevals = int(np.max(maxevals))
        if hasattr(maxevals, '__contains__'):  # i.e. can deal with ``in``
            if len(maxevals) > 1:
                self.minevals = min(maxevals)
                self.evaluations = self.minevals
            if len(maxevals) > 2:
                self.evaluations = np.median(maxevals)
        self.f_aggregate = aggregate
        self.evaluations_just_done = 0  # actually conducted evals, only for documentation
        self.noiseS = 0

    def __call__(self, X, fit, func, ask=None, args=()):
        """proceed with noise measurement, set anew attributes ``evaluations``
        (proposed number of evaluations to "treat" noise) and ``evaluations_just_done``
        and return a factor for increasing sigma.

        Parameters
        ----------
            `X`
                a list/sequence/vector of solutions
            `fit`
                the respective list of function values
            `func`
                the objective function, ``fit[i]`` corresponds to ``func(X[i], *args)``
            `ask`
                a method to generate a new, slightly disturbed solution. The argument
                is (only) mandatory if ``epsilon`` is not zero, see `__init__()`.
            `args`
                optional additional arguments to `func`

        Details
        -------
        Calls the methods ``reeval()``, ``update_measure()`` and ``treat()`` in this order.
        ``self.evaluations`` is adapted within the method `treat()`.

        """
        self.evaluations_just_done = 0
        if not self.maxevals or self.lam_reeval == 0:
            return 1.0
        res = self.reeval(X, fit, func, ask, args)
        if not len(res):
            return 1.0
        self.update_measure()
        return self.treat()

    def get_evaluations(self):
        """return ``self.evaluations``, the number of evalutions to get a single fitness measurement"""
        return self.evaluations

    def treat(self):
        """adapt self.evaluations depending on the current measurement value
        and return ``sigma_fac in (1.0, self.alphasigma)``

        """
        if self.noiseS > 0:
            self.evaluations = min((self.evaluations * self.alphaevals, self.maxevals))
            return self.alphasigma
        else:
            self.evaluations = max((self.evaluations * self.alphaevalsdown, self.minevals))
            return 1.0  # / self.alphasigma

    def reeval(self, X, fit, func, ask, args=()):
        """store two fitness lists, `fit` and ``fitre`` reevaluating some
        solutions in `X`.
        ``self.evaluations`` evaluations are done for each reevaluated
        fitness value.
        See `__call__()`, where `reeval()` is called.

        """
        self.fit = list(fit)
        self.fitre = list(fit)
        self.idx = self.indices(fit)
        if not len(self.idx):
            return self.idx
        evals = int(self.evaluations) if self.f_aggregate else 1
        fagg = np.median if self.f_aggregate is None else self.f_aggregate
        for i in self.idx:
            X_i = X[i]
            if self.epsilon:
                if self.parallel:
                    self.fitre[i] = fagg(func(ask(evals, X_i, self.epsilon), *args))
                else:
                    self.fitre[i] = fagg([func(ask(1, X_i, self.epsilon)[0], *args)
                                            for _k in range(evals)])
            else:
                self.fitre[i] = fagg([func(X_i, *args) for _k in range(evals)])
        self.evaluations_just_done = evals * len(self.idx)
        return self.fit, self.fitre, self.idx

    def update_measure(self):
        """updated noise level measure using two fitness lists ``self.fit`` and
        ``self.fitre``, return ``self.noiseS, all_individual_measures``.

        Assumes that `self.idx` contains the indices where the fitness
        lists differ

        """
        lam = len(self.fit)
        idx = np.argsort(self.fit + self.fitre)
        ranks = np.argsort(idx).reshape((2, lam))
        rankDelta = ranks[0] - ranks[1] - np.sign(ranks[0] - ranks[1])

        # compute rank change limits using both ranks[0] and ranks[1]
        r = np.arange(1, 2 * lam)  # 2 * lam - 2 elements
        limits = [0.5 * (Mh.prctile(np.abs(r - (ranks[0, i] + 1 - (ranks[0, i] > ranks[1, i]))),
                                      self.theta * 50) +
                         Mh.prctile(np.abs(r - (ranks[1, i] + 1 - (ranks[1, i] > ranks[0, i]))),
                                      self.theta * 50))
                    for i in self.idx]
        # compute measurement
        #                               max: 1 rankchange in 2*lambda is always fine
        s = np.abs(rankDelta[self.idx]) - Mh.amax(limits, 1)  # lives roughly in 0..2*lambda
        self.noiseS += self.cum * (np.mean(s) - self.noiseS)
        return self.noiseS, s

    def indices(self, fit):
        """return the set of indices to be reevaluated for noise measurement,
        taking the ``lam_reeval`` best from the first ``2 * lam_reeval + 2``
        values.

        Given the first values are the earliest, this is a useful policy also
        with a time changing objective.

        """
        lam = self.lam_reeval if self.lam_reeval else 2 + len(fit) / 20
        reev = int(lam) + ((lam % 1) > np.random.rand())
        return np.argsort(array(fit, copy=False)[:2 * (reev + 1)])[:reev]

# ____________________________________________________________
# ____________________________________________________________
class Sections(object):
    """plot sections through an objective function.

    A first rational thing to do, when facing an (expensive)
    application. By default 6 points in each coordinate are evaluated.
    This class is still experimental.

    Examples
    --------

    >>> import cma, numpy as np
    >>> s = cma.Sections(cma.Fcts.rosen, np.zeros(3)).do(plot=False)
    >>> s.do(plot=False)  # evaluate the same points again, i.e. check for noise
    >> try:
    ...     s.plot()
    ... except:
    ...     print('plotting failed: matplotlib.pyplot package is missing?')

    Details
    -------
    Data are saved after each function call during `do()`. The filename
    is attribute ``name`` and by default ``str(func)``, see `__init__()`.

    A random (orthogonal) basis can be generated with
    ``cma.Rotation()(np.eye(3))``.

    CAVEAT: The default name is unique in the function name, but it
    should be unique in all parameters of `__init__()` but `plot_cmd`
    and `load`. If, for example, a different basis is chosen, either
    the name must be changed or the ``.pkl`` file containing the
    previous data must first be renamed or deleted.

    ``s.res`` is a dictionary with an entry for each "coordinate" ``i``
    and with an entry ``'x'``, the middle point. Each entry ``i`` is
    again a dictionary with keys being different dx values and the
    value being a sequence of f-values. For example ``s.res[2][0.1] ==
    [0.01, 0.01]``, which is generated using the difference vector ``s
    .basis[2]`` like

    ``s.res[2][dx] += func(s.res['x'] + dx * s.basis[2])``.

    :See: `__init__()`

    """
    def __init__(self, func, x, args=(), basis=None, name=None,
                 plot_cmd=pyplot.plot if pyplot else None, load=True):
        """
        Parameters
        ----------
            `func`
                objective function
            `x`
                point in search space, middle point of the sections
            `args`
                arguments passed to `func`
            `basis`
                evaluated points are ``func(x + locations[j] * basis[i]) for i in len(basis) for j in len(locations)``,
                see `do()`
            `name`
                filename where to save the result
            `plot_cmd`
                command used to plot the data, typically matplotlib pyplots `plot` or `semilogy`
            `load`
                load previous data from file ``str(func) + '.pkl'``

        """
        self.func = func
        self.args = args
        self.x = x
        self.name = name if name else str(func).replace(' ', '_').replace('>', '').replace('<', '')
        self.plot_cmd = plot_cmd  # or semilogy
        self.basis = np.eye(len(x)) if basis is None else basis

        try:
            self.load()
            if any(self.res['x'] != x):
                self.res = {}
                self.res['x'] = x  # TODO: res['x'] does not look perfect
            else:
                print(self.name + ' loaded')
        except:
            self.res = {}
            self.res['x'] = x

    def do(self, repetitions=1, locations=np.arange(-0.5, 0.6, 0.2), plot=True):
        """generates, plots and saves function values ``func(y)``,
        where ``y`` is 'close' to `x` (see `__init__()`). The data are stored in
        the ``res`` attribute and the class instance is saved in a file
        with (the weired) name ``str(func)``.

        Parameters
        ----------
            `repetitions`
                for each point, only for noisy functions is >1 useful. For
                ``repetitions==0`` only already generated data are plotted.
            `locations`
                coordinated wise deviations from the middle point given in `__init__`

        """
        if not repetitions:
            self.plot()
            return

        res = self.res
        for i in range(len(self.basis)):  # i-th coordinate
            if i not in res:
                res[i] = {}
            # xx = np.array(self.x)
            # TODO: store res[i]['dx'] = self.basis[i] here?
            for dx in locations:
                xx = self.x + dx * self.basis[i]
                xkey = dx  # xx[i] if (self.basis == np.eye(len(self.basis))).all() else dx
                if xkey not in res[i]:
                    res[i][xkey] = []
                n = repetitions
                while n > 0:
                    n -= 1
                    res[i][xkey].append(self.func(xx, *self.args))
                    if plot:
                        self.plot()
                    self.save()
        return self

    def plot(self, plot_cmd=None, tf=lambda y: y):
        """plot the data we have, return ``self``"""
        if not plot_cmd:
            plot_cmd = self.plot_cmd
        colors = 'bgrcmyk'
        #pyplot.hold(False)
        res = self.res

        flatx, flatf = self.flattened()
        minf = np.inf
        for i in flatf:
            minf = min((minf, min(flatf[i])))
        addf = 1e-9 - minf if minf <= 1e-9 else 0
        for i in sorted(res.keys()):  # we plot not all values here
            if isinstance(i, int):
                color = colors[i % len(colors)]
                arx = sorted(res[i].keys())
                plot_cmd(arx, [tf(np.median(res[i][x]) + addf) for x in arx], color + '-')
                pyplot.text(arx[-1], tf(np.median(res[i][arx[-1]])), i)
                #pyplot.hold(True)
                plot_cmd(flatx[i], tf(np.array(flatf[i]) + addf), color + 'o')
        pyplot.ylabel('f + ' + str(addf))
        pyplot.draw()
        show()
        # input('press return')
        return self

    def flattened(self):
        """return flattened data ``(x, f)`` such that for the sweep through
        coordinate ``i`` we have for data point ``j`` that ``f[i][j] == func(x[i][j])``

        """
        flatx = {}
        flatf = {}
        for i in self.res:
            if isinstance(i, int):
                flatx[i] = []
                flatf[i] = []
                for x in sorted(self.res[i]):
                    for d in sorted(self.res[i][x]):
                        flatx[i].append(x)
                        flatf[i].append(d)
        return flatx, flatf

    def save(self, name=None):
        """save to file"""
        import pickle
        name = name if name else self.name
        fun = self.func
        del self.func  # instance method produces error
        pickle.dump(self, open(name + '.pkl', "wb"))
        self.func = fun
        return self

    def load(self, name=None):
        """load from file"""
        import pickle
        name = name if name else self.name
        s = pickle.load(open(name + '.pkl', 'rb'))
        self.res = s.res  # disregard the class
        return self

#____________________________________________________________
#____________________________________________________________
class _Error(Exception):
    """generic exception of cma module"""
    pass

# ____________________________________________________________
# ____________________________________________________________
#
class ElapsedTime(object):
    """using ``time.clock`` with overflow handling to measure CPU time.

    Example:

    >>> clock = ElapsedTime()  # clock starts here
    >>> t1 = clock()  # get elapsed CPU time

    Details: 32-bit C overflows after int(2**32/1e6) == 4294s about 72 min

    """
    def __init__(self):
        self.tic0 = time.clock()
        self.tic = self.tic0
        self.lasttoc = time.clock()
        self.lastdiff = time.clock() - self.lasttoc
        self.time_to_add = 0
        self.messages = 0
    reset = __init__
    def __call__(self):
        toc = time.clock()
        if toc - self.tic >= self.lasttoc - self.tic:
            self.lastdiff = toc - self.lasttoc
            self.lasttoc = toc
        else:  # overflow, reset self.tic
            if self.messages < 3:
                self.messages += 1
                print('  in cma.ElapsedTime: time measure overflow, last difference estimated from',
                        self.tic0, self.tic, self.lasttoc, toc, toc - self.lasttoc, self.lastdiff)

            self.time_to_add += self.lastdiff + self.lasttoc - self.tic
            self.tic = toc  # reset
            self.lasttoc = toc
        self.elapsedtime = toc - self.tic + self.time_to_add
        return self.elapsedtime

class Misc(object):
    # ____________________________________________________________
    # ____________________________________________________________
    #
    class MathHelperFunctions(object):
        """static convenience math helper functions, if the function name
        is preceded with an "a", a numpy array is returned

        """
        @staticmethod
        def aclamp(x, upper):
            return -Misc.MathHelperFunctions.apos(-x, -upper)
        @staticmethod
        def equals_approximately(a, b, eps=1e-12):
            if a < 0:
                a, b = -1 * a, -1 * b
            return (a - eps < b < a + eps) or ((1 - eps) * a < b < (1 + eps) * a)
        @staticmethod
        def vequals_approximately(a, b, eps=1e-12):
            a, b = array(a), array(b)
            idx = np.where(a < 0)[0]
            if len(idx):
                a[idx], b[idx] = -1 * a[idx], -1 * b[idx]
            return (np.all(a - eps < b) and np.all(b < a + eps)
                    ) or (np.all((1 - eps) * a < b) and np.all(b < (1 + eps) * a))
        @staticmethod
        def expms(A, eig=np.linalg.eigh):
            """matrix exponential for a symmetric matrix"""
            # TODO: check that this works reliably for low rank matrices
            # first: symmetrize A
            D, B = eig(A)
            return np.dot(B, (np.exp(D) * B).T)
        @staticmethod
        def amax(vec, vec_or_scalar):
            return array(Misc.MathHelperFunctions.max(vec, vec_or_scalar))
        @staticmethod
        def max(vec, vec_or_scalar):
            b = vec_or_scalar
            if np.isscalar(b):
                m = [max(x, b) for x in vec]
            else:
                m = [max(vec[i], b[i]) for i in rglen((vec))]
            return m
        @staticmethod
        def minmax(val, min_val, max_val):
            assert min_val <= max_val
            return min((max_val, max((val, min_val))))
        @staticmethod
        def amin(vec_or_scalar, vec_or_scalar2):
            return array(Misc.MathHelperFunctions.min(vec_or_scalar, vec_or_scalar2))
        @staticmethod
        def min(a, b):
            iss = np.isscalar
            if iss(a) and iss(b):
                return min(a, b)
            if iss(a):
                a, b = b, a
            # now only b can be still a scalar
            if iss(b):
                return [min(x, b) for x in a]
            else:  # two non-scalars must have the same length
                return [min(a[i], b[i]) for i in rglen((a))]
        @staticmethod
        def norm(vec, expo=2):
            return sum(vec**expo)**(1 / expo)
        @staticmethod
        def apos(x, lower=0):
            """clips argument (scalar or array) from below at lower"""
            if lower == 0:
                return (x > 0) * x
            else:
                return lower + (x > lower) * (x - lower)
        @staticmethod
        def prctile(data, p_vals=[0, 25, 50, 75, 100], sorted_=False):
            """``prctile(data, 50)`` returns the median, but p_vals can
            also be a sequence.

            Provides for small samples better values than matplotlib.mlab.prctile,
            however also slower.

            """
            ps = [p_vals] if np.isscalar(p_vals) else p_vals

            if not sorted_:
                data = sorted(data)
            n = len(data)
            d = []
            for p in ps:
                fi = p * n / 100 - 0.5
                if fi <= 0:  # maybe extrapolate?
                    d.append(data[0])
                elif fi >= n - 1:
                    d.append(data[-1])
                else:
                    i = int(fi)
                    d.append((i + 1 - fi) * data[i] + (fi - i) * data[i + 1])
            return d[0] if np.isscalar(p_vals) else d
        @staticmethod
        def sround(nb):  # TODO: to be vectorized
            """return stochastic round: floor(nb) + (rand()<remainder(nb))"""
            return nb // 1 + (np.random.rand(1)[0] < (nb % 1))

        @staticmethod
        def cauchy_with_variance_one():
            n = np.random.randn() / np.random.randn()
            while abs(n) > 1000:
                n = np.random.randn() / np.random.randn()
            return n / 25
        @staticmethod
        def standard_finite_cauchy(size=1):
            try:
                l = len(size)
            except TypeError:
                l = 0

            if l == 0:
                return array([Mh.cauchy_with_variance_one() for _i in range(size)])
            elif l == 1:
                return array([Mh.cauchy_with_variance_one() for _i in range(size[0])])
            elif l == 2:
                return array([[Mh.cauchy_with_variance_one() for _i in range(size[1])]
                             for _j in range(size[0])])
            else:
                raise _Error('len(size) cannot be large than two')


    @staticmethod
    def likelihood(x, m=None, Cinv=None, sigma=1, detC=None):
        """return likelihood of x for the normal density N(m, sigma**2 * Cinv**-1)"""
        # testing: MC integrate must be one: mean(p(x_i)) * volume(where x_i are uniformely sampled)
        # for i in range(3): print mean([cma.likelihood(20*r-10, dim * [0], None, 3) for r in rand(10000,dim)]) * 20**dim
        if m is None:
            dx = x
        else:
            dx = x - m  # array(x) - array(m)
        n = len(x)
        s2pi = (2 * np.pi)**(n / 2.)
        if Cinv is None:
            return exp(-sum(dx**2) / sigma**2 / 2) / s2pi / sigma**n
        if detC is None:
            detC = 1. / np.linalg.linalg.det(Cinv)
        return  exp(-np.dot(dx, np.dot(Cinv, dx)) / sigma**2 / 2) / s2pi / abs(detC)**0.5 / sigma**n

    @staticmethod
    def loglikelihood(self, x, previous=False):
        """return log-likelihood of `x` regarding the current sample distribution"""
        # testing of original fct: MC integrate must be one: mean(p(x_i)) * volume(where x_i are uniformely sampled)
        # for i in range(3): print mean([cma.likelihood(20*r-10, dim * [0], None, 3) for r in rand(10000,dim)]) * 20**dim
        # TODO: test this!!
        # c=cma.fmin...
        # c[3]['cma'].loglikelihood(...)

        if previous and hasattr(self, 'lastiter'):
            sigma = self.lastiter.sigma
            Crootinv = self.lastiter._Crootinv
            xmean = self.lastiter.mean
            D = self.lastiter.D
        elif previous and self.countiter > 1:
            raise _Error('no previous distribution parameters stored, check options importance_mixing')
        else:
            sigma = self.sigma
            Crootinv = self._Crootinv
            xmean = self.mean
            D = self.D

        dx = array(x) - xmean  # array(x) - array(m)
        n = self.N
        logs2pi = n * log(2 * np.pi) / 2.
        logdetC = 2 * sum(log(D))
        dx = np.dot(Crootinv, dx)
        res = -sum(dx**2) / sigma**2 / 2 - logs2pi - logdetC / 2 - n * log(sigma)
        if 1 < 3:  # testing
            s2pi = (2 * np.pi)**(n / 2.)
            detC = np.prod(D)**2
            res2 = -sum(dx**2) / sigma**2 / 2 - log(s2pi * abs(detC)**0.5 * sigma**n)
            assert res2 < res + 1e-8 or res2 > res - 1e-8
        return res

    # ____________________________________________________________
    # ____________________________________________________________
    #
    # C and B are arrays rather than matrices, because they are
    # addressed via B[i][j], matrices can only be addressed via B[i,j]

    # tred2(N, B, diagD, offdiag);
    # tql2(N, diagD, offdiag, B);


    # Symmetric Householder reduction to tridiagonal form, translated from JAMA package.
    @staticmethod
    def eig(C):
        """eigendecomposition of a symmetric matrix, much slower than
        `numpy.linalg.eigh`, return ``(EVals, Basis)``, the eigenvalues
        and an orthonormal basis of the corresponding eigenvectors, where

            ``Basis[i]``
                the i-th row of ``Basis``
            columns of ``Basis``, ``[Basis[j][i] for j in range(len(Basis))]``
                the i-th eigenvector with eigenvalue ``EVals[i]``

        """

    # class eig(object):
    #     def __call__(self, C):

    # Householder transformation of a symmetric matrix V into tridiagonal form.
        # -> n             : dimension
        # -> V             : symmetric nxn-matrix
        # <- V             : orthogonal transformation matrix:
        #                    tridiag matrix == V * V_in * V^t
        # <- d             : diagonal
        # <- e[0..n-1]     : off diagonal (elements 1..n-1)

        # Symmetric tridiagonal QL algorithm, iterative
        # Computes the eigensystem from a tridiagonal matrix in roughtly 3N^3 operations
        # -> n     : Dimension.
        # -> d     : Diagonale of tridiagonal matrix.
        # -> e[1..n-1] : off-diagonal, output from Householder
        # -> V     : matrix output von Householder
        # <- d     : eigenvalues
        # <- e     : garbage?
        # <- V     : basis of eigenvectors, according to d


        #  tred2(N, B, diagD, offdiag); B=C on input
        #  tql2(N, diagD, offdiag, B);

        #  private void tred2 (int n, double V[][], double d[], double e[]) {
        def tred2 (n, V, d, e):
            #  This is derived from the Algol procedures tred2 by
            #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
            #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
            #  Fortran subroutine in EISPACK.

            num_opt = False  # factor 1.5 in 30-D

            for j in range(n):
                d[j] = V[n - 1][j]  # d is output argument

            # Householder reduction to tridiagonal form.

            for i in range(n - 1, 0, -1):
                # Scale to avoid under/overflow.
                h = 0.0
                if not num_opt:
                    scale = 0.0
                    for k in range(i):
                        scale = scale + abs(d[k])
                else:
                    scale = sum(abs(d[0:i]))

                if scale == 0.0:
                    e[i] = d[i - 1]
                    for j in range(i):
                        d[j] = V[i - 1][j]
                        V[i][j] = 0.0
                        V[j][i] = 0.0
                else:

                    # Generate Householder vector.
                    if not num_opt:
                        for k in range(i):
                            d[k] /= scale
                            h += d[k] * d[k]
                    else:
                        d[:i] /= scale
                        h = np.dot(d[:i], d[:i])

                    f = d[i - 1]
                    g = h**0.5

                    if f > 0:
                        g = -g

                    e[i] = scale * g
                    h = h - f * g
                    d[i - 1] = f - g
                    if not num_opt:
                        for j in range(i):
                            e[j] = 0.0
                    else:
                        e[:i] = 0.0

                    # Apply similarity transformation to remaining columns.

                    for j in range(i):
                        f = d[j]
                        V[j][i] = f
                        g = e[j] + V[j][j] * f
                        if not num_opt:
                            for k in range(j + 1, i):
                                g += V[k][j] * d[k]
                                e[k] += V[k][j] * f
                            e[j] = g
                        else:
                            e[j + 1:i] += V.T[j][j + 1:i] * f
                            e[j] = g + np.dot(V.T[j][j + 1:i], d[j + 1:i])

                    f = 0.0
                    if not num_opt:
                        for j in range(i):
                            e[j] /= h
                            f += e[j] * d[j]
                    else:
                        e[:i] /= h
                        f += np.dot(e[:i], d[:i])

                    hh = f / (h + h)
                    if not num_opt:
                        for j in range(i):
                            e[j] -= hh * d[j]
                    else:
                        e[:i] -= hh * d[:i]

                    for j in range(i):
                        f = d[j]
                        g = e[j]
                        if not num_opt:
                            for k in range(j, i):
                                V[k][j] -= (f * e[k] + g * d[k])
                        else:
                            V.T[j][j:i] -= (f * e[j:i] + g * d[j:i])

                        d[j] = V[i - 1][j]
                        V[i][j] = 0.0

                d[i] = h
            # end for i--

            # Accumulate transformations.

            for i in range(n - 1):
                V[n - 1][i] = V[i][i]
                V[i][i] = 1.0
                h = d[i + 1]
                if h != 0.0:
                    if not num_opt:
                        for k in range(i + 1):
                            d[k] = V[k][i + 1] / h
                    else:
                        d[:i + 1] = V.T[i + 1][:i + 1] / h

                    for j in range(i + 1):
                        if not num_opt:
                            g = 0.0
                            for k in range(i + 1):
                                g += V[k][i + 1] * V[k][j]
                            for k in range(i + 1):
                                V[k][j] -= g * d[k]
                        else:
                            g = np.dot(V.T[i + 1][0:i + 1], V.T[j][0:i + 1])
                            V.T[j][:i + 1] -= g * d[:i + 1]

                if not num_opt:
                    for k in range(i + 1):
                        V[k][i + 1] = 0.0
                else:
                    V.T[i + 1][:i + 1] = 0.0


            if not num_opt:
                for j in range(n):
                    d[j] = V[n - 1][j]
                    V[n - 1][j] = 0.0
            else:
                d[:n] = V[n - 1][:n]
                V[n - 1][:n] = 0.0

            V[n - 1][n - 1] = 1.0
            e[0] = 0.0


        # Symmetric tridiagonal QL algorithm, taken from JAMA package.
        # private void tql2 (int n, double d[], double e[], double V[][]) {
        # needs roughly 3N^3 operations
        def tql2 (n, d, e, V):

            #  This is derived from the Algol procedures tql2, by
            #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
            #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
            #  Fortran subroutine in EISPACK.

            num_opt = False  # using vectors from numpy makes it faster

            if not num_opt:
                for i in range(1, n):  # (int i = 1; i < n; i++):
                    e[i - 1] = e[i]
            else:
                e[0:n - 1] = e[1:n]
            e[n - 1] = 0.0

            f = 0.0
            tst1 = 0.0
            eps = 2.0**-52.0
            for l in range(n):  # (int l = 0; l < n; l++) {

                # Find small subdiagonal element

                tst1 = max(tst1, abs(d[l]) + abs(e[l]))
                m = l
                while m < n:
                    if abs(e[m]) <= eps * tst1:
                        break
                    m += 1

                # If m == l, d[l] is an eigenvalue,
                # otherwise, iterate.

                if m > l:
                    iiter = 0
                    while 1:  # do {
                        iiter += 1  # (Could check iteration count here.)

                        # Compute implicit shift

                        g = d[l]
                        p = (d[l + 1] - g) / (2.0 * e[l])
                        r = (p**2 + 1)**0.5  # hypot(p,1.0)
                        if p < 0:
                            r = -r

                        d[l] = e[l] / (p + r)
                        d[l + 1] = e[l] * (p + r)
                        dl1 = d[l + 1]
                        h = g - d[l]
                        if not num_opt:
                            for i in range(l + 2, n):
                                d[i] -= h
                        else:
                            d[l + 2:n] -= h

                        f = f + h

                        # Implicit QL transformation.

                        p = d[m]
                        c = 1.0
                        c2 = c
                        c3 = c
                        el1 = e[l + 1]
                        s = 0.0
                        s2 = 0.0

                        # hh = V.T[0].copy()  # only with num_opt
                        for i in range(m - 1, l - 1, -1):  # (int i = m-1; i >= l; i--) {
                            c3 = c2
                            c2 = c
                            s2 = s
                            g = c * e[i]
                            h = c * p
                            r = (p**2 + e[i]**2)**0.5  # hypot(p,e[i])
                            e[i + 1] = s * r
                            s = e[i] / r
                            c = p / r
                            p = c * d[i] - s * g
                            d[i + 1] = h + s * (c * g + s * d[i])

                            # Accumulate transformation.

                            if not num_opt:  # overall factor 3 in 30-D
                                for k in range(n):  # (int k = 0; k < n; k++) {
                                    h = V[k][i + 1]
                                    V[k][i + 1] = s * V[k][i] + c * h
                                    V[k][i] = c * V[k][i] - s * h
                            else:  # about 20% faster in 10-D
                                hh = V.T[i + 1].copy()
                                # hh[:] = V.T[i+1][:]
                                V.T[i + 1] = s * V.T[i] + c * hh
                                V.T[i] = c * V.T[i] - s * hh
                                # V.T[i] *= c
                                # V.T[i] -= s * hh

                        p = -s * s2 * c3 * el1 * e[l] / dl1
                        e[l] = s * p
                        d[l] = c * p

                        # Check for convergence.
                        if abs(e[l]) <= eps * tst1:
                            break
                    # } while (Math.abs(e[l]) > eps*tst1);

                d[l] = d[l] + f
                e[l] = 0.0


            # Sort eigenvalues and corresponding vectors.
        # tql2

        N = len(C[0])
        if 1 < 3:
            V = [[x[i] for i in range(N)] for x in C]  # copy each "row"
            d = N * [0.]
            e = N * [0.]

        tred2(N, V, d, e)
        tql2(N, d, e, V)
        return (array(d), array(V))
Mh = Misc.MathHelperFunctions

# from new_stuff import *

def pprint(to_be_printed):
    """nicely formated print"""
    try:
        import pprint as pp
        # generate an instance PrettyPrinter
        # pp.PrettyPrinter().pprint(to_be_printed)
        pp.pprint(to_be_printed)
    except ImportError:
        if isinstance(to_be_printed, dict):
            print('{')
            for k, v in to_be_printed.items():
                print("'" + k + "'" if isinstance(k, str) else k,
                      ': ',
                      "'" + v + "'" if isinstance(k, str) else v,
                      sep="")
            print('}')
        else:
            print('could not import pprint module, will apply regular print')
            print(to_be_printed)

pp = pprint

class Rotation(object):
    """Rotation class that implements an orthogonal linear transformation,
    one for each dimension. Used to implement non-separable test functions.

    Example:

    >>> import cma, numpy as np
    >>> R = cma.Rotation()
    >>> R2 = cma.Rotation() # another rotation
    >>> x = np.array((1,2,3))
    >>> print(R(R(x), inverse=1))
    [ 1.  2.  3.]

    """
    dicMatrices = {}  # store matrix if necessary, for each dimension
    def __init__(self):
        self.dicMatrices = {}  # otherwise there might be shared bases which is probably not what we want
    def __call__(self, x, inverse=False):  # function when calling an object
        """Rotates the input array `x` with a fixed rotation matrix
           (``self.dicMatrices['str(len(x))']``)
        """
        N = x.shape[0]  # can be an array or matrix, TODO: accept also a list of arrays?
        if str(N) not in self.dicMatrices:  # create new N-basis for once and all
            B = np.random.randn(N, N)
            for i in range(N):
                for j in range(0, i):
                    B[i] -= np.dot(B[i], B[j]) * B[j]
                B[i] /= sum(B[i]**2)**0.5
            self.dicMatrices[str(N)] = B
        if inverse:
            return np.dot(self.dicMatrices[str(N)].T, x)  # compute rotation
        else:
            return np.dot(self.dicMatrices[str(N)], x)  # compute rotation
# Use rotate(x) to rotate x
rotate = Rotation()

# ____________________________________________________________
# ____________________________________________________________
#
class FitnessFunctions(object):
    """ versatile container for test objective functions """

    def __init__(self):
        self.counter = 0  # number of calls or any other practical use
    def rot(self, x, fun, rot=1, args=()):
        """returns ``fun(rotation(x), *args)``, ie. `fun` applied to a rotated argument"""
        if len(np.shape(array(x))) > 1:  # parallelized
            res = []
            for x in x:
                res.append(self.rot(x, fun, rot, args))
            return res

        if rot:
            return fun(rotate(x, *args))
        else:
            return fun(x)
    def somenan(self, x, fun, p=0.1):
        """returns sometimes np.NaN, otherwise fun(x)"""
        if np.random.rand(1) < p:
            return np.NaN
        else:
            return fun(x)
    def rand(self, x):
        """Random test objective function"""
        return np.random.random(1)[0]
    def linear(self, x):
        return -x[0]
    def lineard(self, x):
        if 1 < 3 and any(array(x) < 0):
            return np.nan
        if 1 < 3 and sum([ (10 + i) * x[i] for i in rglen(x)]) > 50e3:
            return np.nan
        return -sum(x)
    def sphere(self, x):
        """Sphere (squared norm) test objective function"""
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        return sum((x + 0)**2)
    def sphere_pos(self, x):
        """Sphere (squared norm) test objective function"""
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        c = 0.0
        if x[0] < c:
            return np.nan
        return -c**2 + sum((x + 0)**2)
    def spherewithoneconstraint(self, x):
        return sum((x + 0)**2) if x[0] > 1 else np.nan
    def elliwithoneconstraint(self, x, idx=[-1]):
        return self.ellirot(x) if all(array(x)[idx] > 1) else np.nan

    def spherewithnconstraints(self, x):
        return sum((x + 0)**2) if all(array(x) > 1) else np.nan
    # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    def noisysphere(self, x, noise=2.10e-9, cond=1.0, noise_offset=0.10):
        """noise=10 does not work with default popsize, noise handling does not help """
        return self.elli(x, cond=cond) * (1 + noise * np.random.randn() / len(x)) + noise_offset * np.random.rand()
    def spherew(self, x):
        """Sphere (squared norm) with sum x_i = 1 test objective function"""
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        # s = sum(abs(x))
        # return sum((x/s+0)**2) - 1/len(x)
        # return sum((x/s)**2) - 1/len(x)
        return -0.01 * x[0] + abs(x[0])**-2 * sum(x[1:]**2)
    def partsphere(self, x):
        """Sphere (squared norm) test objective function"""
        self.counter += 1
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        dim = len(x)
        x = array([x[i % dim] for i in range(2 * dim)])
        N = 8
        i = self.counter % dim
        # f = sum(x[i:i + N]**2)
        f = sum(x[np.random.randint(dim, size=N)]**2)
        return f
    def sectorsphere(self, x):
        """asymmetric Sphere (squared norm) test objective function"""
        return sum(x**2) + (1e6 - 1) * sum(x[x < 0]**2)
    def cornersphere(self, x):
        """Sphere (squared norm) test objective function constraint to the corner"""
        nconstr = len(x) - 0
        if any(x[:nconstr] < 1):
            return np.NaN
        return sum(x**2) - nconstr
    def cornerelli(self, x):
        """ """
        if any(x < 1):
            return np.NaN
        return self.elli(x) - self.elli(np.ones(len(x)))
    def cornerellirot(self, x):
        """ """
        if any(x < 1):
            return np.NaN
        return self.ellirot(x)
    def normalSkew(self, f):
        N = np.random.randn(1)[0]**2
        if N < 1:
            N = f * N  # diminish blow up lower part
        return N
    def noiseC(self, x, func=sphere, fac=10, expon=0.8):
        f = func(self, x)
        N = np.random.randn(1)[0] / np.random.randn(1)[0]
        return max(1e-19, f + (float(fac) / len(x)) * f**expon * N)
    def noise(self, x, func=sphere, fac=10, expon=1):
        f = func(self, x)
        # R = np.random.randn(1)[0]
        R = np.log10(f) + expon * abs(10 - np.log10(f)) * np.random.rand(1)[0]
        # sig = float(fac)/float(len(x))
        # R = log(f) + 0.5*log(f) * random.randn(1)[0]
        # return max(1e-19, f + sig * (f**np.log10(f)) * np.exp(R))
        # return max(1e-19, f * np.exp(sig * N / f**expon))
        # return max(1e-19, f * normalSkew(f**expon)**sig)
        return f + 10**R  # == f + f**(1+0.5*RN)
    def cigar(self, x, rot=0, cond=1e6, noise=0):
        """Cigar test objective function"""
        if rot:
            x = rotate(x)
        x = [x] if np.isscalar(x[0]) else x  # scalar into list
        f = [(x[0]**2 + cond * sum(x[1:]**2)) * np.exp(noise * np.random.randn(1)[0] / len(x)) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar
    def tablet(self, x, rot=0):
        """Tablet test objective function"""
        if rot:
            x = rotate(x)
        x = [x] if np.isscalar(x[0]) else x  # scalar into list
        f = [1e6 * x[0]**2 + sum(x[1:]**2) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar
    def cigtab(self, y):
        """Cigtab test objective function"""
        X = [y] if np.isscalar(y[0]) else y
        f = [1e-4 * x[0]**2 + 1e4 * x[1]**2 + sum(x[2:]**2) for x in X]
        return f if len(f) > 1 else f[0]
    def twoaxes(self, y):
        """Cigtab test objective function"""
        X = [y] if np.isscalar(y[0]) else y
        N2 = len(X[0]) // 2
        f = [1e6 * sum(x[0:N2]**2) + sum(x[N2:]**2) for x in X]
        return f if len(f) > 1 else f[0]
    def ellirot(self, x):
        return fcts.elli(array(x), 1)
    def hyperelli(self, x):
        N = len(x)
        return sum((np.arange(1, N + 1) * x)**2)
    def elli(self, x, rot=0, xoffset=0, cond=1e6, actuator_noise=0.0, both=False):
        """Ellipsoid test objective function"""
        if not np.isscalar(x[0]):  # parallel evaluation
            return [self.elli(xi, rot) for xi in x]  # could save 20% overall
        if rot:
            x = rotate(x)
        N = len(x)
        if actuator_noise:
            x = x + actuator_noise * np.random.randn(N)

        ftrue = sum(cond**(np.arange(N) / (N - 1.)) * (x + xoffset)**2)

        alpha = 0.49 + 1. / N
        beta = 1
        felli = np.random.rand(1)[0]**beta * ftrue * \
                max(1, (10.**9 / (ftrue + 1e-99))**(alpha * np.random.rand(1)[0]))
        # felli = ftrue + 1*np.random.randn(1)[0] / (1e-30 +
        #                                           np.abs(np.random.randn(1)[0]))**0
        if both:
            return (felli, ftrue)
        else:
            # return felli  # possibly noisy value
            return ftrue  # + np.random.randn()
    def elliconstraint(self, x, cfac=1e8, tough=True, cond=1e6):
        """ellipsoid test objective function with "constraints" """
        N = len(x)
        f = sum(cond**(np.arange(N)[-1::-1] / (N - 1)) * x**2)
        cvals = (x[0] + 1,
                 x[0] + 1 + 100 * x[1],
                 x[0] + 1 - 100 * x[1])
        if tough:
            f += cfac * sum(max(0, c) for c in cvals)
        else:
            f += cfac * sum(max(0, c + 1e-3)**2 for c in cvals)
        return f
    def rosen(self, x, alpha=1e2):
        """Rosenbrock test objective function"""
        x = [x] if np.isscalar(x[0]) else x  # scalar into list
        f = [sum(alpha * (x[:-1]**2 - x[1:])**2 + (1. - x[:-1])**2) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar
    def diffpow(self, x, rot=0):
        """Diffpow test objective function"""
        N = len(x)
        if rot:
            x = rotate(x)
        return sum(np.abs(x)**(2. + 4.*np.arange(N) / (N - 1.)))**0.5
    def rosenelli(self, x):
        N = len(x)
        return self.rosen(x[:N / 2]) + self.elli(x[N / 2:], cond=1)
    def ridge(self, x, expo=2):
        x = [x] if np.isscalar(x[0]) else x  # scalar into list
        f = [x[0] + 100 * np.sum(x[1:]**2)**(expo / 2.) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar
    def ridgecircle(self, x, expo=0.5):
        """happy cat by HG Beyer"""
        a = len(x)
        s = sum(x**2)
        return ((s - a)**2)**(expo / 2) + s / a + sum(x) / a
    def happycat(self, x, alpha=1. / 8):
        s = sum(x**2)
        return ((s - len(x))**2)**alpha + (s / 2 + sum(x)) / len(x) + 0.5
    def flat(self, x):
        return 1
        return 1 if np.random.rand(1) < 0.9 else 1.1
        return np.random.randint(1, 30)
    def branin(self, x):
        # in [0,15]**2
        y = x[1]
        x = x[0] + 5
        return (y - 5.1 * x**2 / 4 / np.pi**2 + 5 * x / np.pi - 6)**2 + 10 * (1 - 1 / 8 / np.pi) * np.cos(x) + 10 - 0.397887357729738160000
    def goldsteinprice(self, x):
        x1 = x[0]
        x2 = x[1]
        return (1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)) * (
                30 + (2 * x1 - 3 * x2)**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)) - 3
    def griewank(self, x):
        # was in [-600 600]
        x = (600. / 5) * x
        return 1 - np.prod(np.cos(x / sqrt(1. + np.arange(len(x))))) + sum(x**2) / 4e3
    def rastrigin(self, x):
        """Rastrigin test objective function"""
        if not np.isscalar(x[0]):
            N = len(x[0])
            return [10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]
            # return 10*N + sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
        N = len(x)
        return 10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x))
    def schaffer(self, x):
        """ Schaffer function x0 in [-100..100]"""
        N = len(x)
        s = x[0:N - 1]**2 + x[1:N]**2
        return sum(s**0.25 * (np.sin(50 * s**0.1)**2 + 1))

    def schwefelelli(self, x):
        s = 0
        f = 0
        for i in rglen(x):
            s += x[i]
            f += s**2
        return f
    def schwefelmult(self, x, pen_fac=1e4):
        """multimodal Schwefel function with domain -500..500"""
        y = [x] if np.isscalar(x[0]) else x
        N = len(y[0])
        f = array([418.9829 * N - 1.27275661e-5 * N - sum(x * np.sin(np.abs(x)**0.5))
                + pen_fac * sum((abs(x) > 500) * (abs(x) - 500)**2) for x in y])
        return f if len(f) > 1 else f[0]
    def optprob(self, x):
        n = np.arange(len(x)) + 1
        f = n * x * (1 - x)**(n - 1)
        return sum(1 - f)
    def lincon(self, x, theta=0.01):
        """ridge like linear function with one linear constraint"""
        if x[0] < 0:
            return np.NaN
        return theta * x[1] + x[0]
    def rosen_nesterov(self, x, rho=100):
        """needs exponential number of steps in a non-increasing f-sequence.

        x_0 = (-1,1,...,1)
        See Jarre (2011) "On Nesterov's Smooth Chebyshev-Rosenbrock Function"

        """
        f = 0.25 * (x[0] - 1)**2
        f += rho * sum((x[1:] - 2 * x[:-1]**2 + 1)**2)
        return f
    def powel_singular(self, x):
        # ((8 * np.sin(7 * (x[i] - 0.9)**2)**2 ) + (6 * np.sin()))
        res = np.sum((x[i - 1] + 10 * x[i])**2 + 5 * (x[i + 1] - x[i + 2])**2 +
                     (x[i] - 2 * x[i + 1])**4 + 10 * (x[i - 1] - x[i + 2])**4
                     for i in range(1, len(x) - 2))
        return 1 + res
    def styblinski_tang(self, x):
        return (39.1661657037714171054273576010019 * len(x))**1 + sum(x**4 - 16*x**2 + 5*x) / 2
    def bukin(self, x):
        """Bukin function from Wikipedia, generalized simplistically from 2-D.

        http://en.wikipedia.org/wiki/Test_functions_for_optimization"""
        s = 0
        for k in range((1+len(x)) // 2):
            z = x[2 * k]
            y = x[min((2*k + 1, len(x)-1))]
            s += 100 * np.abs(y - 0.01 * z**2)**0.5 + 0.01 * np.abs(z + 10)
        return s

fcts = FitnessFunctions()
Fcts = fcts  # for cross compatibility, as if the functions were static members of class Fcts
def felli(x):
    """unbound test function, needed to test multiprocessor"""
    return sum(1e6**(np.arange(len(x)) / (len(x) - 1)) * (np.array(x, copy=False))**2)


# ____________________________________________
# ____________________________________________________________
def _test(module=None):  # None is fine when called from inside the module
    import doctest
    print(doctest.testmod(module))  # this is pretty coool!
def process_doctest_output(stream=None):
    """ """
    import fileinput
    s1 = ""
    s2 = ""
    s3 = ""
    state = 0
    for line in fileinput.input(stream):  # takes argv as file or stdin
        if 1 < 3:
            s3 += line
            if state < -1 and line.startswith('***'):
                print(s3)
            if line.startswith('***'):
                s3 = ""

        if state == -1:  # found a failed example line
            s1 += '\n\n*** Failed Example:' + line
            s2 += '\n\n\n'  # line
            # state = 0  # wait for 'Expected:' line

        if line.startswith('Expected:'):
            state = 1
            continue
        elif line.startswith('Got:'):
            state = 2
            continue
        elif line.startswith('***'):  # marks end of failed example
            state = 0
        elif line.startswith('Failed example:'):
            state = -1
        elif line.startswith('Exception raised'):
            state = -2

        # in effect more else:
        if state == 1:
            s1 += line + ''
        if state == 2:
            s2 += line + ''

# ____________________________________________________________
# ____________________________________________________________
#
def main(argv=None):
    """to install and/or test from the command line use::

        python cma.py [options | func dim sig0 [optkey optval][optkey optval]...]

    with options being

    ``--test`` (or ``-t``) to run the doctest, ``--test -v`` to get (much) verbosity.

    ``install`` to install cma.py (uses setup from distutils.core).

    ``--doc`` for more infos.

    Or start Python or (even better) ``ipython -pylab`` and::

        import cma
        cma.main('--test')
        help(cma)
        help(cma.fmin)
        res = fmin(cma.fcts.rosen, 10 * [0], 1)
        cma.plot()

    Examples
    ========
    Testing with the local python distribution from a command line
    in a folder where ``cma.py`` can be found::

        python cma.py --test

    And a single run on the Rosenbrock function::

        python cma.py rosen 10 1  # dimension initial_sigma
        python cma.py plot

    In the python shell::

        import cma
        cma.main('--test')

    """
    if argv is None:
        argv = sys.argv  # should have better been sys.argv[1:]
    else:
        if isinstance(argv, list):
            argv = ['python'] + argv  # see above
        else:
            argv = ['python'] + [argv]

    # uncomment for unit test
    # _test()
    # handle input arguments, getopt might be helpful ;-)
    if len(argv) >= 1:  # function and help
        if len(argv) == 1 or argv[1].startswith('-h') or argv[1].startswith('--help'):
            print(main.__doc__)
            fun = None
        elif argv[1].startswith('-t') or argv[1].startswith('--test'):
            import doctest
            if len(argv) > 2 and (argv[2].startswith('--v') or argv[2].startswith('-v')):  # verbose
                print('doctest for cma.py: due to different platforms and python versions')
                print('and in some cases due to a missing unique random seed')
                print('many examples will "fail". This is OK, if they give a similar')
                print('to the expected result and if no exception occurs. ')
                # if argv[1][2] == 'v':
                doctest.testmod(sys.modules[__name__], report=True)  # this is quite cool!
            else:  # was: if len(argv) > 2 and (argv[2].startswith('--qu') or argv[2].startswith('-q')):
                print('doctest for cma.py: launching...') # not anymore: (it might be necessary to close the pop up window to finish)
                fn = '_cma_doctest_.txt'
                stdout = sys.stdout
                try:
                    with open(fn, 'w') as f:
                        sys.stdout = f
                        clock = ElapsedTime()
                        doctest.testmod(sys.modules[__name__], report=True)  # this is quite cool!
                        t_elapsed = clock()
                finally:
                    sys.stdout = stdout
                process_doctest_output(fn)
                # clean up
                try:
                    import os
                    for name in os.listdir('.'):
                        if (name.startswith('bound_method_FitnessFunctions.rosen_of_cma.FitnessFunctions_object_at_')
                            and name.endswith('.pkl')):
                            os.remove(name)
                except:
                    pass
                print('doctest for cma.py: finished (no other output should be seen after launching, more in file _cma_doctest_.txt)')
                print('  elapsed time [s]:', t_elapsed)
            return
        elif argv[1] == '--doc':
            print(__doc__)
            print(CMAEvolutionStrategy.__doc__)
            print(fmin.__doc__)
            fun = None
        elif argv[1] == '--fcts':
            print('List of valid function names:')
            print([d for d in dir(fcts) if not d.startswith('_')])
            fun = None
        elif argv[1] in ('install', '--install'):
            from distutils.core import setup
            setup(name="cma",
                  long_description=__doc__,
                  version=__version__.split()[0],
                  description="CMA-ES, Covariance Matrix Adaptation Evolution Strategy for non-linear numerical optimization in Python",
                  author="Nikolaus Hansen",
                  author_email="hansen at lri.fr",
                  maintainer="Nikolaus Hansen",
                  maintainer_email="hansen at lri.fr",
                  url="https://www.lri.fr/~hansen/cmaes_inmatlab.html#python",
                  license="MIT",
                  classifiers = [
                    "Intended Audience :: Science/Research",
                    "Intended Audience :: Education",
                    "Intended Audience :: Other Audience",
                    "Topic :: Scientific/Engineering",
                    "Topic :: Scientific/Engineering :: Mathematics",
                    "Topic :: Scientific/Engineering :: Artificial Intelligence",
                    "Operating System :: OS Independent",
                    "Programming Language :: Python :: 2.6",
                    "Programming Language :: Python :: 2.7",
                    "Programming Language :: Python :: 3",
                    "Development Status :: 4 - Beta",
                    "Environment :: Console",
                    "License :: OSI Approved :: MIT License",
                  ],
                  keywords=["optimization", "CMA-ES", "cmaes"],
                  py_modules=["cma"],
                  requires=["numpy"],
            )
            fun = None
        elif argv[1] in ('plot',):
            plot(name=argv[2] if len(argv) > 2 else None)
            input('press return')
            fun = None
        elif len(argv) > 3:
            fun = eval('fcts.' + argv[1])
        else:
            print('try -h option')
            fun = None

    if fun is not None:

        if len(argv) > 2:  # dimension
            x0 = np.ones(eval(argv[2]))
        if len(argv) > 3:  # sigma
            sig0 = eval(argv[3])

        opts = {}
        for i in range(5, len(argv), 2):
            opts[argv[i - 1]] = eval(argv[i])

        # run fmin
        if fun is not None:
            tic = time.time()
            fmin(fun, x0, sig0, opts)  # ftarget=1e-9, tolfacupx=1e9, verb_log=10)
            # plot()
            # print ' best function value ', res[2]['es'].best[1]
            print('elapsed time [s]: + %.2f', round(time.time() - tic, 2))

    elif not len(argv):
        fmin(fcts.elli, np.ones(6) * 0.1, 0.1, {'ftarget':1e-9})


# ____________________________________________________________
# ____________________________________________________________
#
# mainly for testing purpose
# executed when called from an OS shell
if __name__ == "__main__":
    # for i in range(1000):  # how to find the memory leak
    #     main(["cma.py", "rastrigin", "10", "5", "popsize", "200", "maxfevals", "24999", "verb_log", "0"])
    main()
