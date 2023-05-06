"""Tools

This file includes:
    * :code:`multi_index`
"""

# Load libraries
import numpy.random as npr
cimport numpy as np
import os


# Load C functions
cdef extern from "math.h":
    double log (double x)
    double exp (double x)
    double lgamma (double x)

cdef int multi_index(list pylist, double sum_):
    cdef int return_index = 0
    cdef double sum_prop = 0.0
    cdef double u = npr.uniform(0, 1) * sum_

    assert(len(pylist) > 0), "vector length is not > 0"

    while True:
        sum_prop += pylist[return_index]

        if u < sum_prop:
            break

        return_index += 1

    assert(return_index <= len(pylist) - 1)

    return return_index


def folder_check(str path):
    """Check a folder
    
    Check a folder's existence and make it if not.
    """
    if os.path.isdir(path) is False:
        os.makedirs(path)

"""
Distributions
"""
cdef double gammapdfln(double x, double a, double b):
    # a: shape, b: scale
    return - a * log(b) - lgamma(a) + (a-1.0) * log(x) - x/b

cdef double betapdfln(double x, double a, double b):
    return lgamma(a+b) - lgamma(a) - lgamma(b) + (a-1.0) * log(x) + (b-1.0) * log(1.0-x) 


"""
Slice Sampling
"""
cdef double shrink(double x, double A=0.5):
    # x --> p
    return 1.0 / (1.0 + exp(-A * x) )

cdef double expand(double p, double A=0.5):
    # p --> x
    return -(1.0/A) * log((1.0/p) - 1.0)

cdef double shrinkp(double x):
    # (x > 0) --> p
    return x / (1.0 + x)

cdef double expandp(double p):
    # p --> (x > 0)
    return p /(1.0 - p)
