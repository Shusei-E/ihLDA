cdef int multi_index(list pylist, double sum__)

# Distributions
cdef double gammapdfln(double x, double a, double b)
cdef double betapdfln(double x, double a, double b)

# Slice Sampling
cdef double shrink(double x, double A=*)
cdef double expand(double p, double A=*)
cdef double shrinkp(double x)
cdef double expandp(double p)
