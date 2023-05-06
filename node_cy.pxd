# cython: language_level=3

cdef void data_add_cy(object node, int word_id, int word_position)
cdef void data_remove_cy(object node, int word_id, int word_position)
cdef list path_cy(object node)
cdef list path_reset_cy(object node)

cdef void customer_add_vertical(object node)
cdef void customer_add_horizontal(object node)

cdef double prob_stop_cy(object node)
cdef double prob_stop_horizontal_cy(object node)

cdef double alpha(object node)

cdef double nu(object node)
cdef double psi(object node)

cdef double calc_omega(object node, double value)
cdef double calc_psi(object node, double value)

cdef double gamma(object node)