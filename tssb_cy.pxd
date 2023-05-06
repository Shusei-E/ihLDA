#cython: language_level=3
cdef object node_create_cy(object tssb, object node_parent)

cdef find_node_cy(double u, object node, object tssb)

cdef node_from_path_cy(object tssb, list path)

cdef int path_compare_cy(list a, list b)

cdef void node_cull_cy(object tssb, object node=*)

cdef void node_cull_recursive(object node)
