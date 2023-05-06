cdef class HPYTable:
    cdef public list  tables  # a list for tables
    cdef public double tuw  # number of tables for a word `w` in node `u`
    cdef public double cuw  # number of word `w` in node `u`