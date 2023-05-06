# cython: profile=True
# htssb_cy.pyx

"""Cythonized functions for htssb

Important Functions:
    * :code:`node_corres_find_cy`: Find a corresponding node\
            in the parent TSSB. :py:meth:`tssb_cy.node_create` calls\
            this function.
"""

# Load files
cimport tssb_cy
cimport node_cy

cdef node_corres_find_cy(object node):
    """Find a corresponding node in the parent TSSB

    :py:meth:`tssb_cy.node_create` calls this function.

    Return:
        NODE object (a correspondng node)
    """
    cdef object new_node

    # root_node
    if node.node_parent is None:
        return node.htssb.tssb_root.node_root

    cdef object node_parent_corres = node.node_parent.node_corres

    cdef int k_corres  # index of my corresponding node in root TSSB


    if len(node.node_parent.node_corres.children) == 0:
        # The corresponding node of my parent node does not have
        # children. My corresting node will be the first child.

        new_node = tssb_cy.node_create_cy(node_parent_corres.tssb_mine,
                                          node_parent_corres
                                         )
        k_corres = 0

    elif len(node.node_parent.children) == 1:
        # If I am a first child of my parent's node, I will be
        # the first child of my parent's corrsponding node
        k_corres = 0

    else:
        # I am not the first child of my parent's node
        k_corres = node.node_parent.children.index(node)

        while not k_corres < len(node_parent_corres.children):
            # We need a new node
            new_node = tssb_cy.node_create_cy(node_parent_corres.tssb_mine,
                                              node_parent_corres
                                             )

    return node_parent_corres.children[k_corres]
