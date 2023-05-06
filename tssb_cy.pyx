# cython: profile=True
# tssb_cy.pyx
"""Cythonized functions for tssb
Most functions are defined as :code:`cdef` and do not
appear in documents.
"""

# Load libraries
import numpy.random as npr
from numpy import argsort
cimport numpy as np

# Load files
from node import NODE
cimport htssb_cy
cimport node_cy
from node_cy cimport nu, psi


def node_create(object tssb, object node_parent):
    """Create node
    Create a new node in HTSSB

    1. Create a NODE instance
    2. Find a corresponding node in a parent TSSB\
             (if it is the root_tssb, skip this part)
    3. Initialize nu and psi of the created node\
        using the parent-children relationship. (See corrected Eq.(22) in\
        Mochihashi and Noji (2016) paper.): This part is not needed if\
        we always recalculate nu and psi
    """

    return node_create_cy(tssb, node_parent)


cdef object node_create_cy(object tssb, object node_parent):
    ### 1. Create a NODE instance
    cdef object node = NODE(tssb.ihLDA, tssb.htssb,
                            tssb_mine=tssb, node_parent=node_parent)
    tssb.node_list.append(node)

    if node_parent is not None:
       node_parent.children.append(node)

    # Create path
    node_cy.path_cy(node)

    ### 2. Find a corresponding node in a parent TSSB
    cdef object node_corres
    if tssb.tssb_parent is not None:

        # This node is not in the root_tssb
        # then, it should have a corresponding node
        node_corres = htssb_cy.node_corres_find_cy(node)
        node_corres.ref_corres += 1
        node_corres.node_corres_lower.append(node)
        node.node_corres = node_corres

        assert(node_corres.tssb_parent is None) # In ihLDA
        assert(node.path() == node.node_corres.path())

    ### 3. Initialize Parameters
    if node.node_parent is not None:
        # This is not root node
        node.index_in_parent = node.node_parent.children.index(node)

    return node


def initialize(object tssb, mode):
    """Initialize TSSB

    An important break points is whether it is a tssb_root or not.
    """

    if mode == "random":
        return initialize_cy_random(tssb)

    if mode == "dgp":
        return initialize_cy_dgp(tssb)

"""
Initialize TSSB
"""

cdef void initialize_cy_random(object tssb):
    """Random initialization
    """
    cdef double u
    cdef int word_id

    tssb.node_root = node_create_cy(tssb, node_parent=None)

    if tssb.tssb_parent is None:
        # This is a root_tssb
        # No need to add node when we initialize root_tssb
        return

    # Add data to nodes
    tssb.doc_len = tssb.ihLDA.data.each_doc_len[tssb.doc_id]
    tssb.node_assignments = list(range(tssb.doc_len))

    for index in range(tssb.doc_len):
        word_id = tssb.ihLDA.data.documents[tssb.doc_id][index]

        u = npr.uniform(0, 1)
        node = find_node_cy(u, tssb.node_root, tssb=tssb)

        node_cy.data_add_cy(node, word_id, index)


cdef void initialize_cy_dgp(object tssb):
    """Initialization following the DGP of TSSB
    """

    cdef int word_id
    tssb.node_root = node_create_cy(tssb, node_parent=None)

    if tssb.tssb_parent is None:
        # This is a root_tssb
        # No need to add node when we initialize root_tssb
        return

    # Add data to nodes
    tssb.doc_len = tssb.ihLDA.data.each_doc_len[tssb.doc_id]
    tssb.node_assignments = list(range(tssb.doc_len))

    for index in range(tssb.doc_len):
        word_id = tssb.ihLDA.data.documents[tssb.doc_id][index]
        node = dgp_node(tssb.node_root, word_id, initialize=1)

        node_cy.data_add_cy(node, word_id, index)


cdef object dgp_node(object node, int word_id, int initialize=0):
    """Sample node following the DGP
    First, check whether or not to stop vertically.
    If it needs to go down, check where to go horizontally.
    Recursively check the node.

    * initialize: this function is called in `tssb_cy.pyx` as well.
    """
    cdef double stop_prob
    cdef double pass_prob
    cdef dict stop_dict
    cdef dict pass_dict
    cdef double u = npr.rand()
    cdef double u2 = npr.rand()

    cdef double prob_vertical

    cdef double prob_horizontal
    cdef double prob_horizontal_remaining

    cdef double temp
    cdef int index
    cdef double prob
    cdef double prob_sum = 0.0

    if node.tssb_mine.htssb.params["depth_max"] <= node.level:
        # Reached at the bottom
        return node

    while len(node.children) < len(node.node_corres.children):
        # Match with the parent TSSB

        node_create_cy(node.tssb_mine, node_parent=node)
        node_cy.nu(node.children[-1])

    if node.node_parent is None:
        # This is a root node
        prob_vertical = node_cy.nu(node)
        stop_prob = prob_word(node, word_id) * prob_vertical

        if len(node.children) == 0 or node.children[-1].num_data != 0:
            # Add a new node
            node_create_cy(node.tssb_mine, node_parent=node)
            node_cy.nu(node.children[-1])

        pass_prob = 0.0
        prob_horizontal_remaining = 1.0
        pass_dict = {"sum": 0.0, "horizontal":[]}
        for child in node.children:
            prob_horizontal = node_cy.psi(child)
            temp = prob_word(child, word_id) * (prob_horizontal_remaining) * prob_horizontal
            pass_prob += temp
            pass_dict["horizontal"].append(temp)

            prob_horizontal_remaining *= (1 - prob_horizontal)


        pass_dict["sum"] = sum(pass_dict["horizontal"])
        pass_prob *= (1 - prob_vertical)

        stop_prob = stop_prob / (stop_prob + pass_prob)
        if initialize:
            stop_prob = pow(stop_prob, 0.4)
            pass_prob = pow(1.0 - stop_prob, 0.4)
            stop_prob = stop_prob / (stop_prob + pass_prob)

        if u < stop_prob:
            return node
        else:
            # Select where to stop horizontally
            u2 *= pass_dict["sum"]  # normalize
            prob_sum = 0.0

            for index, prob in enumerate(pass_dict["horizontal"]):
                prob_sum += prob
                if u2 < prob_sum:
                    return dgp_node(node.children[index], word_id, initialize)

    else:
        # Check stop vertically or go ahead
        stop_dict = getdict_node_stop(node, word_id)
        pass_dict = getdict_node_pass(node, word_id)

        stop_prob = stop_dict["sum"] / (stop_dict["sum"] + pass_dict["sum"])
        if initialize:
            stop_prob = pow(stop_prob, 0.6)
            pass_prob = pow(1.0 - stop_prob, 0.6)
            stop_prob = stop_prob / (stop_prob + pass_prob)

        if u < stop_prob:
            # Stop at this node
            return node
        else:
            # choose from pass horizontal
            u2 *= pass_dict["sum"]  # normalize
            prob_sum = 0.0

            for index, prob in enumerate(pass_dict["horizontal"]):
                prob_sum += prob
                if u2 < prob_sum:
                    return dgp_node(node.children[index], word_id, initialize)


cdef dict getdict_node_stop(object node, int word_id):
    cdef dict stop_dict = {"sum": 0.0, "horizontal":[]}
    stop_dict["sum"] = prob_word(node, word_id) * node_cy.nu(node)

    return stop_dict


cdef dict getdict_node_pass(object node, int word_id):
    cdef dict pass_dict = {"sum": 0.0, "horizontal":[]}
    cdef double pass_vertical = 1.0 - node_cy.nu(node)
    cdef double pass_prob
    cdef double prob_horizontal_remaining

    # Add a new node
    if len(node.children) == 0 or node.children[-1].num_data != 0:
        node_create_cy(node.tssb_mine, node_parent=node)
        node_cy.nu(node.children[-1])

    pass_prob = 0.0
    prob_horizontal_remaining = 1.0
    for child in node.children:
        prob_horizontal = node_cy.psi(child)
        temp = prob_word(child, word_id) *\
                (prob_horizontal_remaining) * prob_horizontal *\
                pass_vertical

        pass_prob += temp
        pass_dict["horizontal"].append(temp)

        prob_horizontal_remaining *= (1 - prob_horizontal)


    pass_dict["sum"] = sum(pass_dict["horizontal"])

    return pass_dict


cdef double prob_word(object node, int word_id):
    return(node.node_corres.tw_dist.wordprob_py(word_id))

"""
Find a node at a specific path location
"""
def node_from_path(object tssb, list path):
    return node_from_path_cy(tssb, path)

cdef node_from_path_cy(object tssb, list path):

    return node_from_path_search(tssb.node_root, tssb, path, 0)

cdef node_from_path_search(object node, object tssb, list path, int level):

    if node.path() == path:
        return node

    index_children = path[level + 1]
    while not index_children < len(node.children):
        # Add new_node
        new_node = node_create_cy(tssb, node)

    return node_from_path_search(node.children[index_children], tssb, path, level + 1)


"""
Find node
"""

def find_node(double u, object node, object tssb):
    """ Find a corresponding node in a tree

    Arguments:
        * tssb: a TSSB class you are looking into
    """
    return find_node_cy(u, node, tssb)


cdef find_node_cy(double u, object node, object tssb):
    cdef dict params = tssb.htssb.params
    cdef double prod
    cdef int k
    cdef double val_psi = -1.0
    cdef double v = nu(node)

    if params["depth_max"] <= node.level:
        return node
    if u <  v:
        return node
    else:
        u = (u - v) / (1.0 - v)
        prod = 1.0
        k = 0

        while True:
            if not k < len(node.children):
                node_create_cy(tssb, node_parent=node)

            val_psi = psi(node.children[k])
            prod *= (1.0 - val_psi)
            if u < 1.0-prod:
                break

            k += 1

            if k > params["child_max"][node.children[k-1].level]:
                return node.children[k-1]

        try:
            u = ((u-1.0) * (1.0-val_psi) + prod) / (prod * val_psi)
        except ZeroDivisionError:
            # If it tries to divide by 0,
            # val_psi is close to 1 so (1.0-val_psi) is almost 0
            u = ((u-1.0) * (1.0 - 2.2250738585072e-300) + prod) / (prod * 2.2250738585072e-300)

        return find_node_cy(u, node.children[k], tssb)


"""
Compare path
"""

def path_compare(list a, list b):
    """
    Compare path and return two values
        * 0: a is left part of the tree compared to b
        * 1: b is left part of the tree compared to a
        * 2: same path

    Check :py:meth:`node.NODE.path` for the path.

    .. figure:: figures/path_tree.png
        :scale: 50 %

    * a = [0 0 0] and b = [0 2 0]: return 0
    * a = [0 0 1 0] and b = [0 2 1]: return 0
    * a = [0 2] and b = [0 1]: return 1
    * a = [0 2 0] and b = [0 2 0 1]: return 0
    * a = [0 1] and b = [0 1]: return 2
    """

    if a == b:
        return 2

    return path_compare_cy(a, b)

cdef int path_compare_cy(list a, list b):
    cdef int a_head = a[0]
    cdef int b_head = b[0]
    cdef int a_len = len(a)
    cdef int b_len = len(b)

    if a_head > b_head:
        return 1
    if b_head > a_head:
        return 0

    if a_len == 1 or b_len == 1:
        if a_len < b_len:
            return 0
        else:
            return 1

    return path_compare_cy(a[1:], b[1:])



"""
Cull nodes
"""

def node_cull(object tssb):
    """Remove Unused nodes

    We focus of the stop counters. If the coutner is\
            not 0, we need it.

    In ihLDA_cy.pyx, we can directly call `node_cull_cy()`
    """

    return node_cull_cy(tssb)


cdef void node_cull_cy(object tssb, node=None):

    if node is None:
        node_cull_recursive(tssb.node_root)
    else:
        node_cull_recursive(node)

cdef void node_cull_recursive(object node):
    """Cull nodes

    Conditions (should satisfy all):
        * Not in the root_node
        * The last element of the children
        * Vertical counter is 0 (no customer uses)
        * Horizontal counter is 0 (no customer uses)
    """

    # Down all the way down first
    for child in node.children[::-1]:
        node_cull_recursive(child)

    # Check conditions

    if node.num_data != 0:
        # Node is used
        return

    # Check whether cull this node
    if node.node_parent is None:
        # This is the root_node
        return

    if node != node.node_parent.children[-1]:
        # This node is not the last element of children
        return

    if len(node.children) != 0:
        # Have children
        return

    if node.nv0 != 0 or node.nh0 != 0:
        # Stop counter is not 0
        return

    if node.nv1 != 0 or node.nh1 != 0:
        # Stop counter is not 0
        return

    # If this is a node in root_TSSB
    # check HPY info
    if node.tssb_parent is None:
        if node.tw_dist.cu != 0:
            # If there is a customer,
            # do not cull -> return
            return
        assert(node.tw_dist.tu == 0)

        # If this node (in parent TSSB) is
        # referred by a node in child TSSB,
        # do not cull it
        if len(node.node_corres_lower) != 0:
            return

        assert(len(node.node_corres_lower) == 0)

    # If the node is in document (child) TSSB,
    # remove it from the corresponding node info
    if node.tssb_parent is not None:
        node.node_corres.node_corres_lower.remove(node)

    # Remove from the parent
    node.node_parent.children.pop()

    # Remove from the node_list
    node.tssb_mine.node_list.remove(node)

    # If cull node, update parameters
    #node_cull_update_params(node.htssb.tssb_root)  # not sure the use of it
    return


cdef void node_cull_update_params(object tssb_root):
    """
    If we have unused hpy_d and hpy_theta as a result of\
            cull_node(), remove it.

    Note that level starts from 0
    """
    cdef int current_max_level
    current_max_level = node_level_recursive(tssb_root.node_root, 0)
    tssb_root.htssb.params["current_max"] = current_max_level

    if tssb_root.ihLDA.mode == "hpy":

        while current_max_level < (len(tssb_root.htssb.params["hpy_d"]) - 1):
            tssb_root.htssb.params["hpy_d"].pop()
            tssb_root.htssb.params["hpy_theta"].pop()
            tssb_root.htssb.params["alpha_l"].pop()

        assert(current_max_level == len(tssb_root.htssb.params["hpy_d"]) - 1)
        assert(current_max_level == len(tssb_root.htssb.params["hpy_theta"]) - 1)
        assert(current_max_level == len(tssb_root.htssb.params["alpha_l"]) - 1)


cdef int node_level_recursive(object node, int current_max):
    cdef int temp

    if current_max < node.level:
        current_max = node.level

    for child in node.children:
        temp = node_level_recursive(child, current_max)

        if current_max < temp:
            current_max = temp

    return current_max
