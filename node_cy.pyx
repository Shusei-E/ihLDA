# cython: profile=True
# cython: language_level=3
# node_cy.pyx
"""Cythonized Node
Most functions are defined as :code:`cdef` and do not
appear in documents.
"""

"""
Functions related to updating node parameters
"""

# Load libraries
import numpy.random as npr
cimport numpy as np
import pathlib

# Load files
cimport tool

cdef extern from "math.h":
    double pow(double x, double y)

def alpha_(object node):
    return alpha(node)

cdef double alpha(object node):
    cdef double alpha
    cdef double alpha0 = -1.0
    cdef double decay = 1.0

    if node.ihLDA.si_direction == 1:
        # Scale-invariance to both directions
        if node.node_parent is not None:
            # node is not the root_node
            decay = vertical_length(node.node_parent)
        alpha0 = node.htssb.params["alpha"]
        alpha =  alpha0 * decay
    else:
        # alpha by level
        while len(node.htssb.params["alpha_l"])-1 < node.level:
            node.htssb.params["alpha_l"].append(0.1)

        alpha = node.htssb.params["alpha_l"][node.level] *\
                pow(node.htssb.params["lambda"], node.level)

    assert(alpha > 0)
    return alpha


cdef double vertical_length(object node):
    # the horizontal length of the node
    cdef double length = 1.0
    if node.node_parent is None:
        return nu(node)

    parent = node.node_parent
    length = nu(node)
    while parent:
        length *= 1.0 - nu(parent)
        parent = parent.node_parent

    assert(length > 0)

    return length


cdef double gamma(object node):
    cdef double gamma
    cdef double gamma0 = -1.0
    cdef double decay = 1.0

    ## decay
    ##   decay is the same as it's parent's horizontal length
    if node.node_parent is not None:
        decay = horizontal_length(node.node_parent)

    # gamma
    # Use single gamma
    gamma0 = node.htssb.params["gamma"]

    gamma = gamma0 * decay

    assert(gamma > 0)
    return gamma


cdef double horizontal_length(object node):
    # the horizontal length of the node
    cdef double length = 1.0
    if node.node_parent is None:
        return psi(node)

    assert(node.node_parent.children[node.index_in_parent] == node)

    for k in range(len(node.node_parent.children)):
        if k == node.index_in_parent:
            length *= psi(node.node_parent.children[k])  # stops
        else:
            length *= (1.0 - psi(node.node_parent.children[k]))  # passes

    assert(length > 0)

    return length


def nu_(object node):
    return nu(node)


cdef double calc_omega(object node, double value):
    if node == None:
        return value
    else:
        value *= (1.0 - nu(node))
        return calc_omega(node.node_parent, value)


cdef double nu(object node):
    cdef double aH = node.htssb.params["aH"]
    cdef double numerator
    cdef double denominator
    cdef double new_nu
    cdef double multiplier  # remaining length of the stick

    if node.tssb_parent is None:
        # Node in a root TSSB
        # Use Eq.(16) in Mochihashi & Noji (2016)
        new_nu = (1.0 + node.nv0) / (1.0 + alpha(node) + node.nv0 + node.nv1)
        return new_nu

    multiplier = calc_omega(node.node_corres.node_parent, 1.0)  # pass all parent nodes
    numerator = aH * nu(node.node_corres) * multiplier + node.nv0

    denominator = aH * multiplier + node.nv0 + node.nv1

    new_nu = numerator / denominator

    assert(new_nu <= 1.0)
    return new_nu


def psi_(object node):
    return psi(node)


cdef double calc_psi(object node, double value):
    if node.node_parent == None:
        return value
    else:
        for child in node.node_parent.children:
            if child == node:
                break
            else:
                value *= (1.0 - psi(child))

        return value


cdef double psi(object node):
    cdef double bH = node.htssb.params["aH"]
    cdef double numerator
    cdef double denominator
    cdef double new_psi
    cdef double multiplier  # remaining length of the stick

    if node.tssb_parent is None:
        # Node in a root TSSB
        # Use Eq.(17) in Mochihashi & Noji (2016)
        new_psi = (1.0 + node.nh0) / (1.0 + gamma(node) + node.nh0 + node.nh1)

        if not new_psi < 1:  # clipping gamma if it is tiny
            print("  psi: " + str(node.path()) + ", " + str(node.nh0) + ", " + str(node.nh1) + " / " + "{:.2e}".format(gamma(node)) + " / " + "{:.2e}".format(new_psi))
            # clip value of gamma
            new_psi = (1.0 + node.nh0) / (1.0 + 1e-9 + node.nh0 + node.nh1)

        return new_psi

    multiplier = calc_psi(node.node_corres, 1.0)
    numerator = bH * psi(node.node_corres) * multiplier + node.nh0

    denominator = bH * multiplier + node.nh0 + node.nh1

    if denominator < 1e-15:  # tiny denominator: (when no word uses this topic)
        denominator = 1e-9

    new_psi = numerator / denominator

    if new_psi >= 1.0:
        numerator = bH * psi(node.node_corres) * multiplier + node.nh0
        denominator = bH * multiplier + node.nh0 + 1.0
        new_psi = numerator / denominator

    assert(new_psi < 1.0)
    return new_psi


"""
Functions related to path
"""

def path(object node):
    """Return node path

    The idea is that path does not change once
    it is set in a TSSB.

    :py:meth:`tssb_cy.node_create` calles this first.
    Note that node is already added to its parent node
    when this function is called.
    """

    return path_cy(node)


cdef list path_cy(object node):
    cdef list path

    if node.path_stored is None:  # Node

        if node.node_parent is None:
            path = [0]
        else:
            path = path_cy(node.node_parent).copy()
            # path.append(len(node.node_parent.children)-1)
            path.append(node.node_parent.children.index(node))
                # The node is already added to its parent if there is.
                # That's why we need to subtract 1 from the length
                # to get its index.

        node.path_stored = path

    return node.path_stored


cdef list path_reset_cy(object node):
    # Ignore the curret stored path and get a new one
    # Works only if your parent has a correct path

    cdef list path

    if node.node_parent is None:
        path = [0]
    else:
        path = path_reset_cy(node.node_parent).copy()
        path.append(node.node_parent.children.index(node))

    node.path_stored = path

    return node.path_stored


"""
Add data
"""

def data_add(object node, int word_id, int word_position):
    """Add data to a node

    We need to do two things:
        1. Update a dictionary that stores word count
        2. Add a customer to a node CRP counter
        3. Add a word to HPY or Gaussian Kernel

    Arguments:
        node: a NODE class object
        word_id: word_id
    """

    return data_add_cy(node, word_id, word_position)

cdef void data_add_cy(object node, int word_id, int word_position):
    # Word Dictionary Update
    if word_id in node.words_dict:
        node.words_dict[word_id] += 1
    else:
        node.words_dict[word_id] = 1

    node.num_data += 1
    assert(node.num_data == sum(node.words_dict.values()))

    # Node Counter
    customer_add_vertical(node)
    customer_add_horizontal(node)

    # Add to node's topic-word distribution
    assert(node.node_corres is not None)
    node.node_corres.tw_dist.data_add(word_id)

    # Update Node Assignments
    node.tssb_mine.node_assignments[word_position] = node

    return

def data_remove(object node, int word_id, int word_position):
    """Remove data from a node

    We need to do two things:
        1. Update a dictionary that stores word count
        2. Remove a customer to a node CRP counter
        3. Remove a word from HPY or Gaussian Kernel

    Arguments:
        node: a node class object
        word_id: word_id
    """

    return data_remove_cy(node, word_id, word_position)

cdef void data_remove_cy(object node, int word_id, int word_position):
    # Word Dictionary Update
    assert(node is not None)
    assert(word_id in node.words_dict)

    node.words_dict[word_id] -= 1
    if node.words_dict[word_id] == 0:
        del node.words_dict[word_id]

    node.num_data -= 1
    assert(node.num_data == sum(node.words_dict.values()))

    # Node Counter
    customer_remove_vertical(node)
    customer_remove_horizontal(node)

    # Remove from node's topic-word distribution
    assert(node.node_corres is not None)
    node.node_corres.tw_dist.data_remove(word_id)


    return

"""
Count customers for node
"""

#
# Add my TSSB vertical
#
cdef void customer_add_vertical(object node):
    """Add a customer vertically

    1. Update my TSSB
    2. If my TSSB is not the root TSSB, consider sending\
            a proxy customer.
    """

    customer_add_vertical_mytssb(node)

    if node.tssb_parent is not None:
        """
        If tssb is not the root TSSB,
        consider sending a proxy customer
        """
        customer_add_vertical_tssbparent(node)

    return


cdef void customer_add_vertical_mytssb(object node):
    """Add a vertical customer to my TSSB

    1. Add 1 to the stop counter of myself
    2. Add 1 to the pass counter of all parents (= ansestors)
    """

    # Stop at my self
    node.nv0 += 1

    # Add up parents until reaches to the root
    customer_add_vertical_mytssb_parents(node.node_parent)


cdef void customer_add_vertical_mytssb_parents(object node):
    if node is None:
        return

    node.nv1 += 1
    customer_add_vertical_mytssb_parents(node.node_parent)


#
# Add my TSSB horizontal
#
cdef void customer_add_horizontal(object node):
    """Add a customer to a horizontal CDP
    """

    customer_add_horizontal_mytssb(node)


cdef void customer_add_horizontal_mytssb(object node):
    """Add a horizontal customer to my TSSB

    1. Update counter of my level
    2. Update upper level counter
    """

    if node.node_parent is None:
        # At the node_root
        node.nh0 += 1
    else:
        for node_passed in node.node_parent.children:
            if node_passed == node:
                # Actually, it stops
                node.nh0 += 1
                break

            # Passed this node
            node_passed.nh1 += 1

    # Whenever it stops, check whether to
    # send a proxy customer
    if node.tssb_parent is not None:
        customer_add_horizontal_tssbparent(node)

    # Go to its parent
    if node.node_parent is not None:
        customer_add_horizontal_mytssb(node.node_parent)


#
# Remove my TSSB vertical
#
cdef void customer_remove_vertical(object node):
    customer_remove_vertical_mytssb(node)

    if node.tssb_parent is not None:
        """
        If tssb is not the root TSSB,
        consider removing a proxy customer
        """
        customer_remove_vertical_tssbparent(node)


cdef void customer_remove_vertical_mytssb(object node):
    """Remove a vertical customer from my TSSB

    1. Remove 1 to the stop counter of myself
    2. Remove 1 to the pass counter of all parents (= ansestors)
    """

    # Stop at my self
    node.nv0 -= 1
    assert(node.nv0 >= 0)

    # Remove up parents until reaches to the root
    customer_remove_vertical_mytssb_parents(node.node_parent)


cdef void customer_remove_vertical_mytssb_parents(object node):
    if node is None:
        return

    node.nv1 -= 1
    assert(node.nv1 >= 0)
    customer_remove_vertical_mytssb_parents(node.node_parent)


#
# Remove my TSSB horizontal
#
cdef void customer_remove_horizontal(object node):

    customer_remove_horizontal_mytssb(node)


cdef void customer_remove_horizontal_mytssb(object node):
    """Remove a horizontal customer from my TSSB

    1. Update counter of my level
    2. Update upper level counter
    """

    if node.node_parent is None:
        # At the node_root
        node.nh0 -= 1
    else:
        for node_passed in node.node_parent.children:
            if node_passed == node:
                # It originally stopped here
                node.nh0 -= 1
                assert(node.nh0 >= 0)
                break

            # Passed this node
            node_passed.nh1 -= 1
            assert(node.nh1 >= 0)

    # Check whether or not to remove
    # the proxy customer
    if node.tssb_parent is not None:
        customer_remove_horizontal_tssbparent(node)

    # Go to its parent
    if node.node_parent is not None:
        customer_remove_horizontal_mytssb(node.node_parent)


#
# Add parent TSSB vertical
#
cdef void customer_add_vertical_tssbparent(object node):
    """Add a vertical customer to its parent TSSB if necessary

    1. Decide whether we need to create a proxy customer
    2. Add a table to a special CRP if we add a customer to our\
            parent TSSB. Otherwise, we proportionally select one\
            of the existing tables and add a customer.
    """
    ## Decide whether or not to send a proxy customer
    cdef double aH = node.htssb.params["aH"]
    cdef double corres_nu = nu(node.node_corres)
    cdef int index

    node.table_temp = node.nu_table.copy()  # clear the table
    node.table_temp.append( aH * corres_nu )

    index = tool.multi_index(node.table_temp, sum(node.table_temp))

    if(index < len(node.table_temp) - 1):
        node.nu_table[index] += 1
    else:
        node.nu_table.append(1)
        customer_add_vertical(node.node_corres)


#
# Add parent TSSB horizontal
#
cdef void customer_add_horizontal_tssbparent(object node):
    """Add a horizontal customer to its parent TSSB if necessary

    1. Decide whether we need to create a proxy customer
    2. Add a table to a special CRP if we add a customer to our\
            parent TSSB. Otherwise, we proportionally select one\
            of the existing tables and add a customer.
    """

    cdef double bH = node.htssb.params["aH"]
    cdef double corres_psi = psi(node.node_corres)
    cdef int index

    node.table_temp = node.psi_table.copy()
    node.table_temp.append( bH * corres_psi )

    index = tool.multi_index(node.table_temp, sum(node.table_temp))

    if(index < len(node.table_temp) - 1):
        node.psi_table[index] += 1
    else:
        node.psi_table.append(1)
        customer_add_horizontal(node.node_corres)


#
# Remove parent TSSB vertical
#
cdef void customer_remove_vertical_tssbparent(object node):
    cdef int index = tool.multi_index(node.nu_table, sum(node.nu_table))

    node.nu_table[index] -= 1

    if node.nu_table[index] == 0:
        del node.nu_table[index]

        assert(node.tssb_parent is not None)
        assert(node.node_corres is not None)
        customer_remove_vertical(node.node_corres)


#
# Remove parent TSSB horizontal
#
cdef void customer_remove_horizontal_tssbparent(object node):
    cdef int index = tool.multi_index(node.psi_table, sum(node.psi_table))

    node.psi_table[index] -= 1

    if node.psi_table[index] == 0:
        del node.psi_table[index]

        assert(node.tssb_parent is not None)
        assert(node.node_corres is not None)
        customer_remove_horizontal(node.node_corres)


"""
Node probabilities
"""

def prob_stop(node):
    """Calculate the stop probability

    """

    return prob_stop_cy(node)

cdef double prob_stop_cy(object node):
    cdef double prob = nu(node)

    return prob_stop_recursive(node, prob)

cdef double prob_stop_recursive(object node, double prob):
    cdef object child

    if node.node_parent is None:
        # Reached the root
        return prob

    prob *= (1.0 - nu(node.node_parent)) # vertically passed parent

    for child in node.node_parent.children:
        if node == child:
            prob *= psi(child)
            return prob_stop_recursive(node.node_parent, prob)

        else:
            prob *= (1.0 - psi(child))
                # prob of horizontally passing siblings


cdef double prob_stop_horizontal_cy(object node):
    """Calculate the horizontal stop probability

    This is used in size-biased permutation
    """
    cdef double prob = 1.0

    for child in node.node_parent.children:
        if node == child:
            prob *= psi(child)
            return prob
        else:
            prob *= (1.0 - psi(child))

    return prob
