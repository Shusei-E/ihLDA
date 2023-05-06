# cython: profile=True
# hpy_cy.pyx
# cython: language_level=3
"""Cythonized HPY

We use HPY for topic-word distributions.
This is at Parent TSSB level.
"""

# Load libraries
import numpy.random as npr
import numpy as np

# Load C functions
cdef extern from "math.h":
   double log (double x)
   double exp (double x)
   double lgamma (double x)
   double fmax(double x, double y)
   double fmin(double x, double y)
   double pow(double x, double y)

cdef class HPYTable:
    """Cythonized HPY Table

    This class manages tables for a word.
    """

    def __init__(self):
        self.tables = []  # a list for tables
        self.tuw = 0  # number of tables for a word `w` in node `u`
        self.cuw = 0  # number of word `w` in node `u`

cdef class HPY_CY:
    """Cythonized HPY class
    """

    cdef object node
    cdef public double tu  # sum of the number of tables
    cdef public double cu  # sum of the number of words = number of words on this node
    cdef public dict c0w  # special for root_node, to calculate perplexity

    def __init__(self, node):
        assert(node.tssb_parent is None)
        self.node = node
        self.tu = 0  # sum of the number of tables
        self.cu = 0  # sum of the number of words = number of words on this node
        self.c0w = {}  # special for root_node, to calculate perplexity

        # If du or theta_u lacks for this level, add them
        self.get_du()
        self.get_theta_u()

    cpdef data_add(self, word_id):
        """Add a word to HPY on this node
        """
        self.data_add_cy(word_id)

    cpdef data_remove(self, word_id):
        """Remove a word from HPY on this node
        """
        self.data_remove_cy(word_id)

    cpdef wordprob_py(self, word_id):
        """Calculate p(w)
        """
        return self.wordprob(word_id)

    """
    Calculate probabilities
    """
    cdef double wordprob_parent(self, word_id):

        if self.node.node_parent is not None:
            # If there is a parent node, return its prob
            # if not (= you're on parent node), there is no
            # need to use its (nonexisting) parent
            return self.node.node_parent.tw_dist.wordprob_py(word_id)
        else:
            # We are at the root_node
            return self.node.htssb.params["g0"][word_id]


    cdef double wordprob(self, word_id):
        """Return word probability
        """
        cdef double prob = 0.0
        cdef double first_term = 0.0
        cdef double second_coef = 0.0
        cdef double second_term = 0.0
        cdef double cuw
        cdef double tuw
        cdef double d = self.node.htssb.params["hpy_d"][self.node.level]
        cdef double theta = self.node.htssb.params["hpy_theta"][self.node.level]

        prob = 0.0
        first_term = 0.0
        second_coef = 0.0
        second_term = 0.0
        d = self.node.htssb.params["hpy_d"][self.node.level]
        theta = self.node.htssb.params["hpy_theta"][self.node.level]

        if self.node.node_parent is None:
            # Root node
            prob = self.node.htssb.params["g0"][word_id]

        elif word_id not in self.node.words_dict:
            # If word_id is not assigned in this topic
            # This happens when calculating perplexity
            second_coef = (theta + d * self.tu) / (theta + self.cu)
            prob = second_coef * self.wordprob_parent(word_id)

        else:
            cuw = self.node.words_dict[word_id].cuw
            tuw = self.node.words_dict[word_id].tuw

            first_term = (cuw - d * tuw) / (theta + self.cu)
            second_coef = (theta + d * self.tu) / (theta + self.cu)
            second_term = second_coef * self.wordprob_parent(word_id)

            prob = first_term + second_term


        return prob

    """
    Utilities
    """
    cdef double get_du(self):
        cdef int level = self.node.level

        # Add if the level is missing
        while len(self.node.htssb.params["hpy_d"])-1 < level:
            self.node.htssb.params["hpy_d"].append(0.5)

        return self.node.htssb.params["hpy_d"][level]

    cdef double get_theta_u(self):
        cdef int level = self.node.level

        # Add if the level is missing
        while len(self.node.htssb.params["hpy_theta"])-1 < level:
            self.node.htssb.params["hpy_theta"].append(2.0)

        return self.node.htssb.params["hpy_theta"][level]


    """
    Add word
    """

    cdef void data_add_cy(self, word_id):
        # If the node is root_node
        if self.node.level == 0:
            assert(self.node.node_parent is None)
            if word_id in self.c0w:
                self.c0w[word_id] += 1
            else:
                self.c0w[word_id] = 1

            return


        if word_id in self.node.words_dict:
            # Add to an existing table
            self.add_new_or_existing(word_id)
        else:
            # Add to a new table
            self.node.words_dict[word_id] = None
            self.add_new_table(word_id)

        return

    cdef void add_new_or_existing(self, word_id):
        """Decide where the word comes from

        The word comes from either this node\
                or the parent node (HPYLM p.17).
        """
        cdef double parent_Gw
        cdef double sum_prob
        cdef double normalizer
        cdef int k
        cdef double u
        cdef double stack
        cdef double d = self.node.htssb.params["hpy_d"][self.node.level]
        cdef double theta = self.node.htssb.params["hpy_theta"][self.node.level]
        cdef list tables = self.node.words_dict[word_id].tables
        cdef int tuw = self.node.words_dict[word_id].tuw

        d = self.node.htssb.params["hpy_d"][self.node.level]
        theta = self.node.htssb.params["hpy_theta"][self.node.level]
        tables = self.node.words_dict[word_id].tables
        tuw = self.node.words_dict[word_id].tuw

        # Probability in parent
        parent_Gw = self.wordprob_parent(word_id)

        # Consider discounted values
        sum_prob = 0.0
        for k in range(tuw):
            sum_prob += fmax(0.0, tables[k] - d)

        sum_prob += (theta + d * self.tu) * parent_Gw
        normalizer = 1.0 / sum_prob

        u = npr.uniform(0.0, 1.0)
        stack = 0.0
        assert(len(tables) == tuw)
        for k in range(tuw):
            stack += normalizer * fmax(0.0, tables[k] - d)
            if u <= stack:
                self.add_existing_table(word_id, k)
                return

        self.add_new_table(word_id)
        return


    cdef void add_existing_table(self, word_id, k):
        """Add a word to the existing table in the node

        Arguments:
            word_id: word_id
            k: index of the table (for the word)
        """

        self.node.words_dict[word_id].tables[k] += 1

        # Number of tables is the same
        # but the counter should be updated
        self.node.words_dict[word_id].cuw += 1
        self.cu += 1

        return


    cdef void add_new_table(self, word_id):
        """Add a word to the new table

        """

        # A word is new to the node
        if self.node.words_dict[word_id] is None:
            self.node.words_dict[word_id] = HPYTable()

        # Create a new table
        self.node.words_dict[word_id].tables.append(1)

        # Modify counter
        self.node.words_dict[word_id].tuw += 1  # new table for this word
        self.node.words_dict[word_id].cuw += 1  # increase a counter
        self.tu += 1  # new table on this node
        self.cu += 1  # a word added on this node

        # Send the word to its parent
        assert(self.node.node_parent is not None)
        self.node.node_parent.tw_dist.data_add(word_id)

        return


    """
    Remove word
    """

    cdef void data_remove_cy(self, word_id):
        cdef double normalizer
        cdef int k
        cdef double u
        cdef double stack

        if self.node.node_parent is None:
            # At root_node
            assert(self.node.level == 0)
            self.c0w[word_id] -= 1

            assert(self.c0w[word_id] >= 0)

            if self.c0w[word_id] == 0:
                del self.c0w[word_id]

            return

        """
        If this node is not the root_node
        remove proportionally
        """
        cdef list tables = self.node.words_dict[word_id].tables
        cdef int tuw = self.node.words_dict[word_id].tuw
        cdef double cuw = self.node.words_dict[word_id].cuw

        # Consider discounted values
        normalizer = 1.0 / cuw
        assert(cuw == sum(tables))

        u = npr.uniform(0.0, 1.0)
        stack = 0.0
        for k in range(tuw):
            stack += normalizer * tables[k]

            if u <= stack:
                self.node.words_dict[word_id].tables[k] -= 1
                assert(self.node.words_dict[word_id].tables[k] >= 0)

                self.node.words_dict[word_id].cuw -= 1
                assert(self.node.words_dict[word_id].cuw >= 0)

                self.cu -= 1
                assert(self.cu >= 0)

                # Check if remove from the parent
                if self.node.words_dict[word_id].tables[k] == 0:
                    self.node.node_parent.tw_dist.data_remove(word_id)

                    # Remove unoccupied table
                    self.node.words_dict[word_id].tables.pop(k)

                    self.node.words_dict[word_id].tuw -= 1
                    assert(self.node.words_dict[word_id].tuw >= 0)

                    self.tu -= 1
                    assert(self.tu >= 0)

                break

        return
