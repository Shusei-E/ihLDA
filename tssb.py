""" Tree-Structured Stick-beraking Process

A module for TSSB
"""

# Load Libraries
import numpy.random as npr

# Load files
import tssb_cy


class TSSB:
    """
    TSSB class

    Methods:
        * Initialize
            * Assign data to node
        * Data management
            * data_add
            * data_remove
        * Node management
            * find_node
            * cull_node_root
            * cull_node_branch

    Important objects:
        * self.doc_loglik: Log-likelihood of the document

    Arguments:
        * ihLDA: ihLDA class object
        * htssb: HTSSB class object
        * tssb_parent (default: :code:`None`): tssb_root
        * doc_id (default: :code:`None`): should be positive number if it is not root
    """

    def __init__(self,
                 ihLDA,
                 htssb,
                 tssb_parent=None,
                 doc_id=None,
                 ):

        self.ihLDA = ihLDA
        self.htssb = htssb

        # TSSB Info
        self.tssb_parent = tssb_parent
        self.doc_id = doc_id
        self.doc_len = -1  # Initialized in initialize_cy
        self.node_list = []  # Nodes that this TSSB has
        self.node_assignments = []  # corresponds to doc order, initialized later
        self.node_root = None  # initialized in initialize_cy_random or initialize_cy_dgp

        # Initialize
        tssb_cy.initialize(self, mode=ihLDA.iter_setting["initialize_mode"])

    def data_add(self, target_node, word_id, word_position):
        """Add data to TSSB

        Arguments:
            * target_node (node class): which node to add
            * word_id (int): word_id in the corpus
        """

        target_node.data_add(word_id, word_position)

    def data_remove(self, target_node, word_id, word_position):
        """Remove data from TSSB

        Arguments:
            * target_node (node class): from which node to remove
            * word_id (int) word_id in the corpus
        """

        target_node.data_remove(word_id, word_position)
