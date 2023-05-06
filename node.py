"""Node
Node in TSSB
"""

# Load Libraries
import numpy as np

# Load files
import hpy_cy
import node_cy


class NODE:
    """Node class
    Node in TSSB

    Arguments:
        * node_parent (NODE class): default is :code:`None`
        * my_tssb (TSSB class): TSSB class that this node belongs to

    Important Objects:
        * ihLDA
        * htssb
        * tssb_mine
        * tssb_parent: :code:`None` if it is a node in root TSSB
        * node_parent
        * words_dict: in Child TSSBs: it stores {word_id: count}. \
                If you use HPY and it is a node\
                in Parent TSSB: it stores {word_id: HPYTable()}.
        * tw_dist: a topic-word distribution you are going to use.\
                 See :code:`initialization()`.
    """

    def __init__(self, ihLDA, htssb,
                 tssb_mine, node_parent=None):
        """Initialize node

        """

        # Organize Arguments
        self.ihLDA = ihLDA
        self.htssb = htssb
        self.tssb_mine = tssb_mine
        self.tssb_parent = tssb_mine.tssb_parent
        self.node_parent = node_parent

        # Node parameters
        self.children = []  # node_children
        self.level = 0  # depth of the node
        self.path_stored = None  # set later
        self.index_in_parent = -1  # index in node_parent.children

        # Customers
        self.nv0 = 0  # stop vertically
        self.nv1 = 0  # pass vertically

        self.nh0 = 0  # stop horizontally
        self.nh1 = 0  # pass horizontally

        self.nu_table = []  # NodeCRP: nu
        self.psi_table = []  # NodeCRP: psi
        self.table_temp = []  # temp store, used in HPY as well

        # Corresponding nodes
        self.node_corres = None  # set later (tssb_cy.node_create())
        self.ref_corres = 0  # referenced as a corresponding node
        self.node_corres_lower = []  # corresponding nodes for root_tssb

        # Word
        self.num_data = 0
        self.words_dict = {}  # read explanation
        self.tw_dist = None  # topic-word distribution

        # Initialize
        self.initialize()

    def initialize(self):
        """Initialize node

        """

        # Level
        if self.node_parent is None:
            self.level = 0
        else:
            self.level = self.node_parent.level + 1

        # Update current max level
        if self.htssb.params["current_max"] < self.level:
            self.htssb.params["current_max"] = self.level

        # Set topic-word distribution
        if self.tssb_parent is None:
            # This is root node
            self.tw_dist = hpy_cy.HPY_CY(self)


    def data_add(self, word_id, word_position):
        """Call node_cy.data_add

        """

        return node_cy.data_add(self, word_id, word_position)

    def data_remove(self, word_id, word_position):
        """Call node_cy.data_remove

        """

        return node_cy.data_remove(self, word_id, word_position)

    def path(self):
        """Return Node path

        .. figure:: figures/path_tree.png
            :scale: 50 %

            Sample Path. The root node has [0].
        """

        return node_cy.path(self)

    def path_str(self):

        return "_".join([str(i) for i in node_cy.path(self)])

    def prob_stop(self):
        """Calculate Stop probability of the node

        Eq.(11) Mochihashi & Noji (2016)
        """

        return node_cy.prob_stop(self)
