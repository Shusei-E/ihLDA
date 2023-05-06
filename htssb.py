""" Hierarchical Tree-Structured Stick-breaking Process

A module for HTSSB
"""

# Load libraries
import sys

# Load files
import htssb_cy
import tssb_cy
from tssb import TSSB


class HTSSB:
    """
    HTSSB class for managing a root TSSB and children TSSB

    Methods:
        * Initialize
        * Initialize tssb_docs

    Arguments:
        ihLDA: ihLDA class

    Important objects:
        * self.tssb_docs: TSSB for documents
        * self.params: HTSSB, TSSB, and inference parameters
    """

    def __init__(self, ihLDA):
        self.ihLDA = ihLDA

        # HTSSB parameters
        self.tssb_docs = []  # TSSB for documents

        # Initialize
        self.initialize()

    def initialize(self):
        """Initialize HTSSB

        1. Initialize parameters
        2. Create a tssb_root
        3. Assign words into TSSB

        Prameters are stored in a dictionary :code:`self.param`.

        Initial parameters are:
            * tssb_alpha: :math:`\\nu \\sim {\\rm Be}(1, \\alpha)`
            * tssb_gamma: :math:`\\psi \\sim {\\rm Be}(1, \\gamma)`
            * tssb_lambda: :math:`\\alpha(\\epsilon) =`\
            :math:`\\alpha_0 \\cdot \\lambda^{|\\epsilon|}`
            * depth_min
            * depth_max
            * mode

        :code:`lambda` and :code:`lambda_h` are decay parameters.

        If the mode is :code:`hpy`, we have following additional parameters:
            * hpy_d: Initially, it has a value for the root level.\
                    :code:`get_du()` function in :py:mod:`dirtree`\
                    can initialize a value for the new level
            * hpy_theta
        """

        # Initialize parameters (make sure all parameters have the same length)
        # The first element is for the root level so it does not matter.
        self.params = {
                "aH": 0.1,  # HTSSB parameter, `a` and `b` in the paper
                # Sampling Parameters
                "alpha": 2.0,  # single alpha (si_direction: both)
                "alpha_l": [25.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # (si_direction: horizontal)
                "gamma": 3.0,  # single gamma
                "lambda": 0.7,  # vertical decay
                "depth_min": 0,
                "depth_max": self.ihLDA.depth_max,  # only root = 0
                "child_max": [15, 10, 10, 10, 10, 10],  # can change by level
                "current_max": 0  # current max level
                }

        # Add parameters for HPY
        # We estimate this parameter
        self.params.update(
            {
                # d and theta can change by level
                "hpy_d": [1, 0.5, 0.4, 0.3, 0.2, 0.1],  # discount parameter
                "hpy_theta": [1, 200, 130, 60, 50, 40],  # concentration parameter
            }
        )
        self.params["g0"] = self.ihLDA.data.get_g0()

        # Initialize the root TSSB
        self.tssb_root = TSSB(self.ihLDA, self)

        # Initialize TSSBs
        for doc_id in range(self.ihLDA.data.doc_num):
            tssb = TSSB(self.ihLDA, htssb=self,
                        tssb_parent=self.tssb_root, doc_id=doc_id)
            self.tssb_docs.append(tssb)
