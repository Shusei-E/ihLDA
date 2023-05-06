# main.py
"""Main class file of ihLDA

This file contains a main function.
It reads path and initialize the model.
"""

# Load libraries
import argparse
import numpy.random as npr
import sys

# Load files
import ihLDA

# Saving the object
import pickle
import os

# Set seed
npr.seed(225)

# Adjust recursive time
sys.setrecursionlimit(1500)


class MAIN:
    """Main class of ihLDA

    The class firstly made when we run the model.

    * Get arguments from command line
    * Instantiate ihLDA class
    """

    def __init__(self):
        self.get_args()
        self.filename_pkl = self.parser_args.output_path + "/model_temp.pkl"  # temp model file name
        self.initialize()

        # Iteration
        self.ihLDA.iteration_run()

    def get_args(self):
        """Get command line arguments
        """

        parser = argparse.ArgumentParser()
        parser.add_argument("--data_folder",
                            default=("./input/sample/"),
                            help="an absolute path to a folder with .txt files")
        parser.add_argument("--data_object",
                            default="",
                            help="a data object to load instead of text files")
        parser.add_argument("--output_path",
                            default=("./output/"),
                            help="an absolte path to an output folder")
        parser.add_argument("--iter_num", default="10000",
                            help="number of iteration")
        parser.add_argument("--depth_max", default="3",  # 2: root-first-second (existing models)
                            help="Maximum depth of the tree")
        parser.add_argument("--si_direction", default="1",  # 1: vertical and horizontal, 0: horizontal only
                            help="Scale-Invariance direction")
        parser.add_argument("--iter_add_later", default="0",  # 1:add_later
                            help="Check whether run additional iterations")
        self.parser_args = parser.parse_args()

    def initialize(self):
        """Initialize model

        Create ihLDA class (:py:mod:`ihLDA`)
        """

        if os.path.exists(self.filename_pkl):
            with open(self.filename_pkl, "rb") as pkl:
                self.ihLDA = pickle.load(pkl)
            self.ihLDA.iter_setting["iter_start"] = self.ihLDA.save_info["iter_finished"] + 1
            iter_remain = self.ihLDA.iter_setting["iter_end"] - self.ihLDA.iter_setting["iter_start"]

            print("\033[94m" + "Loading the temp object ({iter} remained iterations). Initialized date: {date}.".format(iter=iter_remain, date=self.ihLDA.iter_setting["iter_startdt"]) + "\033[0m")
        else:
            self.ihLDA = ihLDA.ihLDA(self, self.parser_args)

if __name__ == '__main__':
    main = MAIN()
