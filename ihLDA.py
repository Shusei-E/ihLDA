# ihLDA.py
""" Main module of ihLDA

This modules contains :py:class:`ihLDA` class that
manages the main components of the model
including visualization functions.

Important functions are:
    * Create HTSSB
    * Run iteration
    * Calculate perplexity
    * Save results
    * Call visualization
"""

# Load libraries
import pathlib
import pandas as pd
import pickle
import sys
import os
import time

# Load files
import dataread
import htssb
import ihLDA_cy
import save_model
import tool
import visualization


class ihLDA:
    """ihLDA class

    This controls the whole model

    Arguments:
        * parser: parsed command line arguments
    """

    def __init__(self, main, parser_args):
        self.main = main
        self.data_folder = parser_args.data_folder
        self.data_object = parser_args.data_object
        self.output_path = parser_args.output_path
        self.iter_num = int(parser_args.iter_num)
        self.depth_max = int(parser_args.depth_max)
        self.si_direction = int(parser_args.si_direction)
        self.iter_add_later = int(parser_args.iter_add_later)
        self.filename_pkl = main.filename_pkl  # a file for the pickle object

        # Initialization
        self.initialize()

    def initialize(self):
        """Initialize ihLDA

        1. Read data
        2. Instantiate HTSSB class
        """

        # Place to store information
        self.save_info = {
            "output_folder": self.output_path,
            "figure_save_prob": 0,  # randomly store figure (Save all: 1)
            "iter_finished": 0,  # finished iteration
            # Tree depth
            "depth_max": self.depth_max,  # maximum depth of the tree
            # Scale-invariance direction
            "Scale-Invariance": self.si_direction,  # 1: both, 0: horizontal only
            # Perplexity
            "perplexity_iter": [],  # iteration numbers that keep perplexity
            "perplexity": [],  # train perplexity
            "perplexity_test": [],  # test perplexity
            # Hyperparameters
            "param_iter": [],  # iteration numbers that keep parameters
            "alpha0": [],
            "lambda": [],
            "lambda_h": [],
            "lambda_h_single": [],
            # Save gamma
            "gamma": [],  # single gamma
            "gamma0": [],  # gamma by level
            "gamma_h": [],  # gamma decay by level
            "gamma0_iter": [],
            "gamma0_level": [],
            # Topic-Word distribution
            "num_topwords": 200,  # the number of topwords to save
            "num_topics": [],
            # For parameters (newer version)
            "iteration": [],
            "parameter": [],  # name of the parameter
            "level": [],  # level of the parameter (None if level invariant)
            "value": [],  # value of thee parameter
            # Sampler
            "alpha_l_mh": [0, 0]  # [accept, reject]
        }

        self.save_info.update(
            {
                "hpy_iter": [],
                "hpy_level": [],
                "hpy_d": [],
                "hpy_theta": []
            }
        )

        # Settings for inference
        self.iter_setting = {
            "save_model": 1,
            "save_model_interval": 1000,  # how often do we save the model
            # Initialization
            "initialize_mode": "dgp",  # dgp or random
            # Other settings
            "ppl_cache": 1,  # Use cache when calculating perplexity
            "iter_start": 1,
            "iter_end": self.iter_num + 1,
            "time_total": 0,  # keep time
            "time_temp": 0,  # keep time
            "iter_startdt": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "iter_memo": ""  # memo for the iteration
        }

        # Make an output folder
        if not os.path.exists(self.save_info["output_folder"]):
            os.makedirs(self.save_info["output_folder"])

        # Read data
        self.data_read()

        # Instantiate HTSSB
        self.htssb = htssb.HTSSB(self)

        # Save settings
        self.save_settings()

    def save_settings(self):
        # Save settings
        params = self.htssb.params.copy()
        del params["g0"]

        params_str = str(params) + "\n\n" + str(self.iter_setting)
        save_folder = pathlib.Path(self.save_info["output_folder"]).resolve()
        save_path = str(save_folder /
                        pathlib.Path("settings.txt"))
        del params

        with open(save_path, "w") as f:
            f.write(params_str)


    def save_filenames(self):
        # Save doc_id and filenames
        doc_id = list(range(len(self.data_files)))
        filenames = [self.data.docid_to_filename[id_] for id_ in doc_id]
        id_file = pd.DataFrame({"doc_id": doc_id, "filename": filenames})

        save_folder = pathlib.Path(self.save_info["output_folder"]).resolve()
        save_path = str(save_folder /
                        pathlib.Path("filenames.csv"))
        id_file.to_csv(save_path, index=False)

    def data_read(self):
        """Read Data

        Read data (:py:mod:`dataread`)
        """

        if self.data_object != "" and os.path.exists(self.data_object):
            with open(self.data_object, "rb") as pkl:
                self.data = pickle.load(pkl)

            print("\033[94m" + "Loading the text data object: {filename} ({docs} documents).".format(filename = self.data_object, docs=len(self.data.documents)) + "\033[0m")

        else:
            self.data_files = self.get_files()
            self.data = dataread.DATA_READ(self.data_files, True)

            save_folder = pathlib.Path(self.save_info["output_folder"]).resolve()
            save_path = str(save_folder /
                            pathlib.Path("textdata.pkl"))

            with open(save_path, "wb") as f:
                pickle.dump(self.data, f)
            self.save_filenames()


    def get_files(self):
        """Get a list of files to read

        Returns:
            data_files: a list of files to read
        """

        folder = pathlib.Path(self.data_folder)

        if folder.exists():
            # Get a file list, make it an absolute path, and convert it to str
            data_files = [str(path.resolve()) for path in folder.glob("*.txt")]
            data_files.sort()

            if len(data_files) == 0:
                sys.exit("Error: Number of files found is 0")

        else:
            sys.exit("Error: Input folder doesn't exist")

        return data_files

    def iteration_run(self):
        """Run iterations

        Runs iteration, asks to add iterations, and\
                save figures before finishes.
        """

        ihLDA_cy.iteration(self,
                           self.iter_setting["iter_start"],
                           self.iter_setting["iter_end"])

        # Add iterations if needed
        if self.iter_add_later != 0:
            iter_add = -1
            while True:
                while iter_add < 0:
                    try:
                        iter_add = int(input("Adding iteration number (0 to finish): "))
                    except ValueError:
                        continue

                if iter_add <= 0:
                    break

                # Update Iteration number
                self.iter_setting["iter_start"] = self.iter_setting["iter_end"]
                self.iter_setting["iter_end"] =\
                    self.iter_setting["iter_start"] + iter_add

                iter_add = -1  # reset

                # Run additional iteration
                ihLDA_cy.iteration(self,
                                   self.iter_setting["iter_start"],
                                   self.iter_setting["iter_end"])

        # Create Figures before finish
        print("Saving TSSB figures...", end="")
        self.save_visualization()
        print("Done")

        return

    def save(self):
        """Save results

        1. Save topics
        2. Save topic assignments

        Parameters and perplexity are saved every time\
                they are estimated.
        """

        print("Saving sampled parameters and topics...", end="")
        self.save_params()
        print("Done")

        self.save_topics()

    def save_params(self):
        """Save estimated parameters

        Save estimated parameters in the file
        """

        # Hyperparameters
        param_dict = {
            "Iteration": self.save_info["param_iter"],
            "NumTopics": self.save_info["num_topics"]
        }

        param_column = ["Iteration", "NumTopics"]


        df_params = pd.DataFrame(param_dict, columns=param_column)

        save_folder = pathlib.Path(self.save_info["output_folder"]).resolve()
        save_path = str(save_folder /
                        pathlib.Path("info.csv"))
        df_params.to_csv(save_path, index=False)

        df_new = pd.DataFrame(
            {
              "Iteration": self.save_info["iteration"],
              "Parameter": self.save_info["parameter"],
              "Level": self.save_info["level"],
              "Value": self.save_info["value"]
            },
            columns=["Iteration", "Parameter", "Level", "Value"]
        )

        save_folder = pathlib.Path(self.save_info["output_folder"]).resolve()
        save_path = str(save_folder /
                        pathlib.Path("parameters.csv"))
        df_new.to_csv(save_path, index=False)

    def save_perplexity(self, iter_num, time_avg=None, print_str=""):
        """Calculate, show and save perplexity

        """

        self.save_info["perplexity_iter"].append(iter_num)

        # Perplexity
        perplexity = ihLDA_cy.calc_perplexity(self)
        self.save_info["perplexity"].append(perplexity)

        # Test Perplexity
        test_perplexity = ihLDA_cy.calc_testperplexity(self)
        self.save_info["perplexity_test"].append(test_perplexity)

        # Print Perplexity
        print("\033[95m[" + str(iter_num) + "]",
              "Train:", str(round(perplexity, 3)),
              "/ Test:", str(round(test_perplexity, 3)),
              "/ Avg Time:", "{time} sec/iter".format(time=str(time_avg)),
              print_str,
              "\033[0m")

        # Save Perplexity
        save_folder = pathlib.Path(self.save_info["output_folder"]).resolve()
        save_path = str(save_folder /
                        pathlib.Path("perplexity.csv"))
        lik_df = pd.DataFrame({
            "Iteration": self.save_info["perplexity_iter"],
            "Perplexity": self.save_info["perplexity"],
            "Test Perplexity": self.save_info["perplexity_test"]
            })
        lik_df.to_csv(save_path, index=False)

    def save_topics(self):
        """Save topic-word distribution

        """

        print("Organizing Top Words...", end="")
        ihLDA_cy.save_topic_word_dist(self)
        print("Done")

    def save_visualization(self):
        """Visualize trees and save

        """

        visualization.fig_tssbs(self)

    def save_model(self, i):
        save_folder = pathlib.Path(self.save_info["output_folder"]).resolve() / "model"
        tool.folder_check(str(save_folder))
        save_path = str(save_folder /
                        pathlib.Path("model_" + str(i) + ".pkl"))

        with open(save_path, "wb") as f:
            pickle.dump(save_model.make_pickle_obj(self), f)
