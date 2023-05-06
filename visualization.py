"""Visualize ihLDA"""

# Load libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pathlib
import numpy.random as npr

# Load files
import tool


def fig_tssbs(ihLDA):
    """Save TSSBs as figures
    """

    def node_iterate(node, store):

        store = store_info(node, store)

        for child in node.children:
            store = node_iterate(child, store)

        return store

    def store_info(node, store):
        info = [
            node.level,
            node.prob_stop()
        ]
        store["length"].append(info)

        if node.level > store["deepest"]:
            store["deepest"] = node.level

        return store

    def save_fig(store, ihLDA, save_path):
        fig = plt.figure()
        ax = plt.axes()

        # Setting values
        height = 1
        interval = 0.4
        fill_ = True
        linewidth = 2.0  # edgewidth
        fillcolor = "#619CFF"
        depth_max = ihLDA.htssb.params["depth_max"]

        # Calculate the initial position and add a rectangle
        initial_y = (height + interval) * (depth_max) + interval
        initial_length = store["length"][0][1]
        plt.ylim(0, initial_y + height + interval)

        r = patches.Rectangle(
            xy=(0, initial_y), width=initial_length,
            height=height,
            edgecolor='#000000', linewidth=linewidth,
            facecolor=fillcolor,
            fill=fill_
        )
        ax.add_patch(r)

        # Add other nodes
        length_cumulative = initial_length
        for info in store["length"][1:]:
            level = info[0]
            length = info[1]

            x = length_cumulative
            y = initial_y - height - (height + interval) * level + height

            r = patches.Rectangle(
                xy=(x, y), width=length,
                height=height,
                edgecolor='#000000',
                facecolor=fillcolor,
                linewidth=linewidth,
                fill=fill_
            )
            ax.add_patch(r)

            length_cumulative += length

        # Remove lines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Remove ticks
        plt.tick_params(
            axis="x",
            top=False,
            bottom=False,
            labeltop=True,
            labelbottom=False
         )
        plt.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False
        )

        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure

    # Prepare a save folder
    save_folder = pathlib.Path(ihLDA.save_info["output_folder"]).resolve() / "fig_tssb"
    tool.folder_check(str(save_folder))

    # Make a TSSBs list
    check_tssbs = ihLDA.htssb.tssb_docs.copy()
    check_tssbs.append(ihLDA.htssb.tssb_root)

    for tssb in check_tssbs:

        # Check if randomly save figures
        if ihLDA.save_info["figure_save_prob"] != 1 and tssb.doc_id is not None:
            u = npr.uniform(0, 1)
            if ihLDA.save_info["figure_save_prob"] <= u:
                continue

        store = {
            "deepest": 0,
            "length": []
        }

        store = node_iterate(tssb.node_root, store)

        # Name
        if tssb.doc_id is None:
            save_name = "RootTSSB"
        else:
            save_name = "Doc" + str(tssb.doc_id) +\
                        "_" + ihLDA.data.get_filename_from_docid(tssb.doc_id)

        save_name = save_name + ".pdf"
        save_path = str(save_folder /
                        pathlib.Path(save_name))

        save_fig(store, ihLDA, save_path)
