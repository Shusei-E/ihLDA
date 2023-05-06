# cython: profile=True
# save_model.pyx
# cython: language_level=3

import pandas as pd
import ihLDA_cy

cimport node_cy

"""Create an object for Pickel
    * Parent Tree with realized probabilities
    * Child Trees with realized probabilities
    * Topic information
"""

def make_pickle_obj(ihLDA):
    save_obj = {}
    save_obj["RootTSSB"] = make_pickle_RootTSSB(ihLDA.htssb.tssb_root)
    return save_obj


cdef dict make_pickle_RootTSSB(object tssb):
    cdef dict info = {}
    return pickle_tssb(tssb.node_root, info)


cdef dict pickle_tssb(object node, dict info):
    cdef list words = []
    cdef list topics = []
    cdef int word_id
    cdef int word_position
    cdef str path

    info["children"] = []
    info["nu"] = node_cy.nu(node)
    info["psi"] = node_cy.psi(node)
    info["path"] = node_cy.path_cy(node)
    info["pi"] = node_cy.prob_stop_cy(node)

    info["type"] = "parentTSSB"
    df_temp = pd_topic_word_dist(node)
    df_temp.sort_values(by=["Probability", "Word"],
                        ascending=[False, True],
                        inplace=True)
    info["tw_dist"] = df_temp

    # Recursively go down
    for child in node.children:
        info["children"].append(pickle_tssb(child, {}))

    return info


cdef object pd_topic_word_dist(object node):
    cdef list words_temp
    cdef list count_temp
    cdef list path_temp
    cdef list prob_temp
    cdef int word_id
    cdef str word
    cdef int count
    cdef str path

    path = "_".join([str(i) for i in node.path()])  # path in the root TSSB
    words_temp = []
    count_temp = []
    path_temp = []
    prob_temp = []

    if path == "0":
        assert(node.words_dict == {})

        for word_id in range(node.ihLDA.data.num_vocab):
            word = node.ihLDA.data.get_word_from_wordid(word_id)
            words_temp.append(word)

            prob_temp.append(node.tw_dist.wordprob_py(word_id))
            path_temp.append(path)

            if word_id in node.tw_dist.c0w:
                count_temp.append(node.tw_dist.c0w[word_id])
            else:
                count_temp.append(0)
    else:
        assert(node.tw_dist.c0w == {})

        for word_id in range(node.ihLDA.data.num_vocab):
            word = node.ihLDA.data.get_word_from_wordid(word_id)
            words_temp.append(word)

            prob_temp.append(node.tw_dist.wordprob_py(word_id))
            path_temp.append(path)

            if word_id in node.words_dict:
                count = node.words_dict[word_id].cuw
                count_temp.append(count)
            else:
                count_temp.append(0)

    df_temp = pd.DataFrame({
        "Path": path_temp,
        "Word": words_temp,
        "Count": count_temp,
        "Probability": prob_temp
        },
        columns=["Path", "Word", "Count", "Probability"])

    return(df_temp)
