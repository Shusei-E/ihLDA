# cython: profile=True
# cython: language_level=3
# ihLDA_cy.pyx
"""Cythonized functions for ihLDA

"""

# Load libraries
from cpython cimport bool
import numpy.random as npr
import numpy as np
cimport numpy as np
import pandas as pd
import pathlib
import time

# Saving the object
import pickle
import os

# Load files
cimport node_cy
cimport save_model
cimport tool
cimport tssb_cy

# Load C functions
cdef extern from "math.h":
   double log (double x)
   double exp (double x)
   double lgamma (double x)
   double pow(double x, double y)
   double fmin(double x, double y)


"""
Iteration
"""

def iteration(object ihLDA, int iter_start, int iter_end):
    """Run Iteration
    """

    cdef int i
    cdef double time_avg
    cdef int num_topics_prev = -1
    cdef str print_str

    # Check initialized model
    if iter_start == 1:
        print("// Initial Perplexity //")
        ihLDA.save_perplexity(0, 0)

    print("// Start Iteration //")
    ihLDA.iter_setting["time_total"] = time.time()
    ihLDA.iter_setting["time_temp"] = time.time()

    # Remove unncessary nodes
    tssb_cy.node_cull_cy(ihLDA.htssb.tssb_root)

    for i in range(iter_start, iter_end):
        iteration_single(ihLDA, i)

        # Save parameters
        if i == 1:
            time_passed = round((time.time() - ihLDA.iter_setting["time_temp"]), 1)
            ihLDA.save_perplexity(i, time_passed)
            ihLDA.save_params()

        elif i == iter_end - 1 and i % 10 != 0:
            count = i % 10
            time_avg = round((time.time() - ihLDA.iter_setting["time_temp"])/count, 1)
            ihLDA.save_perplexity(i, time_avg)

        elif i % 10 == 0:
            time_avg = round((time.time() - ihLDA.iter_setting["time_temp"])/10, 1)
            ihLDA.save_perplexity(i, time_avg)
            ihLDA.save_params()

            ihLDA.iter_setting["time_temp"] = time.time()

        ihLDA.save_info["iter_finished"] = i  # update the finished iteration

        # Save model
        if ihLDA.iter_setting["save_model"] and (i == 1 or i % ihLDA.iter_setting["save_model_interval"] == 0):
            ihLDA.save_model(i)
            ihLDA.save_topics()

        # Saving the temp object for resume
        if i == 1 or i % 10 == 0:
            with open(ihLDA.filename_pkl, "wb") as pkl:
                pickle.dump(ihLDA, pkl)
        if i == iter_end - 1:
            ihLDA.save_model(i)


    time_total = round(time.time() - ihLDA.iter_setting["time_total"], 1)
    print("\033[92m" + "Total time: " + str(time_total) +\
            " seconds for {iter} times".format(iter=str(iter_end-iter_start)) + "\033[0m")

    # Save (parameters and topics)
    ihLDA.save()

    return


cdef void iteration_single(object ihLDA, int iter_index):
    """Single iteration
    Run a single sampling.

    """
    cdef str print_output = "  "
    cdef str print_numtopics = ""

    ### Sampling

    # Topic Assignment: Slice Sampling
    sample_topicassign(ihLDA, iter_index)

    # Cull nodes
    tssb_cy.node_cull_cy(ihLDA.htssb.tssb_root)

    # alpha
    if ihLDA.si_direction == 1:
        sample_alpha(ihLDA)
        ihLDA.save_info["iteration"].append(iter_index)
        ihLDA.save_info["parameter"].append("alpha")
        ihLDA.save_info["level"].append(None)
        ihLDA.save_info["value"].append(ihLDA.htssb.params["alpha"])
    else:
        # horizontal only
        sample_level_alpha(ihLDA)
        for i in range(len(ihLDA.htssb.params["alpha_l"])):
            ihLDA.save_info["iteration"].append(iter_index)
            ihLDA.save_info["parameter"].append("alpha_l")
            ihLDA.save_info["level"].append(i)
            ihLDA.save_info["value"].append(ihLDA.htssb.params["alpha_l"][i])

    # lambda
    if ihLDA.si_direction == 1:
        sample_lambda(ihLDA)
        ihLDA.save_info["iteration"].append(iter_index)
        ihLDA.save_info["parameter"].append("lambda")
        ihLDA.save_info["level"].append(None)
        ihLDA.save_info["value"].append(ihLDA.htssb.params["lambda"])

    # gamma
    sample_gamma(ihLDA)
    ihLDA.save_info["iteration"].append(iter_index)
    ihLDA.save_info["parameter"].append("gamma")
    ihLDA.save_info["level"].append(None)
    ihLDA.save_info["value"].append(ihLDA.htssb.params["gamma"])

    # HPY
    sample_hpy_d_theta(ihLDA)
    # Save info
    for i in range(len(ihLDA.htssb.params["hpy_d"])):
        ihLDA.save_info["iteration"].append(iter_index)
        ihLDA.save_info["parameter"].append("hpy_d")
        ihLDA.save_info["level"].append(i)
        ihLDA.save_info["value"].append(ihLDA.htssb.params["hpy_d"][i])

        ihLDA.save_info["iteration"].append(iter_index)
        ihLDA.save_info["parameter"].append("hpy_theta")
        ihLDA.save_info["level"].append(i)
        ihLDA.save_info["value"].append(ihLDA.htssb.params["hpy_theta"][i])

    # HTSSB Hyper Parameters
    sample_htssb_hyper(ihLDA, iter_index)
    ihLDA.save_info["iteration"].append(iter_index)
    ihLDA.save_info["parameter"].append("HTSSB_hyper")
    ihLDA.save_info["level"].append(None)
    ihLDA.save_info["value"].append(ihLDA.htssb.params["aH"])

    # Save values
    ihLDA.save_info["param_iter"].append(iter_index)
    ihLDA.save_info["lambda"].append(ihLDA.htssb.params["lambda"])
    ihLDA.save_info["num_topics"].append(len(ihLDA.htssb.tssb_root.node_list))

    # Print values
    print_numtopics = "NumTopics: " + str(len(ihLDA.htssb.tssb_root.node_list))
    print_maxlevel = " / MaxLevel: " + str(ihLDA.htssb.params["current_max"])

    print(print_output + print_numtopics + print_maxlevel)


"""
Sample Topic Assignment (Slice Sampling)
"""

cdef void sample_topicassign(object ihLDA, int iter_index):
    """Topic Reassignment
    """
    cdef list tssb_ids = list(range(ihLDA.data.doc_num))
    cdef int doc_id
    cdef object tssb
    assert(len(tssb_ids) == len(ihLDA.data.documents))
    npr.shuffle(tssb_ids)

    cdef object node_current
    cdef object node_new

    cdef list word_indexes
    cdef int word_position

    cdef double llk_current
    cdef double llk_new

    cdef list path_current
    cdef list path_new
    cdef int path_compare

    cdef double u_min
    cdef double u_max
    cdef double u

    for doc_id in tssb_ids:
        tssb = ihLDA.htssb.tssb_docs[doc_id]

        word_indexes = list(range(ihLDA.data.each_doc_len[doc_id]))
        npr.shuffle(word_indexes)

        for word_position in word_indexes:
            word_id = ihLDA.data.documents[doc_id][word_position]
            node_current = tssb.node_assignments[word_position]

            # Remove Data
            node_cy.data_remove_cy(node_current, word_id, word_position)

            # Create a slice
            llk_current = llk_word(node_current, word_id) + log(npr.rand())
            u_min = 0.0
            u_max = 1.0

            while True:
                u = (u_max - u_min) * npr.rand() + u_min
                node_new = tssb_cy.find_node_cy(u, tssb.node_root, tssb)

                llk_new = llk_word(node_new, word_id)

                if llk_new > llk_current:
                    break
                elif abs(u_max - u_min) < 1e-10:
                    node_new = node_current
                    break
                else:
                    path_current = node_cy.path_cy(node_current)
                    path_new = node_cy.path_cy(node_new)

                    if path_current == path_new:
                        node_new = node_current
                        break

                    path_compare = tssb_cy.path_compare_cy(path_current, path_new)

                    if path_compare == 0:
                        # path_current < path_new
                        u_max = u
                    elif path_compare == 1:
                        # path_new < path_current
                        u_min = u

            # Add data again
            node_cy.data_add_cy(node_new, word_id, word_position)
            # Cull
            if node_current.num_data == 0:
                if node_new.node_parent is not None:
                    tssb_cy.node_cull_cy(tssb, node=node_new.node_parent)
                    tssb_cy.node_cull_cy(ihLDA.htssb.tssb_root,
                                         node=node_new.node_parent.node_corres)

        tssb_cy.node_cull_cy(tssb)

    return


cdef double llk_word(object node, int word_id):
    cdef double llk = 0.0

    llk = log(node.node_corres.tw_dist.wordprob_py(word_id))

    return llk


cdef double prob_word(object node, int word_id):
    return(node.node_corres.tw_dist.wordprob_py(word_id))



"""
Sample alpha (using scale-invariance in both vertical and horizontal directions)
"""
cdef void sample_alpha(object ihLDA, int shrink_max=1000):
    cdef double alpha_current = ihLDA.htssb.params["alpha"]
    cdef double alpha_new
    cdef double alpha_use = alpha_current

    # Slice sampling
    cdef double start
    cdef double end
    cdef double p_old
    cdef double p_new
    cdef double llk_sliced
    cdef double llk_new
    cdef int shrink_time

    start = 0.0
    end = tool.shrinkp(30.0)
    p_old = tool.shrinkp(alpha_current)
    llk_sliced = llk_alpha(ihLDA, alpha_current) - 2.0 * log(1.0 - p_old) + log(npr.uniform())

    for shrink_time in range(shrink_max):
        p_new = npr.uniform(start, end)
        alpha_new = tool.expandp(p_new)
        ihLDA.htssb.params["alpha"] = alpha_new

        llk_new = llk_alpha(ihLDA, alpha_new) - 2.0 * log(1.0 - p_new)

        if llk_sliced < llk_new:
            break
        elif p_old < p_new:
            end = p_new
        elif p_new < p_old:
            start = p_new
        else:
            print("sample_alpha: shrinked too much")
            ihLDA.htssb.params["alpha"] = alpha_current
            break

    return

cdef double llk_alpha(object ihLDA, double alpha):
    return llk_alpha_descend(ihLDA.htssb.tssb_root.node_root) +\
            tool.gammapdfln(alpha, 1.5, 1.0)  # prior

cdef double llk_alpha_descend(object node):
    cdef double llk = 0.0
    cdef double alpha = node_cy.alpha(node)

    llk += log(alpha) +\
           lgamma(node.nv0 + 1.0) +\
           lgamma(node.nv1 + alpha) -\
           lgamma(node.nv0 + node.nv1 + alpha + 1.0)

    for child in node.children:
        llk += llk_alpha_descend(child)

    return llk


"""
Sample alpha_l (using scale-invariance only in horizontal direction)
"""
cdef void sample_level_alpha(object ihLDA, int shrink_max=1000):
    cdef double alpha_current
    cdef double alpha_new

    # Slice Sampling
    cdef double start
    cdef double end
    cdef double p_old
    cdef double p_new
    cdef double llk_sliced
    cdef double llk_new
    cdef int shrink_time

    cdef list levels = list(range(ihLDA.htssb.params["current_max"] + 1))

    for level in levels:
        start = 0.0
        end = tool.shrinkp(30.0)
        alpha_current = ihLDA.htssb.params["alpha_l"][level]
        p_old = tool.shrinkp(alpha_current)

        llk_sliced = llk_alpha_l(ihLDA, alpha_current, level) -\
                     2.0 * log(1.0 - p_old) + log(npr.uniform())

        for shrink_time in range(shrink_max):
            p_new = npr.uniform(start, end)
            alpha_new = tool.expandp(p_new)

            llk_new = llk_alpha_l(ihLDA, alpha_new, level) - 2.0 * log(1.0 - p_new)

            if llk_sliced < llk_new:
                ihLDA.htssb.params["alpha_l"][level] = alpha_new
                break
            elif p_old < p_new:
                end = p_new
            elif p_new < p_old:
                start = p_new
            else:
                print("sample_alpha_l: shrinked too much")
                ihLDA.htssb.params["alpha_l"][level] = alpha_current
                break

    return


cdef double llk_alpha_l(object ihLDA, double alpha, int level):
    return llk_alpha_l_descend(ihLDA.htssb.tssb_root.node_root,
                               alpha,
                               ihLDA.htssb.params["lambda"],
                               level) +\
            tool.gammapdfln(alpha, 1.5, 1.0)  # prior


cdef double llk_alpha_l_descend(object node, double alpha, double lambda_, int level):
    cdef double llk = 0.0
    cdef double alpha_l

    if node.level == level:
        # Only consider a particular level
        alpha_l = pow(lambda_, node.level) * alpha

        llk += log(alpha_l) +\
               lgamma(node.nv0 + 1.0) +\
               lgamma(node.nv1 + alpha_l) -\
               lgamma(node.nv0 + node.nv1 + alpha_l + 1.0)

    for child in node.children:
        llk += llk_alpha_l_descend(child, alpha, lambda_, level)

    return llk


"""
Sapmple gamma
"""

cdef void sample_gamma(object ihLDA, int shrink_max = 1000):
    cdef double gamma_current = ihLDA.htssb.params["gamma"]
    cdef double gamma_new

    # Slice sampling
    cdef double start
    cdef double end
    cdef double p_old
    cdef double p_new
    cdef double llk_sliced
    cdef double llk_new
    cdef int shrink_time

    start = 0.0
    end = tool.shrinkp(30.0)
    p_old = tool.shrinkp(gamma_current)
    llk_sliced = llk_gamma(ihLDA, gamma_current) - 2.0 * log(1.0 - p_old) + log(npr.uniform())

    for shrink_time in range(shrink_max):
        p_new = npr.uniform(start, end)
        gamma_new = tool.expandp(p_new)
        ihLDA.htssb.params["gamma"] = gamma_new

        llk_new = llk_gamma(ihLDA, gamma_new) - 2.0 * log(1.0 - p_new)

        if llk_sliced < llk_new:
            break
        elif p_old < p_new:
            end = p_new
        elif p_new < p_old:
            start = p_new
        else:
            print("sample_gamma: shrinked too much")
            ihLDA.htssb.params["gamma"] = gamma_current
            break

    return


cdef double llk_gamma(object ihLDA, double gamma):
    return llk_gamma_descend(ihLDA.htssb.tssb_root.node_root) +\
           tool.gammapdfln(gamma, 2, 4)  # prior

cdef double llk_gamma_descend(object node):
    cdef double llk = 0.0
    cdef double gamma_use

    if node.node_parent is not None:
        # root_node is not related to the horizontal parameter
        # Check derivation note: `hyperparameter_alpha.pdf`
        # This is a special case (category = 2) of Polya
        gamma_use = node_cy.gamma(node)
        llk += log(gamma_use) +\
               lgamma(node.nh0 + 1.0) +\
               lgamma(node.nh1 + gamma_use) -\
               lgamma(node.nh0 + node.nh1 + gamma_use + 1.0)


    for child in node.children:
        llk += llk_gamma_descend(child)

    return llk


"""
Sapmple lambda
"""

cdef void sample_lambda(object ihLDA, int shrink_max = 1000):
    cdef double lambda_current = ihLDA.htssb.params["lambda"]
    cdef double lambda_new
    cdef double lambda_use = lambda_current

    # Slice sampling
    cdef double start
    cdef double end
    cdef double p_old
    cdef double p_new
    cdef double llk_sliced
    cdef double llk_new
    cdef int shrink_time

    start = tool.shrinkp(0.0)
    end = tool.shrinkp(1.0)
    p_old = tool.shrinkp(lambda_current)

    llk_sliced = llk_lambda(ihLDA, lambda_current) - 2.0 * log(1.0 - p_old) + log(npr.uniform())

    for shrink_time in range(shrink_max):
        p_new = npr.uniform(start, end)
        lambda_new = tool.expandp(p_new)

        llk_new = llk_lambda(ihLDA, lambda_new) - 2.0 * log(1.0 - p_new)

        if llk_sliced < llk_new:
            lambda_use = lambda_new
            break
        elif p_old < p_new:
            end = p_new
        elif p_new < p_old:
            start = p_new
        else:
            print("sample_lambda: shrinked too much")
            break

    ihLDA.htssb.params["lambda"] = lambda_use
    return


cdef double llk_lambda(object ihLDA, double lambda_):
    return llk_lambda_descend(ihLDA.htssb.tssb_root.node_root,
                              lambda_
                             ) +\
           tool.betapdfln(lambda_, 1.0, 10.0)  # prior


cdef double llk_lambda_descend(object node, double lambda_):
    cdef double llk = 0.0
    cdef double alpha = node_cy.alpha(node)

    llk += log(alpha) +\
           lgamma(node.nv0 + 1.0) +\
           lgamma(node.nv1 + alpha) -\
           lgamma(node.nv0 + node.nv1 + alpha + 1.0)

    for child in node.children:
        llk += llk_lambda_descend(child, lambda_)

    return llk

"""
Sapmple hpy_d and hpy_theta
"""

cdef void sample_hpy_d_theta(object ihLDA):
    # Prepare Nodes by levels
    cdef dict nodes_level = {}
    cdef int max_level
    cdef int level

    for node in ihLDA.htssb.tssb_root.node_list:
        level = node.level

        if level in nodes_level:
            nodes_level[level].append(node)
        else:
            nodes_level[level] = [node]

    max_level = max(nodes_level.keys())
    assert(max_level <= ihLDA.htssb.params["current_max"])

    for level in range(max_level):
        if level == 0:
            # d and theta are not used if level == 0 (HPYLM p.17)
            # we do not need to update values
            continue

        sample_hpy_d_theta_level(ihLDA, level, nodes_level[level])

cdef void sample_hpy_d_theta_level(object ihLDA, int level, list nodes_list):
    cdef double bern
    cdef double bern_p
    cdef double xu
    cdef int cuwk
    cdef int j

    # For d
    cdef double beta_a = 1.0
    cdef double beta_b = 1.0
    # For theta
    cdef double gamma_a = 7.0  # shape
    cdef double gamma_b = 1.0  # scale

    # New value
    cdef double d_current = ihLDA.htssb.params["hpy_d"][level]
    cdef double theta_current = ihLDA.htssb.params["hpy_theta"][level]
    cdef double d_new
    cdef double theta_new

    for node in nodes_list:

        if node.tw_dist.tu >= 2:
            for i in range(1, int(node.tw_dist.tu)):
                bern_p = theta_current / (theta_current + d_current * i)
                bern = npr.uniform(0.0, 1.0)

                if bern > bern_p: # bern_p: probability of being 1
                    # yui = 0
                    beta_a += 1.0
                else:
                    # yui = 1
                    gamma_a += 1.0

            xu = npr.beta(theta_current+1.0, node.tw_dist.cu-1.0) # for x
            gamma_b -= log(xu)

        for hpytable in node.words_dict.values():
            for cuwk in hpytable.tables:
                if cuwk >= 2:
                    for j in range(1, cuwk):
                        bern_p = (j-1.0) / (j - d_current)
                        bern = npr.uniform(0.0, 1.0)

                        if bern > bern_p:
                            # zuwkj = 0
                            beta_b += 1.0


    # Sample d
    d_new = npr.beta(beta_a, beta_b)

    # Sample theta
    theta_new = npr.gamma(gamma_a, 1.0/gamma_b)

    ihLDA.htssb.params["hpy_d"][level] = d_new
    ihLDA.htssb.params["hpy_theta"][level] = theta_new


"""
Sample HTSSB Hyperparameters
"""
cdef void sample_htssb_hyper(object ihLDA, int iter_index):
    sample_htssb_hyper_slice(ihLDA)
    return

cdef void sample_htssb_hyper_slice(object ihLDA):
    ## Slice Sampling
    cdef double start = 0.0
    cdef double end = tool.shrinkp(15.0)
    cdef double val_current = ihLDA.htssb.params["aH"]
    cdef double p_old = tool.shrinkp(val_current)
    cdef double p_new
    cdef double val_proposal
    cdef double llk_sliced
    cdef double llk_new
    cdef int shrink_time

    llk_sliced = llk_htssb_hyper(ihLDA, val_current) -\
                 2.0 * log(1.0 - p_old) + log(npr.uniform())

    for shrink_time in range(200):
        p_new = npr.uniform(start, end)
        val_proposal = tool.expandp(p_new)

        llk_new = llk_htssb_hyper(ihLDA, val_proposal) - 2.0 * log(1.0 - p_new)

        if llk_sliced < llk_new:
            ihLDA.htssb.params["aH"] = val_proposal
            break
        elif p_old < p_new:
            end = p_new
        elif p_new < p_old:
            start = p_new
        else:
            print("sample_alpha_l: shrinked too much")
            ihLDA.htssb.params["aH"] = val_current
            break
    return


cdef double llk_htssb_hyper(object ihLDA, double aH):
    cdef double llk = 0.0
    cdef double x
    cdef double a
    cdef double b
    cdef double numerator
    cdef double denominator
    cdef object tssb
    cdef object node

    # Prior
    llk += tool.gammapdfln(aH, 1.5, 1.0)

    for tssb in ihLDA.htssb.tssb_docs:
        for node in tssb.node_list:
            # Vertical
            b = aH * node_cy.calc_omega(node.node_corres.node_parent, 1.0)
            a = b * node_cy.nu(node.node_corres)
            x = (a + node.nv0) / (b + node.nv0 + node.nv1)
            llk += tool.betapdfln(x, a, b)

            # Horizontal
            b = aH * node_cy.calc_psi(node.node_corres, 1.0)
            a = b * node_cy.psi(node.node_corres)
            x = (a + node.nh0) / (b + node.nh0 + node.nh1)
            if x >= 1.0:
                x = 1.0 - 1e-5
            llk += tool.betapdfln(x, a, b)

    return llk


"""
Calculate Perplexity
"""

def calc_perplexity(object ihLDA):
    """Calculate, show and save test perplexity

    """
    return calc_likelihood_hpy(ihLDA, perplexity = True)


cdef double calc_likelihood_hpy(object ihLDA, bool perplexity = False):
    cdef double ppl = 0.0
    cdef double llk = 0.0
    cdef double total_len = ihLDA.data.total_len
    cdef double node_stop  # node stop probability

    cdef double word_total
    cdef double stack

    cdef list doc_words

    cdef int word_id
    cdef double word_prob

    cdef dict cache_stop  # cache for node stop probability
    cdef dict cache_hpy
    cdef object node_corres

    cdef int cache_use
    if perplexity:
        cache_use = ihLDA.iter_setting["ppl_cache"]
    else:
        cache_use = 0  # do not use

    cache_hpy = {}

    for tssb in ihLDA.htssb.tssb_docs:
        doc_words = ihLDA.data.documents[tssb.doc_id]
        cache_stop = {}  # cache for node stop probability

        for word_id in doc_words:
            stack = 0.0  # stack the stop probability for each word
            word_total = 0.0

            for node in tssb.node_list:

                # Node stop probabilty
                # It corresponds to p(\varepsilon)
                if cache_use:
                    if node not in cache_stop:
                        cache_stop[node] = node_cy.prob_stop_cy(node)

                    node_stop = cache_stop[node]
                else:
                    node_stop = node_cy.prob_stop_cy(node)

                stack += node_stop  # stack nod_stop probability

                # Get word probability from HPY
                if cache_use:
                    node_corres = node.node_corres

                    if node_corres not in cache_hpy:
                        cache_hpy[node_corres] = {}

                    if word_id not in cache_hpy[node_corres]:
                        cache_hpy[node_corres][word_id] =\
                                node_corres.tw_dist.wordprob_py(word_id)

                    word_prob = cache_hpy[node_corres][word_id]

                else:
                    node_corres = node.node_corres
                    word_prob = node_corres.tw_dist.wordprob_py(word_id)

                word_total += node_stop * word_prob
                assert(stack <= 1.0)

            # Instantiated nodes
            llk += log(word_total)

    if perplexity:
        ppl = exp(-llk / total_len)
        return ppl
    else:
        return llk



"""
Calculate Test Perplexity
"""

def calc_testperplexity(object ihLDA):
    """Calculate, show and save test perplexity

    """
    return calc_testperplexity_hpy(ihLDA)


cdef double calc_testperplexity_hpy(object ihLDA):
    cdef double perplexity = 0.0
    cdef double llk = 0.0
    cdef double total_len = 0.0
    cdef double node_stop  # node stop probability
    cdef double stack
    cdef double word_total

    cdef list doc_words

    cdef str word_raw
    cdef int word_id
    cdef double word_prob

    cdef dict cache_stop  # cache for node stop probability
    cdef dict cache_hpy
    cdef object node_corres

    cache_hpy = {}

    for tssb in ihLDA.htssb.tssb_docs:
        doc_words = ihLDA.data.documents_test[tssb.doc_id]
        cache_stop = {}  # cache for node stop probability

        for word_raw in doc_words:
            if word_raw in ihLDA.data.word_to_wordid:
                word_id = ihLDA.data.word_to_wordid[word_raw]
            else:
                # Not in train document
                continue

            stack = 0.0  # stack the stop probability for each word
            word_total = 0.0
            total_len += 1.0  # count only used word

            for node in tssb.node_list:

                # Node stop probabilty
                # It corresponds to p(\varepsilon)
                if ihLDA.iter_setting["ppl_cache"]:
                    if node not in cache_stop:
                        cache_stop[node] = node_cy.prob_stop_cy(node)

                    node_stop = cache_stop[node]
                else:
                    node_stop = node_cy.prob_stop_cy(node)

                stack += node_stop  # stack nod_stop probability

                # Get word probability from HPY
                if ihLDA.iter_setting["ppl_cache"]:
                    node_corres = node.node_corres

                    if node_corres not in cache_hpy:
                        cache_hpy[node_corres] = {}

                    if word_id not in cache_hpy[node_corres]:
                        cache_hpy[node_corres][word_id] =\
                                node_corres.tw_dist.wordprob_py(word_id)

                    word_prob = cache_hpy[node_corres][word_id]

                else:
                    node_corres = node.node_corres
                    word_prob = node_corres.tw_dist.wordprob_py(word_id)

                word_total += node_stop * word_prob
                assert(stack <= 1.0)

            # Instantiated nodes
            llk += log(word_total)

    perplexity = exp(-llk / total_len)

    return perplexity


"""
Save Topic-Word Distribution
"""

def save_topic_word_dist(object ihLDA):
    return save_topic_word_dist_cy(ihLDA)

cdef object save_topic_word_dist_cy(object ihLDA):
    """
    Count created in this section includes proxy counts
    """
    # Save
    cdef list store = []  # list based on probability
    cdef int num_topwords = ihLDA.save_info["num_topwords"]


    for node in ihLDA.htssb.tssb_root.node_list:
        df_temp = save_model.pd_topic_word_dist(node)

        # Sort
        df_temp.sort_values(by=["Probability", "Word"],
                            ascending=[False, True],
                            inplace=True)

        # Store
        if num_topwords < df_temp.shape[0]:
            store.append(df_temp[:num_topwords])
        else:
            store.append(df_temp)

        # Create a new list only use words
        df_temp.sort_values(by=["Count", "Word"],
                            ascending=[False, True],
                            inplace=True)

    # Merge
    df = pd.concat(store, ignore_index=True)
    df.sort_values(
        by=["Path", "Probability", "Word"],
        ascending=[True, False, True],
        inplace=True
    )

    # Save
    save_folder = pathlib.Path(ihLDA.save_info["output_folder"]).resolve()
    save_path = str(save_folder /
                    pathlib.Path("TopWords_prob.csv"))
    df.to_csv(save_path, index=False)

    return df