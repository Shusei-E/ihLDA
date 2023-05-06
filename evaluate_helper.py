import pickle
import pandas as pd
pd.set_option('display.max_rows', 200)

def show_allpath(node, res):
    res.append(node["path"])
    for child in node["children"]:
        res = show_allpath(child, res)
    return res


def str_path(x):
    return "_".join([str(y) for y in x])


def show_twdist(model, path, top_n, min_count):
    # Show Topic-Word distribution
    return(show_twdist_recursive(node = model["RootTSSB"], path=path, top_n=top_n, min_count=min_count))


def show_twdist_recursive(node, path, top_n, min_count, detail = False):
    if node["path"] == path:
        if sum(node["tw_dist"].iloc[:, 2]) >= min_count:
            words = node["tw_dist"].iloc[0:top_n, 1]
            s = ", ".join(words)
            return s
    else:
        for child in node["children"]:
            s = show_twdist_recursive(child, path, top_n, min_count, detail)
            if s is not None:
                return s


def topwords_itstm(file, min_count=0):
    """
    Show topwords of each topic in the tree
    :param file: file path of the model
    :param min_count: minimum count of the words in topic (`100` for replicating existing models)
    """

    with open(file, "rb") as f:
        model = pickle.load(f)

    all_path = show_allpath(model["RootTSSB"], [])
    topwords = []
    all_path_str = []
    level = []
    parent = []

    for path in all_path:
        tw = show_twdist(model, path, top_n=10, min_count=min_count)

        if tw is None:
            continue

        all_path_str.append(str_path(path))
        level.append(len(path))
        parent.append(str_path(path[0:-1]))

        topwords.append(tw)

    topwords_df = pd.DataFrame({
        "level": level,
        "path": all_path_str,
        "topwords": topwords
    })

    return topwords_df