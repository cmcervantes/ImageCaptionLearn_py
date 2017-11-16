import numpy as np
from scipy import sparse

import core as util

__author__ = 'ccervantes'


def load_very_sparse_feats(filename, meta_dict=None, ignored_feats=None):
    """
    Loads the given sparse feature data into a SciPy sparse matrix, returning
    a (X, Y, IDs) tuple, which are (sparse_matrix, numpy_array, list),
    respectively
    NOTE: labels Y are a list of actual label values (contrast with
          load_dense_feats, which returns a matrix of one-hots)

    :param filename:        Sparse feature file, in the liblinear format
                            <label> <index_0>:<value_0> ... <index_n>:<value_n> # <ID>
    :param meta_dict:       Dictionary mapping human-readable names to feature
                            indices (used in conjunction with ignored_feats)
    :param ignored_feats:   List of human-readable feature names to ignore
    :return:                (X, Y, IDs) tuple for data, labels, and unique
                            sample IDs, respectively
    """
    ids = list()
    y_vals = list()
    x_cols = list()
    x_rows = list()
    x_vals = list()

    # Store a mapping of an index and an
    # adjustment; when we encounter an index
    # greater than a key, we subtract the
    # value from it
    ignored_adjust_dict = dict()
    if ignored_feats is not None:
        for feat in ignored_feats:
            val = meta_dict[feat]
            if isinstance(val, list):
                ignored_adjust_dict[val[1]] = val[1] - val[0]
            else:
                ignored_adjust_dict[val] = 1
        #endfor
    #endif

    row_count = 0
    n_rows = 0
    with open(filename, 'r') as f:
        # Read the total number of lines
        for n_rows, l in enumerate(f):
            pass
        n_rows += 1

        # Reset the file and read for content
        f.seek(0)
        for line in f:
            # get the ID from the comment string
            id_split = line.split(" # ")
            ids.append(id_split[1].strip())

            # split the non-comments by spaces
            vector_split = id_split[0].strip().split(" ")

            # the label is the first element
            y_vals.append(int(float(vector_split[0].strip())))

            # iterate through the vector
            for i in range(1, len(vector_split)):
                kv_pair = vector_split[i].split(":")
                if float(kv_pair[1]) > 0.0:
                    idx = int(kv_pair[0].strip())
                    adj_idx = idx

                    # If this is one of the ignored features, drop it
                    if ignored_feats is not None and \
                       util.is_ignored_idx(idx, meta_dict, ignored_feats):
                        continue

                    # Adjust the index, to account for the missing
                    # indices
                    for j in ignored_adjust_dict.keys():
                        if j < idx:
                            adj_idx -= ignored_adjust_dict[j]
                    #endfor

                    x_rows.append(row_count)
                    x_cols.append(adj_idx)
                    x_vals.append(float(kv_pair[1].strip()))
            #endfor
            row_count += 1
        #endfor
        f.close()
    #endwith

    # get the number of feats, less the one we've ignored
    num_feats = meta_dict['max_idx'] + 1
    if ignored_feats is not None:
        for feat in ignored_feats:
            val = meta_dict[feat]
            index_count = 1
            if isinstance(val, list):
                index_count = val[1] - val[0]
            num_feats -= index_count
        #endfor
    #endif

    x = sparse.csc_matrix((np.array(x_vals), (x_rows, x_cols)),
                          shape=(n_rows, num_feats))
    y = np.array(y_vals)
    return x, y, ids
#enddef


def load_sparse_feats(filename, meta_dict=None, ignored_feats=None, n_features=None):
    """
    Loads the given sparse feature data into a sparse numpy matrix,
    returning (x, y, ids) tuple; this should be more efficient
    for moderately sparse data, but for very sparse data, it is
    unclear whether load_very_sparse_feats is more efficient
    NOTE: labels Y are a list of actual label values (contrast with
          load_dense_feats, which returns a matrix of one-hots)

    :param filename:        Sparse feature file, in the liblinear format
                            <label> <index_0>:<value_0> ... <index_n>:<value_n> # <ID>
    :param meta_dict:       Dictionary mapping human-readable names to feature
                            indices (used in conjunction with ignored_feats)
    :param ignored_feats:   List of human-readable feature names to ignore
    :return:                (X, Y, IDs) tuple for data, labels, and unique
                            sample IDs, respectively
    """
    # Store a mapping of an index and an
    # adjustment; when we encounter an index
    # greater than a key, we subtract the
    # value from it
    ignored_adjust_dict = dict()
    if ignored_feats is not None:
        for feat in ignored_feats:
            val = meta_dict[feat]
            if isinstance(val, list):
                ignored_adjust_dict[val[1]] = val[1] - val[0]
            else:
                ignored_adjust_dict[val] = 1
        #endfor
    #endif
    ignored_adjust_keys = set(ignored_adjust_dict.keys())

    # get the number of feats, less the one we've ignored
    n_feats = 0
    if n_features is not None:
        n_feats = n_features
    elif meta_dict is not None:
        n_feats = meta_dict['max_idx'] + 1
    if ignored_feats is not None:
        for feat in ignored_feats:
            val = meta_dict[feat]
            index_count = 1
            if isinstance(val, list):
                index_count = val[1] - val[0]
            n_feats -= index_count
        #endfor
    #endif

    with open(filename, 'r') as f:
        # Read the total number of lines
        n_rows = 0
        for n_rows, l in enumerate(f):
            pass
        n_rows += 1

        y = np.zeros(n_rows)
        x = np.zeros([n_rows, n_feats])
        ids = list()

        # Reset the file and read for content
        f.seek(0)
        i = 0
        for line in f:
            # get the ID from the comment string
            id_split = line.split(" # ")
            ids.append(id_split[1].strip())

            # split the non-comments by spaces
            vector_split = id_split[0].strip().split(" ")

            # the label is the first element
            y[i] = int(float(vector_split[0].strip()))

            # iterate through the vector
            for j in range(1, len(vector_split)):
                kv_pair = vector_split[j].split(":")

                if float(kv_pair[1]) != 0.0:
                    idx = int(kv_pair[0].strip())
                    adj_idx = idx

                    # If this is one of the ignored features, drop it
                    if ignored_feats is not None and \
                       util.is_ignored_idx(idx, meta_dict, ignored_feats):
                        continue

                    # Adjust the index, to account for the missing
                    # indices
                    for ignored_f in ignored_adjust_keys:
                        if ignored_f < idx:
                            adj_idx -= ignored_adjust_dict[ignored_f]
                    #endfor

                    if "box" in filename:
                        adj_idx -= 1

                    x[i][adj_idx] = float(kv_pair[1].strip())
                #endif
            #endfor
            i += 1
        #endfor
        f.close()
    #endwith

    return x, y, ids
#enddef


def load_dense_feats(filename):
    """
    Loads the given dense feature data into a numpy matrix, returning
    a (X, Y, IDs) tuple, which are (numpy_matrix, numpy_matrix, list),
    respectively.
    NOTE: Labels Y are returned as a [num_samples, num_classes] matrix,
          where each sample's label is represented as a one-hot vector

    :param filename:        Sparse feature file, in a modified liblinear format
                            <label> <value_0> ... <value_n> # <ID>
    :return:                (X, Y, IDs) tuple for data, labels, and unique
                            sample IDs, respectively
    """
    n_lines = 0
    max_label = -1
    n_feats = -1
    ids = list()

    with open(filename, 'r') as f:
        # Pass through the file once, getting
        # the number of lines, classes, and features
        for line in f.readlines():
            n_lines += 1
            id_split = line.split("#")
            label_split = id_split[0].split(" ")
            label = int(label_split[0].strip())
            if label > max_label:
                max_label = label
            if n_feats == -1:
                n_feats = len(label_split) - 1
        #endfor
        n_lines += 1

        # set up our data arrays
        x = np.zeros((n_lines, n_feats))
        y = np.zeros((n_lines, max_label + 1))

        # reset the file and read for content
        f.seek(0)
        i = 0
        for line in f.readlines():
            id_split = line.split("#")
            label_split = id_split[0].split(",")

            ids.append(id_split[1].strip())
            y[i][int(label_split[0])] = 1.0
            for j in range(0, n_feats):
                x[i][j] = float(label_split[j+1].strip())
            i += 1
        #endfor
    #endwith
    return x, y, ids
#enddef


def load_ablation_file(filename):
    """
    Reads an ablation file (hash comments; pipe separated
    lines of feature groupings) into a set of sets
    for use with our classifiers

    :param filename: Ablation file
    :return: set of sets
    """
    feature_groups = set()
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.strip() == "" or line.startswith("#"):
                continue

            feature_group = set()
            for feat_name in line.split("|"):
                feature_group.add(feat_name.strip())
            feature_groups.add(frozenset(feature_group))
        #endfor
    #endwith
    return feature_groups
#enddef

