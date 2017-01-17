import numpy as np
from scipy import sparse

"""
Loads the data into a scipy sparse matrix from the given filename,
returning an (X, Y, IDs) tuple, or
(sparse_data_matrix, np_array_labels, list_of_example_IDs)
"""
def __dep__load_feats_data(filename, max_feat_idx, log=None):
    ids = list()
    y_vals = list()
    x_cols = list()
    x_rows = list()
    x_vals = list()

    rowCount = 0
    totalRows = 0
    with open(filename, 'r') as f:
        for line in f:
            totalRows += 1
        f.seek(0)
        for line in f:
            #get the ID from the comment string
            commentSplit = line.split(" # ")
            ids.append(commentSplit[1].strip())

            #split the non-comments by spaces
            vectorSplit = commentSplit[0].strip().split(" ")

            #the label is the first element
            y_vals.append(int(float(vectorSplit[0])))

            #iterate through the vector
            for i in range(1,len(vectorSplit)):
                kvPair = vectorSplit[i].split(":")
                if float(kvPair[1]) != -1.0:
                    x_rows.append(rowCount)
                    x_cols.append(int(kvPair[0]))
                    x_vals.append(float(kvPair[1]))
            #endfor
            rowCount += 1

            if rowCount >= 1000 and log is not None:
                log.log_status('info', None,
                   "Read %.1fk vectors (%.2f%%)",
                   rowCount / 1000.0, 100.0 * rowCount / totalRows)
            #endif
        #endfor
        f.close()
    #endwith
    x = sparse.csc_matrix((np.array(x_vals), (x_rows, x_cols)),
                          shape=(totalRows, max_feat_idx))
    y = np.array(y_vals)
    return x, y, ids
#enddef


"""
Loads the data into a scipy sparse matrix from the given filename,
returning an (X, Y, IDs) tuple, or
(sparse_data_matrix, np_array_labels, list_of_example_IDs)
"""
def load_feats_data(filename, meta_dict=None, ignored_feats=None, log=None):
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
    for feat in ignored_feats:
        val = meta_dict[feat]
        if isinstance(val, list):
            ignored_adjust_dict[val[1]] = val[1] - val[0]
        else:
            ignored_adjust_dict[val] = 1
    #endfor

    rowCount = 0
    totalRows = 0
    with open(filename, 'r') as f:
        for line in f:
            totalRows += 1
        f.seek(0)
        for line in f:
            #get the ID from the comment string
            commentSplit = line.split(" # ")
            ids.append(commentSplit[1].strip())

            #split the non-comments by spaces
            vectorSplit = commentSplit[0].strip().split(" ")

            #the label is the first element
            y_vals.append(int(float(vectorSplit[0])))

            #iterate through the vector
            for i in range(1,len(vectorSplit)):
                kvPair = vectorSplit[i].split(":")
                if float(kvPair[1]) > 0.0:
                    idx = int(kvPair[0])
                    adj_idx = idx

                    # If this is one of the ignored features, drop it
                    if is_ignored_idx(idx, meta_dict, ignored_feats):
                        continue

                    # Adjust the index, to account for the missing
                    # indices
                    for j in ignored_adjust_dict.keys():
                        if j < idx:
                            adj_idx -= ignored_adjust_dict[j]
                    #endfor

                    x_rows.append(rowCount)
                    x_cols.append(adj_idx)
                    x_vals.append(float(kvPair[1]))
            #endfor
            rowCount += 1

            if rowCount >= 1000 and log is not None:
                log.log_status('info', None,
                               "Read %.1fk vectors (%.2f%%)",
                               rowCount / 1000.0, 100.0 * rowCount / totalRows)
                #endif
        #endfor
        f.close()
    #endwith

    #get the number of feats, less the one we've ignored
    num_feats = meta_dict['max_idx'] + 1
    for feat in ignored_feats:
        val = meta_dict[feat]
        index_count = 1
        if isinstance(val, list):
            index_count = val[1] - val[0]
        num_feats -= index_count
    #endfor

    x = sparse.csc_matrix((np.array(x_vals), (x_rows, x_cols)),
                          shape=(totalRows, num_feats))
    y = np.array(y_vals)
    return x, y, ids
#enddef

def is_ignored_idx(idx, meta_dict, ignored_feats):
    for feat in ignored_feats:
        val = meta_dict[feat]
        if isinstance(val, list):
            return val[0] <= idx <= val[1]
        else:
            return idx == val
#enddef

"""
Dumps all of the argument values to the console, so we know
how we ran this experiment
"""
def dump_args(arg_dict, log):
    for arg in arg_dict.keys():
        log.debug(None, "%s: %s", arg, str(arg_dict[arg]))
        #endfor
#enddef
