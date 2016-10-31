import numpy as np
from scipy import sparse

"""
Loads the data into a scipy sparse matrix from the given filename,
returning an (X, Y, IDs) tuple, or
(sparse_data_matrix, np_array_labels, list_of_example_IDs)
"""
def load_feats_data(filename, max_feat_idx, log=None):
    IDs = list()
    Y_val = list()
    X_col = list()
    X_row = list()
    X_data = list()

    rowCount = 0
    totalRows = 0
    with open(filename, 'r') as f:
        for line in f:
            totalRows += 1
        f.seek(0)
        for line in f:
            #get the ID from the comment string
            commentSplit = line.split(" # ")
            IDs.append(commentSplit[1])

            #split the non-comments by spaces
            vectorSplit = commentSplit[0].strip().split(" ")

            #the label is the first element
            Y_val.append(int(float(vectorSplit[0])))

            #iterate through the vector
            for i in range(1,len(vectorSplit)):
                kvPair = vectorSplit[i].split(":")
                if float(kvPair[1]) != -1.0:
                    X_row.append(rowCount)
                    X_col.append(int(kvPair[0]))
                    X_data.append(float(kvPair[1]))
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
    X = sparse.csc_matrix((np.array(X_data), (X_row, X_col)),
                          shape=(totalRows, max_feat_idx))
    Y = np.array(Y_val)
    return X, Y, IDs
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
