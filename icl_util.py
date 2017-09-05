import numpy as np
from scipy import sparse
from os.path import abspath, expanduser, exists

__author__ = "ccervantes"


def load_sparse_feats(filename, meta_dict=None, ignored_feats=None):
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
    for feat in ignored_feats:
        val = meta_dict[feat]
        if isinstance(val, list):
            ignored_adjust_dict[val[1]] = val[1] - val[0]
        else:
            ignored_adjust_dict[val] = 1
    #endfor

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
                    if is_ignored_idx(idx, meta_dict, ignored_feats):
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
    for feat in ignored_feats:
        val = meta_dict[feat]
        index_count = 1
        if isinstance(val, list):
            index_count = val[1] - val[0]
        num_feats -= index_count
    #endfor

    x = sparse.csc_matrix((np.array(x_vals), (x_rows, x_cols)),
                          shape=(n_rows, num_feats))
    y = np.array(y_vals)
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


"""
Whether this idx is one of our ignored features (checks
the ranges and values in the meta_dict)
"""
def is_ignored_idx(idx, meta_dict, ignored_feats):
    is_ignored = False
    for feat in ignored_feats:
        val = meta_dict[feat]
        if isinstance(val, list):
            is_ignored |= val[0] <= idx <= val[1]
        else:
            is_ignored |= idx == val
        #endif
    return is_ignored
#enddef


def dump_args(arg_dict, log):
    """
    Dumps all of the argument values to the logger
    so it's easier to keep track of experiment params

    :param arg_dict: Argparse dict
    :param log: LogUtil obect
    :return:
    """
    for arg in arg_dict.keys():
        log.debug(None, "%s: %s", arg, str(arg_dict[arg]))
    #endfor
#enddef


"""
Returns the list as an evenly-divided
list of lists (for processing as a table)
"""
def list_to_rows(lst, num_cols):
    lst = list(lst)

    #add a row if the list isn't neatly divisible
    num_rows = len(lst) / num_cols
    if len(lst) % num_rows > 0:
        num_rows += 1

    #partition the list into a list of rows
    rows = list()
    for i in range(0, num_rows):
        row = list()
        for j in range(i * num_cols, (i+1) * num_cols):
            if j < len(lst):
                row.append(str(lst[j]))
            else:
                row.append("")
        rows.append(row)
    return rows
#enddef


def rows_to_str(rows, has_headers=False, use_latex=False):
    """
    Returns a string representation of the given
    row list (a list of lists) as a formatted table

    :param rows: List of string lists
    :param has_headers: Whether the given rows include column
                        and rows headers (exclusive with use_latex)
    :param use_latex: Whether to use latex table formatting
    :return: Single string for the formatted table
    """
    # Get the number of rows / columns
    num_rows = len(rows)
    num_cols = 0
    for row in rows:
        if len(row) > num_cols:
            num_cols = len(row)

    # Get the width of each column
    col_widths = [0] * num_cols
    for row in rows:
        for c in range(0, len(row)):
            col = row[c]
            if len(col) > col_widths[c]:
                col_widths[c] = len(col)
        #endfor
    #endfor

    table_str = ""
    if use_latex:
        header = '\\begin{tabular}{'
        for i in range(0, num_cols):
            header += 'l'
        header += '}'
        table_str += header + "\n"

        for i in range(0, len(rows)):
            row_str = '\t'+' & '.join(rows[i])
            if i < len(rows) - 1:
                row_str += "\\\\"
            table_str += row_str + "\n"
        table_str += "\\end{tabular}"
    else:
        # Specify the formatting string, including the
        # row header separation (where applicable)
        format_str = ""
        start_idx = 0
        if has_headers:
            format_str = "%-" + str(col_widths[0]+1) + "s | "
            start_idx += 1
        #endif
        for i in range(start_idx, num_cols):
            format_str += "%-" + str(col_widths[i] + 1) + "s "
        format_str += "\n"

        # Create the table string from the given rows
        # starting with the first row (in case it contains
        # column headers)
        first_row = list()
        for c in range(0, num_cols):
            if c < len(rows[0]):
                first_row.append(rows[0][c])
            else:
                first_row.append("")
        #endfor
        table_str = format_str % tuple(first_row)

        # If headers were specified, add a row of dashes
        if has_headers:
            for c in range(0, num_cols):
                for w in range(0, col_widths[c] + 2):
                    table_str += "-"
                # first column has a pipe and extra space
                if c == 0:
                    table_str += "|-"
            #endfor
            table_str += "\n"
        #endif

        # Add the other rows
        for r in range(1, num_rows):
            row_str = list()
            for c in range(0, num_cols):
                if c < len(rows[r]):
                    row_str.append(rows[r][c])
                else:
                    row_str.append("")
                #endif
            #endfor
            table_str += format_str % tuple(row_str)
        #endfor
        table_str = table_str[0:len(table_str)-1]
    #endif

    return table_str
#enddef


"""
Returns the idx with the max value (in the arr)
"""
def get_max_idx(arr):
    idx = -1
    if isinstance(arr, list):
        max_idx = -float('inf')
        for i in range(0, len(arr)):
            if arr[i] > max_idx:
                max_idx = arr[i]
                idx = i
            #endif
        #endfor
    #endif
    return idx
#enddef


def kv_str_to_dict(kv_str):
    """
    Parses a key-value string (key_0:val_0;key_1:val_1)
    to a dictionary, mapping keys to values
    :param kv_str: Key value string
    :return: Dictionary of keys and values
    """
    kv_dict = dict()
    for kv_pair in kv_str.split(';'):
        kv_split = kv_pair.split(":")
        kv_dict[kv_split[0]] = kv_split[1]
    #endfor
    return kv_dict
#enddef


def list_to_index_dict(l):
    """
    Converts a list of items to
    an index dictionary, mapping
    the item at position i to
    the value i
    :param l: List of items
    :return: Mapping of items to list index
    """
    d = dict()
    for i in range(0, len(l)):
        d[l[i]] = i
    return d
#enddef


def arg_file_exists(parser, filename):
    """
    Expands the given filename and checks if
    the file exists; returning the expanded,
    absolute path if so and causing a parser
    error if not

    :param parser:   Argument parser with which
                     to cause an error
    :param filename: Filename to check
    :return:         Absolute path of filename
    """
    if filename is not None:
        filename = abspath(expanduser(filename))
        if exists(filename):
            return filename
        #endif
    #endif
    parser.error("\nCould not find " + filename)
#enddef