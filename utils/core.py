from os.path import abspath, expanduser, exists

__author__ = "ccervantes"


def is_ignored_idx(idx, meta_dict, ignored_feats):
    """
    Returns whether the given idx is one of the
    ignored features (it checks the ranges and
    values in the meta_dict)
    :param idx:
    :param meta_dict:
    :param ignored_feats:
    :return:
    """
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


def get_max_idx(arr):
    """
    Returns the index with the maximum
    value in the array
    :param arr:
    :return:
    """
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


def arg_path_exists(parser, filename):
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


def get_full_path(path):
    """
    Returns the abspath/expanduser version
    of the given path, appended with a forward
    slash (for directories) if one was provided
    (since abspath strips it)

    :param path:
    :return:
    """
    full_path = abspath(expanduser(path))
    if path.endswith('/'):
        full_path += '/'
    return full_path
#enddef
