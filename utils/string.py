
__author__ = 'ccervantes'


def list_to_rows(lst, num_cols):
    """
    Returns the list as an evenly-divided
    list of lists for processing as a table
    :param lst:
    :param num_cols:
    :return:
    """
    lst = list(lst)

    # Add a row if the list isn't neatly divisible
    num_rows = len(lst) / num_cols
    if len(lst) % num_rows > 0:
        num_rows += 1

    # Partition the list into a list of rows
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


def dict_to_kv_str(dct):
    """
    Returns the key-value string representation
    of the given dictionary in the format
    (key_0:val_0;key_1:val_1)
    :param dct: Dictionary of keys and values
    :return: Key value string
    """
    kv_str = ""
    keys = list(dct.keys())
    for i in range(0, len(keys)):
        k = keys[i]
        v = dct[k]
        kv_str += k + ":" + v
        if i < len(keys) - 1:
            kv_str += ";"
    #endfor
    return kv_str
#enddef
