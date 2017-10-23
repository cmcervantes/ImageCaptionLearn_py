import numpy as np
from sklearn import metrics as sk_metrics
from utils.ScoreDict import ScoreDict
from utils import string as str_util
from utils.Logger import Logger

__author__ = 'ccervantes'


def evaluate_relations(mention_pairs, pred_labels,
                       gold_label_dict, log=None):
    """
    Evaluates relation predictions as complete pairs, incorporating
    information about nonvisual mentions and invalid link pairs

    :param mention_pairs: List of mention pair IDs corresponding to
                          pred_labels order
    :param pred_labels: List of predicted labels with order corresponding
                        to mention_pairs
    :param gold_label_dict: Dictionary of (m_i_id, m_j_id) tuples mapped to
                            gold labels
    :param gold_nonvis: Whether to use gold nonvisual information in reporting scores
    :param log: optional logger to log status messages to
    :return: Complete score_dict for these mention pairs
    """
    score_dict = ScoreDict()

    # Associate mention pairs with their predicted labels
    if log is not None:
        log.info("Mapping mention pair IDs to labels")
    pred_label_dict = dict()
    for i in range(0, len(mention_pairs)):
        pred_label_dict[mention_pairs[i]] = pred_labels[i]

    # Iterate through the gold label dict, retrieving and parsing
    # the predicted link labels on a pairwise basis
    # UPDATE: .keys() returns a _list_, not a set!
    if log is not None:
        log.info("Predicting")
    gold_pairs = set(gold_label_dict.keys())
    pred_pairs = set(pred_label_dict.keys())
    for pair in gold_pairs:
        if pair[0] not in pred_pairs or pair[1] not in pred_pairs:
            continue
        gold = gold_label_dict[pair]
        pred_ij = pred_label_dict[pair[0]]
        pred_ji = pred_label_dict[pair[1]]

        # switch on the ji / ij vals
        pred = "invalid"
        if pred_ij == pred_ji == 0:
            pred = "null"
        elif pred_ij == pred_ji == 1:
            pred = "coref"
        elif pred_ij + pred_ji == 5:
            if pred_ij == 2:
                pred = "subset_ij"
            elif pred_ji == 2:
                pred = "subset_ji"
        #endif

        # Handle subset pairs according to whether their direction
        # is correct; if both links are subset and their direction
        # matches, this is a correct prediction; otherwise this is
        # a mismatched link
        if gold.startswith("subset_") and pred.startswith("subset_"):
            if gold == pred:
                pred = "subset"
            else:
                pred = "-reverse_sub-"
            gold = "subset"
        if gold.startswith("subset_"):
            gold = "subset"
        if pred.startswith("subset_"):
            pred = "subset"
        #endif

        # increment the score dict
        score_dict.increment(gold, pred)
    #endfor

    # Print the score dict scores
    label_str = ["-invalid-", "-reverse_sub-",
                 "null", "coref", "subset"]
    for l in label_str:
        print "%10s: %s" % (l, score_dict.get_score(l).to_string())
    score_dict.print_confusion()

    return score_dict
#enddef


def evaluate_multiclass(gold_labels, pred_labels, class_names, log=None):
    """
    Logs the counts and evaluation metrics, based on the given
    counts and labels
    :param gold_labels: numpy array of the data's labels (y)
    :param pred_labels: numpy array of the predict labels
    :param class_names: human label names corresponding to int labels
    :param log: The logging object to use (if None prints to console)
    """

    # Store the labels as a score dict, to keep our evaluation functions
    # consistent
    score_dict = ScoreDict(gold_labels, pred_labels)

    # Print the counts
    gold_counts = np.bincount(gold_labels)
    pred_bins = np.bincount(pred_labels)
    pred_counts = np.pad(pred_bins, len(class_names)-len(pred_bins), 'constant')
    count_tbl = list()
    count_tbl.append(['gold', 'pred'])
    for l in range(0, len(class_names)):
        count_tbl.append([str(gold_counts[l]), str(pred_counts[l])])
    log.info('Counts\n'+str_util.rows_to_str(count_tbl))

    # Followed by the precision/recall/f1 for our labels
    precision_scores = sk_metrics.precision_score(y_true=gold_labels,
                                                  y_pred=pred_labels,
                                                  average=None)
    recall_scores = sk_metrics.recall_score(y_true=gold_labels,
                                            y_pred=pred_labels, average=None)
    f1_scores = sk_metrics.f1_score(y_true=gold_labels,
                                    y_pred=pred_labels, average=None)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=gold_labels, y_pred=pred_labels,
                                                   labels=range(0, len(class_names)))
    accuracy = sk_metrics.accuracy_score(y_true=gold_labels, y_pred=pred_labels)

    # Log the metrics
    metrics = list()
    metrics.append(["", "P", "R", "F1"])
    for l in range(0, len(class_names)):
        row = list()
        row.append(class_names[l])

        if l < len(precision_scores):
            p = 100.0 * precision_scores[l]
            row.append("%.2f%%" % p)
        else:
            row.append("0.00%")
        if l < len(recall_scores):
            r = 100.0 * recall_scores[l]
            row.append("%.2f%%" % r)
        else:
            row.append("0.00%")
        if l < len(f1_scores):
            f1 = 100.0 * f1_scores[l]
            row.append("%.2f%%" % f1)
        else:
            row.append("0.00%")
        metrics.append(row)
    #endfor
    metrics_str = str_util.rows_to_str(metrics, True)
    accuracy_str = "Accuracy: " + str(100.0 * accuracy)

    # Log the confusion matrix as well
    conf_matrix_table = list()
    col_headers = list()
    col_headers.append("g| p->")
    col_headers.extend(class_names)
    conf_matrix_table.append(col_headers)
    for i in range(0, len(class_names)):
        row = list()
        row.append(class_names[i])
        for j in range(0, len(class_names)):
            row.append(str(int(confusion_matrix[i][j])))
        conf_matrix_table.append(row)
    #endfor
    conf_str = str_util.rows_to_str(conf_matrix_table, True, False)

    if log is None:
        print accuracy_str
        print metrics_str
        print conf_str
    else:
        log.info("\n" + accuracy_str)
        log.info("\n" + metrics_str)
        log.info("\n" + conf_str)

    return score_dict
#enddef

