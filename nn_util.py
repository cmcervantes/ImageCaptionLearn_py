import math
import tensorflow as tf
import numpy as np
import random
import json
from sklearn import metrics as sk_metrics

import icl_util as util
from Word2VecUtil import Word2VecUtil

__author__ = 'ccervantes'

__WORD_2_VEC_PATH = '/shared/projects/word2vec/GoogleNews-vectors-negative300.bin.gz'

# Global nn_util vars
__w2v = None


def init_w2v():
    """
    Initializes this utility's word2vec module
    """
    global __w2v, __WORD_2_VEC_PATH
    __w2v = Word2VecUtil(__WORD_2_VEC_PATH)
#enddef


def set_random_seeds(seed=20170927):
    """
    Sets the tensorflow and numpy random seeds to a
    fixed number for experimental consistency
    :return:
    """
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
#enddef


def get_weights(shape, init_type='xavier', trainable=True):
    """
    Returns the tensorflow variable for a weight tensor
    :param shape: Number of dimensions (for standard problems, this is
                  typically the number of hidden nodes x number of classes)
    :param init_type:  Type of initialization (uniform, normal, xavier)
    :param trainable:  Whether this vector is trainable
    :return:           Tensorflow variable for the weight tensor
    """

    if init_type == 'xavier':
        init_range = math.sqrt(6.0 / sum(shape))
        return tf.Variable(tf.random_uniform(shape, minval=-init_range, maxval=init_range,
                                             seed=20170801), trainable=trainable)
    elif init_type == "normal":
        return tf.Variable(tf.random_normal(shape, stddev=1.0 / math.sqrt(shape[0]),
                                            seed=20170801), trainable=trainable)
    elif init_type == "uniform":
        return tf.Variable(tf.random_uniform(shape, minval=-0.05, maxval=0.05,
                                             seed=20170801), trainable=trainable)
    return None
#enddef


def get_biases(shape, init_type="xavier", trainable=True):
    """
    Returns a tensorflow variable for the model's biases
    :param shape: Number of dimensions (for standard problems, this is
                  typically the number of classes)
    :param init_type:  Type of initialization (uniform, normal, xavier)
    :param trainable: Whether this variable is trainable
    :return: Tensorflow variable for the bias tensor
    """

    if init_type == 'xavier':
        init_range = math.sqrt(6.0 / sum(shape))
        return tf.Variable(tf.random_uniform(shape, minval=-init_range,
                                             maxval=init_range, seed=20170801),
                                             trainable=trainable)
    elif init_type == 'normal':
        return tf.Variable(tf.random_normal(shape, stddev=1.0 / math.sqrt(shape[0]),
                                            seed=20170801), trainable = trainable)
    elif init_type == 'uniform':
        return tf.Variable(tf.random_uniform(shape, minval=-0.05, maxval=0.05,
                                             seed=20170801), trainable=trainable)
    return None
#enddef


def load_mention_pair_data(sentence_file, mention_idx_file,
                           feats_file=None, feats_meta_file=None):
    """
    Reads the given sentence and mention pair index files, mapping
    sentence IDs to word2vec matrices, and mention pair IDs to
    first/last word indices, sentence IDs, labels, and
    normalization vectors (for averages, where appropriate).
    Optionally reads a feature and feature meta file and adds these
    vectors to the data dict

    :param sentence_file:   File containing captions and IDs
    :param mention_idx_file: File containing mention pair IDs and index tuples
    :param feats_file: File containing engineered (sparse) features
    :param feats_meta_file: File associating engineered feature indices with
                            human-readable names
    :return Dictionary storing the aforementioned dictionaries
    """
    global __w2v
    data_dict = dict()

    # Load the sentence file, which we assume is
    # in the format
    #   <img_id>#<cap_idx>    <caption_less_punc>
    data_dict['sentences'] = dict()
    if sentence_file is not None:
        with open(sentence_file, 'r') as f:
            for line in f.readlines():
                id_split = line.split("\t")
                data_dict['sentences'][id_split[0].strip()] = \
                    __w2v.get_w2v_matrix(id_split[1].split(" "))
                #endfor
                #endwith
    #endif

    # Get the maximum sentence length (max seq length)
    data_dict['max_seq_len'] = -1
    for sentence_matrix in data_dict['sentences'].values():
        if len(sentence_matrix) > data_dict['max_seq_len']:
            data_dict['max_seq_len'] = len(sentence_matrix)
    #endfor

    # Load the mention index file, which
    # we assume is in the format
    #   <pair_id>     0,<cap_end_idx>,<m1_start>,<m1_end>,<m2_start>,<m2_end>   <label>
    # for the intra-caption case, and
    #   <pair_id>     0,<cap_1_end_idx>,0,<cap_2_end_idx>,<m1_start>,<m1_end>,<m2_start>,<m2_end>   <label>
    # in the cross-caption case
    # where the pair ID is
    #   doc:<img_id>;caption_1:<cap_1_idx>;mention_1:<mention_1_idx>;caption_2:<cap_2_idx>;mention_2:<m_2_idx>
    # UPDATE: In this representation, we're going to ignore the caption
    # token indices and focus entirely on mention indices; our goal is to produce
    #   <m1_start> <m1_range> <m1_end> <m2_start> <m2_range> <m2_end>
    # so for now we only need <m1_start>, <m1_end>, <m2_start>, <m2_end>
    data_dict['mention_pair_cap_ids'] = dict()
    data_dict['mention_pair_indices'] = dict()
    data_dict['mention_pair_norm_vecs'] = dict()
    data_dict['mention_pair_labels'] = dict()
    if mention_idx_file is not None:
        with open(mention_idx_file, 'r') as f:
            for line in f.readlines():
                # Parse the ID to get the caption(s) from it
                id_split = line.split("\t")
                pair_id = id_split[0].strip()

                # Associate each mention pair with a tuple of its caption IDs
                id_dict = util.kv_str_to_dict(pair_id)
                cap_1 = id_dict['doc'] + "#" + id_dict['caption_1']
                cap_2 = id_dict['doc'] + "#" + id_dict['caption_2']
                data_dict['mention_pair_cap_ids'][pair_id] = (cap_1, cap_2)

                # Parse the mention pair indices as actual integers
                # and store those lists
                indices_str = id_split[1].strip().split(",")
                indices = list()
                for i in indices_str:
                    indices.append(int(i))
                data_dict['mention_pair_indices'][pair_id] = indices

                # Create the normalization vectors for each mention, associating
                # each with 2*n_max_seq 0s, except for indices corresponding
                # to the mention (twice, because we need both forward and backward);
                # Since we're averaging the forward and backward outputs of a mention,
                # we want the norm value to be 1 / 2|m|
                # NOTE: If we're running into memory problems, we should have one
                # of these arrays per unique mention, not one per appearance in a mention
                # pair
                norm_vec_i = np.zeros(2*data_dict['max_seq_len'])
                norm_vec_j = np.zeros(2*data_dict['max_seq_len'])
                norm_i = 1 / (2 * (1 + indices[1] - indices[0]))
                norm_j = 1 / (2 * (1 + indices[3] - indices[2]))
                for idx in range(indices[0], indices[1]+1):
                    norm_vec_i[idx] = norm_i
                    norm_vec_i[data_dict['max_seq_len'] + idx] = norm_i
                for idx in range(indices[2], indices[3]+1):
                    norm_vec_j[idx] = norm_j
                    norm_vec_j[data_dict['max_seq_len'] + idx] = norm_j
                #endfor
                data_dict['mention_pair_norm_vecs'][pair_id] = (norm_vec_i, norm_vec_j)

                # Represent the label as a one-hot (and since this is for
                # relations, we know this should be 4-dimensional
                label = np.zeros([4])
                label[int(id_split[2].strip())] = 1.0
                data_dict['mention_pair_labels'][pair_id] = label
            #endfor
        #endwith
    #endif

    # If feature files have been provided, add those to the data dict too
    if feats_file is not None and feats_meta_file is not None:
        meta_dict = None
        if feats_meta_file is not None:
            meta_dict = json.load(open(feats_meta_file, 'r'))
            data_dict['max_feat_idx'] = meta_dict['max_idx']
        X, _, IDs = util.load_sparse_feats(feats_file, meta_dict)
        data_dict['mention_pair_feats'] = dict()
        for i in range(0, len(IDs)):
            data_dict['mention_pair_feats'][IDs[i]] = X[i]
    #endif
    return data_dict
#enddef


def log_eval(gold_labels, pred_labels, class_names, log=None):
    """
    Logs the counts and evaluation metrics, based on the given
    counts and labels
    :param gold_labels: numpy array of the data's labels (y)
    :param pred_labels: numpy array of the predict labels
    :param class_names: human label names corresponding to int labels
    :param log: The logging object to use (if None prints to console)
    """

    # Print the counts
    gold_counts = np.bincount(gold_labels)
    pred_bins = np.bincount(pred_labels)
    pred_counts = np.pad(pred_bins, 4-len(pred_bins), 'constant')
    log.info(None, "Gold Counts -- n: %d; c: %d; b: %d; p: %d",
             int(gold_counts[0]), int(gold_counts[1]),
             int(gold_counts[2]), int(gold_counts[3]))
    log.info(None, "Pred Counts -- n: %d; c: %d; b: %d; p: %d",
             int(pred_counts[0]), int(pred_counts[1]),
             int(pred_counts[2]), int(pred_counts[3]))

    # Followed by the precision/recall/f1 for our labels
    precision_scores = sk_metrics.precision_score(y_true=gold_labels,
                                                  y_pred=pred_labels,
                                                  average=None)
    recall_scores = sk_metrics.recall_score(y_true=gold_labels,
                                            y_pred=pred_labels, average=None)
    f1_scores = sk_metrics.f1_score(y_true=gold_labels,
                                    y_pred=pred_labels, average=None)
    confusion_matrix = sk_metrics.confusion_matrix(y_true=gold_labels, y_pred=pred_labels)

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
    metrics_str = util.rows_to_str(metrics, True)

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
    conf_str = util.rows_to_str(conf_matrix_table, True, False)

    if log is None:
        print metrics_str
        print conf_str
    else:
        log.info("\n" + metrics_str)
        log.info("\n" + conf_str)
#enddef


def leaky_relu(logits, alpha=0.01):
    """
    Simple (ideally efficient) representation of the
    leaky relu activation function
    :param logits:
    :param alpha:
    :return:
    """
    return tf.maximum(logits, alpha*logits)
#enddef
