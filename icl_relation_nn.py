import tensorflow as tf
import numpy as np
import os
from sklearn import metrics as sk_metrics
from argparse import ArgumentParser
from os.path import abspath, expanduser

from Word2VecUtil import Word2VecUtil
from LogUtil import LogUtil
import icl_util as util
import nn_util

___author___ = "ccervantes"

WORD_2_VEC_PATH = '/shared/projects/word2vec/GoogleNews-vectors-negative300.bin.gz'
N_EMBEDDING_FEATS = 300
CLASSES = ['n', 'c', 'b', 'p']
RELATION_TYPES = ['intra', 'cross']


def set_tf_seed():
    """
    Sets the tensorflow random seed to a fixed number
    for experimental consistency
    :return:
    """
    tf.set_random_seed(20170927)
#enddef


def load_data(sentence_file, mention_idx_file):
    """
    Reads the given sentence and mention index files, and returns
    four dictionaries, mapping
        - caption IDs to captions
        - mention pair IDs to word index tuples
        - mention pair IDs to the captions from which they came
        - mention pair IDs to one-hot labels

    :param sentence_file:   File containing captions and IDs
    :param mention_idx_file: File containing mention pair IDs and index tuples
    :return: A four-tuple of dictionaries used by load_batch()
    """

    # Load the sentence file, which we assume is
    # in the format
    #   <img_id>#<cap_idx>    <caption_less_punc>
    sentence_dict = dict()
    if sentence_file is not None:
        with open(sentence_file, 'r') as f:
            for line in f.readlines():
                id_split = line.split("\t")
                sentence_dict[id_split[0].strip()] = id_split[1].split(" ")
            #endfor
        #endwith
    #endif

    # Load the mention index file, which
    # we assume is in the format
    #   <pair_id>     0,<cap_end_idx>,<m1_start>,<m1_end>,<m2_start>,<m2_end>   <label>
    # for the intra-caption case, and
    #   <pair_id>     0,<cap_1_end_idx>,0,<cap_2_end_idx>,<m1_start>,<m1_end>,<m2_start>,<m2_end>   <label>
    # in the cross-caption case
    # where the pair ID is
    #   doc:<img_id>;caption_1:<cap_1_idx>;mention_1:<mention_1_idx>;caption_2:<cap_2_idx>;mention_2:<m_2_idx>
    # Here, we want to map the mention pair IDs to the tuple of relevant indices,
    # the pair IDs to the original caption indices, and the pair IDs
    # to their label
    mention_pair_indices = dict()
    mention_pair_caps = dict()
    mention_pair_labels = dict()
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
                mention_pair_caps[pair_id] = (cap_1, cap_2)

                # Parse the mention pair indices as actual integers
                # and store those lists
                indices_str = id_split[1].strip().split(",")
                indices = list()
                for i in indices_str:
                    indices.append(int(i))
                mention_pair_indices[pair_id] = indices

                # Represent the label as a one-hot (and since this is for
                # relations, we know this should be 4-dimensional
                label = np.zeros([4])
                label[int(id_split[2].strip())] = 1.0
                mention_pair_labels[pair_id] = label
            #endfor
        #endwith
    #endif

    return sentence_dict, mention_pair_indices, mention_pair_caps, mention_pair_labels
#enddef


def load_batch(rel_type, mention_pairs, sentence_dict,
               mention_pair_indices, mention_pair_caps,
               mention_pair_labels):
    """
    Loads a batch of mention pairs, given a set of pair IDs;
    returns an array of sentences, the array of their lengths,
    a matrix of mention pair word indices (which will later
    be split into sets of six or eight) and an array of labels

    :param rel_type: Intra/cross relations
    :param mention_pairs: List of mention pair IDs
    :param sentence_dict: Dictionary mapping caption IDs
                          to the captions themselves
    :param mention_pair_indices: Dictionary mapping mention
                                 pair IDs to word-index tuples
    :param mention_pair_caps: Dictionary mapping mention pair
                              IDs to source caption IDs
    :param mention_pair_labels: Dictionary mapping mention
                                pair IDs to labels
    :return: Four tuple of sentences [n_batch_size, n_max_seq],
             sentence lengths [n_batch_size], mention pair word indices
             [(6 or 8)*n_batch_size, 3], and labels [batch_size, n_classes]
    """
    global w2v, N_EMBEDDING_FEATS, CLASSES, RELATION_TYPES

    # Build the sentence tensor
    n_max_seq = get_max_seq_size(sentence_dict)
    batch_size = len(mention_pairs)
    n_seq = None
    if rel_type == RELATION_TYPES[0]:
        n_seq = batch_size
    elif rel_type == RELATION_TYPES[1]:
        n_seq = 2 * batch_size
    sentence_batch = np.zeros([n_seq, n_max_seq, N_EMBEDDING_FEATS])
    sentence_lengths = np.zeros([n_seq])
    batch_idx = 0   # Index into the sentence batch tensor and
                    # sentence lengths vector
    for i in range(0, batch_size):
        # set this sentence idx to the sentence embeddings,
        # implicitly padding the end of the sequence with 0s
        sentence_ids = list()
        if rel_type == RELATION_TYPES[0]:
            sentence_ids.append(mention_pair_caps[mention_pairs[i]][0])
        elif rel_type == RELATION_TYPES[1]:
            for s_id in mention_pair_caps[mention_pairs[i]]:
                sentence_ids.append(s_id)
        for sentence_id in sentence_ids:
            sentence = sentence_dict[sentence_id]
            sentence_matrix = w2v.get_w2v_matrix(sentence)
            for j in range(0, len(sentence_matrix)):
                row = sentence_matrix[j]
                for k in range(0, len(row)):
                    sentence_batch[batch_idx][j][k] = row[k]
            sentence_lengths[batch_idx] = len(sentence)
            batch_idx += 1
        #endfor
    #endfor

    # In order to load a batch of mentions pairs, we need a
    # [6*batch_size, 3] or [8*batch_size, 3] matrix, where
    # the three elements are (lstm_direction, sentence_idx, word_idx)
    # So first, put this batch's mentions in a matrix
    n_indices = -1
    for pair_id in mention_pairs:
        l = len(mention_pair_indices[pair_id])
        if l > n_indices:
            n_indices = l
    #endfor
    mention_pair_batch = np.zeros([n_indices * batch_size, 3])
    label_batch = np.zeros([batch_size, len(CLASSES)])
    batch_idx = 0   # Index into the mention pair batch matrix
    for i in range(0, batch_size):
        # Add this pair's label to the label batch
        pair_id = mention_pairs[i]
        label_batch[i] = mention_pair_labels[pair_id]

        # Get the referred sentence indices
        sentence_idx_1 = None
        sentence_idx_2 = None
        if rel_type == RELATION_TYPES[0]:
            # In the intra caption case, the sentence index
            # is the batch index
            sentence_idx_1 = i
            sentence_idx_2 = i
        elif rel_type == RELATION_TYPES[1]:
            # In the cross caption case, they should be the
            # first and second sentences after the current
            # batch size * 2
            sentence_idx_1 = i * 2
            sentence_idx_2 = i * 2 + 1
        #endif

        # Note: this isn't going to be elegant, but it'll
        # be clear; I tried the elegant way and it was
        # not that great and kind of confusing
        j = 0   # index into the word index tuple
        word_indices = mention_pair_indices[pair_id]

        # backward direction of the first word of the first sentence
        mention_pair_batch[batch_idx] = np.array((1, sentence_idx_1, word_indices[j]))
        batch_idx += 1
        j += 1

        # forward direction of the last word of the first sentence
        mention_pair_batch[batch_idx] = np.array((0, sentence_idx_1, word_indices[j]))
        batch_idx += 1
        j += 1

        # If this is a cross-caption batch, the next two indices are
        # for the second sentence
        if rel_type == RELATION_TYPES[1]:
            # backward direction of the first word of the second sentence
            mention_pair_batch[batch_idx] = np.array((1, sentence_idx_2, word_indices[j]))
            batch_idx += 1
            j += 1

            # forward direction of the last word of the second sentence
            mention_pair_batch[batch_idx] = np.array((0, sentence_idx_2, word_indices[j]))
            batch_idx += 1
            j += 1
        #endif

        # Finally, get the four indices for this mention pair
        # Backward direction; first word of the first mention
        mention_pair_batch[batch_idx] = np.array((1, sentence_idx_1, word_indices[j]))
        batch_idx += 1
        j += 1

        # Forward direction; last word of the first mention
        mention_pair_batch[batch_idx] = np.array((0, sentence_idx_1, word_indices[j]))
        batch_idx += 1
        j += 1

        # Backward direction; first word of the second mention
        mention_pair_batch[batch_idx] = np.array((1, sentence_idx_2, word_indices[j]))
        batch_idx += 1
        j += 1

        # Forward direction; last word of the second mention
        mention_pair_batch[batch_idx] = np.array((0, sentence_idx_2, word_indices[j]))
        batch_idx += 1
    #endfor

    return sentence_batch, sentence_lengths, mention_pair_batch, label_batch
#enddef


def __dep__load_batch(mention_pairs, sentence_dict, mention_pair_indices,
               mention_pair_caps, mention_pair_labels):
    """
    Loads a batch of mention pairs, given a set of pair IDs; returns
    an array of sentences, the array of their lengths, a matrix
    of mention pair word indices (which will later be split into
    sets of six or eight), and an array of labels

    :param mention_pairs: List of mention pair IDs
    :param sentence_dict: Dictionary mapping caption IDs to the captions themselves
    :param mention_pair_indices: Dictionary mapping mention pair IDs to
                                 word-index tuples
    :param mention_pair_caps:    Dictionary mapping mention pair IDs to source caption IDs
    :param mention_pair_labels:  Dictionary mapping mention pair IDs to labels
    :return: Four tuple of sentences [n_sentences, n_max_seq], sentence lengths [n_sentences],
             mention pair word indices [6*n_batch_size, 3], and labels [batch_size, n_classes]
    """
    global w2v, N_EMBEDDING_FEATS, CLASSES

    # Given the set of mention pair IDs, collect a list of unique caption IDs
    sentence_ids = set()
    for pair_id in mention_pairs:
        sentence_pair = mention_pair_caps[pair_id]
        sentence_ids.append(sentence_pair[0])
        #sentence_ids.append(sentence_pair[1])
    #endfor
    sentence_ids = list(sentence_ids)
    sentence_indices = util.list_to_index_dict(sentence_ids)

    # Build the sentence tensor from these IDs
    n_max_seq = get_max_seq_size(sentence_dict)
    sentence_batch = np.zeros([len(sentence_ids), n_max_seq,
                               N_EMBEDDING_FEATS])
    sentence_lengths = np.zeros([len(sentence_ids)])
    for s in range(0, len(sentence_ids)):
        # set this sentence idx to the sentence embeddings,
        # implicitly padding the end of the sequence with 0s
        sentence = sentence_dict[sentence_ids[s]]
        sentence_matrix = w2v.get_w2v_matrix(sentence)
        for w in range(0, len(sentence_matrix)):
            row = sentence_matrix[w]
            for j in range(0, len(row)):
                sentence_batch[s][w][j] = row[j]
            #endfor
        #endfor
        sentence_lengths[s] = len(sentence)
    #endfor

    # In order to load a batch of mentions pairs, we need a
    # [6*batch_size, 3] or [8*batch_size, 3] matrix, where
    # the three elements are (lstm_direction, sentence_idx, word_idx)
    # So first, put this batch's mentions in a matrix
    n_indices = -1
    for pair_id in mention_pairs:
        l = len(mention_pair_indices[pair_id])
        if l > n_indices:
            n_indices = l
    #endfor
    mention_pair_batch = np.zeros([n_indices * len(mention_pairs), 3])
    label_batch = np.zeros([len(mention_pairs), len(CLASSES)])
    batch_idx = 0
    for pair_idx in range(0, len(mention_pairs)):
        pair_id = mention_pairs[pair_idx]

        # Add this pair's label to the label batch
        label_batch[pair_idx] = mention_pair_labels[pair_id]

        # Get the tuple of word indices
        word_indices = mention_pair_indices[pair_id]
        w = 0   # index into the tuple of word indices

        # Get the tuple of sentences, which we then convert to
        # sentence indices
        sentence_id_pair = mention_pair_caps[pair_id]
        sentence_idx_1 = sentence_indices[sentence_id_pair[0]]
        sentence_idx_2 = sentence_indices[sentence_id_pair[1]]

        # backward direction of the first word of the first sentence
        mention_pair_batch[batch_idx] = np.array((1, sentence_idx_1, word_indices[w]))
        batch_idx += 1
        w += 1

        # forward direction of the last word of the first sentence
        mention_pair_batch[batch_idx] = np.array((0, sentence_idx_1, word_indices[w]))
        batch_idx += 1
        w += 1

        # If this is a cross-caption batch, the next two indices are
        # for the second sentence
        if len(word_indices) == 8:
            # backward direction of the first word of the second sentence
            mention_pair_batch[batch_idx] = np.array((1, sentence_idx_2, word_indices[w]))
            batch_idx += 1
            w += 1

            # forward direction of the last word of the second sentence
            mention_pair_batch[batch_idx] = np.array((0, sentence_idx_2, word_indices[w]))
            batch_idx += 1
            w += 1
        #endif

        # Finally, get the four indices for this mention pair
        # Backward direction; first word of the first mention
        mention_pair_batch[batch_idx] = np.array((1, sentence_idx_1, word_indices[w]))
        batch_idx += 1
        w += 1

        # Forward direction; last word of the first mention
        mention_pair_batch[batch_idx] = np.array((0, sentence_idx_1, word_indices[w]))
        batch_idx += 1
        w += 1

        # Backward direction; first word of the second mention
        mention_pair_batch[batch_idx] = np.array((1, sentence_idx_2, word_indices[w]))
        batch_idx += 1
        w += 1

        # Forward direction; last word of the second mention
        mention_pair_batch[batch_idx] = np.array((0, sentence_idx_2, word_indices[w]))
        batch_idx += 1
    #endfor

    return sentence_batch, sentence_lengths, mention_pair_batch, label_batch
#enddef


def get_max_seq_size(sentence_dict):
    """
    Reads through a sentence dictionary
    and returns the length of the longest
    sentence
    :param sentence_dict: Sentence dictionary (from
                          load_sentence_files)
    :return:              Length of longest sentence
    """
    max_len = -1
    for sentence in sentence_dict.values():
        if len(sentence) > max_len:
            max_len = len(sentence)
    return max_len
#enddef


def setup_bidirectional_lstm(param_dict, n_max_seq, n_feats, n_hidden, n_parallel=32):
    """
    Sets up the tensorflow placeholders, adding relevant variables
    to the given param_dict (where variable names have the 'bidirectional_lstm'
    prefix

    :param param_dict: Dictionary of tensorflow names and variables
    :param n_max_seq:  Size of the largest sequence in the LSTM
    :param n_feats:    Number of word features
    :param n_hidden:   Size of the hidden layer in the LSTM cells
    :param n_parallel: Number of parallel tasks the RNN may run (def: 32)
    """
    # input
    x = tf.placeholder(tf.float32, [None, n_max_seq, n_feats])
    param_dict[tf.get_variable_scope().name+'/x'] = x

    # sequence lengths
    seq_lengths = tf.placeholder(tf.int32, [None])
    param_dict[tf.get_variable_scope().name+'/seq_lengths'] = seq_lengths

    # dropout percentage
    dropout = tf.placeholder(tf.float32)
    param_dict[tf.get_variable_scope().name+'/dropout'] = dropout

    # set up the cells
    lstm_cell = {}
    for direction in ["fw", "bw"]:
        with tf.variable_scope(direction):
            lstm_cell[direction] = \
                tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
            lstm_cell[direction] = \
                tf.nn.rnn_cell.DropoutWrapper(lstm_cell[direction],
                                              output_keep_prob=dropout)
            #endwith
    #endfor

    # Define the bidirectional dynamic-sized RNN
    #   inputs: [batch_size, max_sequence_length, num_features]; the inputs
    #   sequence length: [batch_size]; the sequence lengths
    #   parallel_iterations: (Default: 32); number of iterations to run in parallel;
    #                        values >> 1 run faster but take more memory
    #   swap_memory: Whether to swap from CPU to GPU (use if model can't fit on the GPU)
    #   time_major: Whether the shape of the input tensors' first dimension is
    #               time (sequence) or -- as is typical -- batch (example)
    outputs, states = \
        tf.nn.bidirectional_dynamic_rnn(lstm_cell["fw"],
                                        lstm_cell["bw"],
                                        x, dtype=tf.float32,
                                        parallel_iterations=n_parallel,
                                        sequence_length=seq_lengths,
                                        time_major=False)
    param_dict[tf.get_variable_scope().name+'/outputs'] = outputs
#enddef


def setup_relation(rel_type, batch_size, n_max_seq, n_hidden_lstm,
                   n_hidden_rel, weighted_classes, n_parallel,
                   lrn_rate):
    """
    Sets up the relation classification network, which -- each batch --
    passes a (<= batch_size) set of sentences to a bidirectional LSTM
    and feeds selected output of that LSTM to a hidden layer; which
    output nodes are passed depends on the mention pairs in the batch.

    The functionality of this setup is highly dependent on the synergy between this
    function and the input its given: in particular, the input sentences must
    be in the same order as the mention indices expect them to be, and the mention
    indices must implicitly define a 6 or 8 tuple of words (every six or eight
    elements must refer to a mention pair, depending on the intra/cross setting)

    :param rel_type: intra or cross relations
    :param batch_size: Number of mention pairs to run each batch
    :param n_max_seq: Size of the largest sentence
    :param n_hidden_lstm: Number of hidden units in the lstm cells
    :param n_hidden_rel: Number of hidden units to which the mention pairs'
                         representation is passed
    :param weighted_classes: Whether to weight the examples by their class inversely
                    with the frequency of that class
    :param n_parallel: Number of parallel jobs the LSTM may perform at once
    :param lrn_rate: Learning rate of the optimizer
    :return: Parameter dict, associating tensorflow placeholders with their names
    """
    global CLASSES, N_EMBEDDING_FEATS

    param_dict = dict()

    # Each mention pair is a one-hot for its label (n,c,b,p)
    y = tf.placeholder(tf.float32, [batch_size, len(CLASSES)])
    param_dict['y'] = y

    # Set up the mention pair indices, which are our examples (we only send
    # relevant sentences to the LSTM); note here that for tensor manipulation
    # reasons, we are actually given a list of (lstm_dir,sentence,word) indices,
    # and since each mention pair is a concatenation of six or eight
    # words, the mention pair indices are of size [num_words_per_pair*batch_size, 3]
    num_words_per_pair = 0
    if rel_type == RELATION_TYPES[0]:
        num_words_per_pair = 6
    elif rel_type == RELATION_TYPES[1]:
        num_words_per_pair = 8
    mention_pair_indices = tf.placeholder(tf.int32,
                                          [num_words_per_pair * batch_size, 3])
    param_dict['mention_pair_indices'] = mention_pair_indices

    # Set up the bidirectional LSTM
    with tf.variable_scope('bidirectional_lstm'):
        setup_bidirectional_lstm(param_dict, n_max_seq, N_EMBEDDING_FEATS,
                                 n_hidden_lstm, n_parallel)

    # Get the outputs, which are a (fw,bw) tuple of
    # [batch_size, seq_length, n_hidden_lstm matrices
    outputs = param_dict['bidirectional_lstm/outputs']

    # Recall that the outputs are a tuple of forward and backward
    # directions, and thus the true size of outputs is
    # [2, n_seq, n_max_seq, n_hidden_lstm]
    # and since mention pair indices are
    # [num_words_per_pair*batch_size, 3]
    # we want to gather num_words_per_pair * batch_size vectors
    # of size n_hidden_lstm
    # -- where each mention pair index is indexing the output matrix --
    # and then we want to reshape those vectors into batch_size vectors
    # of size num_words_per_pair*n_hidden_lstm,
    # resulting in a batch_input vector of size
    # [batch_size, num_words_per_pair*n_hidden_lstm]
    n_hidden_ingress = num_words_per_pair * n_hidden_lstm
    batch_input = tf.reshape(tf.gather_nd(outputs, mention_pair_indices),
                             [batch_size, n_hidden_ingress])
    param_dict['batch_input'] = batch_input

    # dropout percentage
    dropout = tf.placeholder(tf.float32)
    param_dict['dropout'] = dropout

    # Set up the final hidden layer
    with tf.variable_scope("hdn_ingress"):
        weights = nn_util.get_weights([n_hidden_ingress, n_hidden_rel])
        param_dict[tf.get_variable_scope().name+'/weights'] = weights
        biases = nn_util.get_biases([1, n_hidden_rel])
        param_dict[tf.get_variable_scope().name+'/biases'] = biases
        hdn_logits = tf.nn.tanh(tf.matmul(batch_input, weights) + biases)
    with tf.variable_scope("hdn_egress"):
        weights = nn_util.get_weights([n_hidden_rel, len(CLASSES)])
        param_dict[tf.get_variable_scope().name+'/weights'] = weights
        biases = nn_util.get_biases([1, len(CLASSES)])
        param_dict[tf.get_variable_scope().name+'/biases'] = biases
        logits = tf.nn.dropout(tf.nn.tanh(tf.matmul(hdn_logits, weights) + biases), dropout)
    #endwith
    param_dict['logits'] = logits

    # In order to use the weighted softmax cross entropy, we need to
    # get the weights, which first requires us to get label counts for this batch
    # Note: this results in a [1, n_classes] vector
    batch_identity = tf.constant(np.ones([1, batch_size]), dtype=tf.float32)
    gold_label_counts = tf.matmul(batch_identity, y)
    param_dict['gold_label_counts'] = gold_label_counts

    # prediction is just the minimization of the softmax cross entropy, unless
    # we're balancing class weights
    if weighted_classes:
        # Once we have counts, we want the weights to be the inverse frequency
        class_weights = tf.subtract(
            tf.constant(np.ones([1, len(CLASSES)]), dtype=tf.float32),
            tf.scalar_mul(tf.constant(1/batch_size, dtype=tf.float32),
                          gold_label_counts))

        # Class weights is now a [1,n_classes] vector containing inverse class frequency
        # percentages; now we need to get example weights
        # Note: This operation is
        #       1) Take our [1, n_classes] class weights and transpose them
        #          to [n_classes, 1]
        #       2) Multiply y -- [batch_size, n_classes] -- by the transposed
        #          class weights, resulting in [batch_size, 1]
        #       3) Squeeze that last dimension out, turning the matrix into a vector;
        #          this is necessary for the softmax operation
        example_weights = tf.squeeze(tf.matmul(y, tf.transpose(class_weights)))

        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits,
                                                        weights=example_weights)
    else:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_sum(cross_entropy)
    param_dict['loss'] = loss

    # Adam Optimizer is a stochastic gradient descent style algo from 2014 that's
    # apparently amazing
    train_op = tf.train.AdamOptimizer(lrn_rate).minimize(loss)
    param_dict['train_op'] = train_op

    # Evaluate model
    pred = tf.argmax(logits, 1)
    correct_pred = tf.equal(pred, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    pred_label_counts = tf.matmul(batch_identity,
                                  tf.one_hot(tf.cast(pred, dtype=tf.int32), len(CLASSES)))
    param_dict['pred_label_counts'] = pred_label_counts
    param_dict['pred'] = pred
    param_dict['accuracy'] = accuracy

    # Print sizes, just to make sure of everything
    for p in param_dict.keys():
        if p == 'bidirectional_lstm/outputs':
            log.debug(None, "%-20s: %s", "bidi_outputs_fw",
                      str(param_dict[p][0].get_shape().as_list()))
            log.debug(None, "%-20s: %s", "bidi_outputs_bw",
                      str(param_dict[p][1].get_shape().as_list()))
        elif p not in ('bidirectional_lstm/dropout', 'dropout', 'pred', 'train_op', 'accuracy', 'loss'):
            log.debug(None, "%-20s: %s",
                      p.replace("bidirectional_lstm", "bidi"),
                      str(param_dict[p].get_shape().as_list()))
        #endif
    #endfor

    # return the parameter dictionary, so we can use feed_dict and run
    # operations in the session
    return param_dict
#enddef


def train(rel_type, sentence_file, mention_idx_file, n_iter,
          batch_size, n_hidden_lstm, n_hidden_rel,
          weighted_classes, dropout_p, n_parallel, lrn_rate):
    """
    Trains a relation model, as defined in setup_relation

    :param rel_type: intra or cross relations
    :param sentence_file: File with captions
    :param mention_idx_file: File with mention pair word indices
    :param n_iter: Number of iterations for which to train
    :param batch_size: Number of mention pairs to run each batch
    :param n_hidden_lstm: Number of hidden units in the lstm cells
    :param n_hidden_rel: Number of hidden units to which the mention pairs'
                         representation is passed
    :param weighted_classes: Whether to weight the examples by their
                             class inversely with the frequency of
                             that class
    :param dropout_p: Dropout probability (percent to keep)
    :param n_parallel: Number of parallel jobs the LSTM may perform at once
    :param lrn_rate: Learning rate of the optimizer
    :return:
    """
    log.info("Loading data from " + sentence_file +
             " and " + mention_idx_file)
    sentence_dict, mention_pair_indices, mention_pair_caps, mention_pair_labels = \
        load_data(sentence_file, mention_idx_file)
    mention_pairs = mention_pair_indices.keys()
    n_max_seq = get_max_seq_size(sentence_dict)
    # gold_labels = list()
    # gold_counts = [0, 0, 0, 0]
    # pred_labels = list()
    # pred_counts = [0, 0, 0, 0]

    log.info("Setting up network architecture")
    tf_vars = setup_relation(rel_type, batch_size, n_max_seq, n_hidden_lstm,
                             n_hidden_rel, weighted_classes, n_parallel, lrn_rate)

    # Get the relevant tensorflow variables from the dictionary
    lstm_x = tf_vars['bidirectional_lstm/x']
    lstm_seq_lengths = tf_vars['bidirectional_lstm/seq_lengths']
    lstm_dropout = tf_vars['bidirectional_lstm/dropout']
    pair_indices = tf_vars['mention_pair_indices']
    dropout = tf_vars['dropout']
    y = tf_vars['y']
    loss = tf_vars['loss']
    train_op = tf_vars['train_op']
    accuracy = tf_vars['accuracy']
    pred = tf_vars['pred']

    log.info("Training")
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        # Initialize all our variables
        sess.run(tf.global_variables_initializer())

        # Set the random seed
        set_tf_seed()

        # Iterate for n_iter iterations
        for i in range(0, n_iter):
            log.log_status('info', None,
                           "Training; %d iterations complete (%.2f%%)",
                           i, 100.0 * i / n_iter)

            # Grab a random batch of the data
            batch_mention_pairs = \
                np.random.choice(mention_pairs, batch_size, False)
            batch_sentences, batch_sent_lengths, \
                batch_pair_indices, batch_labels = \
                load_batch(rel_type, batch_mention_pairs, sentence_dict,
                           mention_pair_indices, mention_pair_caps,
                           mention_pair_labels)

            # Train
            sess.run(train_op, feed_dict={lstm_x: batch_sentences,
                                          lstm_seq_lengths: batch_sent_lengths,
                                          lstm_dropout: dropout_p,
                                          pair_indices: batch_pair_indices,
                                          dropout: dropout_p,
                                          y: batch_labels})

            # Every thousand iterations, run accuracy
            # on our batches and save the model
            if (i+1) % 500 == 0 or i >= n_iter - 1:
                saver.save(sess, model_file)
                acc = sess.run(accuracy, feed_dict={lstm_x: batch_sentences,
                                                    lstm_seq_lengths: batch_sent_lengths,
                                                    lstm_dropout: dropout_p,
                                                    pair_indices: batch_pair_indices,
                                                    dropout: dropout_p,
                                                    y: batch_labels})
                batch_loss = sess.run(loss, feed_dict={lstm_x: batch_sentences,
                                                       lstm_seq_lengths: batch_sent_lengths,
                                                       lstm_dropout: dropout_p,
                                                       pair_indices: batch_pair_indices,
                                                       dropout: dropout_p,
                                                       y: batch_labels})
                log.info(None, "Iter: %d; Batch loss: %.2f; Batch accuracy: %.2f%%",
                         i+1, batch_loss, 100.0 * acc)

                # Collect stats every batch, so we can print average scores
                #gold_labels.extend(np.argmax(batch_labels, 1))
                #pred_labels.extend(sess.run(pred, feed_dict={lstm_x: batch_sentences,
                #                                             lstm_seq_lengths: batch_sent_lengths,
                #                                             lstm_dropout: dropout_p,
                #                                             pair_indices: batch_pair_indices,
                #                                             dropout: dropout_p,
                #                                             y: batch_labels}))
                #gc = sess.run(tf_vars['gold_label_counts'], feed_dict={y: batch_labels})[0]
                #pc = sess.run(tf_vars['pred_label_counts'],
                #              feed_dict={lstm_x: batch_sentences,
                #                         lstm_seq_lengths: batch_sent_lengths,
                #                         lstm_dropout: dropout_p,
                #                         pair_indices: batch_pair_indices,
                #                         dropout: dropout_p,
                #                         y: batch_labels})[0]
                #for j in range(0, len(CLASSES)):
                #    gold_counts[j] += gc[j]
                #    pred_counts[j] += pc[j]
                #endfor

                gold_labels = np.argmax(batch_labels, 1)
                pred_labels = sess.run(pred, feed_dict={lstm_x: batch_sentences,
                                                        lstm_seq_lengths: batch_sent_lengths,
                                                        lstm_dropout: dropout_p,
                                                        pair_indices: batch_pair_indices,
                                                        dropout: dropout_p,
                                                        y: batch_labels})
                gold_label_counts = tf_vars['gold_label_counts']
                gold_counts = sess.run(gold_label_counts,
                                       feed_dict={lstm_x: batch_sentences,
                                                  lstm_seq_lengths: batch_sent_lengths,
                                                  lstm_dropout: dropout_p,
                                                  pair_indices: batch_pair_indices,
                                                  dropout: dropout_p,
                                                  y: batch_labels})[0]
                pred_label_counts = tf_vars['pred_label_counts']
                pred_counts = sess.run(pred_label_counts,
                                       feed_dict={lstm_x: batch_sentences,
                                                  lstm_seq_lengths: batch_sent_lengths,
                                                  lstm_dropout: dropout_p,
                                                  pair_indices: batch_pair_indices,
                                                  dropout: dropout_p,
                                                  y: batch_labels})[0]

                log.info(None, "Gold Counts -- n: %d; c: %d; b: %d; p: %d",
                         int(gold_counts[0]), int(gold_counts[1]),
                         int(gold_counts[2]), int(gold_counts[3]))
                log.info(None, "Pred Counts -- n: %d; c: %d; b: %d; p: %d",
                         int(pred_counts[0]), int(pred_counts[1]),
                         int(pred_counts[2]), int(pred_counts[3]))

                # Get the metrics for this batch
                precision_scores = sk_metrics.precision_score(y_true=gold_labels,
                                                              y_pred=pred_labels,
                                                              average=None)
                recall_scores = sk_metrics.recall_score(y_true=gold_labels,
                                                        y_pred=pred_labels, average=None)
                f1_scores = sk_metrics.f1_score(y_true=gold_labels,
                                                y_pred=pred_labels, average=None)
                metrics = list()
                metrics.append(["", "P", "R", "F1"])
                for l in range(0, len(CLASSES)):
                    row = list()
                    row.append(CLASSES[l])

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
                log.info("\n" + util.rows_to_str(metrics, True))
            #endif
        #endfor
        log.info("Saving final model")
        saver.save(sess, model_file)
    #endwith
#enddef


def predict(rel_type, sentence_file, mention_idx_file, scores_file,
            batch_size, n_hidden_lstm, n_hidden_rel):
    log.info("Loading data from " + sentence_file +
             " and " + mention_idx_file)
    sentence_dict, mention_pair_indices, mention_pair_caps, mention_pair_labels = \
        load_data(sentence_file, mention_idx_file)
    mention_pairs = mention_pair_indices.keys()
    n_max_seq = get_max_seq_size(sentence_dict)

    log.info("Setting up network architecture")
    tf_vars = setup_relation(rel_type, batch_size, n_max_seq, n_hidden_lstm,
                             n_hidden_rel, False, 32, 0.001)

    # Get the relevant tensorflow variables from the dictionary
    lstm_x = tf_vars['bidirectional_lstm/x']
    lstm_seq_lengths = tf_vars['bidirectional_lstm/seq_lengths']
    lstm_dropout = tf_vars['bidirectional_lstm/dropout']
    pair_indices = tf_vars['mention_pair_indices']
    dropout = tf_vars['dropout']
    y = tf_vars['y']
    logits = tf_vars['logits']

    saver = tf.train.Saver(max_to_keep=100)

    with open(scores_file, 'w') as f:
        with tf.Session() as sess:
            # Restore our variables
            saver.restore(sess, model_file)

            # Set the random seed (though it shouldn't
            # matter for prediction)
            set_tf_seed()

            # Run until we don't have any more mention
            # pairs to predict for
            mention_pair_matrix = np.reshape(mention_pairs,
                                             [int(len(mention_pairs)/batch_size) + 1, batch_size])
            for i in range(0, mention_pair_matrix.shape[0]):
                log.log_status('info', None, 'Predicting; %d batches complete (%.2f%%)',
                               i, 100.0 * i / mention_pair_matrix.shape[0])

                # Grab a random batch of the data
                batch_sentences, batch_sent_lengths, \
                    batch_pair_indices, batch_labels = \
                    load_batch(rel_type, mention_pair_matrix[i], sentence_dict,
                               mention_pair_indices, mention_pair_caps,
                               mention_pair_labels)

                # Predict
                pred_scores = sess.run(logits, feed_dict={lstm_x: batch_sentences,
                                                          lstm_seq_lengths: batch_sent_lengths,
                                                          lstm_dropout: 1.0,
                                                          pair_indices: batch_pair_indices,
                                                          dropout: 1.0, y: batch_labels})
                for j in range(0, len(pred_scores)):
                    f.write(mention_pair_matrix[i][j] + "\t" + ",".join(pred_scores[j]))
            #endfor
        #endwith
    #endwith
#enddef

# Set up the global logger
#log = LogUtil('debug', 180)

# Parse arguments
parser = ArgumentParser("ImageCaptionLearn_py: Neural Network for Relation "
                        "Prediction; Bidirectional LSTM to hidden layer "
                        "to softmax over (n)ull, (c)oreference, su(b)set, "
                        "and su(p)erset labels")
parser.add_argument("--iter", type=int, default=1000,
                    help="train opt; number of iterations")
parser.add_argument("--batch_size", type=int, default=100,
                    help="train opt; number of random mention pairs per batch")
parser.add_argument("--hidden_lstm", type=int, default=300,
                    help="train opt; number of hidden units within "
                         "the LSTM cells")
parser.add_argument("--hidden_rel", type=int, default=300,
                    help="train opt; number of hidden units in the "
                         "layer after the LSTM")
parser.add_argument("--weighted_classes", action="store_true",
                    help="Whether to inversely weight the classes "
                         "in the loss")
parser.add_argument("--parallel", type=int, default=64,
                    help="train opt; number of tasks the LSTM may "
                         "run in parallel")
parser.add_argument("--learn_rate", type=float, default=0.001,
                    help="train opt; optimizer learning rate")
parser.add_argument("--dropout", type=float, default=1.0,
                    help="train opt; Dropout rate (probability to keep)")
parser.add_argument("--sentence_file", #required=True,
                    type=lambda f: util.arg_file_exists(parser, f),
                    help="File associating caption IDs with their captions "
                         "(punctuation tokens removed)")
parser.add_argument("--mention_idx_file", #required=True,
                    type=lambda f: util.arg_file_exists(parser, f),
                    help="File associating mention pair IDs with "
                         "the set of indices that define them")
parser.add_argument("--train", action='store_true', help='Trains a model')
parser.add_argument("--predict", action='store_true',
                    help='Predicts using pre-trained model')
parser.add_argument("--rel_type", choices=['intra', 'cross'],
                    help="Whether we're dealing with intra-caption or "
                         "cross-caption relations")
# parser.add_argument("--model_file",
#                    type=str, help="Model file to save/load")
# parser.add_argument("--log_file", type=str, help="File to which logs are written")
args = parser.parse_args()
arg_dict = vars(args)

# For quick experimentation purposes, we're going to set
# some default values here. Note that this is ONLY for
# the grid search over parameters and should be
# removed as we move forward
arg_dict['sentence_file'] = abspath(expanduser("~/data/tacl201708/nn/flickr30k_train_captions.txt"))
arg_dict['mention_idx_file'] = abspath(expanduser("~/data/tacl201708/nn/flickr30k_train_mentionPairs_intra.txt"))
model_file = abspath(expanduser("~/models/tacl201708/"))
log_file = abspath(expanduser("~/data/tacl201708/logs/flickr30k_train_"))
out_file_root = "relation_" + arg_dict['rel_type'] + "_nn_"
out_file_root += "iter" + str(int(arg_dict['iter']/1000)) + "k_"
out_file_root += "batch" + str(int(arg_dict['batch_size'])) + "_"
out_file_root += "dropout" + str(int(arg_dict['dropout'] * 100)) + "_"
out_file_root += "lstm" + str(int(arg_dict['hidden_lstm'])) + "_"
out_file_root += "rel" + str(int(arg_dict['hidden_rel']))
if arg_dict['weighted_classes']:
    out_file_root += "_weighted"
model_file += out_file_root + ".model"
# log_file += out_file_root + ".log"

# Set up the global logger
log = LogUtil('debug', 180)
util.dump_args(arg_dict, log)

# Grab the possibly-not-yet-created model file from the args
# model_file = abspath(expanduser(arg_dict['model_file']))

# Set up the word2vec utility once
log.info("Initializing word2vec")
w2v = Word2VecUtil(WORD_2_VEC_PATH)

# Set up the minimum tensorflow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Train, if training was specified
if arg_dict['train']:
    train(rel_type=arg_dict['rel_type'],
          sentence_file=arg_dict['sentence_file'],
          mention_idx_file=arg_dict['mention_idx_file'],
          n_iter=arg_dict['iter'], batch_size=arg_dict['batch_size'],
          n_hidden_lstm=arg_dict['hidden_lstm'],
          n_hidden_rel=arg_dict['hidden_rel'],
          weighted_classes=arg_dict['weighted_classes'],
          dropout_p=arg_dict['dropout'],
          n_parallel=arg_dict['parallel'],
          lrn_rate=arg_dict['learn_rate'])
#endif



