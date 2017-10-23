import math
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from nn_utils import core as nn_util
from nn_utils import data as nn_data
from nn_utils import eval as nn_eval
from utils import core as util
from utils.Logger import Logger

___author___ = "ccervantes"

N_EMBEDDING_FEATS = 300
CLASSES = ['n', 'c', 'b', 'p']
RELATION_TYPES = ['intra', 'cross']

rel_type = None
pair_enc_scheme = None
data_norm = False
embedding_type = None

# Set up the global logger
log = Logger('debug', 180)


def load_batch_first_avg_last(mention_pairs, data_dict):
    """
    Loads a batch of data, given a list of mention pairs
    and the data dictionary. This version of load_batch
    returns the necessary index matrices for producing a
        first_i avg_i last_i first_j avg_j last_j
    mention pair representation via lstm output
    tensor transformations

    :param mention_pairs: List of mention pair IDs
    :param data_dict: Dictionary of all data dictionaries (for sentences, etc)
    :return: dictionary of batch tensors with aforementioned keys
    """
    global N_EMBEDDING_FEATS, CLASSES, RELATION_TYPES, rel_type
    batch_tensors = dict()

    # We're either looking at batch_size sequences or 2*batch_size sequences,
    # depending on whether we're dealing with intra-caption or cross-caption
    # relations
    batch_size = len(mention_pairs)
    n_seq = None
    if rel_type == RELATION_TYPES[0]:
        n_seq = batch_size
    elif rel_type == RELATION_TYPES[1]:
        n_seq = 2 * batch_size

    # Populate our sentence tensor and sequence length array; since
    # we may have 2*batch_size sentences, we need a separate batch index
    batch_tensors['sentences'] = np.zeros([n_seq, data_dict['max_seq_len'], N_EMBEDDING_FEATS])
    batch_tensors['seq_lengths'] = np.zeros([n_seq])
    batch_idx = 0
    for i in range(0, batch_size):
        # set this sentence idx to the sentence embeddings,
        # implicitly padding the end of the sequence with 0s
        sentence_ids = list()
        if rel_type == RELATION_TYPES[0]:
            sentence_ids.append(data_dict['mention_pair_cap_ids'][mention_pairs[i]][0])
        elif rel_type == RELATION_TYPES[1]:
            for s_id in data_dict['mention_pair_cap_ids'][mention_pairs[i]]:
                sentence_ids.append(s_id)
        for sentence_id in sentence_ids:
            sentence_matrix = data_dict['sentences'][sentence_id]
            for j in range(0, len(sentence_matrix)):
                batch_tensors['sentences'][batch_idx][j] = sentence_matrix[j]
            batch_tensors['seq_lengths'][batch_idx] = len(sentence_matrix)
            batch_idx += 1
        #endfor
    #endfor

    # In our new scheme, we need four matrices of size [batch_size, 3]
    # corresponding to the first words of i mentions, last words
    # of i mentions, first words of j mentions, and last words of j mentions.
    # In addition, we'll also need four [batch_size, 2] matrices -- which will really just
    # pull all the outputs of each sentence, forward and back -- and two
    # [batch_size, 1, 2 * n_max_seq] normalization tensors for averaging
    batch_tensors['first_i_indices'] = np.zeros([batch_size, 3])
    batch_tensors['last_i_indices'] = np.zeros([batch_size, 3])
    batch_tensors['first_j_indices'] = np.zeros([batch_size, 3])
    batch_tensors['last_j_indices'] = np.zeros([batch_size, 3])
    batch_tensors['sent_i_fw_indices'] = np.zeros([batch_size, 2])
    batch_tensors['sent_i_bw_indices'] = np.zeros([batch_size, 2])
    batch_tensors['sent_j_fw_indices'] = np.zeros([batch_size, 2])
    batch_tensors['sent_j_bw_indices'] = np.zeros([batch_size, 2])
    batch_tensors['norm_i'] = np.zeros([batch_size, 1, 2 * data_dict['max_seq_len']])
    batch_tensors['norm_j'] = np.zeros([batch_size, 1, 2 * data_dict['max_seq_len']])
    batch_tensors['pair_feats'] = np.zeros([batch_size, data_dict['max_feat_idx']+1])
    batch_tensors['labels'] = np.zeros([batch_size, len(CLASSES)])
    for i in range(0, batch_size):
        pair_id = mention_pairs[i]

        # Add this pair's label to the label batch
        batch_tensors['labels'][i] = data_dict['mention_pair_labels'][pair_id]

        # get this mention pair's word indices and caption indices
        first_i, last_i, first_j, last_j = data_dict['mention_pair_indices'][pair_id]

        # Get the referred sentence indices
        sentence_idx_i = None
        sentence_idx_j = None
        if rel_type == RELATION_TYPES[0]:
            # In the intra caption case, the sentence index is the batch index
            sentence_idx_i = i
            sentence_idx_j = i
        elif rel_type == RELATION_TYPES[1]:
            # In the cross caption case, they should be the first and
            # second sentences after the current batch size * 2
            sentence_idx_i = i * 2
            sentence_idx_j = i * 2 + 1
        #endif

        # Output index arrays
        batch_tensors['first_i_indices'][i] = np.array((1, sentence_idx_i, first_i))
        batch_tensors['last_i_indices'][i] = np.array((0, sentence_idx_i, last_i))
        batch_tensors['first_j_indices'][i] = np.array((1, sentence_idx_j, first_j))
        batch_tensors['last_j_indices'][i] = np.array((0, sentence_idx_j, last_j))
        batch_tensors['sent_i_fw_indices'][i] = np.array((0, sentence_idx_i))
        batch_tensors['sent_i_bw_indices'][i] = np.array((1, sentence_idx_i))
        batch_tensors['sent_j_fw_indices'][i] = np.array((0, sentence_idx_j))
        batch_tensors['sent_j_bw_indices'][i] = np.array((1, sentence_idx_j))

        # Normalization arrays
        norm_vec_i, norm_vec_j = data_dict['mention_pair_norm_vecs'][pair_id]
        batch_tensors['norm_i'][i] = norm_vec_i
        batch_tensors['norm_j'][i] = norm_vec_j

        # Feature arrays, if specified; we don't have to
        # add anything to the tensors, because they're already 0s
        if pair_id in data_dict['mention_pair_feats']:
            batch_tensors['pair_feats'][i] = data_dict['mention_pair_feats'][pair_id]
    #endfor
    return batch_tensors
#enddef


def load_batch_first_last_sentence(mention_pairs, data_dict):
    """
    Loads a batch of data, given a list of mention pairs
    and the data dictionary. This version of load_batch
    returns the necessary index matrices for producing a
        first_i last_i sent_first_i sent_last_i first_j last_j sent_first_j sent_first_j
    mention pair representation via lstm output
    tensor transformations

    :param mention_pairs: List of mention pair IDs
    :param data_dict: Dictionary of all data dictionaries (for sentences, etc)
    :return: dictionary of batch tensors with aforementioned keys
    """
    global N_EMBEDDING_FEATS, CLASSES, RELATION_TYPES, rel_type
    batch_tensors = dict()

    # We're either looking at batch_size sequences or 2*batch_size sequences,
    # depending on whether we're dealing with intra-caption or cross-caption
    # relations
    batch_size = len(mention_pairs)
    n_seq = None
    if rel_type == RELATION_TYPES[0]:
        n_seq = batch_size
    elif rel_type == RELATION_TYPES[1]:
        n_seq = 2 * batch_size

    # Populate our sentence tensor and sequence length array; since
    # we may have 2*batch_size sentences, we need a separate batch index
    batch_tensors['sentences'] = np.zeros([n_seq, data_dict['max_seq_len'], N_EMBEDDING_FEATS])
    batch_tensors['seq_lengths'] = np.zeros([n_seq])
    batch_idx = 0
    for i in range(0, batch_size):
        # set this sentence idx to the sentence embeddings,
        # implicitly padding the end of the sequence with 0s
        sentence_ids = list()
        if rel_type == RELATION_TYPES[0]:
            sentence_ids.append(data_dict['mention_pair_cap_ids'][mention_pairs[i]][0])
        elif rel_type == RELATION_TYPES[1]:
            for s_id in data_dict['mention_pair_cap_ids'][mention_pairs[i]]:
                sentence_ids.append(s_id)
        for sentence_id in sentence_ids:
            sentence_matrix = data_dict['sentences'][sentence_id]
            for j in range(0, len(sentence_matrix)):
                batch_tensors['sentences'][batch_idx][j] = sentence_matrix[j]
            batch_tensors['seq_lengths'][batch_idx] = len(sentence_matrix)
            batch_idx += 1
        #endfor
    #endfor

    # In our new scheme, we need four matrices of size [batch_size, 3]
    # corresponding to the first words of i mentions, last words
    # of i mentions, first words of j mentions, and last words of j mentions.
    # In addition, we'll also need four [batch_size, 2] matrices -- which will really just
    # pull all the outputs of each sentence, forward and back -- and two
    # [batch_size, 1, 2 * n_max_seq] normalization tensors for averaging
    batch_tensors['first_i_indices'] = np.zeros([batch_size, 3])
    batch_tensors['last_i_indices'] = np.zeros([batch_size, 3])
    batch_tensors['sent_i_first_indices'] = np.zeros([batch_size, 3])
    batch_tensors['sent_i_last_indices'] = np.zeros([batch_size, 3])
    batch_tensors['first_j_indices'] = np.zeros([batch_size, 3])
    batch_tensors['last_j_indices'] = np.zeros([batch_size, 3])

    # We only need sentence j indices if we're training cross-caption relations
    if rel_type == RELATION_TYPES[1]:
        batch_tensors['sent_j_first_indices'] = np.zeros([batch_size, 3])
        batch_tensors['sent_j_last_indices'] = np.zeros([batch_size, 3])
    batch_tensors['pair_feats'] = np.zeros([batch_size, data_dict['max_feat_idx']+1])

    # Iterate through batch_size mention pairs, storing the appropriate
    # vectors into the tensors
    batch_tensors['labels'] = np.zeros([batch_size, len(CLASSES)])
    for i in range(0, batch_size):
        pair_id = mention_pairs[i]

        # Add this pair's label to the label batch
        batch_tensors['labels'][i] = data_dict['mention_pair_labels'][pair_id]

        # get this mention pair's word indices and caption indices
        first_i, last_i, first_j, last_j = data_dict['mention_pair_indices'][pair_id]

        # Get the referred sentence indices
        sentence_idx_i = None
        sentence_idx_j = None
        if rel_type == RELATION_TYPES[0]:
            # In the intra caption case, the sentence index is the batch index
            sentence_idx_i = i
            sentence_idx_j = i
        elif rel_type == RELATION_TYPES[1]:
            # In the cross caption case, they should be the first and
            # second sentences after the current batch size * 2
            sentence_idx_i = i * 2
            sentence_idx_j = i * 2 + 1
        #endif

        # Output index arrays
        batch_tensors['first_i_indices'][i] = np.array((1, sentence_idx_i, first_i))
        batch_tensors['last_i_indices'][i] = np.array((0, sentence_idx_i, last_i))
        batch_tensors['first_j_indices'][i] = np.array((1, sentence_idx_j, first_j))
        batch_tensors['last_j_indices'][i] = np.array((0, sentence_idx_j, last_j))
        batch_tensors['sent_i_first_indices'][i] = np.array((1, sentence_idx_i, 0))
        batch_tensors['sent_i_last_indices'][i] = \
            np.array((0, sentence_idx_i, batch_tensors['seq_lengths'][sentence_idx_i] - 1))
        if rel_type == RELATION_TYPES[1]:
            batch_tensors['sent_j_first_indices'][i] = np.array((1, sentence_idx_j, 0))
            batch_tensors['sent_j_last_indices'][i] = \
                np.array((0, sentence_idx_j, batch_tensors['seq_lengths'][sentence_idx_j] - 1))

        # Feature arrays, if specified; we don't have to
        # add anything to the tensors, because they're already 0s
        if pair_id in data_dict['mention_pair_feats']:
            batch_tensors['pair_feats'][i] = data_dict['mention_pair_feats'][pair_id]
    #endfor
    return batch_tensors
#enddef


def setup_batch_input_first_avg_last(batch_size, lstm_outputs):
    """
    Sets up the input placeholders necessary to handle the
    various indices necessary to perform lstm output manipulations
    for the fist_avg_last formatting
        first_i avg_i last_i first_j avg_j last_j

    :param batch_size: Size of the batches
    :param lstm_outputs: Tensorflow variables for lstm output tensors
    """

    # Input placeholders for the lstm output indices for the first
    # and last words for mentions i and j (we assume here (i,j) pairs)
    first_i_indices = tf.placeholder(tf.int32, [batch_size, 3])
    last_i_indices = tf.placeholder(tf.int32, [batch_size, 3])
    first_j_indices = tf.placeholder(tf.int32, [batch_size, 3])
    last_j_indices = tf.placeholder(tf.int32, [batch_size, 3])
    tf.add_to_collection('first_i_indices', first_i_indices)
    tf.add_to_collection('last_i_indices', last_i_indices)
    tf.add_to_collection('first_j_indices', first_j_indices)
    tf.add_to_collection('last_j_indices', last_j_indices)

    # Input placeholders for the lstm output indices for the
    # i and j sentences (both backward and forward)
    sent_i_fw_indices = tf.placeholder(tf.int32, [batch_size, 2])
    sent_i_bw_indices = tf.placeholder(tf.int32, [batch_size, 2])
    sent_j_fw_indices = tf.placeholder(tf.int32, [batch_size, 2])
    sent_j_bw_indices = tf.placeholder(tf.int32, [batch_size, 2])
    tf.add_to_collection('sent_i_fw_indices', sent_i_fw_indices)
    tf.add_to_collection('sent_i_bw_indices', sent_i_bw_indices)
    tf.add_to_collection('sent_j_fw_indices', sent_j_fw_indices)
    tf.add_to_collection('sent_j_bw_indices', sent_j_bw_indices)

    # Input placeholder for the normalizing constants used in
    # averaging mentions' words' outputs
    norm_i = tf.placeholder(tf.float32, [batch_size, 1, None])
    norm_j = tf.placeholder(tf.float32, [batch_size, 1, None])
    tf.add_to_collection('norm_i', norm_i)
    tf.add_to_collection('norm_j', norm_j)

    # Input placeholder for mention pair feats, if specified
    ij_feats = tf.placeholder(tf.float32, [batch_size, None])
    tf.add_to_collection('ij_feats', ij_feats)

    # We want to pass a concatenation of tensors to the next layer
    # such that each individual item represents
    #       first_i_bw, avg_i, last_i_fw, first_j_bw, avg_j, last_j_fw
    # and size of this tensor is
    #       [batch_size, 6 * n_hidden_lstm
    # To get there, we're going to concatenate six individual tensors,
    # containing each of those pieces, detailed below
    # UPDATE: If we're incorporating engineered features, the size
    #         of the resulting tensor will be larger
    #         (6 * n_hidden_lstm + n_engr_feats)

    # a) Gather the outputs for all the first and last words of i and j
    #    mentions and sentences from the outputs
    outputs_first_i = tf.gather_nd(lstm_outputs, first_i_indices)
    outputs_last_i = tf.gather_nd(lstm_outputs, last_i_indices)
    outputs_first_j = tf.gather_nd(lstm_outputs, first_j_indices)
    outputs_last_j = tf.gather_nd(lstm_outputs, last_j_indices)
    outputs_sent_i_fw = tf.gather_nd(lstm_outputs, sent_i_fw_indices)
    outputs_sent_i_bw = tf.gather_nd(lstm_outputs, sent_i_bw_indices)
    outputs_sent_j_fw = tf.gather_nd(lstm_outputs, sent_j_fw_indices)
    outputs_sent_j_bw = tf.gather_nd(lstm_outputs, sent_j_bw_indices)

    # b) Concatenate the forward and backward sentences together
    outputs_sent_i = tf.concat([outputs_sent_i_fw, outputs_sent_i_bw], 1)
    outputs_sent_j = tf.concat([outputs_sent_j_fw, outputs_sent_j_bw], 1)

    # c) Multiply these concatenated sentences together by our mention-dependant
    #    normalization arrays, resulting in an averaged mention output value
    #       [batch_size, 2 * n_max_seq, n_hidden_lstm] * [batch_size, 1, 2*n_max_seq]
    #    resulting in
    #       [batch_size, 1, n_hidden_lstm]
    #    which we then squeeze the second dimension out of
    avg_i = tf.squeeze(tf.matmul(norm_i, outputs_sent_i))
    avg_j = tf.squeeze(tf.matmul(norm_j, outputs_sent_j))

    # d) Collect these tensors into a list, including the engineered feats
    #    if specified
    tensor_list = [outputs_first_i, avg_i, outputs_last_i,
                   outputs_first_j, avg_j, outputs_last_j,
                   ij_feats]

    # d) Finally, concatenate all these together along the final dimension
    batch_input = tf.concat(tensor_list, 1)
    tf.add_to_collection('batch_input', batch_input)
#enddef


def setup_batch_input_first_last_sentence(batch_size, lstm_outputs):
    """
    Sets up the input placeholders necessary to handle the
    various indices necessary to perform lstm output manipulations
    for the first_last_sentence formatting
        sent_first_i sent_last_i first_i last_i sent_first_j sent_last_j first_j last_j

    :param batch_size: Size of the batches
    :param lstm_outputs: Tensorflow variables for lstm output tensors
    """

    # Input placeholders for the lstm output indices for the first
    # and last words for mentions i and j (we assume here (i,j) pairs)
    first_i_indices = tf.placeholder(tf.int32, [batch_size, 3])
    last_i_indices = tf.placeholder(tf.int32, [batch_size, 3])
    first_j_indices = tf.placeholder(tf.int32, [batch_size, 3])
    last_j_indices = tf.placeholder(tf.int32, [batch_size, 3])
    tf.add_to_collection('first_i_indices', first_i_indices)
    tf.add_to_collection('last_i_indices', last_i_indices)
    tf.add_to_collection('first_j_indices', first_j_indices)
    tf.add_to_collection('last_j_indices', last_j_indices)

    # Input placeholders for the lstm output indices for the
    # first and last words for the i and j sentences
    sent_i_first_indices = tf.placeholder(tf.int32, [batch_size, 3])
    sent_i_last_indices = tf.placeholder(tf.int32, [batch_size, 3])
    sent_j_first_indices = tf.placeholder(tf.int32, [batch_size, 3])
    sent_j_last_indices = tf.placeholder(tf.int32, [batch_size, 3])
    tf.add_to_collection('sent_i_first_indices', sent_i_first_indices)
    tf.add_to_collection('sent_i_last_indices', sent_i_last_indices)
    tf.add_to_collection('sent_j_first_indices', sent_j_first_indices)
    tf.add_to_collection('sent_j_last_indices', sent_j_last_indices)

    # Input placeholder for mention pair feats, if specified
    ij_feats = tf.placeholder(tf.float32, [batch_size, None])
    tf.add_to_collection('ij_feats', ij_feats)

    # We want to pass a concatenation of eight tensors to the next layer
    # such that each individual item represents
    #       sent_i_first_bw, sent_i_last_fw, first_i_bw, last_i_fw,
    #       first_j_bw, last_j_fw, sent_j_first_bw, sent_j_last_fw
    # and size of this tensor is
    #       [batch_size, 8 * n_hidden_lstm
    # To get there, we're going to concatenate eight individual tensors
    # and concatenate them
    # UPDATE: If we're incorporating engineered features, the size
    #         of the resulting tensor will be larger
    #         (8 * n_hidden_lstm + n_engr_feats)
    # UPDATE 2: This actually varies depending on whether we're intra/cross
    #           relations
    outputs_first_i = tf.gather_nd(lstm_outputs, first_i_indices)
    outputs_last_i = tf.gather_nd(lstm_outputs, last_i_indices)
    outputs_first_j = tf.gather_nd(lstm_outputs, first_j_indices)
    outputs_last_j = tf.gather_nd(lstm_outputs, last_j_indices)
    outputs_sent_i_first = tf.gather_nd(lstm_outputs, sent_i_first_indices)
    outputs_sent_i_last = tf.gather_nd(lstm_outputs, sent_i_last_indices)
    tensor_list = [outputs_sent_i_first, outputs_sent_i_last, outputs_first_i,
                   outputs_last_i, outputs_first_j, outputs_last_j]
    if rel_type == RELATION_TYPES[1]:
        outputs_sent_j_first = tf.gather_nd(lstm_outputs, sent_j_first_indices)
        outputs_sent_j_last = tf.gather_nd(lstm_outputs, sent_j_last_indices)
        tensor_list.append(outputs_sent_j_first)
        tensor_list.append(outputs_sent_j_last)
    tensor_list.append(ij_feats)

    batch_input = tf.concat(tensor_list, 1)
    tf.add_to_collection('batch_input', batch_input)
#endif


def setup_relation(batch_size, lstm_hidden_width, start_hidden_width,
                   hidden_depth, weighted_classes, lrn_rate,
                   clip_norm, adam_epsilon, activation,
                   n_engr_feats=None):
    """
    Sets up the relation classifier network, which passes sentences to
    a bidirectional LSTM, and passes the transformed outputs to a hidden layer.

    :param batch_size: Number of mention pairs to run each batch
    :param lstm_hidden_width: Number of hidden units in the lstm cells
    :param start_hidden_width: Number of hidden units to which the mention pairs'
                         representation is passed
    :param hidden_depth: Number of hidden layers after the lstm
    :param weighted_classes: Whether to weight the examples by their class inversely
                    with the frequency of that class
    :param lrn_rate: Learning rate of the optimizer
    :param clip_norm: Global gradient clipping norm
    :param adam_epsilon: Adam optimizer epsilon value
    :param activation: Nonlinear activation function (sigmoid, tanh, relu)
    :param n_engr_feats: The maximum feature vector value for our engineered
                         features
    """
    global CLASSES, N_EMBEDDING_FEATS, pair_enc_scheme

    # Each mention pair is a one-hot for its label (n,c,b,p)
    y = tf.placeholder(tf.float32, [batch_size, len(CLASSES)])
    tf.add_to_collection('y', y)

    # Set up the bidirectional LSTM
    with tf.variable_scope('bidirectional_lstm'):
        nn_util.setup_bidirectional_lstm(lstm_hidden_width, data_norm)

    # Get the outputs, which are a (fw,bw) tuple of
    # [batch_size, seq_length, n_hidden_lstm matrices
    lstm_outputs = (tf.get_collection('bidirectional_lstm/outputs_fw')[0],
                    tf.get_collection('bidirectional_lstm/outputs_bw')[0])

    # Set up the batch input tensors, which is a manipulation of
    # lstm outputs based on graph inputs
    hidden_input_width = None
    if pair_enc_scheme == 'first_avg_last':
        hidden_input_width = 6 * lstm_hidden_width
        setup_batch_input_first_avg_last(batch_size, lstm_outputs)
    elif pair_enc_scheme == 'first_last_sentence':
        if rel_type == RELATION_TYPES[0]:
            hidden_input_width = 6 * lstm_hidden_width
        elif rel_type == RELATION_TYPES[1]:
            hidden_input_width = 8 * lstm_hidden_width
        setup_batch_input_first_last_sentence(batch_size, lstm_outputs)
    #endif
    hidden_input_width += n_engr_feats
    batch_input = tf.get_collection('batch_input')[0]

    # dropout percentage
    dropout = tf.placeholder(tf.float32)
    tf.add_to_collection('dropout', dropout)

    # Set up the hidden layer(s)
    hidden_inputs = [batch_input]
    n_hidden_widths = [start_hidden_width]
    for depth in range(1, hidden_depth):
        n_hidden_widths.append(n_hidden_widths[depth-1] / 2)
    for depth in range(0, hidden_depth):
        with tf.variable_scope("hdn_" + str(depth+1)):
            weights = nn_util.get_weights([hidden_input_width, n_hidden_widths[depth]])
            biases = nn_util.get_biases([1, n_hidden_widths[depth]])
            logits = tf.matmul(hidden_inputs[depth], weights) + biases
            if activation == 'sigmoid':
                logits = tf.nn.sigmoid(logits)
            elif activation == 'tanh':
                logits = tf.nn.tanh(logits)
            elif activation == 'relu':
                logits = tf.nn.relu(logits)
            elif activation == 'leaky_relu':
                logits = nn_util.leaky_relu(logits)
            logits = tf.nn.dropout(logits, dropout)
            hidden_inputs.append(logits)
            hidden_input_width = n_hidden_widths[depth]
        #endwith
    #endfor
    with tf.variable_scope("softmax"):
        weights = nn_util.get_weights([n_hidden_widths[hidden_depth - 1], len(CLASSES)])
        biases = nn_util.get_biases([1, len(CLASSES)])
        # Because our label distribution is so skewed, we have to
        # add a constant epsilon to all of the values to prevent
        # the loss from being NaN
        epsilon = np.nextafter(0, 1)
        constant_epsilon = tf.constant([epsilon, epsilon, epsilon, epsilon], dtype=tf.float32)
        final_logits = tf.matmul(hidden_inputs[hidden_depth], weights) + biases + constant_epsilon
        predicted_proba = tf.nn.softmax(final_logits)
    #endwith
    tf.add_to_collection('predicted_proba', predicted_proba)

    # In order to use the weighted softmax cross entropy, we need to
    # get the weights, which first requires us to get label counts for this batch
    # Note: this results in a [1, n_classes] vector
    batch_identity = tf.constant(np.ones([1, batch_size]), dtype=tf.float32)
    gold_label_counts = tf.matmul(batch_identity, y)

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

        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=final_logits,
                                                        weights=example_weights)
    else:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=final_logits, labels=y)
    #endif
    loss = tf.reduce_sum(cross_entropy)
    tf.add_to_collection('loss', loss)

    # Add the training operation
    nn_util.add_train_op(loss, lrn_rate, adam_epsilon, clip_norm)

    # Evaluate model
    pred = tf.argmax(predicted_proba, 1)
    correct_pred = tf.equal(pred, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.add_to_collection('pred', pred)
    tf.add_to_collection('accuracy', accuracy)

    # Dump all the tf variables, just to double check everything's
    # the right size
    for name in tf.get_default_graph().get_all_collection_keys():
        coll = tf.get_collection(name)
        if len(coll) >= 1:
            coll = coll[0]
        print "%-20s: %s" % (name, coll)
    #endfor
#enddef


def run_op_first_avg_last(tf_session, tf_op, batch_tensors,
                          input_dropout, other_dropout,
                          include_labels):
    """
    Runs an operation with the variables necessary for
    the first_avg_last models

    :param tf_session: Tensorflow session
    :param tf_op: Tensorflow operation
    :param batch_tensors: Data batch
    :param input_dropout: Probability to keep for lstm inputs
    :param other_dropout: Probability to keep for all other nodes
    :param include_labels: Whether to send the labels
                           to the operation
    :return: The result of the operation
    """

    # Get the relevant tensorflow variables
    lstm_x = tf.get_collection('bidirectional_lstm/x')[0]
    lstm_seq_lengths = tf.get_collection('bidirectional_lstm/seq_lengths')[0]
    lstm_input_keep_prob = tf.get_collection('bidirectional_lstm/input_keep_prob')[0]
    lstm_output_keep_prob = tf.get_collection('bidirectional_lstm/output_keep_prob')[0]
    first_i_indices = tf.get_collection('first_i_indices')[0]
    last_i_indices = tf.get_collection('last_i_indices')[0]
    first_j_indices = tf.get_collection('first_j_indices')[0]
    last_j_indices = tf.get_collection('last_j_indices')[0]
    sent_i_fw_indices = tf.get_collection('sent_i_fw_indices')[0]
    sent_i_bw_indices = tf.get_collection('sent_i_bw_indices')[0]
    sent_j_fw_indices = tf.get_collection('sent_j_fw_indices')[0]
    sent_j_bw_indices = tf.get_collection('sent_j_bw_indices')[0]
    norm_i = tf.get_collection('norm_i')[0]
    norm_j = tf.get_collection('norm_j')[0]
    ij_feats = tf.get_collection('ij_feats')[0]
    dropout = tf.get_collection('dropout')[0]
    y = tf.get_collection('y')[0]

    # Run the operation
    feed_dict = {lstm_x: batch_tensors['sentences'],
                 lstm_seq_lengths: batch_tensors['seq_lengths'],
                 lstm_input_keep_prob: input_dropout,
                 lstm_output_keep_prob: other_dropout,
                 first_i_indices: batch_tensors['first_i_indices'],
                 last_i_indices: batch_tensors['last_i_indices'],
                 first_j_indices: batch_tensors['first_j_indices'],
                 last_j_indices: batch_tensors['last_j_indices'],
                 sent_i_fw_indices: batch_tensors['sent_i_fw_indices'],
                 sent_i_bw_indices: batch_tensors['sent_i_bw_indices'],
                 sent_j_fw_indices: batch_tensors['sent_j_fw_indices'],
                 sent_j_bw_indices: batch_tensors['sent_j_bw_indices'],
                 norm_i: batch_tensors['norm_i'],
                 norm_j: batch_tensors['norm_j'],
                 dropout: other_dropout}
    if include_labels:
        feed_dict[y] = batch_tensors['labels']
    feed_dict[ij_feats] = batch_tensors['pair_feats']
    return tf_session.run(tf_op, feed_dict=feed_dict)
#endif


def run_op_first_last_sentence(tf_session, tf_op, batch_tensors,
                               input_dropout, other_dropout,
                               include_labels):
    """
    Runs an operation with the variables necessary for
    the first_last_sentence models

    :param tf_session: Tensorflow session
    :param tf_op: Tensorflow operation
    :param batch_tensors: Data batch
    :param input_dropout: Probability to keep for lstm inputs
    :param other_dropout: Probability to keep for all other nodes
    :param include_labels: Whether to send the labels
                           to the operation
    :return: The result of the operation
    """

    # Get the relevant tensorflow variables
    lstm_x = tf.get_collection('bidirectional_lstm/x')[0]
    lstm_seq_lengths = tf.get_collection('bidirectional_lstm/seq_lengths')[0]
    lstm_input_keep_prob = tf.get_collection('bidirectional_lstm/input_keep_prob')[0]
    lstm_output_keep_prob = tf.get_collection('bidirectional_lstm/output_keep_prob')[0]
    first_i_indices = tf.get_collection('first_i_indices')[0]
    last_i_indices = tf.get_collection('last_i_indices')[0]
    first_j_indices = tf.get_collection('first_j_indices')[0]
    last_j_indices = tf.get_collection('last_j_indices')[0]
    sent_i_first_indices = tf.get_collection('sent_i_first_indices')[0]
    sent_i_last_indices = tf.get_collection('sent_i_last_indices')[0]
    sent_j_first_indices = tf.get_collection('sent_j_first_indices')[0]
    sent_j_last_indices = tf.get_collection('sent_j_last_indices')[0]
    dropout = tf.get_collection('dropout')[0]
    ij_feats = tf.get_collection('ij_feats')[0]
    y = tf.get_collection('y')[0]

    # Run the operation
    feed_dict = {lstm_x: batch_tensors['sentences'],
                 lstm_seq_lengths: batch_tensors['seq_lengths'],
                 lstm_input_keep_prob: input_dropout,
                 lstm_output_keep_prob: other_dropout,
                 first_i_indices: batch_tensors['first_i_indices'],
                 last_i_indices: batch_tensors['last_i_indices'],
                 first_j_indices: batch_tensors['first_j_indices'],
                 last_j_indices: batch_tensors['last_j_indices'],
                 sent_i_first_indices: batch_tensors['sent_i_first_indices'],
                 sent_i_last_indices: batch_tensors['sent_i_last_indices'],
                 dropout: other_dropout}
    if rel_type == RELATION_TYPES[1]:
        feed_dict[sent_j_first_indices] = batch_tensors['sent_j_first_indices']
        feed_dict[sent_j_last_indices] = batch_tensors['sent_j_last_indices']
    if include_labels:
        feed_dict[y] = batch_tensors['labels']
    feed_dict[ij_feats] = batch_tensors['pair_feats']
    return tf_session.run(tf_op, feed_dict=feed_dict)
#endif


def train(sentence_file, mention_idx_file, feature_file,
          feature_meta_file, epochs, batch_size, lstm_hidden_width,
          start_hidden_width, hidden_depth, weighted_classes,
          input_dropout, other_dropout, lrn_rate,
          adam_epsilon, clip_norm, activation, model_file=None,
          eval_sentence_file=None, eval_mention_idx_file=None,
          eval_feature_file=None, eval_feature_meta_file=None,
          eval_label_file=None, early_stopping=False):
    """
    Trains a relation model

    :param sentence_file: File with captions
    :param mention_idx_file: File with mention pair word indices
    :param feature_file: File with sparse mention pair features
    :param feature_meta_file: File associating sparse indices with feature names
    :param epochs: Number of times to run over the data
    :param batch_size: Number of mention pairs to run each batch
    :param lstm_hidden_width: Number of hidden units in the lstm cells
    :param start_hidden_width: Number of hidden units to which the mention pairs'
                               representation is passed
    :param hidden_depth: Number of hidden layers after the lstm
    :param weighted_classes: Whether to weight the examples by their
                             class inversely with the frequency of
                             that class
    :param input_dropout: Probability to keep for lstm inputs
    :param other_dropout: Probability to keep for all other nodes
    :param lrn_rate: Learning rate of the optimizer
    :param clip_norm: Global gradient clipping norm
    :param adam_epsilon: Adam optimizer epsilon value
    :param activation: Nonlinear activation function (sigmoid,tanh,relu)
    :param model_file: File to which the model is periodically saved
    :param eval_sentence_file: Sentence file against which the model
                               should be evaluated
    :param eval_mention_idx_file: Mention index file against which
                                  the model should be evaluated
    :param eval_label_file: Relation label file for eval data
    :return:
    """
    global pair_enc_scheme, embedding_type

    log.info("Loading data from " + sentence_file + " and " + mention_idx_file)
    data_dict = nn_data.load_mention_pair_data(mention_idx_file,
                                               feature_file, feature_meta_file)
    sentence_dict = nn_data.load_sentences(sentence_file, embedding_type)
    for key in sentence_dict.keys():
        data_dict[key] = sentence_dict[key]

    log.info("Loading data from " + eval_sentence_file + " and " + eval_mention_idx_file)
    eval_data_dict = nn_data.load_mention_pair_data(eval_mention_idx_file,
                                                    eval_feature_file, eval_feature_meta_file)
    eval_sentence_dict = nn_data.load_sentences(eval_sentence_file, embedding_type)
    for key in eval_sentence_dict.keys():
        eval_data_dict[key] = eval_sentence_dict[key]

    mention_pairs = data_dict['mention_pair_indices'].keys()
    n_pairs = len(mention_pairs)

    # Load the gold labels from the label file once, and we can just reuse them every epoch
    gold_label_dict = nn_data.load_relation_labels(eval_label_file)

    # We want to keep track of the best coref and subset scores, along
    # with the epoch that they originated from
    best_coref_subset_avg = -1
    best_coref_subset_epoch = -1

    log.info("Setting up network architecture")
    n_engr_feats = data_dict['max_feat_idx'] + 1
    setup_relation(batch_size, lstm_hidden_width, start_hidden_width,
                   hidden_depth, weighted_classes, lrn_rate,
                   clip_norm, adam_epsilon, activation, n_engr_feats)

    # Get our model-independent tensorflow operations
    loss = tf.get_collection('loss')[0]
    train_op = tf.get_collection('train_op')[0]
    accuracy = tf.get_collection('accuracy')[0]

    log.info("Training")
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        # Initialize all our variables
        sess.run(tf.global_variables_initializer())

        # Iterate through the data [epochs] number of times
        for i in range(0, epochs):
            log.info(None, "--- Epoch %d ----", i+1)
            losses = list()
            accuracies = list()

            # Shuffle the data once for this epoch
            np.random.shuffle(mention_pairs)

            # Iterate through the entirety of the data
            start_idx = 0
            end_idx = start_idx + batch_size
            n_iter = n_pairs / batch_size
            for j in range(0, n_iter):
                log.log_status('info', None, 'Training; %d (%.2f%%) batches complete',
                               j, 100.0 * j / n_iter)

                # Retrieve this batch
                batch_mention_pairs = mention_pairs[start_idx:end_idx]

                if pair_enc_scheme == 'first_avg_last':
                    batch_tensors = load_batch_first_avg_last(batch_mention_pairs, data_dict)
                elif pair_enc_scheme == 'first_last_sentence':
                    batch_tensors = load_batch_first_last_sentence(batch_mention_pairs, data_dict)

                # Train
                if pair_enc_scheme == 'first_avg_last':
                    run_op_first_avg_last(sess, train_op, batch_tensors,
                                          input_dropout, other_dropout, True)
                elif pair_enc_scheme == 'first_last_sentence':
                    run_op_first_last_sentence(sess, train_op, batch_tensors,
                                               input_dropout, other_dropout, True)

                # Store the losses and accuracies every 100 batches
                if (j+1) % 100 == 0:
                    if pair_enc_scheme == 'first_avg_last':
                        losses.append(
                            run_op_first_avg_last(sess, loss, batch_tensors,
                                                  input_dropout, other_dropout, True))
                        accuracies.append(
                            run_op_first_avg_last(sess, accuracy, batch_tensors,
                                                  input_dropout, other_dropout, True))
                    elif pair_enc_scheme == 'first_last_sentence':
                        losses.append(
                            run_op_first_last_sentence(sess, loss, batch_tensors,
                                                       input_dropout, other_dropout, True))
                        accuracies.append(
                            run_op_first_last_sentence(sess, accuracy, batch_tensors,
                                                       input_dropout, other_dropout, True))
                    #endif
                #endif
                start_idx = end_idx
                end_idx = start_idx + batch_size
            #endfor

            # Every epoch, evaluate and save the model
            log.info(None, "Saving model; Average Loss: %.2f; Acc: %.2f%%",
                     sum(losses) / float(len(losses)),
                     100.0 * sum(accuracies) / float(len(accuracies)))
            saver.save(sess, model_file)
            if eval_sentence_file is not None and eval_mention_idx_file is not None:
                pred_scores = get_pred_scores(sess, batch_size, eval_data_dict)

                # If we do an argmax on the scores, we get the predicted labels
                eval_mention_pairs = pred_scores.keys()
                pred_labels = list()
                for pair in eval_mention_pairs:
                    pred_labels.append(np.argmax(pred_scores[pair]))

                # Evaluate the predictions
                score_dict = nn_eval.evaluate_relations(eval_mention_pairs, pred_labels, gold_label_dict)

                # Get the current coref / subset and see if their average beats our best
                coref_subset_avg = score_dict.get_score('coref').f1 + score_dict.get_score('subset').f1
                coref_subset_avg /= 2.0
                if coref_subset_avg >= best_coref_subset_avg - 0.005:
                    log.info(None, "Previous best coref/subset average F1 of %.2f%% after %d epochs",
                             100.0 * best_coref_subset_avg, best_coref_subset_epoch)
                    best_coref_subset_avg = coref_subset_avg
                    best_coref_subset_epoch = i
                    log.info(None, "New best at current epoch (%.2f%%)",
                             100.0 * best_coref_subset_avg)
                #endif

                # Implement early stopping; if it's been 10 epochs since our best, stop
                if early_stopping and i >= (best_coref_subset_epoch + 10):
                    log.info(None, "Stopping early; best scores at %d epochs", best_coref_subset_epoch)
                    break
                #endif
            #endif
        #endfor
        log.info("Saving final model")
        saver.save(sess, model_file)
    #endwith
#enddef


def get_pred_scores(tf_session, batch_size, data_dict):
    """
    Evaluates the model loaded in the given session and logs the counts
    and p/r/f1 on the given data

    :param tf_session: Tensorflow session in which the model has
                       been loaded
    :param batch_size: Size of the groups of mention pairs to pass
                       to the network at once\
    :param data_dict: Sentence and mention pair data
    """
    global log, pair_enc_scheme
    pred_scores = dict()
    mention_pairs = data_dict['mention_pair_indices'].keys()

    # Get the predict operation
    predicted_proba = tf.get_collection('predicted_proba')[0]

    # Run until we don't have any more mention pairs to predict for
    pad_length = batch_size * (len(mention_pairs) / batch_size + 1) - len(mention_pairs)
    mention_pair_arr = np.pad(mention_pairs, (0, pad_length), 'edge')
    mention_pair_matrix = np.reshape(mention_pair_arr, [-1, batch_size])
    for i in range(0, mention_pair_matrix.shape[0]):
        log.log_status('info', None, 'Predicting; %d batches complete (%.2f%%)',
                       i, 100.0 * i / mention_pair_matrix.shape[0])
        if pair_enc_scheme == 'first_avg_last':
            batch_tensors = load_batch_first_avg_last(mention_pair_matrix[i], data_dict)
        elif pair_enc_scheme == 'first_last_sentence':
            batch_tensors = load_batch_first_last_sentence(mention_pair_matrix[i], data_dict)

        # Add the labels to the list
        if pair_enc_scheme == 'first_avg_last':
            predicted_scores = run_op_first_avg_last(tf_session, predicted_proba, batch_tensors,
                                                     1.0, 1.0, False)
        elif pair_enc_scheme == 'first_last_sentence':
            predicted_scores = run_op_first_last_sentence(tf_session, predicted_proba, batch_tensors,
                                                          1.0, 1.0, False)

        # Take all the scores unless this is the last row
        row_length = len(predicted_scores)
        if i == mention_pair_matrix.shape[0] - 1:
            row_length = batch_size - pad_length
        for j in range(0, row_length):
            pred_scores[mention_pair_matrix[i][j]] = predicted_scores[j]
    #endfor
    return pred_scores
#enddef


def predict(tf_session, batch_size, sentence_file,
            mention_idx_file, feature_file,
            feature_meta_file, label_file, scores_file=None):
    """
    Wrapper for making predictions on a pre-trained model, already loaded into
    the session
    :param tf_session:
    :param batch_size:
    :param sentence_file:
    :param mention_idx_file:
    :param feature_file:
    :param feature_meta_file:
    :param label_file:
    :param scores_file:
    :return:
    """
    global embedding_type

    # Load the data
    log.info("Loading data from " + sentence_file + " and " + mention_idx_file)
    data_dict = nn_data.load_mention_pair_data(mention_idx_file,
                                               feature_file, feature_meta_file)
    sentence_dict = nn_data.load_sentences(sentence_file, embedding_type)
    for key in sentence_dict.keys():
        data_dict[key] = sentence_dict[key]

    # Get the predicted scores, given our arguments
    log.info("Predictiong scores")
    pred_scores = get_pred_scores(tf_session, batch_size, data_dict)

    # log.warning("Skipping evaluation, since it takes way too long right now for some reason")
    log.info("Loading data from " + label_file)
    gold_label_dict = nn_data.load_relation_labels(label_file)

    # If we do an argmax on the scores, we get the predicted labels
    log.info("Getting labels from scores")
    mention_pairs = pred_scores.keys()
    pred_labels = list()
    for pair in mention_pairs:
        pred_labels.append(np.argmax(pred_scores[pair]))

    # Evaluate the predictions
    log.info("Evaluating against the gold")
    nn_eval.evaluate_relations(mention_pairs, pred_labels, gold_label_dict, log)

    # If a scores file was specified, write the scores
    log.info("Writing scores file")
    if scores_file is not None:
        with open(scores_file, 'w') as f:
            for pair_id in pred_scores.keys():
                score_line = list()
                score_line.append(pair_id)
                for score in pred_scores[pair_id]:
                    score_line.append(str(math.log(score)))
                f.write(",".join(score_line) + "\n")
            f.close()
        #endwith
    #endif
#enddef


def __init__():
    global rel_type, pair_enc_scheme, data_norm, log, embedding_type

    # Parse arguments
    parser = ArgumentParser("ImageCaptionLearn_py: Neural Network for Relation "
                            "Prediction; Bidirectional LSTM to hidden layer "
                            "to softmax over (n)ull, (c)oreference, su(b)set, "
                            "and su(p)erset labels")
    parser.add_argument("--epochs", type=int, default=20,
                        help="train opt; number of times to iterate over the dataset")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="train opt; number of random mention pairs per batch")
    parser.add_argument("--lstm_hidden_width", type=int, default=200,
                        help="train opt; number of hidden units within "
                             "the LSTM cells")
    parser.add_argument("--start_hidden_width", type=int, default=150,
                        help="train opt; number of hidden units in the "
                             "layer after the LSTM")
    parser.add_argument("--hidden_depth", type=int, default=1,
                        help="train opt; number of hidden layers after the "
                             "lstm, where each is last_width/2 units wide, "
                             "starting with start_hidden_width")
    parser.add_argument("--weighted_classes", action="store_true",
                        help="Whether to inversely weight the classes "
                             "in the loss")
    parser.add_argument("--learn_rate", type=float, default=0.001,
                        help="train opt; optimizer learning rate")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="train opt; Adam optimizer epsilon value")
    parser.add_argument("--clip_norm", type=float, default=5.0,
                        help='train opt; global clip norm value')
    parser.add_argument("--data_norm", action='store_true',
                        help="train opt; Whether to L2-normalize the w2v word vectors")
    parser.add_argument("--input_keep_prob", type=float, default=1.0,
                        help="train opt; probability to keep lstm input nodes")
    parser.add_argument("--other_keep_prob", type=float, default=1.0,
                        help="train opt; probability to keep all other nodes")
    parser.add_argument("--pair_enc_scheme", choices=["first_avg_last", "first_last_sentence"],
                        default="first_last_sentence",
                        help="train opt; specifies how lstm outputs are transformed")
    parser.add_argument("--data_dir", required=True,
                        type=lambda f: util.arg_path_exists(parser, f),
                        help="Directory containing raw/, feats/, and scores/ directories")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Data file root (eg. flickr30k_train)")
    parser.add_argument("--eval_data_root", type=str,
                        help="Data file root for eval data (eg. flickr30k_dev)")
    parser.add_argument("--train", action='store_true', help='Trains a model')
    parser.add_argument("--activation", choices=['sigmoid', 'tanh', 'relu', 'leaky_relu'],
                        default='relu',
                        help='train opt; which nonlinear activation function to use')
    parser.add_argument("--predict", action='store_true',
                        help='Predicts using pre-trained model')
    parser.add_argument("--rel_type", choices=['intra', 'cross'], required=True,
                        help="Whether we're dealing with intra-caption or "
                             "cross-caption relations")
    parser.add_argument("--model_file", #required=True,
                        type=str, help="Model file to save/load")
    parser.add_argument("--embedding_type", choices=['w2v', 'glove'], default='w2v',
                        help="Word embedding type to use")
    parser.add_argument("--early_stopping", action='store_true',
                        help="Whether to implement early stopping based on the "+
                             "evaluation performance")
    args = parser.parse_args()
    arg_dict = vars(args)

    rel_type = arg_dict['rel_type']
    data_norm = arg_dict['data_norm']
    pair_enc_scheme = arg_dict['pair_enc_scheme']
    if arg_dict['train']:
        arg_dict['model_file'] = "/home/ccervan2/models/tacl201711/" + \
                                 nn_data.build_model_file_name(arg_dict, "rel_lstm")
    model_file = arg_dict['model_file']
    util.dump_args(arg_dict, log)

    # Construct data files from the root directory and filename
    data_dir = arg_dict['data_dir'] + "/"
    data_root = arg_dict['data_root']
    eval_data_root = arg_dict['eval_data_root']
    sentence_file = data_dir + "raw/" + data_root + "_captions.txt"
    mention_idx_file = data_dir + "raw/" + data_root + "_mentionPairs_" + rel_type + ".txt"
    feature_file = data_dir + "feats/" + data_root + "_relation.feats"
    feature_meta_file = data_dir + "feats/" + data_root + "_relation_meta.json"
    label_file = data_dir + "raw/" + data_root + "_mentionPairLabels.txt"
    eval_sentence_file = data_dir + "raw/" + eval_data_root + "_captions.txt"
    eval_mention_idx_file = data_dir + "raw/" + eval_data_root + "_mentionPairs_" + rel_type + ".txt"
    eval_feature_file = data_dir + "feats/" + eval_data_root + "_relation.feats"
    eval_feature_meta_file = data_dir + "feats/" + eval_data_root + "_relation_meta.json"
    eval_label_file = data_dir + "raw/" + eval_data_root + "_mentionPair_labels.txt"

    # Load the appropriate word embeddings
    embedding_type = arg_dict['embedding_type']
    if embedding_type == 'w2v':
        log.info("Initializing word2vec")
        nn_data.init_w2v()
    elif embedding_type == 'glove':
        log.info("Initializing glove")
        nn_data.init_glove()
    #endif

    # Set the random seeds identically every run
    nn_util.set_random_seeds()

    # Set up the minimum tensorflow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Train, if training was specified
    if arg_dict['train']:
        train(sentence_file=sentence_file,
              mention_idx_file=mention_idx_file,
              feature_file=feature_file,
              feature_meta_file=feature_meta_file,
              epochs=arg_dict['epochs'], batch_size=arg_dict['batch_size'],
              lstm_hidden_width=arg_dict['lstm_hidden_width'],
              start_hidden_width=arg_dict['start_hidden_width'],
              hidden_depth=arg_dict['hidden_depth'],
              weighted_classes=arg_dict['weighted_classes'],
              input_dropout=arg_dict['input_keep_prob'],
              other_dropout=arg_dict['other_keep_prob'],
              lrn_rate=arg_dict['learn_rate'], clip_norm=arg_dict['clip_norm'],
              adam_epsilon=arg_dict['adam_epsilon'], activation=arg_dict['activation'],
              model_file=model_file, eval_sentence_file=eval_sentence_file,
              eval_mention_idx_file=eval_mention_idx_file,
              eval_feature_file=eval_feature_file,
              eval_feature_meta_file=eval_feature_meta_file,
              eval_label_file=eval_label_file,
              early_stopping=arg_dict['early_stopping'])
    elif arg_dict['predict']:
        scores_file = data_dir + "scores/" + data_root + \
                      "_relation_" + rel_type + ".scores"

        # Restore our variables
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_file + ".meta")
            saver.restore(sess, model_file)
            predict(sess, batch_size=arg_dict['batch_size'],
                    sentence_file=sentence_file,
                    mention_idx_file=mention_idx_file,
                    feature_file=feature_file, feature_meta_file=feature_meta_file,
                    label_file=label_file, scores_file=scores_file)
        #endwith
    #endif
#enddef


__init__()

