import math
import random
import numpy as np
import tensorflow as tf

__author__ = 'ccervantes'


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


def add_train_op(loss, lrn_rate, adam_epsilon, clip_norm):
    """
    Sets up and adds the "train_op" to the graph, setting
    the adam optimizer's learning rate and adam epsilon value
    as specified; if clip_norm is not None, global gradient
    clipping is used

    :param loss: Loss to optimize
    :param lrn_rate: Learning rate of the optimizer
    :param adam_epsilon: Adam optimizer epsilon value
    :param clip_norm: Global gradient clipping norm
    :return:
    """
    # Adam optimizer sets a variable learning rate for every weight, along
    # with rolling averages; It uses beta1 (def: 0.9), beta2 (def: 0.99),
    # and epsilon (def: 1E-08) to get exponential decay
    optimiz = tf.train.AdamOptimizer(learning_rate=lrn_rate, epsilon=adam_epsilon)
    if clip_norm is None:
        train_op = optimiz.minimize(loss)
    else:
        # We want to perform gradient clipping (by the global norm, to start)
        # which means we need to unstack the steps in minimize and clip
        # the gradients between them
        gradients, variables = zip(*optimiz.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        train_op = optimiz.apply_gradients(zip(gradients, variables))
    #endif
    tf.add_to_collection('train_op', train_op)
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


def setup_bidirectional_lstm(n_hidden, data_norm=False,
                             n_embedding_width=300, n_parallel=64):
    """
    Sets up the tensorflow placeholders, adding relevant variables
    to the graph's collections

    :param n_hidden: Size of the hidden layer in the LSTM cells
    :param data_norm: Whether to apply the L2 norm to our data
    :param n_embedding_width: Width of word embeddings (typically 300)
    :param n_parallel: Number of parallel tasks the RNN may run (64 seems
                       as high as we can set it on our current machines)
    """
    scope_name = tf.get_variable_scope().name + "/"

    # input
    x = tf.placeholder(tf.float32, [None, None, n_embedding_width])
    if data_norm:
        x = tf.nn.l2_normalize(x, dim=2)
    tf.add_to_collection(scope_name + 'x', x)

    # sequence lengths
    seq_lengths = tf.placeholder(tf.int32, [None])
    tf.add_to_collection(scope_name + 'seq_lengths', seq_lengths)

    # dropout percentage
    input_keep_prob = tf.placeholder(tf.float32)
    output_keep_prob = tf.placeholder(tf.float32)
    tf.add_to_collection(scope_name + 'input_keep_prob', input_keep_prob)
    tf.add_to_collection(scope_name + 'output_keep_prob', output_keep_prob)

    # set up the cells
    lstm_cell = {}
    for direction in ["fw", "bw"]:
        with tf.variable_scope(direction):
            lstm_cell[direction] = \
                tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
            lstm_cell[direction] = \
                tf.nn.rnn_cell.DropoutWrapper(lstm_cell[direction], input_keep_prob=input_keep_prob,
                                              output_keep_prob=output_keep_prob)
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
    tf.add_to_collection(scope_name + 'outputs_fw', outputs[0])
    tf.add_to_collection(scope_name + 'outputs_bw', outputs[1])
#enddef


def setup_batch_input_first_last_sentence_mention(batch_size, lstm_outputs):
    """
    Sets up the input placeholders necessary to handle the
    various indices necessary to perform lstm output manipulations
    for the first_last_sentence formatting for a single mention
        sent_first sent_last first last

    :param batch_size: Size of the batches
    :param lstm_outputs: Tensorflow variables for lstm output tensors
    """

    # Input placeholders for the lstm output indices for the first
    # and last words for the mention
    first_indices = tf.placeholder(tf.int32, [batch_size, 3])
    last_indices = tf.placeholder(tf.int32, [batch_size, 3])
    tf.add_to_collection('first_indices', first_indices)
    tf.add_to_collection('last_indices', last_indices)

    # Input placeholders for the lstm output indices for the
    # first and last words for the i and j sentences
    sent_first_indices = tf.placeholder(tf.int32, [batch_size, 3])
    sent_last_indices = tf.placeholder(tf.int32, [batch_size, 3])
    tf.add_to_collection('sent_first_indices', sent_first_indices)
    tf.add_to_collection('sent_last_indices', sent_last_indices)

    # Input placeholder for mention feats, if specified
    m_feats = tf.placeholder(tf.float32, [batch_size, None])
    tf.add_to_collection('m_feats', m_feats)

    # We want to pass a concatenation of four tensors to the next layer
    # such that each individual item represents
    #       sent_first_bw, sent_last_fw, first_bw, last_fw
    # and size of this tensor is
    #       [batch_size, 4 * n_hidden_lstm + n_engr_feats
    outputs_first = tf.gather_nd(lstm_outputs, first_indices)
    outputs_last = tf.gather_nd(lstm_outputs, last_indices)
    outputs_sent_first = tf.gather_nd(lstm_outputs, sent_first_indices)
    outputs_sent_last = tf.gather_nd(lstm_outputs, sent_last_indices)
    tensor_list = [outputs_sent_first, outputs_sent_last, outputs_first,
                   outputs_last, m_feats]

    batch_input = tf.concat(tensor_list, 1)
    tf.add_to_collection('batch_input', batch_input)
#enddef


def setup_batch_input_first_last_sentence_mention_pair(batch_size, lstm_outputs, rel_type):
    """
    Sets up the input placeholders necessary to handle the
    various indices necessary to perform lstm output manipulations
    for the first_last_sentence formatting
        sent_first_i sent_last_i first_i last_i sent_first_j sent_last_j first_j last_j

    :param batch_size: Size of the batches
    :param lstm_outputs: Tensorflow variables for lstm output tensors
    :param rel_type: intra, cross
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
    if rel_type == "cross":
        outputs_sent_j_first = tf.gather_nd(lstm_outputs, sent_j_first_indices)
        outputs_sent_j_last = tf.gather_nd(lstm_outputs, sent_j_last_indices)
        tensor_list.append(outputs_sent_j_first)
        tensor_list.append(outputs_sent_j_last)
    tensor_list.append(ij_feats)

    batch_input = tf.concat(tensor_list, 1)
    tf.add_to_collection('batch_input', batch_input)
#endif



def run_op_first_last_sentence_mention(tf_session, tf_op, batch_tensors,
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
    first_indices = tf.get_collection('first_indices')[0]
    last_indices = tf.get_collection('last_indices')[0]
    sent_first_indices = tf.get_collection('sent_first_indices')[0]
    sent_last_indices = tf.get_collection('sent_last_indices')[0]
    dropout = tf.get_collection('dropout')[0]
    m_feats = tf.get_collection('m_feats')[0]
    y = tf.get_collection('y')[0]

    # Run the operation
    feed_dict = {lstm_x: batch_tensors['sentences'],
                 lstm_seq_lengths: batch_tensors['seq_lengths'],
                 lstm_input_keep_prob: input_dropout,
                 lstm_output_keep_prob: other_dropout,
                 first_indices: batch_tensors['first_indices'],
                 last_indices: batch_tensors['last_indices'],
                 sent_first_indices: batch_tensors['sent_first_indices'],
                 sent_last_indices: batch_tensors['sent_last_indices'],
                 dropout: other_dropout,
                 m_feats: batch_tensors['mention_feats']}
    if include_labels:
        feed_dict[y] = batch_tensors['labels']
    return tf_session.run(tf_op, feed_dict=feed_dict)
#endif

