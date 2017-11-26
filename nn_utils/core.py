import math
import random
import numpy as np
import tensorflow as tf
import nn_utils.data as nn_data

__author__ = 'ccervantes'


def set_random_seeds(seed=20171201):
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
        return tf.Variable(tf.random_uniform(shape, minval=-init_range,
                                             maxval=init_range, seed=20170801),
                           trainable=trainable, dtype=tf.float32)
    elif init_type == "normal":
        return tf.Variable(tf.random_normal(shape, stddev=1.0 / math.sqrt(shape[0]),
                                            seed=20170801),
                           trainable=trainable, dtype=tf.float32)
    elif init_type == "uniform":
        return tf.Variable(tf.random_uniform(shape, minval=-0.05, maxval=0.05,
                                             seed=20170801),
                           trainable=trainable, dtype=tf.float32)
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
                                            seed=20170801), trainable=trainable)
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
    scope_name = tf.get_variable_scope().name
    if scope_name != "":
        scope_name += "/"

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
    tf.add_to_collection(scope_name + 'train_op', train_op)
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


def get_widths(start_width, depth, end_width=None):
    """
    Returns a list of widths, either halving the previous
    (assuming no end width was specified) or halving the
    difference between the previous and the end width

    :param start_width:
    :param depth:
    :param end_width:
    :return:
    """

    widths = list()
    widths.append(int(start_width))
    if end_width is None:
        for d in range(1, depth+1):
            widths.append(int(widths[d-1] / 2))
    else:
        for d in range(1, depth):
            widths.append(int(max(widths[d-1] - end_width, 0) / 2 + end_width))
        widths.append(int(end_width))
    return widths
#enddef


def setup_ffw(input, widths, activation=None, dropout=None):
    """
    Sets up a feed-forward network, starting with
    the given input and with the specified widths.
    Optional arguments specify activation function,
    dropout, and scope_name.

    NOTE: In order to have a single linear
          transformation with trainable weights
          and biases, pass a single width and
          no activation

    :param input:
    :param widths:
    :param activation:
    :param dropout:
    :param scope_name:
    :return:
    """
    scope_name = tf.get_variable_scope().name
    if scope_name != "":
        scope_name += "/"

    inputs = [input]
    widths.insert(0, input.get_shape().as_list()[1])
    for depth in range(1, len(widths)):
        with tf.variable_scope(scope_name + "hdn_" + str(depth)):
            weights = get_weights([widths[depth-1], widths[depth]])
            biases = get_biases([1, widths[depth]])
            logits = tf.matmul(inputs[depth-1], weights) + biases

            # Apply a nonlinearity, if specified
            if activation == "sigmoid":
                logits = tf.nn.sigmoid(logits)
            elif activation == 'tanh':
                logits = tf.nn.tanh(logits)
            elif activation == 'relu':
                logits = tf.nn.relu(logits)
            elif activation == 'leaky_relu':
                logits = leaky_relu(logits)

            # Apply dropout, if specified
            if dropout is not None:
                logits = tf.nn.dropout(logits, dropout)

            # Add this layer's output as the
            # input to the next layer
            inputs.append(logits)
        #endwidth
    #endfor

    # Return the final logits
    return inputs[len(inputs)-1]
#enddef


def apply_softmax(logits, n_classes, add_epsilon=True):
    """
    Performs the final transformation from the given
    logits to the number of classes, by default adding
    a constant epsilon to accomodate very imbalanced classes;
    returns the result of this linear operation (final_logits)
    and the predicted probabilities after applying the softmax

    :param logits:
    :param n_classes:
    :param add_epsilon:
    :return:
    """
    weights = get_weights([logits.get_shape().as_list()[1], n_classes])
    biases = get_biases([1, n_classes])

    constant_eps_arr = np.zeros([n_classes])
    if add_epsilon:
        # Because our label distribution is so skewed, we have to
        # add a constant epsilon to all of the values to prevent
        # the loss from being NaN
        epsilon = np.nextafter(0, 1)
        for i in range(0, n_classes):
            constant_eps_arr[i] = epsilon
    #endif
    constant_epsilon = tf.constant(constant_eps_arr, dtype=tf.float32)

    final_logits = tf.matmul(logits, weights) + biases + constant_epsilon
    predicted_proba = tf.nn.softmax(final_logits)
    return final_logits, predicted_proba
#enddef


def setup_cross_entropy_loss(batch_size, logits, y, n_classes, weighted_classes=False):
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
            tf.constant(np.ones([1, n_classes]), dtype=tf.float32),
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
    #endif
    return tf.reduce_sum(cross_entropy)
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
    scope_name = tf.get_variable_scope().name
    if scope_name != "":
        scope_name += "/"

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
    tf.add_to_collection(scope_name + 'lstm_input_dropout', input_keep_prob)
    tf.add_to_collection(scope_name + 'dropout', output_keep_prob)

    # set up the cells
    lstm_cell = {}
    for direction in ["fw", "bw"]:
        with tf.variable_scope(direction):
            lstm_cell[direction] = \
                tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
            lstm_cell[direction] = \
                tf.nn.rnn_cell.DropoutWrapper(lstm_cell[direction],
                                              input_keep_prob=input_keep_prob,
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
        tf.nn.bidirectional_dynamic_rnn(lstm_cell["fw"], lstm_cell["bw"],
                                        x, dtype=tf.float32,
                                        parallel_iterations=n_parallel,
                                        sequence_length=seq_lengths,
                                        time_major=False)
    tf.add_to_collection(scope_name + 'outputs_fw', outputs[0])
    tf.add_to_collection(scope_name + 'outputs_bw', outputs[1])
#enddef


def __add_lstm_output_placeholder(var_name, batch_size, lstm_outputs, scope_name):
    """
    Adds an lstm output placeholder, which gathers lstm outputs by the various
    indices that are set with the placeholder
    :param var_name:
    :param batch_size:
    :param lstm_outputs:
    :param scope_name:
    :return:
    """

    # All word indices are three-dimensional vectors, representing
    #   [lstm_dir, sentence_idx, word_idx]
    var = tf.placeholder(tf.int32, [batch_size, 3])
    tf.add_to_collection(scope_name + var_name, var)
    return tf.gather_nd(lstm_outputs, var)
#enddef


def setup_batch_inputs(batch_size, lstm_outputs, task, encoding_scheme,
                       n_mention_feats, box_embedding_width=None,
                       n_box_feats=None):
    """

    :param batch_size:
    :param lstm_outputs:
    :param task:
    :param encoding_scheme:
    :param n_mention_feats:
    :param box_embedding_width:
    :param n_box_feats:
    :return:
    """
    scope_name = tf.get_variable_scope().name
    if scope_name != "":
        scope_name += "/"

    tensor_list = list()

    # Regardless of task, we use the backward direction from
    # the first of the i mention words, and the forward direction
    # from the last i mention words
    tensor_list.append(__add_lstm_output_placeholder("first_i_bw", batch_size,
                                                     lstm_outputs, scope_name))
    tensor_list.append(__add_lstm_output_placeholder("last_i_fw", batch_size,
                                                     lstm_outputs, scope_name))

    # Add encoding-specific placeholders
    if encoding_scheme == "first_last_sentence":
        tensor_list.append(__add_lstm_output_placeholder("sent_last_i_fw", batch_size,
                                                         lstm_outputs, scope_name))
        tensor_list.append(__add_lstm_output_placeholder("sent_first_i_bw", batch_size,
                                                         lstm_outputs, scope_name))
    elif encoding_scheme == "first_last_mention":
        tensor_list.append(__add_lstm_output_placeholder("first_i_fw", batch_size,
                                                         lstm_outputs, scope_name))
        tensor_list.append(__add_lstm_output_placeholder("last_i_bw", batch_size,
                                                         lstm_outputs, scope_name))
    #endif

    # Switch on the task / encoding scheme for the rest of the variables
    if "rel" in task:
        tensor_list.append(__add_lstm_output_placeholder("first_j_bw", batch_size,
                                                         lstm_outputs, scope_name))
        tensor_list.append(__add_lstm_output_placeholder("last_j_fw", batch_size,
                                                         lstm_outputs, scope_name))
        ij_feats = tf.placeholder(tf.float32, [batch_size, n_mention_feats])
        tf.add_to_collection(scope_name + "ij_feats", ij_feats)
        tensor_list.append(ij_feats)

        if encoding_scheme == "first_last_sentence" and task == "rel_cross":
            tensor_list.append(__add_lstm_output_placeholder("sent_last_j_fw", batch_size,
                                                             lstm_outputs, scope_name))
            tensor_list.append(__add_lstm_output_placeholder("sent_first_j_bw", batch_size,
                                                             lstm_outputs, scope_name))
        elif encoding_scheme == "first_last_mention":
            tensor_list.append(__add_lstm_output_placeholder("first_j_fw", batch_size,
                                                             lstm_outputs, scope_name))
            tensor_list.append(__add_lstm_output_placeholder("last_j_bw", batch_size,
                                                             lstm_outputs, scope_name))
        #endif
    elif task == "nonvis" or task == "card" or task == "affinity":
        m_feats = tf.placeholder(tf.float32, [batch_size, n_mention_feats])
        tf.add_to_collection(scope_name + "m_feats", m_feats)
        tensor_list.append(m_feats)

        if task == "affinity":
            box_embeddings = tf.placeholder(tf.float32, [batch_size, box_embedding_width])
            tf.add_to_collection(scope_name + "box_embeddings", box_embeddings)
            tensor_list.append(box_embeddings)
            if n_box_feats is not None:
                b_feats = tf.placeholder(tf.float32, [batch_size, n_box_feats])
                tf.add_to_collection(scope_name + "b_feats", b_feats)
                tensor_list.append(b_feats)
        #endif
    #endif

    # Everything should be loaded into the tensor list, so we can concatenate
    # and add the input to the collection
    tf.add_to_collection(scope_name + 'batch_input', tf.concat(tensor_list, 1))
#endif


def setup_core_architecture(task, encoding_scheme, batch_size, start_hidden_width,
                            hidden_depth, weighted_classes, activation,
                            n_classes, n_mention_feats, box_embedding_width=None,
                            n_box_feats=None):
    """
    Sets up the core classification network by adding variables to
    the tensorflow graph; assumes a bidirectional lstm (under that
    namespace) has already been set up

    :param task: {nonvis, card, rel_intra, rel_cross}
    :param encoding_scheme: {first_last_sentence, first_last_mention}
    :param batch_size: Size of minibatches
    :param start_hidden_width: Width of first hidden layer after the lstm
    :param hidden_depth: Number of hidden layers between the lstm and the
                         softmax
    :param weighted_classes: Whether to weigh the examples inversely
                             with their class frequency
    :param activation: Nonlinear activation function for hidden layers
    :param n_classes: Number of classes
    :param n_mention_feats: Number of mention features appended to lstm outputs
    :param box_embedding_width: Width of box embeddings appended to the lstm outputs
                                (affinity only
    :param n_box_feats: Number of box features appended to the lstm outputs
                        (for affinity only; currently coco only)
    :return:
    """

    # Get the current variable scope, if we have one
    scope_name = tf.get_variable_scope().name
    if scope_name != "":
        scope_name += "/"

    # Labels y are simply one-hots for the classes
    y = tf.placeholder(tf.float32, [batch_size, n_classes])
    tf.add_to_collection(scope_name + 'y', y)

    # Get the lstm outputs, which are a (fw,bw) tuple of
    # [batch_size, seq_length, n_hidden_lstm matrices
    lstm_outputs = (tf.get_collection('bidirectional_lstm/outputs_fw')[0],
                    tf.get_collection('bidirectional_lstm/outputs_bw')[0])

    # Set up the batch input tensors
    setup_batch_inputs(batch_size, lstm_outputs, task,
                       encoding_scheme, n_mention_feats,
                       box_embedding_width, n_box_feats)
    batch_input = tf.get_collection(scope_name + 'batch_input')[0]

    # dropout percentage
    dropout = tf.placeholder(tf.float32)
    tf.add_to_collection(scope_name + 'dropout', dropout)

    # Set up the hidden layer(s)
    widths = get_widths(start_hidden_width, hidden_depth)
    logits = setup_ffw(batch_input, widths, activation, dropout)

    # Apply the softmax
    with tf.variable_scope(scope_name + "softmax"):
        final_logits, predicted_proba = apply_softmax(logits, n_classes)
    tf.add_to_collection(scope_name + 'predicted_proba', predicted_proba)

    # Get the cross entropy loss and add it to the collection
    loss = setup_cross_entropy_loss(batch_size, final_logits,
                                            y, n_classes, weighted_classes)
    tf.add_to_collection(scope_name + 'loss', loss)

    # Get the model evaluation operations
    pred = tf.argmax(predicted_proba, 1)
    correct_pred = tf.equal(pred, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.add_to_collection(scope_name + 'pred', pred)
    tf.add_to_collection(scope_name + 'accuracy', accuracy)
#enddef


def run_op(sess, op, batch_tensor_list, lstm_input_dropout,
           dropout, encoding_scheme, tasks, scope_names,
           include_labels=False):
    """
    Runs the specified tensorflow operation, which feeds different
    variables depending on the task and encoding scheme; note that
    for multi-task, multi-scope operations, tasks and scope_names
    specify in-order lists such that tasks[i] corresponds to scope_names[i]
    when retrieving variables; also must correspond with batch_tensors[i[

    :param sess: Tensorflow session
    :param op: Tensorflow operation
    :param batch_tensor_list: List of data batches, corresponding to
                              tasks and scope_names
    :param lstm_input_dropout: Probability to keep for lstm inputs
    :param dropout: Probability to keep for all other nodes
    :param encoding_scheme: {first_last_sentence, first_last_mention}
    :param tasks: A list of tasks (equal to or subset of
                 {nonvis, card, rel_intra, rel_cross, affinity})
                 corresponding to scope_names
    :param scope_names: List of variable scope names, appended to the start
                       of variable names, corresponding to tasks
    :param include_labels: Whether to feed the labels to the operation
    :return:
    """
    feed_dict = dict()

    for i in range(0, len(tasks)):
        task = tasks[i]
        scope_name = scope_names[i]
        if scope_name != "":
            scope_name += "/"
        batch_tensors = batch_tensor_list[i]

        # Feed the common variables, including labels, if specified
        feed_dict[tf.get_collection(scope_name + 'dropout')[0]] = dropout
        if include_labels:
            feed_dict[tf.get_collection(scope_name + 'y')[0]] = \
                batch_tensors['labels']

        # Feed the common lstm variables
        feed_dict[tf.get_collection('bidirectional_lstm/x')[0]] = \
            batch_tensors['sentences']
        feed_dict[tf.get_collection('bidirectional_lstm/seq_lengths')[0]] = \
            batch_tensors['seq_lengths']
        feed_dict[tf.get_collection('bidirectional_lstm/lstm_input_dropout')[0]] = \
            lstm_input_dropout
        feed_dict[tf.get_collection('bidirectional_lstm/dropout')[0]] = \
            dropout

        # Regardless of our task, we have i mentions' forward
        # and backward outputs
        feed_dict[tf.get_collection(scope_name + 'first_i_bw')[0]] = \
            batch_tensors['first_i_bw']
        feed_dict[tf.get_collection(scope_name + 'last_i_fw')[0]] = \
            batch_tensors['last_i_fw']

        if encoding_scheme == "first_last_sentence":
            # If we're encoding using first_last_sentence, all
            # task have the sent_last_i_fw and sent_first_i_bw outputs
            feed_dict[tf.get_collection(scope_name + 'sent_last_i_fw')[0]] = \
                batch_tensors['sent_last_i_fw']
            feed_dict[tf.get_collection(scope_name + 'sent_first_i_bw')[0]] = \
                batch_tensors['sent_first_i_bw']
        elif encoding_scheme == "first_last_mention":
            # If we're encoding using first_last_mention, all tasks
            # have the first_i_fw and last_i_bw outputs
            feed_dict[tf.get_collection(scope_name + 'first_i_fw')[0]] = \
                batch_tensors['first_i_fw']
            feed_dict[tf.get_collection(scope_name + 'last_i_bw')[0]] = \
                batch_tensors['last_i_bw']
        #endif

        # Switch on the task / encoding scheme for the rest of the variables
        if "rel" in task:
            feed_dict[tf.get_collection(scope_name + 'first_j_bw')[0]] = \
                batch_tensors['first_j_bw']
            feed_dict[tf.get_collection(scope_name + 'last_j_fw')[0]] = \
                batch_tensors['last_j_fw']
            feed_dict[tf.get_collection(scope_name + 'ij_feats')[0]] = \
                batch_tensors['ij_feats']

            if encoding_scheme == "first_last_sentence" and task == "rel_cross":
                feed_dict[tf.get_collection(scope_name + 'sent_last_j_fw')[0]] = \
                    batch_tensors['sent_last_j_fw']
                feed_dict[tf.get_collection(scope_name + 'sent_first_j_bw')[0]] = \
                    batch_tensors['sent_first_j_bw']
            elif encoding_scheme == "first_last_mention":
                feed_dict[tf.get_collection(scope_name + 'first_j_fw')[0]] = \
                    batch_tensors['first_j_fw']
                feed_dict[tf.get_collection(scope_name + 'last_j_bw')[0]] = \
                    batch_tensors['last_j_bw']
            #endif
        elif task == "nonvis" or task == "card" or task == "affinity":
            feed_dict[tf.get_collection(scope_name + 'm_feats')[0]] = \
                batch_tensors['m_feats']

            if task == "affinity":
                feed_dict[tf.get_collection(scope_name + 'box_embeddings')[0]] = \
                    batch_tensors['box_embeddings']
                if 'b_feats' in batch_tensors.keys():
                    feed_dict[tf.get_collection(scope_name + 'b_feats')[0]] = \
                        batch_tensors['b_feats']
                #endif
            #endif
        #endif
    #endfor

    return sess.run(op, feed_dict=feed_dict)
#enddef


def get_pred_scores_mcc(task, encoding_scheme, sess, batch_size, ids,
                        data_dict, n_classes, log=None):
    """
    Returns the predicted scores and gold labels for the given list
    of ids and the given data dictionary

    :param task: {rel_intra, rel_cross, nonvis, card, affinity}
    :param encoding_scheme: {first_last_sentence, first_last_mention}
    :param sess: Tensorflow session
    :param batch_size: Size of batches
    :param ids: List of ids for which predictions are made
    :param data_dict: Dictionary containing all our data
    :param n_classes: Number of classes for this task
    :param n_embedding_width: Word embedding width (default: 300)
    :param log: logging utility
    :return: (predicted_scores, gold_labels)
    """

    # Get the current variable scope, if we have one
    scope_name = tf.get_variable_scope().name
    if scope_name != "":
        scope_name += "/"

    pred_scores = dict()
    predicted_proba = tf.get_collection(scope_name + 'predicted_proba')[0]

    # Run until we don't have any more ids for which we need predictions
    n_ids = len(ids)
    pad_length = batch_size * (n_ids / batch_size + 1) - n_ids
    id_arr = np.pad(ids, (0, pad_length), 'edge')
    id_matrix = np.reshape(id_arr, [-1, batch_size])
    for i in range(0, id_matrix.shape[0]):
        if log is not None:
            log.log_status('info', None, 'Predicting; %d batches complete (%.2f%%)',
                           i, 100.0 * i / id_matrix.shape[0])

        # Predict on this batch
        batch_tensors = nn_data.load_batch(id_matrix[i], data_dict, task, n_classes)
        predicted_scores = run_op(sess, predicted_proba, [batch_tensors],
                                  1.0, 1.0, encoding_scheme, [task],
                                  [scope_name.replace("/", "")],
                                  False)

        # Save all the scores unless this is the last row
        row_length = len(predicted_scores)
        if i == id_matrix.shape[0] - 1:
            row_length = batch_size - pad_length
        for j in range(0, row_length):
            pred_scores[id_matrix[i][j]] = predicted_scores[j]
        #endfor
    #endfor
    return pred_scores, data_dict['labels']
#enddef


def dump_tf_vars():
    """
    Dumps the tf variables to the console; intended to
    help sanity check that everything's the right size
    :return:
    """
    for name in tf.get_default_graph().get_all_collection_keys():
        coll = tf.get_collection(name)
        if len(coll) >= 1:
            coll = coll[0]
        print "%-20s: %s" % (name, coll)
    #endfor
#enddef