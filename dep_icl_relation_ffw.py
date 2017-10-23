import os
from argparse import ArgumentParser
from os.path import abspath, expanduser

import nn_util
import numpy as np
import tensorflow as tf

import nn_utils.core
import nn_utils.data
from utils import icl_util as util
from utils.LogUtil import LogUtil

___author___ = "ccervantes"

N_EMBEDDING_FEATS = 300
CLASSES = ['n', 'c', 'b', 'p']
RELATION_TYPES = ['intra', 'cross']

rel_type = None
pair_enc_scheme = None
data_norm = False
use_engr_feats = False

# Set up the global logger
log = LogUtil('debug', 180)


def load_batch(mention_pairs, data_dict):
    """
    Loads a batch of data, given a list of mention pairs
    and the data dictionary; returns a different representation
    of the given mention pairs depending on global
    pair encoding scheme

    :param mention_pairs: List of mention pair IDs
    :param data_dict: Dictionary of all data dictionaries (for sentences, etc)
    :return: dictionary of batch tensors with aforementioned keys
    """
    global N_EMBEDDING_FEATS, CLASSES, RELATION_TYPES, \
        rel_type, pair_enc_scheme, use_engr_feats
    batch_tensors = dict()

    batch_size = len(mention_pairs)
    n_pair_encoding = None
    if pair_enc_scheme == 'first_avg_last':
        n_pair_encoding = 6 * N_EMBEDDING_FEATS
    elif pair_enc_scheme == 'first_last_sentence':
        n_pair_encoding = 8 * N_EMBEDDING_FEATS
    if use_engr_feats:
        n_pair_encoding += data_dict['max_feat_idx'] + 1
    batch_tensors['mention_pairs'] = np.zeros([batch_size, n_pair_encoding])
    batch_tensors['labels'] = np.zeros([batch_size, len(CLASSES)])
    for i in range(0, batch_size):
        pair_id = mention_pairs[i]

        # Add this pair's label to the label batch
        batch_tensors['labels'][i] = data_dict['mention_pair_labels'][pair_id]

        # get this mention pair's word indices and caption indices
        first_i, last_i, first_j, last_j = data_dict['mention_pair_indices'][pair_id]

        # Retrieve the word embeddings for these mentions' sentence(s)
        sentence_id_i, sentence_id_j = data_dict['mention_pair_cap_ids'][pair_id]
        sentence_matrix_i = data_dict['sentences'][sentence_id_i]
        sentence_matrix_j = data_dict['sentences'][sentence_id_j]

        # Retrieve the pair feature vectors (or zeros, if applicable)
        pair_feats = None
        if use_engr_feats:
            pair_feats = np.zeros(data_dict['max_feat_idx']+1)
            if pair_id in data_dict['mention_pair_feats']:
                pair_feats = data_dict['mention_pair_feats'][pair_id]
        #endif

        # Get the relevant word embeddings, depending on our scheme
        first_i_vec = sentence_matrix_i[first_i]
        last_i_vec = sentence_matrix_i[last_i]
        first_j_vec = sentence_matrix_j[first_j]
        last_j_vec = sentence_matrix_j[last_j]
        tensor_list = list()
        if pair_enc_scheme == 'first_avg_last':
            mi_len = last_i + 1 - first_i
            avg_i = np.zeros([mi_len, N_EMBEDDING_FEATS])
            for j in range(0, mi_len):
                avg_i[j] = sentence_matrix_i[first_i + j]
            avg_i = avg_i.mean(0)
            mj_len = last_j + 1 - first_j
            avg_j = np.zeros([mj_len, N_EMBEDDING_FEATS])
            for j in range(0, mj_len):
                avg_j[j] = sentence_matrix_j[first_j + j]
            avg_j = avg_j.mean(0)
            tensor_list = [first_i_vec, avg_i, last_i_vec,
                           first_j_vec, avg_j, last_j_vec]
        elif pair_enc_scheme == 'first_last_sentence':
            sent_first_i = sentence_matrix_i[0]
            sent_last_i = sentence_matrix_i[len(sentence_matrix_i)-1]
            sent_first_j = sentence_matrix_j[0]
            sent_last_j = sentence_matrix_j[len(sentence_matrix_j)-1]

            tensor_list = [sent_first_i, sent_last_i,
                           first_i_vec, last_i_vec,
                           sent_first_j, sent_last_j,
                           first_j_vec, last_j_vec]
        #endif

        # Add the feature vector, if appropriate
        if use_engr_feats:
            tensor_list.append(pair_feats)

        batch_tensors['mention_pairs'][i] = np.concatenate(tensor_list)
    #endfor
    return batch_tensors
#enddef


def setup_relation(batch_size, start_hidden_width, max_depth,
                   weighted_classes, lrn_rate,
                   clip_norm, adam_epsilon, activation,
                   n_engr_feats=None):
    """
    Sets up the relation classifier network, which simply passes
    transformed word2vec embeddings to a hidden layer

    :param batch_size: Number of mention pairs to run each batch
    :param start_hidden_width: Number of hidden units to which the mention pairs'
                         representation is passed
    :param max_depth:
    :param weighted_classes: Whether to weight the examples by their class inversely
                    with the frequency of that class
    :param lrn_rate: Learning rate of the optimizer
    :param clip_norm: Global gradient clipping norm
    :param adam_epsilon: Adam optimizer epsilon value
    :param activation: Nonlinear activation function (sigmoid, tanh, relu)
    :param n_engr_feats: The maximum feature vector value for our engineered
                         features
    """
    global CLASSES, N_EMBEDDING_FEATS, pair_enc_scheme, data_norm, use_engr_feats

    # Each mention pair is a one-hot for its label (n,c,b,p)
    y = tf.placeholder(tf.float32, [batch_size, len(CLASSES)])
    tf.add_to_collection('y', y)

    # Mention pair inputs vary on our setup
    hidden_input_width = None
    if pair_enc_scheme == 'first_avg_last':
        hidden_input_width = 6 * N_EMBEDDING_FEATS
    elif pair_enc_scheme == 'first_last_sentence':
        hidden_input_width = 8 * N_EMBEDDING_FEATS
    if use_engr_feats:
        hidden_input_width += n_engr_feats
    x = tf.placeholder(tf.float32, [batch_size, hidden_input_width])
    if data_norm:
        x = tf.nn.l2_normalize(x, dim=1)
    tf.add_to_collection('x', x)

    # dropout percentage
    dropout = tf.placeholder(tf.float32)
    tf.add_to_collection('dropout', dropout)

    # Set up the hidden layer(s)
    hidden_inputs = list()
    hidden_inputs.append(x)
    n_hidden_widths = [start_hidden_width]
    for depth in range(1, max_depth):
        n_hidden_widths.append(n_hidden_widths[depth-1] / 2)
    for depth in range(0, max_depth):
        with tf.variable_scope("hdn_" + str(depth+1)):
            weights = nn_utils.core.get_weights([hidden_input_width, n_hidden_widths[depth]])
            biases = nn_utils.core.get_biases([1, n_hidden_widths[depth]])
            logits = tf.matmul(hidden_inputs[depth], weights) + biases
            if activation == 'sigmoid':
                logits = tf.nn.sigmoid(logits)
            elif activation == 'tanh':
                logits = tf.nn.tanh(logits)
            elif activation == 'relu':
                logits = tf.nn.relu(logits)
            elif activation == 'leaky_relu':
                logits = nn_utils.core.leaky_relu(logits)
            logits = tf.nn.dropout(logits, dropout)
            hidden_inputs.append(logits)
            hidden_input_width = n_hidden_widths[depth]
        #endwith
    #endfor

    with tf.variable_scope("softmax"):
        weights = nn_utils.core.get_weights([n_hidden_widths[max_depth - 1], len(CLASSES)])
        biases = nn_utils.core.get_biases([1, len(CLASSES)])
        # Because our label distribution is so skewed, we have to
        # add a constant epsilon to all of the values to prevent
        # the loss from being NaN
        epsilon = np.nextafter(0, 1)
        constant_epsilon = tf.constant([epsilon, epsilon, epsilon, epsilon], dtype=tf.float32)
        final_logits = tf.matmul(hidden_inputs[max_depth], weights) + biases + constant_epsilon
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

    # Adam optimizer sets a variable learning rate for every weight, along
    # with rolling averages; It uses beta1 (def: 0.9), beta2 (def: 0.99),
    # and epsilon (def: 1E-08) to get exponential decay
    optimiz = tf.train.AdamOptimizer(learning_rate=lrn_rate,
                                     epsilon=adam_epsilon)
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

    # Evaluate model
    pred = tf.argmax(logits, 1)
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


def train(sentence_file, mention_idx_file, feature_file,
          feature_meta_file, epochs, batch_size, start_hidden_width,
          hidden_depth,
          weighted_classes, dropout_p, lrn_rate, adam_epsilon,
          clip_norm, activation, model_file=None,
          eval_sentence_file=None, eval_mention_idx_file=None):
    """
    Trains a relation model as a simple feed-foward network

    :param sentence_file: File with captions
    :param mention_idx_file: File with mention pair word indices
    :param feature_file: File with sparse mention pair features
    :param feature_meta_file: File associating sparse indices with feature names
    :param epochs: Number of times to run over the data
    :param batch_size: Number of mention pairs to run each batch
    :param start_hidden_width: Number of hidden units to which the mention pairs'
                         representation is passed
    :param weighted_classes: Whether to weight the examples by their
                             class inversely with the frequency of
                             that class
    :param dropout_p: Dropout probability (percent to keep)
    :param lrn_rate: Learning rate of the optimizer
    :param clip_norm: Global gradient clipping norm
    :param adam_epsilon: Adam optimizer epsilon value
    :param activation: Nonlinear activation function (sigmoid,tanh,relu)
    :param model_file: File to which the model is periodically saved
    :param eval_sentence_file: Sentence file against which the model
                               should be evaluated
    :param eval_mention_idx_file: Mention index file against which
                                  the model should be evaluated
    :return:
    """
    global log, use_engr_feats

    log.info("Loading data from " + sentence_file + " and " + mention_idx_file)
    if use_engr_feats:
        data_dict = nn_utils.data.load_mention_pair_data(sentence_file, mention_idx_file,
                                                         feature_file, feature_meta_file)
    else:
        data_dict = nn_utils.data.load_mention_pair_data(sentence_file, mention_idx_file)
    #endif
    mention_pairs = data_dict['mention_pair_indices'].keys()
    n_pairs = len(mention_pairs)

    log.info("Setting up network architecture")
    n_engr_feats = None
    if use_engr_feats:
        n_engr_feats = data_dict['max_feat_idx'] + 1
    setup_relation(batch_size, start_hidden_width, hidden_depth,
                   weighted_classes, lrn_rate, clip_norm,
                   adam_epsilon, activation, n_engr_feats)

    # Get our tensorflow variables and operations
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    dropout = tf.get_collection('dropout')[0]
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
                batch_tensors = load_batch(mention_pairs[start_idx:end_idx],
                                           data_dict)

                # Train
                sess.run(train_op, feed_dict={x: batch_tensors['mention_pairs'],
                                              dropout: dropout_p,
                                              y: batch_tensors['labels']})

                # Store the losses and accuracies every 100 batches
                if (j+1) % 100 == 0:
                    losses.append(sess.run(loss,
                                           feed_dict={x: batch_tensors['mention_pairs'],
                                                      dropout: dropout_p,
                                                      y: batch_tensors['labels']}))
                    accuracies.append(sess.run(accuracy,
                                               feed_dict={x: batch_tensors['mention_pairs'],
                                                          dropout: dropout_p,
                                                          y: batch_tensors['labels']}))
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
                if feature_file is not None and feature_meta_file is not None:
                    predict(sess, eval_sentence_file,
                            eval_mention_idx_file, feature_file.replace('train', 'dev'),
                            feature_meta_file.replace('train', 'dev'), batch_size)
                else:
                    predict(sess, eval_sentence_file, eval_mention_idx_file, None, None,
                            batch_size)
        #endfor
        log.info("Saving final model")
        saver.save(sess, model_file)
    #endwith
#enddef


def predict(tf_session, sentence_file, mention_idx_file,
            feature_file, feature_meta_file,
            batch_size, scores_file=None):
    """
    Evaluates the model loaded in the given session and logs the counts
    and p/r/f1 on the given data; optionally saves the predicted scores
    to the scores_file

    :param tf_session: Tensorflow session in which the model has
                       been loaded
    :param sentence_file: Sentence file against which the model
                          should be evaluated
    :param mention_idx_file: Mention index file against which
                             the model should be evaluated
    :param feature_file: File with sparse mention pair features
    :param feature_meta_file: File associating sparse indices with feature names
    :param batch_size: Size of the groups of mention pairs to pass
                       to the network at once
    :param scores_file: The file in which to save predicted probabilities
    """
    global log, use_engr_feats

    gold_labels = list()
    pred_labels = list()
    pred_scores = dict()

    log.info("Loading data from " + sentence_file + " and " + mention_idx_file)
    if use_engr_feats:
        data_dict = nn_utils.data.load_mention_pair_data(sentence_file, mention_idx_file,
                                                         feature_file, feature_meta_file)
    else:
        data_dict = nn_utils.data.load_mention_pair_data(sentence_file, mention_idx_file)
    #endif
    mention_pairs = data_dict['mention_pair_indices'].keys()

    # Get the predict operation
    predicted_proba = tf.get_collection('predicted_proba')[0]
    x = tf.get_collection('x')[0]
    dropout = tf.get_collection('dropout')[0]

    # Run until we don't have any more mention pairs to predict for
    pad_length = batch_size * (len(mention_pairs) / batch_size + 1) - len(mention_pairs)
    mention_pair_arr = np.pad(mention_pairs, (0, pad_length), 'edge')
    mention_pair_matrix = np.reshape(mention_pair_arr, [-1, batch_size])
    for i in range(0, mention_pair_matrix.shape[0]):
        log.log_status('info', None, 'Predicting; %d batches complete (%.2f%%)',
                       i, 100.0 * i / mention_pair_matrix.shape[0])
        batch_tensors = load_batch(mention_pair_matrix[i], data_dict)

        # Add the labels to the list
        gold_labels.extend(np.argmax(batch_tensors['labels'], 1))
        predicted_scores = tf_session.run(predicted_proba,
                                          feed_dict={x: batch_tensors['mention_pairs'],
                                                     dropout: 1.0})
        pred_labels.extend(np.argmax(predicted_scores, 1))

        # Take all the scores unless this is the last row
        row_length = len(predicted_scores)
        if i == mention_pair_matrix.shape[0] - 1:
            row_length = batch_size - pad_length
        for j in range(0, row_length):
            pred_scores[mention_pair_matrix[i][j]] = predicted_scores[j]
    #endfor

    # Print the evaluation
    nn_util.log_eval(gold_labels, pred_labels, CLASSES, log)

    # If a scores file was specified, write the scores
    if scores_file is not None:
        with open(scores_file, 'w') as f:
            for pair_id in pred_scores.keys():
                scores = list()
                for score in pred_scores[pair_id]:
                    scores.append(str(score))
                f.write(pair_id + "\t" + ",".join(scores) + "\n")
            f.close()
        #endwith
    #endif
#enddef


def __init__():
    global rel_type, pair_enc_scheme, data_norm, log, use_engr_feats

    # Parse arguments
    parser = ArgumentParser("ImageCaptionLearn_py: Neural Network for Relation "
                            "Prediction; Bidirectional LSTM to hidden layer "
                            "to softmax over (n)ull, (c)oreference, su(b)set, "
                            "and su(p)erset labels")
    parser.add_argument("--epochs", type=int, default=100,
                        help="train opt; number of times to iterate over the dataset")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="train opt; number of random mention pairs per batch")
    parser.add_argument("--hidden_rel", type=int, default=300,
                        help="train opt; the starting number of hidden units")
    parser.add_argument("--hidden_depth", type=int, default=1,
                        help="train opt; the number of hidden layers to use")
    parser.add_argument("--weighted_classes", action="store_true",
                        help="Whether to inversely weight the classes "
                             "in the loss")
    parser.add_argument("--learn_rate", type=float, default=0.001,
                        help="train opt; optimizer learning rate")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="train opt; Adam optimizer epsilon value")
    parser.add_argument("--clip_norm", type=float, help='train opt; global clip norm value')
    parser.add_argument("--data_norm", action='store_true',
                        help="train opt; Whether to L2-normalize the w2v word vectors")
    parser.add_argument("--dropout", type=float, default=1.0,
                        help="train opt; Dropout rate (probability to keep)")
    parser.add_argument("--pair_enc_scheme", choices=["first_avg_last", "first_last_sentence"],
                        default="first_avg_last",
                        help="train opt; specifies how mention pairs' vectors "
                             "are represented")
    parser.add_argument("--sentence_file", #required=True,
                        type=lambda f: util.arg_file_exists(parser, f),
                        help="File associating caption IDs with their captions "
                             "(punctuation tokens removed)")
    parser.add_argument("--mention_idx_file", #required=True,
                        type=lambda f: util.arg_file_exists(parser, f),
                        help="File associating mention pair IDs with "
                             "the set of indices that define them")
    parser.add_argument("--feature_file",
                        type=lambda f: util.arg_file_exists(parser, f),
                        help="Mention pair feature file (sparse feature format)")
    parser.add_argument("--feature_meta_file",
                        type=lambda f: util.arg_file_exists(parser, f),
                        help="Mention pair feature meta file (associates indices with names)")
    parser.add_argument("--train", action='store_true', help='Trains a model')
    parser.add_argument("--activation", choices=['sigmoid', 'tanh', 'relu', 'leaky_relu'],
                        default='relu',
                        help='train opt; which nonlinear activation function to use')
    parser.add_argument("--predict", action='store_true',
                        help='Predicts using pre-trained model')
    parser.add_argument("--rel_type", choices=['intra', 'cross'], required=True,
                        help="Whether we're dealing with intra-caption or "
                             "cross-caption relations")
    parser.add_argument("--model_file", type=str, help="Model file to save/load")
    parser.add_argument("--score_file", type=str,
                        help="File to which predicted scores will be saved")
    args = parser.parse_args()
    arg_dict = vars(args)

    scores_file = arg_dict['score_file']
    if scores_file is not None:
        scores_file = abspath(expanduser(scores_file))

    # If a feature and feature meta file were provided, we're
    # using engineered features
    if arg_dict['feature_file'] is not None and \
                    arg_dict['feature_meta_file'] is not None:
        use_engr_feats = True

    # Set our global relation type and pair
    # encoding scheme
    rel_type = arg_dict['rel_type']
    pair_enc_scheme = arg_dict['pair_enc_scheme']
    data_norm = arg_dict['data_norm']

    # For quick experimentation purposes, we're going to set
    # some default values here. Note that this is ONLY for
    # the grid search over parameters and should be
    # removed as we move forward
    #arg_dict['sentence_file'] = abspath(expanduser("~/data/tacl201708/nn/flickr30k_train_tune_captions.txt"))
    #arg_dict['mention_idx_file'] = abspath(expanduser("~/data/tacl201708/nn/flickr30k_train_tune_mentionPairs_intra.txt"))
    arg_dict['sentence_file'] = abspath(expanduser("~/data/tacl201708/nn/flickr30k_train_captions.txt"))
    #arg_dict['mention_idx_file'] = abspath(expanduser("~/data/tacl201708/nn/flickr30k_train_mentionPairs_intra.txt"))
    mention_idx_root = "~/data/tacl201708/nn/flickr30k_train_mentionPairs_" + rel_type + ".txt"
    arg_dict['mention_idx_file'] = abspath(expanduser(mention_idx_root))

    model_file = "~/models/tacl201708/nn/" + \
                 "relation_" + arg_dict['rel_type'] + "_ffw_" + \
                 arg_dict['pair_enc_scheme'] + "_" + \
                 "epochs" + str(int(arg_dict['epochs'])) + "_" + \
                 "lrnRate" + str(arg_dict['learn_rate']) + "_" + \
                 "adamEps" + str(arg_dict['adam_epsilon']) + "_"
    if arg_dict['clip_norm'] is not None:
        model_file += "norm" + str(arg_dict['clip_norm']) + "_"
    model_file += "batch" + str(int(arg_dict['batch_size'])) + "_" + \
                  "dropout" + str(int(arg_dict['dropout'] * 100)) + "_" + \
                  "rel" + str(int(arg_dict['hidden_rel']))
    if arg_dict['weighted_classes']:
        model_file += "_weighted"
    model_file += ".model"
    model_file = abspath(expanduser(model_file))
    arg_dict['model_file'] = model_file
    arg_dict['feature_file'] = abspath(expanduser('~/data/tacl201708/feats/flickr30k_train_relation_nn.feats'))
    arg_dict['feature_meta_file'] = abspath(expanduser('~/data/tacl201708/feats/flickr30k_train_relation_nn_meta.json'))

    util.dump_args(arg_dict, log)

    # Grab the possibly-not-yet-created model file from the args
    # model_file = abspath(expanduser(arg_dict['model_file']))

    # Set up the word2vec utility once
    log.info("Initializing word2vec")
    nn_util.init_w2v()

    # Set the random seeds identically every run
    nn_utils.core.set_random_seeds()

    # Set up the minimum tensorflow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Train, if training was specified
    if arg_dict['train']:
        train(sentence_file=arg_dict['sentence_file'],
              mention_idx_file=arg_dict['mention_idx_file'],
              feature_file=arg_dict['feature_file'],
              feature_meta_file=arg_dict['feature_meta_file'],
              epochs=arg_dict['epochs'], batch_size=arg_dict['batch_size'],
              start_hidden_width=arg_dict['hidden_rel'],
              hidden_depth=arg_dict['hidden_depth'],
              weighted_classes=arg_dict['weighted_classes'],
              dropout_p=arg_dict['dropout'], lrn_rate=arg_dict['learn_rate'],
              clip_norm=arg_dict['clip_norm'], adam_epsilon=arg_dict['adam_epsilon'],
              activation=arg_dict['activation'],
              model_file=model_file, eval_sentence_file=arg_dict['sentence_file'].replace('train', 'dev'),
              eval_mention_idx_file=arg_dict['mention_idx_file'].replace('train', 'dev'))
    elif arg_dict['predict']:
        # Restore our variables
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_file + ".meta")
            saver.restore(sess, model_file)

            predict(sess, sentence_file=arg_dict['sentence_file'].replace('train', 'dev'),
                    mention_idx_file=arg_dict['mention_idx_file'].replace('train', 'dev'),
                    feature_file=arg_dict['feature_file'].replace('train', 'dev'),
                    feature_meta_file=arg_dict['feature_meta_file'].replace('train', 'dev'),
                    batch_size=arg_dict['batch_size'], scores_file=scores_file)
        #endwith
    #endif
#enddef


__init__()

