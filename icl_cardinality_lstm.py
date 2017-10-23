import math
import os
from os.path import abspath, expanduser
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser

from nn_utils import data as nn_data
from nn_utils import eval as nn_eval
from nn_utils import core as nn_util
from utils import core as util
from utils import data as data_util
from utils.Logger import Logger

___author___ = "ccervantes"


'''
import GPflow as gp_flow

np.random.seed(1)
X = np.random.rand(10, 1)
Y = np.round(X*4)
Y = np.asarray(Y, np.int32)

bin_edges = np.arange(np.unique(Y).size - 1)
likelihood = gp_flow.likelihoods.Ordinal(bin_edges)


# build a model with this likelihood
m = gp_flow.vgp.VGP(X, Y,
                   kern=gp_flow.kernels.Matern32(1),
                   likelihood=likelihood)

# fit the model
m.optimize()

mu, var = m.predict_y(X)

print X
print Y
print np.round(mu)
'''

N_EMBEDDING_FEATS = 300
CLASSES = ['0', '1', '2', '3', '4', '5',
           '6', '7', '8', '9', '10', '11+']

data_norm = False
embedding_type = None

# Set up the global logger
log = Logger('debug', 180)


def setup(batch_size, lstm_hidden_width, start_hidden_width,
          hidden_depth, weighted_classes,
          lrn_rate, clip_norm, adam_epsilon, activation,
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
    :param n_parallel: Number of parallel jobs the LSTM may perform at once
    :param lrn_rate: Learning rate of the optimizer
    :param clip_norm: Global gradient clipping norm
    :param adam_epsilon: Adam optimizer epsilon value
    :param activation: Nonlinear activation function (sigmoid, tanh, relu)
    :param n_engr_feats: The maximum feature vector value for our engineered
                         features
    """
    global CLASSES, N_EMBEDDING_FEATS

    # Each mention pair is a one-hot for its label (n,c,b,p)
    y = tf.placeholder(tf.float32, [batch_size, len(CLASSES)])
    tf.add_to_collection('y', y)

    # Set up the bidirectional LSTM
    with tf.variable_scope('bidirectional_lstm'):
        nn_util.setup_bidirectional_lstm(lstm_hidden_width)

    # Get the outputs, which are a (fw,bw) tuple of
    # [batch_size, seq_length, n_hidden_lstm matrices
    lstm_outputs = (tf.get_collection('bidirectional_lstm/outputs_fw')[0],
                    tf.get_collection('bidirectional_lstm/outputs_bw')[0])

    # Set up the batch input tensors, which is a manipulation of
    # lstm outputs based on graph inputs
    hidden_input_width = 4 * lstm_hidden_width + n_engr_feats
    nn_util.setup_batch_input_first_last_sentence_mention(batch_size, lstm_outputs)
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
        eps_list = list()
        for i in range(0, len(CLASSES)):
            eps_list.append(epsilon)
        constant_epsilon = tf.constant(eps_list, dtype=tf.float32)
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

    # Set up the train operation
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


def train(sentence_file, mention_idx_file, feature_file,
          feature_meta_file, epochs, batch_size, lstm_hidden_width,
          start_hidden_width, hidden_depth, weighted_classes,
          input_dropout, other_dropout, lrn_rate,
          adam_epsilon, clip_norm, activation, model_file=None,
          eval_sentence_file=None, eval_mention_idx_file=None,
          eval_feature_file=None, eval_feature_meta_file=None,
          early_stopping=False):
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
    :param n_parallel: Number of parallel jobs the LSTM may perform at once
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
    global embedding_type

    log.info("Loading data from " + sentence_file + " and " + mention_idx_file)
    data_dict = nn_data.load_mention_data(mention_idx_file, len(CLASSES),
                                          feature_file, feature_meta_file)
    sentence_dict = nn_data.load_sentences(sentence_file, embedding_type)
    for key in sentence_dict.keys():
        data_dict[key] = sentence_dict[key]
    log.info("Loading data from " + eval_sentence_file + " and " + eval_mention_idx_file)
    eval_data_dict = nn_data.load_mention_data(eval_mention_idx_file, len(CLASSES),
                                               eval_feature_file, eval_feature_meta_file)
    eval_sentence_dict = nn_data.load_sentences(eval_sentence_file, embedding_type)
    for key in eval_sentence_dict.keys():
        eval_data_dict[key] = eval_sentence_dict[key]

    mentions = list(data_dict['mention_indices'].keys())
    n_pairs = len(mentions)

    log.info("Setting up network architecture")
    n_engr_feats = data_dict['max_feat_idx'] + 1
    setup(batch_size, lstm_hidden_width, start_hidden_width,
          hidden_depth, weighted_classes, lrn_rate,
          clip_norm, adam_epsilon, activation, n_engr_feats)

    # Get our model-independent tensorflow operations
    loss = tf.get_collection('loss')[0]
    train_op = tf.get_collection('train_op')[0]
    accuracy = tf.get_collection('accuracy')[0]

    # We want to keep track of the best scores with
    # the epoch that they originated from
    best_avg_score = -1
    best_epoch = -1

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
            np.random.shuffle(mentions)

            # Iterate through the entirety of the data
            start_idx = 0
            end_idx = start_idx + batch_size
            n_iter = n_pairs / batch_size
            for j in range(0, n_iter):
                log.log_status('info', None, 'Training; %d (%.2f%%) batches complete',
                               j, 100.0 * j / n_iter)

                # Retrieve this batch
                batch_mentions = mentions[start_idx:end_idx]
                batch_tensors = nn_data.load_batch_first_last_sentence_mention(batch_mentions, data_dict, len(CLASSES))

                # Train
                nn_util.run_op_first_last_sentence_mention(sess, train_op, batch_tensors,
                                                           input_dropout, other_dropout, True)

                # Store the losses and accuracies every 100 batches
                if (j+1) % 100 == 0:
                    losses.append(
                        nn_util.run_op_first_last_sentence_mention(sess, loss, batch_tensors,
                                                                   input_dropout, other_dropout,
                                                                   True))
                    accuracies.append(
                        nn_util.run_op_first_last_sentence_mention(sess, accuracy, batch_tensors,
                                                                   input_dropout, other_dropout,
                                                                   True))
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
                pred_scores, gold_label_dict = get_pred_scores(sess, batch_size, eval_data_dict)

                # If we do an argmax on the scores, we get the predicted labels
                eval_mentions = list(pred_scores.keys())
                pred_labels = list()
                gold_labels = list()
                for m in eval_mentions:
                    pred_labels.append(np.argmax(pred_scores[m]))
                    gold_labels.append(np.argmax(gold_label_dict[m]))
                #endfor

                # Evaluate the predictions
                score_dict = nn_eval.evaluate_multiclass(gold_labels, pred_labels, CLASSES, log)


                # Get the current coref / subset and see if their average beats our best
                avg = 0
                for lbl in range(0, len(CLASSES)):
                    avg += score_dict.get_score(lbl).f1
                avg /= len(CLASSES)

                if avg >= best_avg_score - 0.005:
                    log.info(None, "Previous best score average F1 of %.2f%% after %d epochs",
                             100.0 * best_avg_score, best_epoch)
                    best_avg_score = avg
                    best_epoch = i
                    log.info(None, "New best at current epoch (%.2f%%)",
                             100.0 * best_avg_score)
                #endif

                # Implement early stopping; if it's been 10 epochs since our best, stop
                if early_stopping and i >= (best_epoch + 10):
                    log.info(None, "Stopping early; best scores at %d epochs", best_epoch)
                    break
                #endif
            #endif
        #endfor
        log.info("Saving final model")
        saver.save(sess, model_file)
    #endwith
#enddef


def get_pred_scores(tf_session, batch_size, eval_data_dict):
    """
    Evaluates the model loaded in the given session and logs the counts
    and p/r/f1 on the given data

    :param tf_session: Tensorflow session in which the model has
                       been loaded
    :param batch_size: Size of the groups of mention pairs to pass
                       to the network at once
    :param eval_data_dict: Evaluation data dict
    """
    global log
    pred_scores = dict()
    mentions = eval_data_dict['mention_indices'].keys()

    # Get the predict operation
    predicted_proba = tf.get_collection('predicted_proba')[0]

    # Run until we don't have any more mention pairs to predict for
    pad_length = batch_size * (len(mentions) / batch_size + 1) - len(mentions)
    mention_arr = np.pad(mentions, (0, pad_length), 'edge')
    mention_matrix = np.reshape(mention_arr, [-1, batch_size])
    for i in range(0, mention_matrix.shape[0]):
        log.log_status('info', None, 'Predicting; %d batches complete (%.2f%%)',
                       i, 100.0 * i / mention_matrix.shape[0])
        batch_tensors = nn_data.load_batch_first_last_sentence_mention(mention_matrix[i], eval_data_dict, len(CLASSES))

        # Add the labels to the list
        predicted_scores = nn_util.run_op_first_last_sentence_mention(tf_session, predicted_proba,
                                                                      batch_tensors, 1.0, 1.0, False)

        # Take all the scores unless this is the last row
        row_length = len(predicted_scores)
        if i == mention_matrix.shape[0] - 1:
            row_length = batch_size - pad_length
        for j in range(0, row_length):
            pred_scores[mention_matrix[i][j]] = predicted_scores[j]
    #endfor
    return pred_scores, eval_data_dict['mention_labels']
#enddef


def predict(tf_session, batch_size, sentence_file,
            mention_idx_file, feature_file,
            feature_meta_file, scores_file=None):
    global embedding_type

    # Load the data
    log.info("Loading data from " + sentence_file + " and " + mention_idx_file)
    data_dict = nn_data.load_mention_data(mention_idx_file, len(CLASSES),
                                          feature_file, feature_meta_file)
    sentence_dict = nn_data.load_sentences(sentence_file, embedding_type)
    for key in sentence_dict.keys():
        data_dict[key] = sentence_dict[key]

    # Get the predicted scores, given our arguments
    pred_scores, gold_label_dict = get_pred_scores(tf_session, batch_size, data_dict)

    # If we do an argmax on the scores, we get the predicted labels
    mentions = list(pred_scores.keys())
    pred_labels = list()
    gold_labels = list()
    for m in mentions:
        pred_labels.append(np.argmax(pred_scores[m]))
        gold_labels.append(np.argmax(gold_label_dict[m]))
    #endfor

    # Evaluate the predictions
    nn_eval.evaluate_multiclass(gold_labels, pred_labels, CLASSES, log)

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
    global data_norm, log, embedding_type

    # Parse arguments
    parser = ArgumentParser("ImageCaptionLearn_py: Neural Network for Cardinality "
                            "Prediction; Bidirectional LSTM to hidden layer "
                            "to softmax over 0-10+ labels")
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
    parser.add_argument("--model_file", type=str, help="Model file to save/load")
    parser.add_argument("--embedding_type", choices=['w2v', 'glove'], default='w2v',
                        help="Word embedding type to use")
    parser.add_argument("--early_stopping", action='store_true',
                        help="Whether to implement early stopping based on the "+
                             "evaluation performance")
    args = parser.parse_args()
    arg_dict = vars(args)

    data_norm = arg_dict['data_norm']
    if arg_dict['train']:
        arg_dict['model_file'] = "/home/ccervan2/models/tacl201711/" + \
                                 nn_data.build_model_file_name(arg_dict, "card_lstm")
    model_file = arg_dict['model_file']
    util.dump_args(arg_dict, log)

    # Construct data files from the root directory and filename
    data_dir = arg_dict['data_dir'] + "/"
    data_root = arg_dict['data_root']
    eval_data_root = arg_dict['eval_data_root']
    sentence_file = data_dir + "raw/" + data_root + "_captions.txt"
    mention_idx_file = data_dir + "raw/" + data_root + "_mentions_card.txt"
    feature_file = data_dir + "feats/" + data_root + "_card.feats"
    feature_meta_file = data_dir + "feats/" + data_root + "_card_meta.json"
    eval_sentence_file = data_dir + "raw/" + eval_data_root + "_captions.txt"
    eval_mention_idx_file = data_dir + "raw/" + eval_data_root + "_mentions_card.txt"
    eval_feature_file = data_dir + "feats/" + eval_data_root + "_card.feats"
    eval_feature_meta_file = data_dir + "feats/" + eval_data_root + "_card_meta.json"

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
              early_stopping=arg_dict['early_stopping'])
    elif arg_dict['predict']:
        scores_file = data_dir + "scores/" + data_root + "_card.scores"

        # Restore our variables
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_file + ".meta")
            saver.restore(sess, model_file)

            predict(sess, batch_size=arg_dict['batch_size'],
                    sentence_file=sentence_file,
                    mention_idx_file=mention_idx_file,
                    feature_file=feature_file,
                    feature_meta_file=feature_meta_file,
                    scores_file=scores_file)
            #endwith
        #endif
#enddef


__init__()

