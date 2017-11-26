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

N_EMBEDDING_WIDTH = 300
CLASSES = ['n', 'c', 'b', 'p']
RELATION_TYPES = ['intra', 'cross']


def train(rel_type, encoding_scheme, embedding_type,
          sentence_file, mention_idx_file, feature_file,
          feature_meta_file, epochs, batch_size, lstm_hidden_width,
          start_hidden_width, hidden_depth, weighted_classes,
          lstm_input_dropout, dropout, lrn_rate,
          adam_epsilon, clip_norm, data_norm, activation, model_file=None,
          eval_sentence_file=None, eval_mention_idx_file=None,
          eval_feature_file=None, eval_feature_meta_file=None,
          eval_label_file=None, early_stopping=False, log=None):
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
    :param lstm_input_dropout: Probability to keep for lstm inputs
    :param dropout: Probability to keep for all other nodes
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
    global CLASSES

    task = "rel_" + rel_type
    n_classes = len(CLASSES)

    log.info("Loading data from " + sentence_file + " and " + mention_idx_file)
    data_dict = nn_data.load_sentences(sentence_file, embedding_type)
    data_dict.update(nn_data.load_mentions(mention_idx_file, task,
                                           feature_file, feature_meta_file,
                                           n_classes))
    log.info("Loading data from " + eval_sentence_file + " and " + eval_mention_idx_file)
    eval_data_dict = nn_data.load_sentences(eval_sentence_file, embedding_type)
    eval_data_dict.update(nn_data.load_mentions(eval_mention_idx_file, task,
                                                eval_feature_file,
                                                eval_feature_meta_file,
                                                n_classes))
    mentions = list(data_dict['mention_indices'].keys())
    n_pairs = len(mentions)

    # Load the gold labels from the label file once, and we can just reuse them every epoch
    gold_label_dict = nn_data.load_relation_labels(eval_label_file)

    # We want to keep track of the best coref and subset scores, along
    # with the epoch that they originated from
    best_coref_subset_avg = -1
    best_coref_subset_epoch = -1

    log.info("Setting up network architecture")

    # Set up the bidirectional LSTM
    with tf.variable_scope('bidirectional_lstm'):
        nn_util.setup_bidirectional_lstm(lstm_hidden_width, data_norm)
    nn_util.setup_core_architecture(task, encoding_scheme,
                                    batch_size, start_hidden_width,
                                    hidden_depth, weighted_classes, activation,
                                    n_classes, data_dict['n_mention_feats'])
    loss = tf.get_collection('loss')[0]
    accuracy = tf.get_collection('accuracy')[0]
    nn_util.add_train_op(loss, lrn_rate, adam_epsilon, clip_norm)
    train_op = tf.get_collection('train_op')[0]
    nn_util.dump_tf_vars()

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
                batch_tensors = nn_data.load_batch(batch_mentions, data_dict,
                                                   task, n_classes)

                # Train
                nn_util.run_op(sess, train_op, [batch_tensors], lstm_input_dropout,
                               dropout, encoding_scheme, [task], [""], True)

                # Store the losses and accuracies every 100 batches
                if (j+1) % 100 == 0:
                    losses.append(nn_util.run_op(sess, loss, [batch_tensors],
                                                 lstm_input_dropout, dropout,
                                                 encoding_scheme, [task],
                                                 [""], True))
                    accuracies.append(nn_util.run_op(sess, accuracy, [batch_tensors],
                                                     lstm_input_dropout, dropout,
                                                     encoding_scheme, [task],
                                                     [""], True))
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
                eval_mention_pairs = eval_data_dict['mention_indices'].keys()
                pred_scores, _ = nn_util.get_pred_scores_mcc(task, encoding_scheme,
                                                             sess, batch_size,
                                                             eval_mention_pairs,
                                                             eval_data_dict,
                                                             n_classes, log)
                pred_labels = list()
                for pair in eval_mention_pairs:
                    pred_labels.append(np.argmax(pred_scores[pair]))

                # Evaluate the predictions
                score_dict = \
                    nn_eval.evaluate_relations(eval_mention_pairs, pred_labels,
                                               gold_label_dict)

                # Get the current coref / subset and see if their average beats our best
                coref_subset_avg = score_dict.get_score('coref').f1 + \
                                   score_dict.get_score('subset').f1
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


def predict(rel_type, encoding_scheme, embedding_type,
            tf_session, batch_size, sentence_file,
            mention_idx_file, feature_file,
            feature_meta_file, label_file, scores_file=None, log=None):
    """
    Wrapper for making predictions on a pre-trained model, already loaded into
    the session
    :param rel_type:
    :param encoding_scheme:
    :param embedding_type:
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
    global CLASSES
    n_classes = len(CLASSES)
    task = "rel_" + rel_type

    # Load the data
    log.info("Loading data from " + sentence_file + " and " + mention_idx_file)
    data_dict = nn_data.load_sentences(sentence_file, embedding_type)
    data_dict.update(nn_data.load_mentions(mention_idx_file, task,
                                           feature_file, feature_meta_file, n_classes))

    # Get the predicted scores, given our arguments
    log.info("Predictiong scores")
    mention_pairs = data_dict['mention_indices'].keys()
    pred_scores, _ = nn_util.get_pred_scores_mcc(task, encoding_scheme,
                                                 tf_session, batch_size,
                                                 mention_pairs, data_dict,
                                                 n_classes, log)

    # log.warning("Skipping evaluation, since it takes way too long right now for some reason")
    log.info("Loading data from " + label_file)
    gold_label_dict = nn_data.load_relation_labels(label_file)

    # If we do an argmax on the scores, we get the predicted labels
    log.info("Getting labels from scores")
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
                    if score == 0:
                        score = np.nextafter(0, 1)
                    score_line.append(str(np.log(score)))
                f.write(",".join(score_line) + "\n")
            f.close()
        #endwith
    #endif
#enddef


def __init__():
    # Set up the global logger
    log = Logger('debug', 180)


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
    parser.add_argument("--lstm_input_dropout", type=float, default=1.0,
                        help="train opt; probability to keep lstm input nodes")
    parser.add_argument("--dropout", type=float, default=1.0,
                        help="train opt; probability to keep all other nodes")
    parser.add_argument("--encoding_scheme",
                        choices=["first_last_sentence", 'first_last_mention'],
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
    if arg_dict['train']:
        arg_dict['model_file'] = "/home/ccervan2/models/tacl201712/" + \
                                    nn_data.build_model_filename(arg_dict, "rel_lstm")
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
    label_file = data_dir + "raw/" + data_root + "_mentionPair_labels.txt"
    if eval_data_root is not None:
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
        train(rel_type=rel_type,
              encoding_scheme=arg_dict['encoding_scheme'],
              embedding_type=embedding_type,
              sentence_file=sentence_file,
              mention_idx_file=mention_idx_file,
              feature_file=feature_file,
              feature_meta_file=feature_meta_file,
              epochs=arg_dict['epochs'], batch_size=arg_dict['batch_size'],
              lstm_hidden_width=arg_dict['lstm_hidden_width'],
              start_hidden_width=arg_dict['start_hidden_width'],
              hidden_depth=arg_dict['hidden_depth'],
              weighted_classes=arg_dict['weighted_classes'],
              lstm_input_dropout=arg_dict['lstm_input_dropout'],
              dropout=arg_dict['dropout'],
              lrn_rate=arg_dict['learn_rate'], clip_norm=arg_dict['clip_norm'],
              data_norm=arg_dict['data_norm'],
              adam_epsilon=arg_dict['adam_epsilon'], activation=arg_dict['activation'],
              model_file=model_file, eval_sentence_file=eval_sentence_file,
              eval_mention_idx_file=eval_mention_idx_file,
              eval_feature_file=eval_feature_file,
              eval_feature_meta_file=eval_feature_meta_file,
              eval_label_file=eval_label_file,
              early_stopping=arg_dict['early_stopping'], log=log)
    elif arg_dict['predict']:
        scores_file = data_dir + "scores/" + data_root + \
                      "_relation_" + rel_type + ".scores"

        # Restore our variables
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_file + ".meta")
            saver.restore(sess, model_file)
            predict(rel_type=rel_type,
                    encoding_scheme=arg_dict['encoding_scheme'],
                    embedding_type=embedding_type,
                    tf_session=sess, batch_size=arg_dict['batch_size'],
                    sentence_file=sentence_file,
                    mention_idx_file=mention_idx_file,
                    feature_file=feature_file, feature_meta_file=feature_meta_file,
                    label_file=label_file, scores_file=scores_file, log=log)
        #endwith
    #endif
#enddef


__init__()

