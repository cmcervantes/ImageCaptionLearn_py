import math
import random
import os
from argparse import ArgumentParser
from os.path import abspath, expanduser

import numpy as np
import tensorflow as tf

from nn_utils import data as nn_data
from nn_utils import eval as nn_eval
from nn_utils import core as nn_util
from utils import core as util
from utils.Logger import Logger

___author___ = "ccervantes"

N_EMBEDDING_WIDTH = 300
N_BOX_WIDTH = 4096
CLASSES = ['0', '1']
task = "affinity"


def shuffle_mention_box_pairs(mention_box_pairs):
    """
    Returns a shuffled list of mention/box pair IDs, such that
    the images appear in a random order, and mention/box
    pairs _within_ the images are in random order, but the
    mention/box pairs are grouped with images

    :param mention_box_pairs:
    :return:
    """
    image_mention_box_pairs = dict()
    for mb_pair in mention_box_pairs:
        # Mention / box IDs are
        #   <mention_id> | <box_id>
        # and mention IDs are, themselves,
        #   <img_id>#<cap_idx>;mention:<mention_idx>
        img_id = mb_pair.split("#")[0]
        if img_id not in image_mention_box_pairs:
            image_mention_box_pairs[img_id] = list()
        image_mention_box_pairs[img_id].append(mb_pair)
    #endfor

    img_ids = image_mention_box_pairs.keys()
    np.random.shuffle(img_ids)
    mention_box_pairs_shuffled = list()
    for img_id in img_ids:
        mb_pairs = image_mention_box_pairs[img_id]
        np.random.shuffle(mb_pairs)
        for mb_pair in mb_pairs:
            mention_box_pairs_shuffled.append(mb_pair)
    #endfor
    return mention_box_pairs_shuffled
#endfor


def get_valid_mention_box_pairs(data_dict):
    """
    Retrieves a list of valid mention/box pairs, which
    effectively takes the intersection of all mention/box
    pairs with the loaded mentions; this is necessary for
    those rare mentions that are empty when punctuation is removed
    :param data_dict: Data dictionary
    :return: List of mention/box pair IDs
    """
    mention_box_pairs = list()
    loaded_mentions = set(data_dict['mention_indices'].keys())
    for mb_pair in data_dict['labels'].keys():
        if mb_pair.split("|")[0] in loaded_mentions:
            mention_box_pairs.append(mb_pair)
    return mention_box_pairs
#enddef


def train(encoding_scheme, embedding_type,
          sentence_file, mention_idx_file, feature_file,
          feature_meta_file, box_dir, mention_box_label_file,
          epochs, batch_size, lstm_hidden_width,
          start_hidden_width, hidden_depth, weighted_classes,
          input_dropout, other_dropout, lrn_rate,
          adam_epsilon, clip_norm, data_norm,
          activation, model_file=None,
          eval_sentence_file=None, eval_mention_idx_file=None,
          eval_feature_file=None, eval_feature_meta_file=None,
          eval_box_dir=None, eval_mention_box_label_file=None,
          early_stopping=False, box_category_file=None,
          eval_box_category_file=None, log=None):
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
    global CLASSES, N_EMBEDDING_WIDTH, N_BOX_WIDTH, task
    n_classes = len(CLASSES)

    log.tic('info', "Loading training data")
    data_dict = nn_data.load_sentences(sentence_file, embedding_type)
    data_dict.update(nn_data.load_mentions(mention_idx_file, task,
                                           feature_file, feature_meta_file,
                                           n_classes))
    data_dict.update(nn_data.load_boxes(mention_box_label_file, box_dir, box_category_file))
    log.toc('info')

    log.tic('info', "Loading eval data")
    eval_data_dict = nn_data.load_sentences(eval_sentence_file, embedding_type)
    eval_data_dict.update(nn_data.load_mentions(eval_mention_idx_file, task,
                                                eval_feature_file, eval_feature_meta_file,
                                                n_classes))
    eval_data_dict.update(nn_data.load_boxes(eval_mention_box_label_file, eval_box_dir,
                                             eval_box_category_file))
    log.toc('info')

    # Load mention box pairs, ignoring all those mentions
    # that aren't in our loaded data (for punctuation reasons)
    mention_box_pairs = get_valid_mention_box_pairs(data_dict)
    n_pairs = len(mention_box_pairs)

    log.info("Setting up network architecture")
    with tf.variable_scope('bidirectional_lstm'):
        nn_util.setup_bidirectional_lstm(lstm_hidden_width, data_norm)
    nn_util.setup_core_architecture(task, encoding_scheme,
                                    batch_size, start_hidden_width,
                                    hidden_depth, weighted_classes, activation,
                                    n_classes, data_dict['n_mention_feats'],
                                    data_dict['box_embedding_width'],
                                    data_dict['n_box_feats'])
    loss = tf.get_collection('loss')[0]
    accuracy = tf.get_collection('accuracy')[0]
    nn_util.add_train_op(loss, lrn_rate, adam_epsilon, clip_norm)
    train_op = tf.get_collection('train_op')[0]
    nn_util.dump_tf_vars()

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
            mention_box_pairs = shuffle_mention_box_pairs(mention_box_pairs)

            # Iterate through the entirety of the data
            start_idx = 0
            end_idx = start_idx + batch_size
            n_iter = n_pairs / batch_size
            for j in range(0, n_iter):
                log.log_status('info', None, 'Training; %d (%.2f%%) batches complete',
                               j, 100.0 * j / n_iter)

                # Retrieve this batch
                batch_mention_box_pairs = mention_box_pairs[start_idx:end_idx]
                batch_tensors = nn_data.load_batch(batch_mention_box_pairs,
                                                   data_dict, task, n_classes)

                # Train
                nn_util.run_op(sess, train_op, [batch_tensors],
                               input_dropout, other_dropout,
                               encoding_scheme, [task], [""], True)

                # Store the losses and accuracies every 100 batches
                if (j+1) % 100 == 0:
                    losses.append(nn_util.run_op(sess, loss, [batch_tensors],
                                                 input_dropout, other_dropout,
                                                 encoding_scheme, [task], [""], True))
                    accuracies.append(nn_util.run_op(sess, accuracy,[batch_tensors],
                                                     input_dropout, other_dropout,
                                                     encoding_scheme, [task], [""], True))
                #endif
                start_idx = end_idx
                end_idx = start_idx + batch_size
            #endfor

            # Every epoch, evaluate and save the model
            if len(losses) > 0:
                log.info(None, "Saving model; Average Loss: %.2f; Acc: %.2f%%",
                         sum(losses) / float(len(losses)),
                         100.0 * sum(accuracies) / float(len(accuracies)))
            saver.save(sess, model_file)
            if (i+1) % 10 == 0 and eval_sentence_file is not None and eval_mention_idx_file is not None:
                eval_mention_box_pairs = \
                    get_valid_mention_box_pairs(eval_data_dict)
                pred_scores, gold_label_dict = \
                    nn_util.get_pred_scores_mcc(task, encoding_scheme, sess,
                                                batch_size, eval_mention_box_pairs,
                                                eval_data_dict, n_classes, log)

                # If we do an argmax on the scores, we get the predicted labels
                eval_mentions = list(pred_scores.keys())
                pred_labels = list()
                gold_labels = list()
                for m in eval_mentions:
                    pred_labels.append(np.argmax(pred_scores[m]))
                    gold_labels.append(np.argmax(gold_label_dict[m]))
                #endfor

                # Evaluate the predictions
                score_dict = \
                    nn_eval.evaluate_multiclass(gold_labels, pred_labels,
                                                CLASSES, log)

                # Get the current coref / subset and see if their average beats our best or
                # within one point of our best (we keep stopping too early)
                avg = score_dict.get_score(0).f1 + score_dict.get_score(1).f1
                avg /= 2.0
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


def predict(encoding_scheme, embedding_type,
            tf_session, batch_size, sentence_file,
            mention_idx_file, feature_file,
            feature_meta_file, box_dir, mention_box_label_file,
            box_category_file=None, scores_file=None, log=None):
    """

    :param encoding_scheme:
    :param embedding_type:
    :param tf_session:
    :param batch_size:
    :param sentence_file:
    :param mention_idx_file:
    :param feature_file:
    :param feature_meta_file:
    :param box_dir:
    :param mention_box_label_file:
    :param scores_file:
    :param log:
    :return:
    """
    global CLASSES, task
    n_classes = len(CLASSES)

    # Load the data
    log.info("Loading data")
    data_dict = nn_data.load_sentences(sentence_file, embedding_type)
    data_dict.update(nn_data.load_mentions(mention_idx_file, task,
                                           feature_file, feature_meta_file,
                                           n_classes))
    data_dict.update(nn_data.load_boxes(mention_box_label_file,
                                        box_dir, box_category_file))

    # Get the predicted scores, given our arguments
    mention_box_pairs = get_valid_mention_box_pairs(data_dict)
    pred_scores, gold_label_dict = \
        nn_util.get_pred_scores_mcc(task, encoding_scheme, tf_session,
                                    batch_size, mention_box_pairs,
                                    data_dict, n_classes, log)

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
    parser = ArgumentParser("ImageCaptionLearn_py: Neural Network for Nonvisual "
                            "Prediction; Bidirectional LSTM to hidden layer "
                            "to softmax over (v)isual and (n)onvisual labels")
    parser.add_argument("--epochs", type=int, default=20,
                        help="train opt; number of times to iterate over the dataset")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="train opt; number of random mention pairs per batch")
    parser.add_argument("--lstm_hidden_width", type=int, default=200,
                        help="train opt; number of hidden units within "
                             "the LSTM cells")
    parser.add_argument("--start_hidden_width", type=int, default=512,
                        help="train opt; number of hidden units in the "
                             "layer after the LSTM")
    parser.add_argument("--hidden_depth", type=int, default=2,
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
    parser.add_argument("--lstm_input_dropout", type=float, default=0.5,
                        help="train opt; probability to keep lstm input nodes")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="train opt; probability to keep all other nodes")
    parser.add_argument("--data_dir", required=True,
                        type=lambda f: util.arg_path_exists(parser, f),
                        help="Directory containing raw/, feats/, and scores/ directories")
    parser.add_argument("--data", choices=["flickr30k", "mscoco", "coco30k", "flickr30k_v1"],
                        required=True, help="Dataset to use")
    parser.add_argument("--split", choices=["train", "dev", "test", "trainDev"],
                        required=True, help="Dataset split")
    parser.add_argument("--eval_data", choices=["flickr30k", "mscoco", "coco30k"],
                        help="Evaluation dataset to use")
    parser.add_argument("--eval_split", choices=["train", "dev", "test", "trainDev"],
                        help="Evaluation dataset split")
    parser.add_argument("--encoding_scheme",
                        choices=["first_last_sentence", 'first_last_mention'],
                        default="first_last_mention",
                        help="train opt; specifies how lstm outputs are transformed")
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
                        help="Whether to implement early stopping based on the "
                             "evaluation performance")
    args = parser.parse_args()
    arg_dict = vars(args)

    if arg_dict['train'] and arg_dict['model_file'] is None:
        arg_dict['model_file'] = "/home/ccervan2/models/tacl201801//" + \
                                 nn_data.build_model_filename(arg_dict, "affinity_lstm")
    model_file = arg_dict['model_file']
    util.dump_args(arg_dict, log)

    # Construct data files from the root directory and filename
    data_dir = arg_dict['data_dir'] + "/"
    data_root = arg_dict['data'] + "_" + arg_dict['split']
    eval_data_root = None
    if arg_dict['train']:
        eval_data_root = arg_dict['eval_data'] + "_" + arg_dict['eval_split']
    sentence_file = data_dir + "raw/" + data_root + "_captions.txt"
    mention_idx_file = data_dir + "raw/" + data_root + "_mentions_affinity.txt"
    feature_file = data_dir + "feats/" + data_root + "_affinity_neural.feats"
    feature_meta_file = data_dir + "feats/" + data_root + "_affinity_neural_meta.json"
    box_dir = data_dir + "feats/" + arg_dict['data'] + "_boxes/" + arg_dict["split"] + "/"
    mention_box_label_file = data_dir + "raw/" + data_root + "_affinity_labels.txt"
    box_category_file = None
    #if "coco" in data_root:
    #    box_category_file = data_dir + "raw/" + data_root + "_box_cats.txt"
    if eval_data_root is not None:
        eval_box_dir = data_dir + "feats/" + arg_dict['eval_data'] + "_boxes/" + arg_dict["eval_split"] + "/"
        eval_sentence_file = data_dir + "raw/" + eval_data_root + "_captions.txt"
        eval_mention_idx_file = data_dir + "raw/" + eval_data_root + "_mentions_affinity.txt"
        eval_feature_file = data_dir + "feats/" + eval_data_root + "_affinity_neural.feats"
        eval_feature_meta_file = data_dir + "feats/" + eval_data_root + "_affinity_neural_meta.json"
        eval_mention_box_label_file = data_dir + "raw/" + eval_data_root + "_affinity_labels.txt"
        eval_box_category_file = None
    #    if "coco" in eval_data_root:
    #        eval_box_category_file = data_dir + "raw/" + eval_data_root + "_box_cats.txt"
    #endif

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
        train(encoding_scheme=arg_dict['encoding_scheme'],
              embedding_type=embedding_type,
              sentence_file=sentence_file,
              mention_idx_file=mention_idx_file,
              feature_file=feature_file,
              feature_meta_file=feature_meta_file, box_dir=box_dir,
              mention_box_label_file=mention_box_label_file,
              epochs=arg_dict['epochs'], batch_size=arg_dict['batch_size'],
              lstm_hidden_width=arg_dict['lstm_hidden_width'],
              start_hidden_width=arg_dict['start_hidden_width'],
              hidden_depth=arg_dict['hidden_depth'],
              weighted_classes=arg_dict['weighted_classes'],
              input_dropout=arg_dict['lstm_input_dropout'],
              other_dropout=arg_dict['dropout'],
              lrn_rate=arg_dict['learn_rate'], clip_norm=arg_dict['clip_norm'],
              data_norm=arg_dict['data_norm'],
              adam_epsilon=arg_dict['adam_epsilon'], activation=arg_dict['activation'],
              model_file=model_file, eval_sentence_file=eval_sentence_file,
              eval_mention_idx_file=eval_mention_idx_file,
              eval_feature_file=eval_feature_file,
              eval_feature_meta_file=eval_feature_meta_file,
              eval_box_dir=eval_box_dir,
              eval_mention_box_label_file=eval_mention_box_label_file,
              early_stopping=arg_dict['early_stopping'],
              box_category_file=box_category_file,
              eval_box_category_file=eval_box_category_file,
              log=log)
    elif arg_dict['predict']:
        scores_file = data_dir + "scores/" + data_root + "_affinity.scores"

        # Restore our variables
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_file + ".meta")
            saver.restore(sess, model_file)

            predict(encoding_scheme=arg_dict['encoding_scheme'],
                    embedding_type=embedding_type,
                    tf_session=sess,
                    batch_size=arg_dict['batch_size'],
                    sentence_file=sentence_file,
                    mention_idx_file=mention_idx_file,
                    feature_file=feature_file,
                    feature_meta_file=feature_meta_file,
                    box_dir=box_dir,
                    mention_box_label_file=mention_box_label_file,
                    scores_file=scores_file,
                    box_category_file=box_category_file,
                    log=log)
        #endwith
    #endif
#enddef


__init__()

