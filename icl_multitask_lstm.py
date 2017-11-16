import tensorflow as tf
import numpy as np
import os
from os.path import expanduser, abspath
from nn_utils import core as nn_util
from nn_utils import data as nn_data
from nn_utils import eval as nn_eval
from utils import core as util
from utils.Logger import Logger
from argparse import ArgumentParser


___author___ = "ccervantes"


TASK_CLASS_DICT = {'rel_intra': ['n', 'c', 'b', 'p'],
                   'rel_cross': ['n', 'c', 'b', 'p'],
                   'nonvis': ['v', 'n'],
                   'affinity': ['0', '1'],
                   'card': ['0', '1', '2', '3', '4', '5',
                            '6', '7', '8', '9', '10', '11+']}
TASKS = ['rel_intra', 'rel_cross', 'nonvis', 'affinity', 'card']
N_EMBEDDING_WIDTH = 300
N_BOX_WIDTH = 4096


def setup(multitask_scheme, lstm_hidden_width, data_norm, start_hidden_width,
          hidden_depth, weighted_classes, activation, encoding_scheme,
          task_data_dicts, batch_size=None, task_batch_sizes=None):
    """
    Sets up all of the networks using a shared lstm

    :param multitask_scheme:
    :param lstm_hidden_width:
    :param data_norm:
    :param start_hidden_width:
    :param hidden_depth:
    :param weighted_classes:
    :param activation:
    :param encoding_scheme:
    :param task_data_dicts:
    :param batch_size:
    :param task_batch_size:
    :return:
    """
    global TASK_CLASS_DICT, TASKS

    # Set up the shared bidirectional LSTM
    with tf.variable_scope('bidirectional_lstm'):
        nn_util.setup_bidirectional_lstm(lstm_hidden_width, data_norm)

    # Set up each tasks' network
    task_vars = dict()
    for task in TASKS:
        n_classes = len(TASK_CLASS_DICT[task])

        # Set up the core architecture for each task
        with tf.variable_scope(task):
            # Retrieve the number of mention and box features
            n_mention_feats = task_data_dicts[task]['n_mention_feats']
            n_box_feats = None
            if 'n_box_feats' in task_data_dicts[task].keys():
                n_box_feats = task_data_dicts[task]['n_box_feats']

            # We either have a constant batch size or varying batch
            # sizes, based on task
            task_batch_size = batch_size
            if multitask_scheme == 'alternate':
                task_batch_size = task_batch_sizes[task]

            # Set up the core architecture and add the loss and accuracy
            # operations to the task vars
            nn_util.setup_core_architecture(task, encoding_scheme,
                                            task_batch_size, start_hidden_width,
                                            hidden_depth, weighted_classes,
                                            activation, n_classes,
                                            n_mention_feats, n_box_feats)
            task_vars[task] = dict()
            task_vars[task]['loss'] = tf.get_collection(task + '/loss')[0]
            task_vars[task]['accuracy'] = tf.get_collection(task + '/accuracy')[0]
        #endwith
    #endfor

    # Dump all the tensorflow vars
    nn_util.dump_tf_vars()

    return task_vars
#enddef


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


def load_data(data_dir, data, split, embedding_type, log=None):
    """
    Loads all of the data for all tasks
    :param data_dir:
    :param data:
    :param split:
    :param embedding_type:
    :param log:
    :return:
    """
    global TASK_CLASS_DICT, N_EMBEDDING_WIDTH

    task_data_dicts = dict()

    data_root = data + "_" + split
    for task in TASKS:
        # Retrieve the input files for these tasks
        sentence_file = data_dir + "raw/" + data_root + "_captions.txt"
        mention_idx_file = data_dir + "raw/" + data_root + "_"
        feature_file = data_dir + "feats/" + data_root + "_"
        feature_meta_file = data_dir + "feats/" + data_root + "_"
        label_file = None

        if 'rel' in task:
            # Relation files are a little weird,
            # since we have some mixed intra/cross files
            mention_idx_file += "mentionPairs_" + task.split("_")[1]
            feature_file += "relation"
            feature_meta_file += "relation"
            label_file = data_dir + "raw/" + data_root + "_mentionPair_labels.txt"
        else:
            mention_idx_file += "mentions_" + task
            feature_file += task
            feature_meta_file += task
            if task == 'affinity':
                label_file = data_dir + "raw/" + data_root + "_mention_box_labels.txt"
        #endif
        mention_idx_file += ".txt"
        feature_file += ".feats"
        feature_meta_file += "_meta.json"

        log.info("Loading data for " + task)
        task_data_dicts[task] = nn_data.load_sentences(sentence_file, embedding_type)
        task_data_dicts[task].update(nn_data.load_mentions(mention_idx_file, task,
                                                           feature_file, feature_meta_file,
                                                           len(TASK_CLASS_DICT[task])))
        if "rel" in task:
            task_data_dicts[task]['gold_label_dict'] = \
                nn_data.load_relation_labels(label_file)
        elif task == "affinity":
            box_dir = data_dir + "feats/" + data + "_boxes/" + split + "/"
            task_data_dicts[task].update(nn_data.load_boxes(box_dir, label_file))
        #endif
    #endfor
    return task_data_dicts
#enddef


def train_jointly(multitask_scheme, epochs, batch_size, lstm_input_dropout,
                  dropout, lrn_rate, adam_epsilon, clip_norm, encoding_scheme,
                  task_vars, task_data_dicts, eval_task_data_dicts,
                  task_ids, eval_task_ids, model_file, log=None):
    """
    Trains a joint model, either using a simple sum of losses
    or with weights over losses

    :param multitask_scheme:
    :param epochs:
    :param batch_size:
    :param lstm_input_dropout:
    :param dropout:
    :param lrn_rate:
    :param adam_epsilon:
    :param clip_norm:
    :param encoding_scheme:
    :param task_vars:
    :param task_data_dicts:
    :param eval_task_data_dicts:
    :param task_ids:
    :param eval_task_ids:
    :param model_file:
    :param log:
    :return:
    """
    global TASKS

    # We either have a simple sum-of-losses model or we're learning
    # weights over those losses
    if multitask_scheme == "simple_joint":
        joint_loss = task_vars['rel_intra']['loss'] + task_vars['rel_cross']['loss'] + \
                     task_vars['nonvis']['loss'] + task_vars['affinity']['loss'] + \
                     task_vars['card']['loss']
        nn_util.add_train_op(joint_loss, lrn_rate, adam_epsilon, clip_norm)
    elif multitask_scheme == "weighted_joint":
        # tf.stack allows us to vector-ize scalars, and since
        # the ffw function assumes a kind of [batch_size, units]
        # shape, we expand the first dimension to 1
        losses = tf.expand_dims(tf.stack([task_vars['rel_intra']['loss'],
                                          task_vars['rel_cross']['loss'],
                                          task_vars['nonvis']['loss'],
                                          task_vars['affinity']['loss'],
                                          task_vars['card']['loss']]), 0)
        joint_loss = tf.reduce_sum(nn_util.setup_ffw(losses, [5]))
        nn_util.add_train_op(joint_loss, lrn_rate, adam_epsilon, clip_norm)
    #endif
    train_op = tf.get_collection('train_op')[0]

    # TODO: Implement early stopping under this framework
    log.info("Training")
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        # Initialize all our variables
        sess.run(tf.global_variables_initializer())

        # Iterate through the data [epochs] number of times
        for i in range(0, epochs):
            log.info(None, "--- Epoch %d ----", i+1)

            # Shuffle everyone's IDs for this epoch
            max_samples = 0
            sample_indices = dict()
            for task in TASKS:
                np.random.shuffle(task_ids[task])
                max_samples = max(max_samples, len(task_ids[task]))
                sample_indices[task] = 0
            #endfor

            # We iterate until we've seen _every_ tasks' samples
            # at least once
            n_iter = max_samples / batch_size
            for j in range(0, n_iter):
                log.log_status('info', None, "Completed %d (%.2f%%) iterations",
                               j, 100.0 * j / n_iter)
                # For each task, get the next [batch_size] samples,
                # wrapping around to the beginning of the list if
                # necessary
                batch_tensor_dicts = list()
                for task in TASKS:
                    ids = task_ids[task]
                    n_ids = len(ids)
                    start_idx = sample_indices[task]
                    if start_idx + batch_size < n_ids:
                        batch_ids = ids[start_idx:start_idx+batch_size]
                        sample_indices[task] += batch_size
                    else:
                        remainder = start_idx + batch_size - n_ids + 1
                        batch_ids = ids[start_idx:n_ids-1]
                        batch_ids.extend(ids[0:remainder])
                        sample_indices[task] = remainder
                    #endif
                    batch_tensor_dicts.append(nn_data.load_batch(batch_ids,
                                                                 task_data_dicts[task],
                                                                 task, len(TASK_CLASS_DICT[task]),
                                                                 N_EMBEDDING_WIDTH))
                #endfor

                # It so happens that I'm using task names as variable
                # namespaces, which is why I'm passing them twice in the
                # operations, below
                nn_util.run_op(sess, train_op, batch_tensor_dicts, lstm_input_dropout,
                               dropout, encoding_scheme, TASKS, TASKS, True)
            #endfor

            # Every epoch, evaluate and save the model
            log.info(None, "Saving model")
            saver.save(sess, model_file)

            for task in TASKS:
                eval_ids = eval_task_ids[task]
                with tf.variable_scope(task):
                    pred_scores, gold_label_dict = \
                        nn_util.get_pred_scores_mcc(task, encoding_scheme, sess,
                                                    batch_size, eval_ids,
                                                    eval_task_data_dicts[task],
                                                    len(TASK_CLASS_DICT[task]),
                                                    N_EMBEDDING_WIDTH, log)

                # If we do an argmax on the scores, we get the predicted labels
                pred_labels = list()
                gold_labels = list()
                for m in eval_ids:
                    pred_labels.append(np.argmax(pred_scores[m]))
                    if 'rel' not in task:
                        gold_labels.append(np.argmax(gold_label_dict[m]))
                #endfor

                # Evaluate the predictions
                if 'rel' in task:
                    score_dict = nn_eval.evaluate_relations(eval_ids, pred_labels,
                                                            task_data_dicts[task]['gold_label_dict'],
                                                            log)
                else:
                    score_dict = nn_eval.evaluate_multiclass(gold_labels,
                                                             pred_labels,
                                                             TASK_CLASS_DICT[task], log)
                #endif
            #endfor
        #endfor

        log.info("Saving final model")
        saver.save(sess, model_file)
    #endwith
#enddef


def train_alternately(epochs, task_batch_sizes, lstm_input_dropout,
                      dropout, lrn_rate, adam_epsilon, clip_norm, encoding_scheme,
                      task_vars, task_data_dicts, eval_task_data_dicts,
                      task_ids, eval_task_ids, model_file, log=None):
    """

    :param epochs:
    :param task_batch_sizes:
    :param lstm_input_dropout:
    :param dropout:
    :param lrn_rate:
    :param adam_epsilon:
    :param clip_norm:
    :param encoding_scheme:
    :param task_vars:
    :param task_data_dicts:
    :param eval_task_data_dicts:
    :param task_ids:
    :param eval_task_ids:
    :param model_file:
    :param log:
    :return:
    """
    global TASKS

    # We're going to retrieve and optimate each loss individually
    train_ops = dict()
    for task in TASKS:
        loss = task_vars[task]['loss']
        with tf.variable_scope(task):
            nn_util.add_train_op(loss, lrn_rate, adam_epsilon, clip_norm)
        train_ops[task] = tf.get_collection(task + '/train_op')[0]
    #endfor


    # TODO: Implement early stopping under this framework
    log.info("Training")
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        # Initialize all our variables
        sess.run(tf.global_variables_initializer())

        # Iterate through the data [epochs] number of times
        for i in range(0, epochs):
            log.info(None, "--- Epoch %d ----", i+1)

            # Create a list of (<task>, <batch_id_arr>)
            # tuples for this epoch
            batch_ids = list()
            for task in TASKS:
                ids = task_ids[task]
                pad_length = task_batch_sizes[task] * (len(ids) /
                             task_batch_sizes[task] + 1) - len(ids)
                id_arr = np.pad(ids, (0, pad_length), 'wrap')
                id_matrix = np.reshape(id_arr, [-1, task_batch_sizes[task]])
                for row_idx in range(0, id_matrix.shape[0]):
                    batch_ids.append((task, id_matrix[row_idx]))
            #endfor

            # Shuffle that list, feeding examples for whatever
            # task and whatever ids we have
            np.random.shuffle(batch_ids)
            for task_ids_tuple in batch_ids:
                task, ids = task_ids_tuple
                batch_tensor = \
                    nn_data.load_batch(ids, task_data_dicts[task],
                                       task, len(TASK_CLASS_DICT[task]),
                                       N_EMBEDDING_WIDTH)

                # Run the operation for this task
                nn_util.run_op(sess, train_ops[task],
                               [batch_tensor], lstm_input_dropout,
                               dropout, encoding_scheme, [task],
                               [task], True)
            #endfor

            # Every epoch, evaluate and save the model
            log.info(None, "Saving model")
            saver.save(sess, model_file)
            for task in TASKS:
                eval_ids = eval_task_ids[task]
                with tf.variable_scope(task):
                    pred_scores, gold_label_dict = \
                        nn_util.get_pred_scores_mcc(task, encoding_scheme, sess,
                                                    task_batch_sizes[task], eval_ids,
                                                    eval_task_data_dicts[task],
                                                    len(TASK_CLASS_DICT[task]),
                                                    N_EMBEDDING_WIDTH, log)

                # If we do an argmax on the scores, we get the predicted labels
                pred_labels = list()
                gold_labels = list()
                for m in eval_ids:
                    pred_labels.append(np.argmax(pred_scores[m]))
                    if 'rel' not in task:
                        gold_labels.append(np.argmax(gold_label_dict[m]))
                #endfor

                # Evaluate the predictions
                if 'rel' in task:
                    score_dict = nn_eval.evaluate_relations(eval_ids, pred_labels,
                                                            task_data_dicts[task]['gold_label_dict'],
                                                            log)
                else:
                    score_dict = nn_eval.evaluate_multiclass(gold_labels,
                                                             pred_labels,
                                                             TASK_CLASS_DICT[task], log)
                #endif
            #endfor
        #endfor

        log.info("Saving final model")
        saver.save(sess, model_file)
    #endwith
#enddef



def __init__():
    """

    :return:
    """
    global TASKS

    # Set up the global logger
    log = Logger('debug', 180)

    # Parse arguments
    parser = ArgumentParser("ImageCaptionLearn_py: Neural Network for "
                            "multitask learning; shared bidirectional "
                            "LSTM to hidden layers to softmax over "
                            "labels")
    parser.add_argument("--epochs", type=int, default=20,
                        help="train opt; number of times to iterate over the dataset")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="train opt; number of random examples per batch")
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
    parser.add_argument("--encoding_scheme",
                        choices=["first_last_sentence", 'first_last_mention'],
                        default="first_last_mention",
                        help="train opt; specifies how lstm outputs are transformed")
    parser.add_argument("--data_dir", required=True,
                        type=lambda f: util.arg_path_exists(parser, f),
                        help="Directory containing raw/, feats/, and scores/ directories")
    parser.add_argument("--data", choices=["flickr30k", "mscoco"], required=True,
                        help="Dataset to use")
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True,
                        help="Dataset split")
    parser.add_argument("--eval_data", choices=["flickr30k", "mscoco"], required=True,
                        help="Evaluation dataset to use")
    parser.add_argument("--eval_split", choices=["train", "dev", "test"], required=True,
                        help="Evaluation dataset split")
    parser.add_argument("--train", action='store_true', help='Trains a model')
    parser.add_argument("--activation", choices=['sigmoid', 'tanh', 'relu', 'leaky_relu'],
                        default='relu',
                        help='train opt; which nonlinear activation function to use')
    parser.add_argument("--predict", action='store_true',
                        help='Predicts using pre-trained model')
    parser.add_argument("--model_file",
                        type=str, help="Model file to save/load")
    parser.add_argument("--embedding_type", choices=['w2v', 'glove'], default='w2v',
                        help="Word embedding type to use")
    parser.add_argument("--multitask_scheme",
                        choices=["simple_joint", "weighted_joint", "alternate"],
                        default="simple_joint",
                        help="Multitask learning scheme")
    args = parser.parse_args()
    arg_dict = vars(args)

    train_model = arg_dict['train']
    predict_scores = arg_dict['predict']
    multitask_scheme = arg_dict['multitask_scheme']

    if train_model:
        arg_dict['model_file'] = "/home/ccervan2/models/tacl201712/" + \
                                 nn_data.build_model_filename(arg_dict,
                                                              "multitask_" +
                                                              multitask_scheme + "_lstm")
    model_file = arg_dict['model_file']
    util.dump_args(arg_dict, log)

    # Initialize the word embeddings
    embedding_type = arg_dict['embedding_type']
    if embedding_type == 'w2v':
        log.info("Initializing word2vec")
        nn_data.init_w2v()
    elif embedding_type == 'glove':
        log.info("Initializing glove")
        nn_data.init_glove()
    #endif

    # Load the data
    task_data_dicts = load_data(arg_dict['data_dir'] + "/", arg_dict['data'],
                                arg_dict['split'], embedding_type, log)
    eval_task_data_dicts = dict()
    if train_model:
        eval_task_data_dicts = load_data(arg_dict['data_dir'] + "/", arg_dict['eval_data'],
                                         arg_dict['eval_split'], embedding_type, log)

    # Set the random seeds identically every run
    nn_util.set_random_seeds()

    # Set up the minimum tensorflow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # DEBUG: mixing ratios
    task_batch_sizes = {'rel_intra': 512, 'rel_cross': 512,
                        'nonvis': 512, 'card': 512,
                        'affinity': 512}

    # Retrieve our sample IDs, by task
    task_ids = dict()
    for task in TASKS:
        if task == 'affinity':
            task_ids[task] = get_valid_mention_box_pairs(task_data_dicts[task])
        else:
            task_ids[task] = list(task_data_dicts[task]['mention_indices'].keys())
    eval_task_ids = dict()
    if train_model:
        for task in TASKS:
            if task == 'affinity':
                eval_task_ids[task] = get_valid_mention_box_pairs(eval_task_data_dicts[task])
            else:
                eval_task_ids[task] = list(eval_task_data_dicts[task]['mention_indices'].keys())
        #endfor
    #endif

    if train_model:
        # Set up the shared network
        task_vars = setup(multitask_scheme, arg_dict['lstm_hidden_width'],
                          arg_dict['data_norm'], arg_dict['start_hidden_width'],
                          arg_dict['hidden_depth'], arg_dict['weighted_classes'],
                          arg_dict['activation'], arg_dict['encoding_scheme'],
                          task_data_dicts, arg_dict['batch_size'], task_batch_sizes)

        # Train the model
        if 'joint' in multitask_scheme:
            train_jointly(multitask_scheme, arg_dict['epochs'], arg_dict['batch_size'],
                          arg_dict['lstm_input_dropout'], arg_dict['dropout'],
                          arg_dict['learn_rate'], arg_dict['adam_epsilon'],
                          arg_dict['clip_norm'], arg_dict['encoding_scheme'],
                          task_vars, task_data_dicts, eval_task_data_dicts,
                          task_ids, eval_task_ids, model_file, log)
        elif multitask_scheme == 'alternate':
            train_alternately(arg_dict['epochs'], task_batch_sizes,
                              arg_dict['lstm_input_dropout'], arg_dict['dropout'],
                              arg_dict['learn_rate'], arg_dict['adam_epsilon'],
                              arg_dict['clip_norm'], arg_dict['encoding_scheme'],
                              task_vars, task_data_dicts, eval_task_data_dicts,
                              task_ids, eval_task_ids, model_file, log)
        #endif
    elif predict_scores:
        pass

#enddef

__init__()