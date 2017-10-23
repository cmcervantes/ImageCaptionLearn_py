import linecache
import math
import random as r
from argparse import ArgumentParser
from os.path import abspath, expanduser

import numpy as np
import tensorflow as tf

from utils.LogUtil import LogUtil
from utils.ScoreDict import ScoreDict

NUM_FEATS = 4396
NUM_IMG_FEATS = 4096
NUM_TXT_FEATS = 300
#CLASS_DISTRO = [0.49, 0.51]   #actual distro: 87 / 13
CLASS_DISTRO = [0.87, 0.13]
CLASSES = [np.array((1, 0)), np.array((0, 1))]
LEXICAL_TYPES = ["people", "other", "scene", "clothing",
                 "bodyparts", "animals", "vehicles",
                 "instruments", "colors"]

"""
Loads a batch of vectors from a single file, where image and language features are split
"""
def load_split_batch(filename, batch_indices):
    global log, CLASS_DISTRO, CLASSES, NUM_IMG_FEATS

    # for each example, read a numpy array
    x_tensor_img = list(); x_tensor_txt = list()
    y_tensor = list(); example_ids = list()
    indices = list(); example_idx = 0
    for i in range(len(CLASSES)):
        indices.append(list())

    for idx in batch_indices:
        line = linecache.getline(filename, idx+1) # linecache indices start at 1
        line_arr = line.split(",")
        if len(line_arr) < 3:
            log.warning('Found missing vector at idx: ' + str(idx) + "; line: " + line)
            continue
        example_ids.append(line_arr[0])
        y = int(line_arr[1])
        y_tensor.append(CLASSES[y])
        x = np.array(line_arr[2:])
        x_tensor_img.append(x[0:NUM_IMG_FEATS])
        x_tensor_txt.append(x[NUM_IMG_FEATS:])
        indices[y].append(example_idx)
        example_idx += 1
    #endfor

    return example_ids, x_tensor_img, x_tensor_txt, y_tensor
#enddef

"""
Initializes a tf weight variable, given a shape
"""
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,
       stddev=1.0 /math.sqrt(float(shape[0]))), name='weights')
#enddef

"""
Initializes a tf bias variable, given a shape
"""
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape=shape))
#enddef

"""
Returns a ScoreDict containing the evaluation of eval_files
on the network represented by the session and tensorflow vals

    for label in scores.keys:
        log.info(scores.getScore(label).toString() +
                 " %d (%.2f%%)" % (scores.getGoldCount(label), scores.getGoldPercent(label)))
    #endfor
"""
def evaluate_model(sess, x_img, x_txt, y_, retain_prob, eval_file, eval_indices, eval_folds):
    scores = ScoreDict()
    batch_size = len(eval_indices) / eval_folds
    for i in range(eval_folds):
        if (i+1) % 25 == 0:
            log.debug('eval_batch: ' + str(i+1))

        batch = eval_indices[i*batch_size:(i+1)*batch_size]
        eval_ids, eval_x_img, eval_x_txt, eval_y = load_split_batch(eval_file, batch)

        if len(eval_ids) > 0:
            predictions = sess.run(y_, feed_dict={x_img: eval_x_img, x_txt: eval_x_txt, retain_prob: 1.0})
            for j in range(len(predictions)):
                gold = np.argmax(np.array(eval_y[j]))
                pred = np.argmax(np.array(predictions[j]))
                scores.increment(gold, pred)
            #endfor
        #endif
    #endfor
    return scores
#enddef

"""
Trains a model with the following architecture
[box_feats] --> [300 node layer] --|
                                   |--> [hidden_joint_width] --> [softmax] --> y
                 [mention feats] --|
"""
def train(train_file, eval_file, eval_type_file, learning_rate,
          model_file, epochs, batch_size, balance=False, dropout=False):
    global log, CLASSES, NUM_IMG_FEATS, NUM_TXT_FEATS

    # Setup tf vars
    y = tf.placeholder(tf.float32, [None, len(CLASSES)])
    x_img = tf.placeholder(tf.float32, [None, NUM_IMG_FEATS])
    x_txt = tf.placeholder(tf.float32, [None, NUM_TXT_FEATS])
    with tf.name_scope('hidden_img'):
        weights = weight_variable([NUM_IMG_FEATS, NUM_TXT_FEATS])
        biases = bias_variable([NUM_TXT_FEATS])
        hidden_img = tf.nn.relu(tf.matmul(x_img, weights) + biases)
    with tf.name_scope('hidden_joint_1'):
        weights = weight_variable([2 * NUM_TXT_FEATS, NUM_TXT_FEATS])
        biases = bias_variable([NUM_TXT_FEATS])
        hidden_joint_1 = tf.nn.relu(tf.matmul(tf.concat(1, [x_txt, hidden_img]), weights) + biases)
        retain_prob = tf.placeholder(tf.float32)
        hdn_joint_drop = tf.nn.dropout(hidden_joint_1, retain_prob)
    with tf.name_scope('softmax'):
        weights = weight_variable([NUM_TXT_FEATS, len(CLASSES)])
        biases = bias_variable([len(CLASSES)])
        logits = tf.add(tf.matmul(hdn_joint_drop, weights), biases)
        y_ = tf.nn.softmax(logits)
    #endwith

    # Either use straight softmax cross entropy with logits; or multiply
    # by our inverse class distro
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
    if balance:
        cw = list()
        cw[:] = [1-w for w in CLASS_DISTRO]
        class_weights = tf.constant(cw)
        weighted_logits = tf.mul(logits, class_weights) # shape [batch_size, 2]
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(weighted_logits, y))
    #endif

    # Using cross-entropy loss and gradient descent
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y *
    #     tf.log(tf.clip_by_value(y_, 1e-20, 1.0)), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    #initialize our variables and run the application
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    # Read the type lists (ignoring IDs, since the file uses the same
    # indexation as train_file / eval file
    # UPDATE: since train has no IDs, the train type file _is_ the train file,
    #         with types in place of IDs
    train_types = list()
    with open(train_file, 'r') as f:
        for line in f:
            train_types.append(line.split(",")[0])
        f.close()
    eval_types = list()
    with open(eval_type_file, 'r') as f:
        for line in f:
            eval_types.append(line.split(",")[1])
        f.close()
    #endwith

    # Train and evaluate a separate net for each lexical type
    type_score_dict = dict()
    for lex_type in LEXICAL_TYPES:
        linecache.clearcache()
        typed_model_file = model_file.replace(".model", "_" + lex_type + ".model")

        # Get the possible training indices for this type
        train_indices = list()
        for i in range(0, len(train_types)):
            if lex_type in train_types[i]:
                train_indices.append(i)
        #endfor

        # Train for the given number of epochs
        for i in range(epochs):
            # load this epoch's data, randomly selecting
            # batch_size indices from this type's training
            # indices
            rand_indices = r.sample(train_indices, batch_size)
            example_ids, batch_x_img, batch_x_txt, batch_y, = \
                load_split_batch(train_file, rand_indices)

            # Train on this batch, incorporating dropout if specified
            dropout_prob = 1.0
            if dropout:
                dropout_prob = 0.5

            if (i+1) % 1000 == 0:
                log.tic('info', "Type %s; Epoch %d; training" % (lex_type, i+1))
            sess.run(train_step, feed_dict={x_img: batch_x_img, x_txt: batch_x_txt,
                                            y: batch_y, retain_prob: dropout_prob})
            if (i+1) % 1000 == 0:
                log.toc('info')
                saver.save(sess, typed_model_file)
            #endif
        #endfor
        # Save this model at the end
        saver.save(sess, typed_model_file)
        log.info(None, "Model saved to: %s", typed_model_file)

        # Evaluate this type's net
        log.info('Evaluating ' + lex_type)
        eval_indices = list()
        for j in range(0, len(eval_types)):
            if lex_type in eval_types[j]:
                eval_indices.append(j)
        #endfor
        linecache.clearcache()
        type_score_dict[lex_type] = \
            evaluate_model(sess, x_img, x_txt, y_, retain_prob, eval_file, eval_indices, 100)
    #endfor

    overall_scores = ScoreDict()
    for lex_type in type_score_dict.keys():
        log.info('---' + lex_type + '---')
        scores = type_score_dict[lex_type]
        overall_scores.merge(scores)
        log.info('---Confusion Matrix---')
        scores.print_confusion()

        log.info("---Scores---")
        for label in scores.keys:
            print str(label) + "\t" + scores.get_score(label).toString() + \
                  " - %d (%.2f%%)" % (scores.get_gold_count(label), scores.get_gold_percent(label))
            print str(label) + "\t Acc: " + str(scores.get_accuracy(label))
        #endfor
    #endfor

    log.info('Overall Scores')
    log.info('---Confusion Matrix---')
    overall_scores.print_confusion()

    log.info("---Scores---")
    for label in overall_scores.keys:
        print str(label) + "\t" + overall_scores.get_score(label).toString() + " - %d (%.2f%%)" % \
                                                                               (overall_scores.get_gold_count(label), overall_scores.get_gold_percent(label))
        print str(label) + "\t Acc: %.2f%%" % (overall_scores.get_accuracy(label))
        #endfor
    print "Total Acc: %.2f%%" % (overall_scores.get_accuracy())
#enddef


def tune(train_file, eval_file, eval_type_file):
    lrn_rates = [0.1, 0.05, 0.01]
    batch_sizes = [50, 100, 200]
    for lrn in lrn_rates:
        for btch in batch_sizes:
            for balance in [True, False]:
                for dropout in [True, False]:
                    log.info(None, "#############learning_rate:%.3f; batch_size:%d; balance:%s; dropout:%s###############",
                             lrn, btch, str(balance), str(dropout))
                    train(train_file, eval_file, eval_type_file, lrn,
                          "affinity.model", 100, btch, balance, dropout)
                #endfor
            #endfor
        #endfor
    #endfor
#enddef

def init():
    global log, NUM_TXT_FEATS

    #parse args
    parser = ArgumentParser("ImageCaptionLearn_py: Box/Mention Affinity Classifier")
    parser.add_argument('--log_lvl', choices=LogUtil.get_logging_lvls(), default='debug',
                        help="log opt; Default logger level")
    parser.add_argument("--log_delay", type=int, default=30,
                        help="log opt; log status delay")
    parser.add_argument("--train", action='store_true', help="Trains a model; saves in --model")
    #parser.add_argument("--eval", action='store_true', help="Evaluates a model; loads from --model")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="train opt; Sets learning rate")
    parser.add_argument("--batch_size", type=int, default=100, help="train opt; Number of imgs in each batch")
    parser.add_argument("--balance", action='store_true',
                        help="train opt; Whether to load data according to true distro")
    parser.add_argument("--dropout", action='store_true', help='train opt; Use dropout @ 50%%')
    parser.add_argument("--epochs", type=int, default=100, help="train opt; Number of batches to iterate over")
    parser.add_argument('--tune', action='store_true',
                        help='Tunes the model over various learning rates / batch sizes / etc.')
    args = parser.parse_args()
    arg_dict = vars(args)

    # Set up up the logger
    log_lvl = arg_dict['log_lvl']
    log_delay = arg_dict['log_delay']
    log = LogUtil(lvl=log_lvl, delay=log_delay)

    # Hard code the files
    train_file = abspath(expanduser("~/source/data/feats/affinity_feats_train.feats"))
    eval_file = abspath(expanduser("~/source/data/feats/affinity_feats_dev.feats"))
    eval_type_file = eval_file.replace(".feats", "_type.csv")

    if arg_dict['train']:
        train(train_file, eval_file, eval_type_file,
              arg_dict['learning_rate'], 'affinity.model',
              arg_dict['epochs'], arg_dict['batch_size'],
              arg_dict['balance'], arg_dict['dropout'])
    elif arg_dict['tune']:
        tune(train_file, eval_file, eval_type_file)
    #endif
#enddef

init()



"""
-------0.01; 1000; 200; False; False-------
10:35:07 (INFO): ---Confusion Matrix---
   | 0              1
0  | 185552 (48.1%) 33554 (47.1%)
1  | 200195 (51.9%) 37699 (52.9%)
10:35:07 (INFO): ---Scores---
0	P:  84.69% | R:  48.10% | F1:  61.35% - 385747 (84.41%)
1	P:  15.85% | R:  52.91% | F1:  24.39% - 71253 (15.59%)

-------0.01; 1000; 200; False; True-------
11:06:38 (INFO): ---Confusion Matrix---
   | 0              1
0  | 148278 (38.4%) 23775 (33.4%)
1  | 237469 (61.6%) 47478 (66.6%)
11:06:38 (INFO): ---Scores---
0	P:  86.18% | R:  38.44% | F1:  53.17% - 385747 (84.41%)
1	P:  16.66% | R:  66.63% | F1:  26.66% - 71253 (15.59%)

-------0.01; 1000; 200; True; True-------
11:37:00 (INFO): ---Confusion Matrix---
   | 0              1
0  | 90366 (23.4%)  19106 (26.8%)
1  | 295381 (76.6%) 52147 (73.2%)
11:37:00 (INFO): ---Scores---
0	P:  82.55% | R:  23.43% | F1:  36.50% - 385747 (84.41%)
1	P:  15.01% | R:  73.19% | F1:  24.90% - 71253 (15.59%)

-------0.1; 1000; 200; False; True-------
   | 0              1
0  | 206096 (53.4%) 34773 (48.8%)
1  | 179651 (46.6%) 36480 (51.2%)
12:03:46 (INFO): ---Scores---
0	P:  85.56% | R:  53.43% | F1:  65.78% - 385747 (84.41%)
1	P:  16.88% | R:  51.20% | F1:  25.39% - 71253 (15.59%)


"""