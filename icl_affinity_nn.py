from argparse import ArgumentParser
from os.path import abspath, expanduser, isfile

import numpy as np
import tensorflow as tf
from utils.LogUtil import LogUtil

import nn_utils.core
from utils import icl_util as util, icl_data_util

___author___ = "ccervantes"


def load_affinity_data(txt_feat_file, img_feat_file, label_id_file):
    """
    Reads the text and image (box) feature files, along with
    a label/id file mapping mention|box pair IDs to their affinity
    labels and returns the data organized for easy retrieval

    :param txt_feat_file:   Text feature file (dense .feats format)
    :param img_feat_file:   Bounding box feature file (dense .feats format)
    :param label_id_file:   'Feature' file mapping true labels to mention/box IDs
    :return:                x_txt - numpy matrix of size [n_mentions, n_txt_feats]
                            x_img - numpy matrix of size [n_boxes, n_img_feats]
                            y     - numpy matrix of size [n_mention_box_pairs, n_classes]
                            pair_ids - list of mention/box pair IDs
                            pair_to_txt_idx - dict mapping pair ID list indices to
                                              the relevant index in x_txt
                            pair_to_img_idx - dict mapping pair ID list indices to
                                              the relevant index in x_img
    """
    global log

    # Load the data, but throw away the garbage labels in the feature files
    # TODO: If this doesn't work for efficiency reasons, we'll probably need a different loading scheme (like linecache)
    log.info("Loading text data from " + txt_feat_file)
    x_txt, _, ids_txt = icl_data_util.load_dense_feats(txt_feat_file)
    id_txt_dict = dict()
    for i in range(0, len(ids_txt)):
        id_txt_dict[ids_txt[i]] = i

    log.info("Loading img data from " + img_feat_file)
    x_img, _, ids_img = icl_data_util.load_dense_feats(img_feat_file)
    id_img_dict = dict()
    for i in range(0, len(ids_img)):
        id_img_dict[ids_img[i]] = i

    # Load the label / ID file for the true labels
    log.info("Loading img data from " + label_id_file)
    _, y, pair_ids = icl_data_util.load_dense_feats(label_id_file)

    log.info("Mapping pair indices to individual indices")
    pair_to_txt_idx = list()
    pair_to_img_idx = list()
    for pair_id in pair_ids:
        # Recall that pair IDs are in the form
        # <img_id>#<cap_idx>;mention:<mention_idx>|<img_id>;box:<box_idx>
        pair_split = pair_id.split("|")
        pair_to_txt_idx.append(id_txt_dict[pair_split[0]])
        pair_to_img_idx.append(id_img_dict[pair_split[1]])
    #endfor

    return x_txt, x_img, y, pair_ids, pair_to_txt_idx, pair_to_img_idx
#enddef


def train(txt_feat_file, img_feat_file, label_id_file, batch_size, n_txt_hidden,
          n_img_hidden, n_joint_hidden, dropout_p, lrn_rate, n_iter):
    """
    Trains a simple affinity classifier, where txt and box features feed into
    their own hidden layer before a joint hidden layer

    :param txt_feat_file:
    :param img_feat_file:
    :param label_id_file:
    :param batch_size:
    :param n_txt_hidden:
    :param n_img_hidden:
    :param n_joint_hidden:
    :param dropout_p:
    :param lrn_rate:
    :param n_iter:
    :return:
    """
    global log, model_file

    log.info("Loading data")
    data_txt_x, data_img_x, data_y, pair_ids, pair_to_txt_idx, pair_to_img_idx = \
        load_affinity_data(txt_feat_file, img_feat_file, label_id_file)
    n_samples = data_y.shape[0]
    n_classes = data_y.shape[1]
    n_txt_feats = data_txt_x.shape[1]
    n_img_feats = data_img_x.shape[1]

    log.info("Setting up affinity model")
    with tf.variable_scope("affinity"):
        '''
        Basic architecture feeds our text representation
        to a hidden layer; our image representation to a
        hidden layer; concatenates the outputs of those
        hidden layers and feeds them to a joint hidden layer
            [n_txt_feats] ---> [n_txt_hidden]--|
                                               |
                                               |--> [n_joint_hidden] ---> [n_classes]
                                               |
            [n_img_feats] ---> [n_img_hidden]--|
        '''
        with tf.variable_scope('txt_hdn'):
            x_txt = tf.placeholder(tf.float32, [batch_size, n_txt_feats])
            weights = nn_utils.core.get_weights([n_txt_feats, n_txt_hidden])
            biases = nn_utils.core.get_biases([1, n_txt_hidden])
            hdn_txt_logits = tf.nn.tanh(tf.matmul(x_txt, weights) + biases)
            print "x_txt: " + str(x_txt.get_shape().as_list())
            print "txt_hdn/weights: " + str(weights.get_shape().as_list())
            print "txt_hdn/biases: " + str(biases.get_shape().as_list())
            print "hdn_txt_logits: " + str(hdn_txt_logits.get_shape().as_list())
        with tf.variable_scope('img_hdn'):
            x_img = tf.placeholder(tf.float32, [batch_size, n_img_feats])
            weights = nn_utils.core.get_weights([n_img_feats, n_img_hidden])
            biases = nn_utils.core.get_biases([1, n_img_hidden])
            hdn_img_logits = tf.nn.tanh(tf.matmul(x_img, weights) + biases)
            print "x_img: " + str(x_txt.get_shape().as_list())
            print "img_hdn/weights: " + str(weights.get_shape().as_list())
            print "img_hdn/biases: " + str(biases.get_shape().as_list())
            print "hdn_img_logits: " + str(hdn_img_logits.get_shape().as_list())
        with tf.variable_scope('joint_hdn'):
            concat_logits = tf.concat((hdn_txt_logits, hdn_img_logits), 1)
            weights = nn_utils.core.get_weights([n_txt_hidden + n_img_hidden, n_joint_hidden])
            biases = nn_utils.core.get_biases([1, n_joint_hidden])
            hdn_joint_logits = tf.nn.tanh(tf.matmul(concat_logits, weights) + biases)
            print "joint_hdn/concat_logits: " + str(concat_logits.get_shape().as_list())
            print "joint_hdn/weights: " + str(weights.get_shape().as_list())
            print "joint_hdn/biases: " + str(biases.get_shape().as_list())
            print "hdn_joint_logits: " + str(hdn_joint_logits.get_shape().as_list())
        with tf.variable_scope('softmax'):
            dropout = tf.placeholder(tf.float32)
            weights = nn_utils.core.get_weights([n_joint_hidden, n_classes])
            biases = nn_utils.core.get_biases([1, n_classes])
            logits = tf.nn.dropout(tf.nn.tanh(tf.matmul(hdn_joint_logits, weights) + biases), dropout)
            print "softmax/weights: " + str(weights.get_shape().as_list())
            print "softmax/biases: " + str(biases.get_shape().as_list())
            print "logits: " + str(logits.get_shape().as_list())
        #endwith

        # prediction is just the minimization of the softmax cross entropy
        y = tf.placeholder(tf.float32, [batch_size, n_classes])
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        print "y: " + str(y.get_shape().as_list())

        # Adam Optimizer is a stochastic gradient descent style algo from 2014 that's
        # apparently amazing
        train_op = tf.train.AdamOptimizer(lrn_rate).minimize(loss)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(hdn_joint_logits, 1), tf.argmax(y, 1))
        pred = tf.argmax(hdn_joint_logits, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #endwith

    # Set up the tensorflow saver
    saver = tf.train.Saver(max_to_keep=100)

    log.info("Training")
    with tf.Session() as sess:

        # Initialize all our variables
        sess.run(tf.global_variables_initializer())

        # Iterate for n_iter iterations
        for i in range(0, n_iter):
            log.log_status('info', None, "Training; %d iterations complete (%.2f%%)",
                           i, 100.0 * i / n_iter)

            # Grab a random batch of the data
            random_indices = np.random.choice(range(0, n_samples), size=batch_size, replace=False)
            batch_txt_x = np.zeros([batch_size, n_txt_feats])
            batch_img_x = np.zeros([batch_size, n_img_feats])
            batch_y = np.zeros((batch_size, n_classes))
            for j in range(0, len(random_indices)):
                idx = random_indices[j]
                batch_y[j] = data_y[idx]
                batch_txt_x[j] = data_txt_x[pair_to_txt_idx[idx]]
                batch_img_x[j] = data_img_x[pair_to_img_idx[idx]]
            #endfor

            # Train
            sess.run(train_op, feed_dict={x_txt: batch_txt_x,
                                          x_img: batch_img_x,
                                          y: batch_y,
                                          dropout: dropout_p})

            # Print the mini-batch accuracy, to keep track of everything
            if (i+1) % 100 == 0 or i == n_iter - 1:
                saver.save(sess, model_file)
                acc = sess.run(accuracy, feed_dict={x_txt: batch_txt_x,
                                                    x_img: batch_img_x,
                                                    y: batch_y,
                                                    dropout: dropout_p})
                mini_loss = sess.run(loss, feed_dict={x_txt: batch_txt_x,
                                                      x_img: batch_img_x,
                                                      y: batch_y,
                                                      dropout: dropout_p})
                print("Iter: %d; Minibatch loss: %.2f; Minibatch accuracy: %.2f%%" %
                      (i+1, mini_loss, 100.0 * acc))
            #endif
        #endfor
    #endwith
#enddef


# Set up the global logger
log = LogUtil('debug', 180)

# Parse arguments
parser = ArgumentParser("ImageCaptionLearn_py: Neural network affinity classifier")
parser.add_argument("--train_file_txt", type=str, help="train opt; training file with "
                                                       "txt features (as dense .feats file)")
parser.add_argument("--train_file_img", type=str, help="train opt; training file with "
                                                       "img features (as dense .feats file)")
parser.add_argument("--train_file_labels", type=str, help="train opt; training file with "
                                                          "mention|box pair IDs and affinity"
                                                          "labels")
parser.add_argument("--batch_size", type=int, default=100, help="train opt; number of random "
                                                                "samples per batch")
parser.add_argument("--hidden_txt", type=int, default=100, help="train opt; number of hidden units "
                                                                "in the text layer")
parser.add_argument("--hidden_img", type=int, default=100, help="train opt; number of hidden units "
                                                                "in the image layer")
parser.add_argument("--hidden_joint", type=int, default=100, help="train opt; number of hidden units "
                                                                  "in the joint layer")
parser.add_argument("--dropout", type=float, default=1.0, help="train opt; Dropout rate "
                                                               "(probability to keep)")
parser.add_argument("--learn_rate", type=float, default=0.001, help="train opt; learning rate")
parser.add_argument("--iter", type=int, default=1000, help="train opt; number of iterations")
parser.add_argument("--model_file", type=str, help="Model file to save/load")
args = parser.parse_args()
arg_dict = vars(args)
util.dump_args(arg_dict, log)

# grab the model file from the args
model_file = arg_dict['model_file']
if model_file is not None:
    model_file = abspath(expanduser(model_file))

# If the training files were specified, train the model
train_file_txt = arg_dict['train_file_txt']
train_file_img = arg_dict['train_file_img']
train_file_labels = arg_dict['train_file_labels']
if train_file_txt is not None:
    train_file_txt = abspath(expanduser(train_file_txt))
if train_file_img is not None:
    train_file_img = abspath(expanduser(train_file_img))
if train_file_labels is not None:
    train_file_labels = abspath(expanduser(train_file_labels))
if isfile(train_file_txt) and isfile(train_file_img) and isfile(train_file_labels):
    log.info("Training Model")
    train(train_file_txt, train_file_img, train_file_labels,
          batch_size=arg_dict['batch_size'], n_txt_hidden=arg_dict['hidden_txt'],
          n_img_hidden=arg_dict['hidden_img'], n_joint_hidden=arg_dict['hidden_joint'],
          dropout_p=arg_dict['dropout'], lrn_rate=arg_dict['learn_rate'],
          n_iter=arg_dict['iter'])
else:
    log.warning("Could not open " + train_file_txt + ", " +
                train_file_img + ", or " + train_file_labels)
    parser.print_usage()
    quit()
#endif


