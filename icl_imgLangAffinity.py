import tensorflow as tf
import random as r
import numpy as np
from argparse import ArgumentParser
from os import listdir
from os.path import abspath, expanduser, isfile, isdir
import math
from ScoreDict import ScoreDict
from LogUtil import LogUtil
import linecache
from itertools import islice

NUM_FEATS = 4396
NUM_IMG_FEATS = 4096
NUM_TXT_FEATS = 300
#CLASS_DISTRO = [0.49, 0.51]   #actual distro: 87 / 13
CLASS_DISTRO = [0.87, 0.13]
CLASSES = [np.array((1, 0)), np.array((0, 1))]


"""
Returns a batch of examples in the form
(example_ids, x_tensor, y_tensor)
where batch_size/10 random files are read but only
batch_size random vectors are kept

Omitting batch_size returns all vectors for all specified filenames

Specifying balance returns examples according
to LABEL_DISTRO, which in practice randomly culls negative
examples until the distribution is satisfied
"""
def load_batch(filenames, batch_size=None, balance=False):
    global log, CLASS_DISTRO, CLASSES

    # If a batch size was specified, get a sample,
    # otherwise get all of the files
    files = filenames
    if batch_size is not None and batch_size < len(filenames):
        files = r.sample(filenames, batch_size / 10)

    # for each example, read a numpy array
    x_tensor = list(); y_tensor = list()
    example_ids = list(); indices = list()
    example_idx = 0; file_idx = 0
    for i in range(len(CLASSES)):
        indices.append(list())
    for filename in files:
        with open(filename, 'r') as f:
            for line in f:
                line_arr = line.split(",")
                example_ids.append(line_arr[0])
                y = int(line_arr[1])
                y_tensor.append(CLASSES[y])
                x_tensor.append(np.array(line_arr[2:]))
                indices[y].append(example_idx)
                example_idx += 1
            #endfor
        #endwith
        file_idx += 1
    #endfor

    #if no batch size was specified, return everything
    if batch_size is None:
        return example_ids, x_tensor, y_tensor

    # If a batch size was specified, we're only keeping a subset
    # of the examples. Which depends on whether balance was
    # specified
    indices_batch = list()
    if balance:
        indices_batch.extend(r.sample(indices[0], int(batch_size * CLASS_DISTRO[0])))
        indices_batch.extend(r.sample(indices[1], int(batch_size * CLASS_DISTRO[1])))
    else:
        indices_batch.extend(indices[0])
        indices_batch.extend(indices[1])
        indices_batch = r.sample(indices_batch, batch_size)
    r.shuffle(indices_batch)

    # create the batch tensors and return
    x_tensor_batch = list(); y_tensor_batch = list()
    example_ids_batch = list()
    for i in indices_batch:
        x_tensor_batch.append(x_tensor[i])
        y_tensor_batch.append(y_tensor[i])
        example_ids_batch.append(example_ids[i])
    return example_ids_batch, x_tensor_batch, y_tensor_batch
#enddef

"""
Returns a batch of examples in the form
(example_ids, x_tensor_img, x_tensor_txt, y_tensor)
where batch_size refers not to the number of examples
but the number of images from which the examples are drawn

Omitting batch_size returns vectors for all specified filenames

Specifying balance returns examples according
to LABEL_DISTRO, which in practice randomly culls negative
examples until the distribution is satisfied
"""
def load_split_batch(filenames, batch_size=None):
    global log, CLASS_DISTRO, CLASSES, NUM_IMG_FEATS

    # If a batch size was specified, get a sample,
    # otherwise get all of the files
    '''
    files = filenames
    if batch_size is not None and batch_size < len(filenames):
        files = r.sample(filenames, batch_size / 10)
    '''
    r.shuffle(filenames)


    # for each example, read a numpy array
    x_tensor_img = list(); x_tensor_txt = list()
    y_tensor = list(); example_ids = list()
    indices = list(); example_idx = 0; file_idx = 0
    for i in range(len(CLASSES)):
        indices.append(list())
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                line_arr = line.split(",")
                example_ids.append(line_arr[0])
                y = int(line_arr[1])
                y_tensor.append(CLASSES[y])
                x = np.array(line_arr[2:])
                x_tensor_img.append(x[0:NUM_IMG_FEATS])
                x_tensor_txt.append(x[NUM_IMG_FEATS:])
                indices[y].append(example_idx)
                example_idx += 1
            #endfor
        #endwith

        # if we've specified a batch size, stop after
        # we've collected enough examples, according to
        # the distro
        if batch_size is not None:
            enough_ex = True
            for i in range(len(CLASS_DISTRO)):
                if len(indices[i]) < int(batch_size * CLASS_DISTRO[i]):
                    enough_ex = False
            if enough_ex:
                break
        #endif

        file_idx += 1
    #endfor

    #if no batch size was specified, return everything
    if batch_size is None:
        return example_ids, x_tensor_img, x_tensor_txt, y_tensor

    # If a batch size was specified, we're only keeping a subset
    # of the examples. Which depends on whether balance was
    # specified
    indices_batch = list()
    indices_batch.extend(r.sample(indices[0], int(batch_size * CLASS_DISTRO[0])))
    indices_batch.extend(r.sample(indices[1], int(batch_size * CLASS_DISTRO[1])))
    '''
    if balance:
        indices_batch.extend(r.sample(indices[0], int(batch_size * CLASS_DISTRO[0])))
        indices_batch.extend(r.sample(indices[1], int(batch_size * CLASS_DISTRO[1])))
    else:
        indices_batch.extend(indices[0])
        indices_batch.extend(indices[1])
        indices_batch = r.sample(indices_batch, batch_size)
    '''
    r.shuffle(indices_batch)

    # create the batch tensors and return
    x_tensor_img_batch = list(); x_tensor_txt_batch = list()
    y_tensor_batch = list(); example_ids_batch = list()
    for i in indices_batch:
        x_tensor_img_batch.append(x_tensor_img[i])
        x_tensor_txt_batch.append(x_tensor_txt[i])
        y_tensor_batch.append(y_tensor[i])
        example_ids_batch.append(example_ids[i])
    return example_ids_batch, x_tensor_img_batch, x_tensor_txt_batch, y_tensor_batch
#enddef

def load_split_batch_single_file(filename, batch_size, start_idx=0, max_idx=0, rand_sample=True):
    global log, CLASS_DISTRO, CLASSES, NUM_IMG_FEATS

    # for each example, read a numpy array
    x_tensor_img = list(); x_tensor_txt = list()
    y_tensor = list(); example_ids = list()
    indices = list(); example_idx = 0
    for i in range(len(CLASSES)):
        indices.append(list())

    # Select batch_size indices
    if rand_sample:
        for idx in r.sample(range(start_idx, max_idx), batch_size):
            line = linecache.getline(filename, idx)
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
    else:
        end_idx = min(start_idx + batch_size, max_idx)
        with open(filename, 'r') as f:
            lines = list(islice(f, start_idx, end_idx))
            for line in lines:
                line_arr = line.split(",")
                if len(line_arr) < 3:
                    log.warning('Found missing vector at idx line: ' + line)
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
        #endwith
    #endif

    return example_ids, x_tensor_img, x_tensor_txt, y_tensor
#enddef

"""
def load_split_batch_single_file(filename, batch_size, start_idx=0, max_idx=0, rand_sample=True):
    global log, CLASS_DISTRO, CLASSES, NUM_IMG_FEATS

    # for each example, read a numpy array
    x_tensor_img = list(); x_tensor_txt = list()
    y_tensor = list(); example_ids = list()
    indices = list(); example_idx = 0
    for i in range(len(CLASSES)):
        indices.append(list())

    # Select batch_size indices
    vec_indices = list()
    if rand_sample:
        vec_indices = r.sample(range(start_idx, max_idx), batch_size)
    else:
        vec_indices = range(start_idx, start_idx + batch_size)
    vec_indices.sort()


    # read each example from the file
    for idx in vec_indices:
        line = linecache.getline(filename, idx)
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


    with open(filename, 'r') as f:
        idx = 1
        for line in f:
            if idx == vec_indices[0]:
                vec_indices.pop(0)
                line_arr = line.split(",")
                example_ids.append(line_arr[0])
                y = int(line_arr[1])
                y_tensor.append(CLASSES[y])
                x = np.array(line_arr[2:])
                x_tensor_img.append(x[0:NUM_IMG_FEATS])
                x_tensor_txt.append(x[NUM_IMG_FEATS:])
                indices[y].append(example_idx)
                example_idx += 1
                if len(vec_indices) == 0:
                    break
            #endif
            idx += 1
        #endfor
    #endwith

    return example_ids, x_tensor_img, x_tensor_txt, y_tensor
#enddef
"""

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
Evaluates the model in sess, given tf vars x and y_, loading
eval_files in evenly sized eval_folds batches
"""
def evaluateModel(sess, x, y_, eval_files, eval_folds):
    global log

    scores = ScoreDict()
    batch_size = int(len(eval_files) / eval_folds)
    for i in range(eval_folds):
        idx_start = i * batch_size
        idx_end = (i+1) * batch_size
        eval_ids, eval_x, eval_y = load_batch(eval_files[idx_start:idx_end])
        predictions = sess.run(y_, feed_dict={x: eval_x})
        for j in range(len(predictions)):
            gold = np.argmax(np.array(eval_y[j]))
            pred = np.argmax(np.array(predictions[j]))
            scores.increment(gold, pred)
        #endfor
    #endfor

    for label in scores.keys:
        log.info(scores.getScore(label).toString() +
                  " %d (%.2f%%)" % (scores.getGoldCount(label), scores.getGoldPercent(label)))
    #endfor
#enddef

"""
Returns a ScoreDict containing the evaluation of eval_files
on the network represented by the session and tensorflow vals


    for label in scores.keys:
        log.info(scores.getScore(label).toString() +
                 " %d (%.2f%%)" % (scores.getGoldCount(label), scores.getGoldPercent(label)))
    #endfor
"""
def evaluate_model(sess, x_img, x_txt, y_, retain_prob, eval_files, eval_folds):
    scores = ScoreDict()
    #batch_size = int(len(eval_files) / eval_folds)
    max_idx = 457079
    batch_size = max_idx / eval_folds
    for i in range(eval_folds):
        if (i+1) % 25 == 0:
            log.debug('eval_batch: ' + str(i+1))

        idx_start = i * batch_size
        idx_end = (i+1) * batch_size
        #eval_ids, eval_x_img, eval_x_txt, eval_y = load_split_batch(eval_files[idx_start:idx_end])
        if idx_end - idx_start > 1:
            eval_ids, eval_x_img, eval_x_txt, eval_y = \
                load_split_batch_single_file(eval_files, batch_size, idx_start, idx_end, False)
            if len(eval_ids) > 0:
                predictions = sess.run(y_, feed_dict={x_img: eval_x_img, x_txt: eval_x_txt, retain_prob: 1.0})
                for j in range(len(predictions)):
                    gold = np.argmax(np.array(eval_y[j]))
                    pred = np.argmax(np.array(predictions[j]))
                    scores.increment(gold, pred)
                #endfor
            #endif
        #endif
    #endfor
    return scores
#enddef

"""
Trains a model with a single, fully connected hidden layer between
the joint input representation and the softmax layer
"""
def train_one_hidden(train_files, eval_files, hidden_layer_width, learning_rate,
                     model_file=None, epochs=None, batch_size=None, balance=False):
    global NUM_FEATS, CLASSES

    # Set up the tf variables for our layers
    x = tf.placeholder(tf.float32, [None, NUM_FEATS])
    y = tf.placeholder(tf.float32, [None, len(CLASSES)])
    with tf.name_scope('hidden1'):
        weights = weight_variable([NUM_FEATS, hidden_layer_width])
        biases = bias_variable([hidden_layer_width])
        hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)
    with tf.name_scope('softmax'):
        weights = weight_variable([hidden_layer_width, len(CLASSES)])
        biases = bias_variable([len(CLASSES)])
        logits = tf.add(tf.matmul(hidden1, weights), biases)
        y_ = tf.nn.softmax(logits)
    #endwith


    '''
    class_weights = list()
    for c in CLASS_DISTRO:
        class_weights.append(1-c)
    class_weights = tf.constant(class_weights)
    weighted_logits = tf.mul(logits, class_weights) # shape [batch_size, 2]
    '''
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))

    # Using cross-entropy loss and gradient descent
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y *
    #     tf.log(tf.clip_by_value(y_, 1e-20, 1.0)), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # initialize our variables and run the application
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    # Load the model from file, or train a new one
    if model_file is not None and (epochs is None or batch_size is None):
        saver.restore(sess, model_file)
    elif epochs is not None and batch_size is not None:
        for i in range(epochs):
            # load this epoch's data
            example_ids, batch_x, batch_y, = load_batch(train_files, batch_size, balance)

            # for each batch after the first, compute our accuracy on the next batch
            # before using it to train
            if i > 0:
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            #endif

            # Now train on this batch, hopefully improving our performance
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

            # Every 100 epochs, save the model
            if model_file is not None and (i+1) % 100 == 0:
                saver.save(sess, model_file)
                log.info(None, "Model saved in file: %s", model_file)
            #endif
        #endfor
        if model_file is not None:
            saver.save(sess, model_file)
            log.info(None, "Model saved in file: %s", model_file)
        #endif
    #endif

    #evaluate model
    scores = ScoreDict()
    batch_size = int(len(eval_files) / 10)
    for i in range(10):
        idx_start = i * batch_size
        idx_end = (i+1) * batch_size
        eval_ids, eval_x, eval_y = load_batch(eval_files[idx_start:idx_end])
        predictions = sess.run(y_, feed_dict={x: eval_x})
        for j in range(len(predictions)):
            gold = np.argmax(np.array(eval_y[j]))
            pred = np.argmax(np.array(predictions[j]))
            scores.increment(gold, pred)
        #endfor
    #endfor

    log.info('---Confusion Matrix---')
    scores.printConfusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.getScore(label).toString() + " - %d (%.2f%%)" % \
                  (scores.getGoldCount(label), scores.getGoldPercent(label))
    #endfor
#enddef

"""
Trains a model with the following architecture
[box_feats] --> [300 node layer] --|
                                   |--> [hidden_joint_width] --> [softmax] --> y
                 [mention feats] --|
"""
def train_hdn_img_to_hdn_joint(train_files, eval_files, learning_rate,
                               model_file, epochs, batch_size, balance=False,
                               dropout=False):
    global log, CLASSES, NUM_IMG_FEATS, NUM_TXT_FEATS

    # Setup tf vars
    y = tf.placeholder(tf.float32, [None, len(CLASSES)])
    x_img = tf.placeholder(tf.float32, [None, NUM_IMG_FEATS])
    x_txt = tf.placeholder(tf.float32, [None, NUM_TXT_FEATS])
    with tf.name_scope('hidden_img'):
        weights = weight_variable([NUM_IMG_FEATS, NUM_TXT_FEATS])
        biases = bias_variable([NUM_TXT_FEATS])
        hiddenImg = tf.nn.relu(tf.matmul(x_img, weights) + biases)
    with tf.name_scope('hidden_joint_1'):
        weights = weight_variable([2 * NUM_TXT_FEATS, NUM_TXT_FEATS])
        biases = bias_variable([NUM_TXT_FEATS])
        hiddenJoint_1 = tf.nn.relu(tf.matmul(tf.concat(1, [x_txt, hiddenImg]), weights) + biases)
        retain_prob = tf.placeholder(tf.float32)
        hdn_joint_drop = tf.nn.dropout(hiddenJoint_1, retain_prob)
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

    # Load the model from file, or train a new one
    if model_file is not None and (epochs is None or batch_size is None):
        saver.restore(sess, model_file)
    elif epochs is not None and batch_size is not None:
        for i in range(epochs):
            # load this epoch's data
            example_ids, batch_x_img, batch_x_txt, batch_y, = \
                load_split_batch_single_file(train_files, batch_size, 0, 239726)

            """ This is too slow now; only evaluate at the end
            # every hundred epochs, evaluate against the dev
            if (i+1) % 250 == 0:
                log.tic('info', 'Epoch ' + str(i+1) + '; evaluating')
                scores = evaluate_model(sess, x_img, x_txt, y_, retain_prob, eval_files, 100)
                log.toc('info')
                log.log_status('info', None, "Epoch: %5d; pos_f1: %.2f%%; neg_f1: %.2f%%; "+ \
                               "Acc: %.2f%%; gold distro: %.2f%%; pred distro: %.2f%%",
                               i, 100.0*scores.getScore(1).f1, 100.0*scores.getScore(0).f1,
                               scores.getAccuracy(), scores.getGoldPercent(1), scores.getPredPercent(1))
            #endfor
            """

            # Now train on this batch, hopefully improving our performance
            dropout_prob = 1.0
            if dropout:
                dropout_prob = 0.5
            sess.run(train_step, feed_dict={x_img: batch_x_img, x_txt: batch_x_txt,
                                            y: batch_y, retain_prob: dropout_prob})

            # Every 100 epochs, save the model
            if model_file is not None and (i+1) % 100 == 0:
                saver.save(sess, model_file)
                log.info(None, "Model saved in file: %s", model_file)
            #endif
        #endfor
        if model_file is not None:
            saver.save(sess, model_file)
            log.info(None, "Model saved in file: %s", model_file)
        #endif
    #endif

    #evaluate model after the last ieration
    log.info('Evaluating')
    scores = evaluate_model(sess, x_img, x_txt, y_, retain_prob, eval_files, 100) #457079
    log.info('---Confusion Matrix---')
    scores.printConfusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.getScore(label).toString() + " - %d (%.2f%%)" % \
                  (scores.getGoldCount(label), scores.getGoldPercent(label))
    #endfor
#enddef


def train(train_files, eval_files, learning_rate, model_file,
          epochs, batch_size, balance=False, dropout=False):
    global log, CLASSES, NUM_IMG_FEATS, NUM_TXT_FEATS

    # Setup tf vars
    y = tf.placeholder(tf.float32, [None, len(CLASSES)])
    x_img = tf.placeholder(tf.float32, [None, NUM_IMG_FEATS])
    x_txt = tf.placeholder(tf.float32, [None, NUM_TXT_FEATS])

    with tf.name_scope('hidden_img_half'):
        weights = weight_variable([NUM_IMG_FEATS, int(NUM_IMG_FEATS/2)])
        biases = bias_variable([int(NUM_IMG_FEATS/2)])
        hidden_img_half = tf.nn.relu(tf.matmul(x_img, weights) + biases)
    with tf.name_scope('hidden_img_quarter'):
        weights = weight_variable([int(NUM_IMG_FEATS/2), int(NUM_IMG_FEATS/4)])
        biases = bias_variable([int(NUM_IMG_FEATS/4)])
        hidden_img_quarter = tf.nn.relu(tf.matmul(hidden_img_half, weights) + biases)
    with tf.name_scope('hidden_img_pre_joint'):
        weights = weight_variable([int(NUM_IMG_FEATS/4), NUM_TXT_FEATS])
        biases = bias_variable([NUM_TXT_FEATS])
        hidden_img_pre = tf.nn.relu(tf.matmul(hidden_img_quarter, weights) + biases)
    with tf.name_scope('hidden_joint'):
        weights = weight_variable([2 * NUM_TXT_FEATS, NUM_TXT_FEATS])
        biases = bias_variable([NUM_TXT_FEATS])
        hidden_joint = tf.nn.relu(tf.matmul(tf.concat(1, [x_txt, hidden_img_pre]), weights) + biases)
    with tf.name_scope('hidden_joint_half'):
        weights = weight_variable([NUM_TXT_FEATS, int(NUM_TXT_FEATS/2)])
        biases = bias_variable([int(NUM_TXT_FEATS/2)])
        hidden_joint_half = tf.nn.relu(tf.matmul(hidden_joint, weights) + biases)
        retain_prob = tf.placeholder(tf.float32)
        hdn_joint_drop = tf.nn.dropout(hidden_joint_half, retain_prob)
    with tf.name_scope('softmax'):
        weights = weight_variable([int(NUM_TXT_FEATS/2), len(CLASSES)])
        biases = bias_variable([len(CLASSES)])
        logits = tf.add(tf.matmul(hdn_joint_drop, weights), biases)
        y_ = tf.nn.softmax(logits)
    #endwith

    # Either use straight softmax cross entropy with logits; or multiply
    # by our inverse class distro
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
    if balance:
        class_weights = list()
        for c in CLASS_DISTRO:
            class_weights.append(1-c)
        class_weights = tf.constant(class_weights)
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

    # Load the model from file, or train a new one
    if model_file is not None and (epochs is None or batch_size is None):
        saver.restore(sess, model_file)
    elif epochs is not None and batch_size is not None:
        for i in range(epochs):
            log.log_status('info', None, "Epoch: %d", i)

            # load this epoch's data
            example_ids, batch_x_img, batch_x_txt, batch_y, = \
                load_split_batch(train_files, batch_size)

            # every hundred epochs, evaluate against the dev
            if (i+1) % 100 == 0:
                scores = evaluate_model(sess, x_img, x_txt, y_, retain_prob, eval_files, 10)
                log.info(None, "Epoch: %5d; pos_f1: %.2f%%; neg_f1: %.2f%%; "+ \
                         "Acc: %.2f%%; gold distro: %.2f%%; pred distro: %.2f%%",
                         i, 100.0*scores.getScore(1).f1, 100.0*scores.getScore(0).f1,
                         scores.getAccuracy(), scores.getGoldPercent(1), scores.getPredPercent(1))
            #endfor

            # Now train on this batch, hopefully improving our performance
            dropout_prob = 1.0
            if dropout:
                dropout_prob = 0.5
            sess.run(train_step, feed_dict={x_img: batch_x_img, x_txt: batch_x_txt,
                                            y: batch_y, retain_prob: dropout_prob})

            # Every 100 epochs, save the model
            if model_file is not None and (i+1) % 100 == 0:
                saver.save(sess, model_file)
                log.info(None, "Model saved in file: %s", model_file)
            #endif
        #endfor
        if model_file is not None:
            saver.save(sess, model_file)
            log.info(None, "Model saved in file: %s", model_file)
        #endif
    #endif

    #evaluate model after the last ieration
    scores = evaluate_model(sess, x_img, x_txt, y_, retain_prob, eval_files, 10)
    log.info('---Confusion Matrix---')
    scores.printConfusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.getScore(label).toString() + " - %d (%.2f%%)" % \
              (scores.getGoldCount(label), scores.getGoldPercent(label))
    #endfor
#enddef

def tune(train_files, eval_files):
    lrn_rates = [0.1, 0.05, 0.01]
    batch_sizes = [50, 100, 200]
    for lrn in lrn_rates:
        for btch in batch_sizes:
            for balance in [True, False]:
                for dropout in [True, False]:
                    log.info(None, "#############learning_rate:%.3f; batch_size:%d; balance:%s; dropout:%s###############",
                             lrn, btch, str(balance), str(dropout))
                    train_hdn_img_to_hdn_joint(train_files, eval_files, lrn, "affinity.model",
                        1000, btch, balance, dropout)
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
    parser.add_argument("--data", type=str, help="Data directory; expects train/dev/test child dirs")
    parser.add_argument("--model", type=str, help="Specifies a model file to save / load, depending on task")
    parser.add_argument("--train", action='store_true', help="Trains a model; saves in --model")
    parser.add_argument("--eval", action='store_true', help="Evaluates a model; loads from --model")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="train opt; Sets learning rate")
    parser.add_argument("--batch_size", type=int, default=100, help="train opt; Number of imgs in each batch")
    parser.add_argument("--balance", action='store_true',
                        help="train opt; Whether to load data according to true distro")
    parser.add_argument("--dropout", action='store_true', help='train opt; Use dropout @ 50%%')
    parser.add_argument("--epochs", type=int, default=100, help="train opt; Number of batches to iterate over")
    parser.add_argument("--hidden_width", type=int, default=NUM_TXT_FEATS,
                        help="train opt; Size of the hidden layer")
    args = parser.parse_args()
    arg_dict = vars(args)

    # Set up up the logger
    log_lvl = arg_dict['log_lvl']
    log_delay = arg_dict['log_delay']
    log = LogUtil(lvl=log_lvl, delay=log_delay)

    # Deterine if we have valid options
    data_dir = arg_dict['data']
    if data_dir is not None:
        data_dir = abspath(expanduser(data_dir))
    model_file = arg_dict['model']
    if model_file is not None:
        model_file = abspath(expanduser(model_file))
    '''
    if data_dir is None or not isdir(data_dir):
        log.error("Must specify data directory")
        parser.print_usage()
        quit()
    elif model_file is None:
        log.error("Must specify model file")
        parser.print_usage()
        quit()
    '''

    # Retrieve all train and eval files once, if needed
    '''
    train_model = arg_dict['train']
    eval_model = arg_dict['eval']
    train_files = list(); eval_files = list()
    if train_model:
        for d in listdir(data_dir + "/train/"):
            full_path = data_dir + "/train/" + d
            if isfile(full_path):
                train_files.append(full_path)
    if eval_model:
        for d in listdir(data_dir + "/dev/"):
            full_path = data_dir + "/dev/" + d
            if isfile(full_path):
                eval_files.append(full_path)
    '''
    #train(train_files, eval_files, arg_dict['learning_rate'], model_file, arg_dict['epochs'],
    #      arg_dict['batch_size'], arg_dict['balance'], arg_dict['dropout'])
    #tune(train_files, eval_files)

    train_hdn_img_to_hdn_joint('/home/ccervan2/source/data/feats/affinity_feats_train.feats',
                               '/home/ccervan2/source/data/feats/affinity_feats_dev.feats',
                               arg_dict['learning_rate'], model_file, arg_dict['epochs'],
                               arg_dict['batch_size'], arg_dict['balance'], arg_dict['dropout'])
    #train_one_hidden(train_files, eval_files, NUM_FEATS / 2, arg_dict['learning_rate'],
    #                 model_file, arg_dict['epochs'], arg_dict['batch_size'], arg_dict['balance_data'])
#enddef

#init()


log = LogUtil(lvl='debug', delay=30)
train_hdn_img_to_hdn_joint('/home/ccervan2/source/data/feats/affinity_feats_train.feats',
                           '/home/ccervan2/source/data/feats/affinity_feats_dev.feats',
                           0.1, 'affinity.model', 1000, 200, False, True)

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