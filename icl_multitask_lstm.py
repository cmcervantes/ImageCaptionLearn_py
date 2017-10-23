import tensorflow as tf
import numpy as np
from os.path import expanduser, abspath
from nn_utils import core as nn_util
from nn_utils import data as nn_data
from utils.Logger import Logger

CLASSES_REL = ['n', 'c', 'b', 'p']
CLASSES_NONVIS = ['v', 'n']
CLASSES_CARD = ['0', '1', '2', '3', '4', '5',
                '6', '7', '8', '9', '10', '11+']
TASKS = ["rel_intra", "rel_cross", "nonvis", "card"]

# Params that we've tuned from individual learning
BATCH_SIZE = 512
LSTM_HIDDEN_WIDTH = 200
START_HIDDEN_WIDTH = 512
HIDDEN_DEPTH = 2
ACTIVATION = 'relu'
LEARN_RATE = 0.001
ADAM_EPSILON = 1E-08
CLIP_NORM = 5.0
EMBEDDING_TYPE = 'w2v'
EPOCHS = 100

# File roots
raw_root = abspath(expanduser("~/data/tacl201712/raw")) + "/"
feats_root = abspath(expanduser("~/data/tacl201712/feats")) + "/"
model_root = abspath(expanduser("~/models/tacl201712/")) + "/"

# Training files for our tasks
sentence_file = raw_root + "flickr30k_train_captions.txt"
mention_idx_file_intra = raw_root + "flickr30k_train_mentionPairs_intra.txt"
mention_idx_file_cross = raw_root + "flickr30k_train_mentionPairs_cross.txt"
mention_idx_file_nonvis = raw_root + "flickr30k_train_mentions_nonvis.txt"
mention_idx_file_card = raw_root + "flickr30k_train_mentions_card.txt"
feats_file_rel = feats_root + "flickr30k_train_relation.feats"
feats_meta_file_rel = feats_root + "flickr30k_train_relation_meta.json"
feats_file_nonvis = feats_root + "flickr30k_train_nonvis.feats"
feats_meta_file_nonvis = feats_root + "flickr30k_train_nonvis_meta.json"
feats_file_card = feats_root + "flickr30k_train_card.feats"
feats_meta_file_card = feats_root + "flickr30k_train_card_meta.json"

# Evaluation files
eval_sentence_file = raw_root + "flickr30k_dev_captions.txt"
eval_mention_idx_file_intra = raw_root + "flickr30k_dev_mentionPairs_intra.txt"
eval_mention_idx_file_cross = raw_root + "flickr30k_dev_mentionPairs_cross.txt"
eval_mention_idx_file_nonvis = raw_root + "flickr30k_dev_mentions_nonvis.txt"
eval_mention_idx_file_card = raw_root + "flickr30k_dev_mentions_card.txt"
eval_feats_file_rel = feats_root + "flickr30k_dev_relation.feats"
eval_feats_meta_file_rel = feats_root + "flickr30k_dev_relation_meta.json"
eval_feats_file_nonvis = feats_root + "flickr30k_dev_nonvis.feats"
eval_feats_meta_file_nonvis = feats_root + "flickr30k_dev_nonvis_meta.json"
eval_feats_file_card = feats_root + "flickr30k_dev_card.feats"
eval_feats_meta_file_card = feats_root + "flickr30k_dev_card_meta.json"
eval_relation_label_file = raw_root + "flickr30k_dev_mentionPair_labels.txt"


log = Logger('debug', 180)
embedding_type = None


def setup(n_feats_rel, n_feats_nonvis, n_feats_card):
    """

    :return:
    """
    global CLASSES_REL, CLASSES_NONVIS, CLASSES_CARD, TASKS, \
        BATCH_SIZE, LSTM_HIDDEN_WIDTH, START_HIDDEN_WIDTH, \
        HIDDEN_DEPTH, ACTIVATION, LEARN_RATE, ADAM_EPSILON, CLIP_NORM

    # dropout percentage
    dropout = tf.placeholder(tf.float32)
    tf.add_to_collection('dropout', dropout)

    # Set up the bidirectional LSTM
    with tf.variable_scope('bidirectional_lstm'):
        nn_util.setup_bidirectional_lstm(LSTM_HIDDEN_WIDTH)

    # Get the outputs, which are a (fw,bw) tuple of
    # [batch_size, seq_length, n_hidden_lstm matrices
    lstm_outputs = (tf.get_collection('bidirectional_lstm/outputs_fw')[0],
                    tf.get_collection('bidirectional_lstm/outputs_bw')[0])

    # Each task has a separate set of input tensors to their hidden layers
    for task in TASKS:
        # Vary the hidden input width by the task
        hidden_input_width = None
        n_classes = None
        if task == 'rel_intra':
            hidden_input_width = 6 * LSTM_HIDDEN_WIDTH + n_feats_rel
            n_classes = len(CLASSES_REL)
        elif task == 'rel_cross':
            hidden_input_width = 8 * LSTM_HIDDEN_WIDTH + n_feats_rel
            n_classes = len(CLASSES_REL)
        elif task == 'nonvis':
            hidden_input_width = 4 * LSTM_HIDDEN_WIDTH + n_feats_nonvis
            n_classes = len(CLASSES_NONVIS)
        elif task == 'card':
            hidden_input_width = 4 * LSTM_HIDDEN_WIDTH + n_feats_card
            n_classes = len(CLASSES_CARD)

        # Each task needs its own variable scope
        with tf.variable_scope(task):
            # each task has a separate label placeholder
            y = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes])

            nn_util.setup_batch_input_first_last_sentence_mention_pair(BATCH_SIZE, lstm_outputs)
            batch_input = tf.get_collection(task + '/batch_input')[0]

            # Set up the hidden layer(s)
            hidden_inputs = [batch_input]
            n_hidden_widths = [START_HIDDEN_WIDTH]
            for depth in range(1, HIDDEN_DEPTH):
                n_hidden_widths.append(n_hidden_widths[depth-1] / 2)
            for depth in range(0, HIDDEN_DEPTH):
                with tf.variable_scope("hdn_" + str(depth+1)):
                    weights = nn_util.get_weights([hidden_input_width, n_hidden_widths[depth]])
                    biases = nn_util.get_biases([1, n_hidden_widths[depth]])
                    logits = tf.matmul(hidden_inputs[depth], weights) + biases
                    if ACTIVATION == 'sigmoid':
                        logits = tf.nn.sigmoid(logits)
                    elif ACTIVATION == 'tanh':
                        logits = tf.nn.tanh(logits)
                    elif ACTIVATION == 'relu':
                        logits = tf.nn.relu(logits)
                    elif ACTIVATION == 'leaky_relu':
                        logits = nn_util.leaky_relu(logits)
                    logits = tf.nn.dropout(logits, dropout)
                    hidden_inputs.append(logits)
                    hidden_input_width = n_hidden_widths[depth]
                #endwith
            #endfor
            with tf.variable_scope("softmax"):
                weights = nn_util.get_weights([n_hidden_widths[HIDDEN_DEPTH - 1], n_classes])
                biases = nn_util.get_biases([1, n_classes])

                # Because our label distribution is so skewed, we have to
                # add a constant epsilon to all of the values to prevent
                # the loss from being NaN
                epsilon = np.nextafter(0, 1)
                eps_list = list()
                for i in range(0, n_classes):
                    eps_list.append(epsilon)
                constant_epsilon = tf.constant(eps_list, dtype=tf.float32)
                final_logits = tf.matmul(hidden_inputs[HIDDEN_DEPTH], weights) + biases + constant_epsilon
                predicted_proba = tf.nn.softmax(final_logits)
            #endwith
            tf.add_to_collection('predicted_proba', predicted_proba)

            # Cross entropy loss
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=final_logits, labels=y)
            loss = tf.reduce_sum(cross_entropy)
            tf.add_to_collection('loss', loss)

            # Add the training operation
            nn_util.add_train_op(loss, LEARN_RATE, ADAM_EPSILON, CLIP_NORM)

            # Evaluate model
            pred = tf.argmax(predicted_proba, 1)
            correct_pred = tf.equal(pred, tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.add_to_collection('pred', pred)
            tf.add_to_collection('accuracy', accuracy)
        #endwith
    #endfor

    # Set up the joint optimization
    loss_intra = tf.get_collection("rel_intra/loss")[0]
    loss_cross = tf.get_collection("rel_cross/loss")[0]
    loss_nonvis = tf.get_collection("nonvis/loss")[0]
    loss_card = tf.get_collection("card/loss")[0]
    joint_loss = tf.add(tf.add(loss_intra, loss_cross), tf.add(loss_nonvis, loss_card))
    tf.add_to_collection('joint_loss', joint_loss)
    nn_util.add_train_op(joint_loss, LEARN_RATE, ADAM_EPSILON, CLIP_NORM)

    # Dump all the tf variables, just to double check everything's
    # the right size
    for name in tf.get_default_graph().get_all_collection_keys():
        coll = tf.get_collection(name)
        if len(coll) >= 1:
            coll = coll[0]
        print "%-20s: %s" % (name, coll)
    #endfor
#endef


def train():


    global log, TASKS, EPOCHS, BATCH_SIZE

    # Load all of our data
    log.info("Loading training data")
    sentence_data = nn_data.load_sentences(sentence_file, embedding_type)
    mention_pair_data_intra = nn_data.load_mention_pair_data(mention_idx_file_intra,
                                                             feats_file_rel,
                                                             feats_meta_file_rel)
    mention_pair_data_cross = nn_data.load_mention_pair_data(mention_idx_file_cross,
                                                             feats_file_rel,
                                                             feats_meta_file_rel)
    mention_data_nonvis = nn_data.load_mention_data(mention_idx_file_nonvis,
                                                    feats_file_nonvis, feats_meta_file_nonvis)
    mention_data_card = nn_data.load_mention_data(mention_idx_file_card,
                                                  feats_file_card, feats_meta_file_card)
    mention_pairs_intra = list(mention_pair_data_intra['mention_pair_indices'].keys())
    n_mention_pairs_intra = len(mention_pairs_intra)
    mention_pairs_cross = list(mention_pair_data_cross['mention_pair_indices'].keys())
    n_mention_pairs_cross = len(mention_pairs_cross)
    mentions_nonvis = list(mention_data_nonvis['mention_indices'].keys())
    n_mentions_nonvis = len(mentions_nonvis)
    mentions_card = list(mention_data_card['mention_indices'].keys())
    n_mentions_card = len(mentions_card)

    log.info("Loading eval data")
    eval_mention_pair_data_intra = nn_data.load_mention_pair_data(eval_mention_idx_file_intra,
                                                                  eval_feats_file_rel,
                                                                  eval_feats_meta_file_rel)
    eval_mention_pair_data_cross = nn_data.load_mention_pair_data(eval_mention_idx_file_cross,
                                                                  eval_feats_file_rel,
                                                                  eval_feats_meta_file_rel)
    eval_mention_data_nonvis = nn_data.load_mention_data(eval_mention_idx_file_nonvis,
                                                         eval_feats_file_nonvis,
                                                         eval_feats_meta_file_nonvis)
    eval_mention_data_card = nn_data.load_mention_data(eval_mention_idx_file_card,
                                                       eval_feats_file_card,
                                                       eval_feats_meta_file_card)

    log.info("Setting up network architecture")
    n_feats_rel = mention_pair_data_cross['max_feat_idx'] + 1
    n_feats_nonvis = mention_data_nonvis['max_feat_idx'] + 1
    n_feats_card = mention_data_card['max_feat_idx'] + 1
    setup(n_feats_rel, n_feats_nonvis, n_feats_card)

    # Get the various operations for our tasks
    tf_ops = dict()
    for task in TASKS:
        loss_name = task + "/loss"
        train_op_name = task + "/train_op"
        accuracy_name = task + "/accuracy"
        tf_ops[loss_name] = tf.get_collection(loss_name)[0]
        tf_ops[train_op_name] = tf.get_collection(train_op_name)[0]
        tf_ops[accuracy_name] = tf.get_collection(accuracy_name)[0]
    #endfor

    # We want to keep track of the best scores with
    # the epoch that they originated from
    best_avg_score = -1
    best_epoch = -1

    # Train
    log.info("Training")
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        # Initialize all our variables
        sess.run(tf.global_variables_initializer())

        # Iterate through the data [epochs] number of times
        for i in range(0, EPOCHS):
            log.info(None, "--- Epoch %d ----", i+1)
            losses = list()
            accuracies = list()

            # Shuffle the data once for this epoch
            np.random.shuffle(mention_pairs_intra)
            np.random.shuffle(mention_pairs_cross)
            np.random.shuffle(mentions_nonvis)
            np.random.shuffle(mentions_card)

            # Iterate through the entirety of the data
            start_idx = 0
            end_idx = start_idx + BATCH_SIZE

            n_iter = n_pairs / BATCH_SIZE
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

                if avg >= best_avg_score - 0.01:
                    log.info(None, "Previous best score average F1 of %.2f%% after %d epochs",
                             100.0 * best_avg_score, best_epoch)
                    best_avg_score = avg
                    best_epoch = i
                    log.info(None, "New best at current epoch (%.2f%%)",
                             100.0 * best_avg_score)
                #endif

                # Implement early stopping; if it's been 10 epochs since our best, stop
                if i >= (best_epoch + 10):
                    log.info(None, "Stopping early; best scores at %d epochs", best_epoch)
                    break
                    #endif
                    #endif
        #endfor
        log.info("Saving final model")
        saver.save(sess, model_file)
        #endwith


