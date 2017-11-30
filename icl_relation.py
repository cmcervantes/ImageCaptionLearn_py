import cPickle
import json
from argparse import ArgumentParser
from os.path import abspath, expanduser
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from utils.Logger import Logger
from utils import core as util
from utils import data as data_util
from utils import string as string_util
from utils.ScoreDict import ScoreDict


def train(solver, max_iter, balance, norm, warm_start,
          multiclass_mode, ignored_feats=set()):
    """
    Trains the relation model as a multinomial logistic regression model
    :param solver:
    :param max_iter:
    :param balance:
    :param norm:
    :param warm_start:
    :param multiclass_mode:
    :param ignored_feats:
    :return:
    """
    global log, meta_dict, train_file, model_file

    log.tic('info', "Loading training data")
    x, y, ids = \
        data_util.load_very_sparse_feats(train_file, meta_dict,
                                         ignored_feats)
    if norm is not None:
        normalize(x, norm=norm, copy=False)
    log.toc('info')

    log.tic('info', "Training")
    class_weight = None
    if balance:
        class_weight = 'balanced'
    #endif
    logistic = LogisticRegression(class_weight=class_weight, solver=solver,
                 max_iter=max_iter, multi_class=multiclass_mode, n_jobs=-1,
                 warm_start=warm_start)
    logistic.fit(x, y)
    log.toc('info')

    log.info("Saving model")
    with open(model_file, 'wb') as pickle_file:
        cPickle.dump(logistic, pickle_file)
#enddef


def induce_ji_predictions(pred_scores):
    for ij_pair in pred_scores.keys():
        # Split the ij pair label into its constituent elements so we
        # can construct the ji pair; Recall that a mention pair ID is
        #   doc:<ID>;caption_1:<idx>;mention_1:<idx>;caption_2:<idx>;mention_2:<idx>
        ij_pair_dict = string_util.kv_str_to_dict(ij_pair)
        ji_pair = "doc:" + ij_pair_dict['doc'] + \
                  ";caption_1:" + ij_pair_dict['caption_2'] + \
                  ";mention_1:" + ij_pair_dict['mention_2'] + \
                  ";caption_2:" + ij_pair_dict['caption_1'] + \
                  ";mention_2:" + ij_pair_dict['mention_1']

        # The scores for ji are the same predictions for coref and null,
        # but flipped for sub/supset
        pred_scores[ji_pair] = np.zeros(4)
        pred_scores[ji_pair][:] = pred_scores[ij_pair]
        pred_scores[ji_pair][2] = pred_scores[ij_pair][3]
        pred_scores[ji_pair][3] = pred_scores[ij_pair][2]
    #endfor
    return pred_scores
#enddef


def evaluate(norm, ignored_feats=set()):
    """
    Evaluates the saved model against the eval file
    :param norm:
    :param ignored_feats:
    :return:
    """
    global log, eval_file, model_file, scores_file, meta_dict

    log.info("Loading model from file")
    logistic = cPickle.load(open(model_file, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = \
        data_util.load_very_sparse_feats(eval_file, meta_dict,
                                         ignored_feats)
    if norm is not None:
        normalize(x_eval, norm=norm, copy=False)
    #endif

    log.info("Evaluating")
    y_pred_probs = logistic.predict_log_proba(x_eval)

    # Though we don't evaluate against them here, we want
    # to store scores for both ij and ji pairs
    pred_scores = dict()
    for i in range(len(ids_eval)):
        pred_scores[ids_eval[i]] = y_pred_probs[i]
    pred_scores = induce_ji_predictions(pred_scores)

    # We evaluate here only on ij pairs, but since this script
    # is not our final evaluation (and because the score should be
    # identical anyway) that's fine; this is functionally an estimate
    # of the true score
    scores = ScoreDict()
    for i in range(len(y_eval)):
        scores.increment(y_eval[i], np.argmax(y_pred_probs[i]))

    log.info("---Confusion matrix---")
    scores.print_confusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.get_score(label).to_string() + " - %d (%.2f%%)" % \
              (scores.get_gold_count(label), scores.get_gold_percent(label))
    print "Acc: " + str(scores.get_accuracy()) + "%"

    if scores_file is not None:
        log.info("Writing probabilities to " + scores_file)
        with open(scores_file, 'w') as f:
            for id in pred_scores.keys():
                line = list()
                line.append(id)
                for j in range(len(pred_scores[id])):
                    line.append(str(pred_scores[id][j]))
                f.write(','.join(line) + '\n')
            #endfor
            f.close()
        #endwith
    #endif
#enddef


# At one time I had more arguments, but in the end it's much
# easier not to specify all this on the command line
log = Logger(lvl='debug', delay=45)

# Parse what arguments remain
solvers = ['lbfgs', 'newton-cg', 'sag']
multiclass_modes = ['multinomial', 'ovr']
parser = ArgumentParser("ImageCaptionLearn_py: Pairwise Relation Classifier")
parser.add_argument("--norm", type=str, help="preproc opt; Specify data normalization")
parser.add_argument("--solver", choices=solvers, default=solvers[0],
                    help="train opt; Multiclass solver to use")
parser.add_argument("--warm", action='store_true', help="train opt; Uses previous solution as init")
parser.add_argument("--max_iter", type=int, default=100, help="train opt; Specifies the max iterations")
parser.add_argument("--balance", action='store_true',
                    help="train_opt; Whether to use class weights inversely proportional to the data distro")
parser.add_argument("--mcc_mode", choices=multiclass_modes, default=multiclass_modes[0],
                    help="train opt; multiclass mode")
parser.add_argument("--nonvis_file", type=str, help="Retrieves nonvis labels from file; excludes from eval")
parser.add_argument("--ablation_file", type=str, help="Performs ablation, using the groupings specified "
                                                      "in the given ablation config file ")
parser.add_argument("--data_dir", required=True,
                    type=lambda f: util.arg_path_exists(parser, f),
                    help="Directory containing feats/, and scores/ directories")
parser.add_argument("--data_root", type=str, required=True,
                    help="Data file root (eg. flickr30k_train)")
parser.add_argument("--eval_data_root", type=str,
                    help="Data file root for eval data (eg. flickr30k_dev)")
parser.add_argument("--train", action='store_true', help='Trains a model')
parser.add_argument("--predict", action='store_true',
                    help='Predicts using pre-trained model')
parser.add_argument("--rel_type", choices=['intra', 'intra_ij', 'cross'],
                    required=True,
                    help="Whether we're dealing with intra-caption or "
                         "cross-caption relations")
parser.add_argument("--model_file", type=str, required=True,
                    help="Model file to save/load")
args = parser.parse_args()
arg_dict = vars(args)
util.dump_args(arg_dict, log)

# Construct data files from the root directory and filename
data_dir = arg_dict['data_dir'] + "/"
data_root = arg_dict['data_root']
eval_data_root = arg_dict['eval_data_root']
rel_type = arg_dict['rel_type']
file_suffix = "_relation_classifier_" + rel_type
train_file = data_dir + "feats/" + data_root + file_suffix + ".feats"
eval_file = data_dir + "feats/" + eval_data_root + file_suffix + ".feats"
scores_file = data_dir + "scores/" + eval_data_root + file_suffix + ".scores"
meta_file = data_dir + "feats/" + data_root + file_suffix + "_meta.json"
meta_dict = json.load(open(meta_file, 'r'))
model_file = arg_dict['model_file']
if model_file is not None:
    model_file = abspath(expanduser(model_file))
ablation_file = arg_dict['ablation_file']
ablation_groups = None
if ablation_file is not None:
    ablation_file = abspath(expanduser(ablation_file))
    ablation_groups = data_util.load_ablation_file(ablation_file)

# Parse the other args
max_iter = arg_dict['max_iter']
balance = arg_dict['balance']
warm_start = arg_dict['warm']
solver_type = arg_dict['solver']
mcc_mode = arg_dict['mcc_mode']
normalize_data = arg_dict['norm']

if arg_dict['train']:
    train(solver_type, max_iter, balance, normalize_data, warm_start, mcc_mode, set())
if arg_dict['predict']:
    evaluate(normalize_data, set())
