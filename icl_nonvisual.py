import cPickle
import json
from argparse import ArgumentParser
from os.path import abspath, expanduser

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utils.Logger import Logger
from utils import core as util
from utils import data as data_util
from utils.ScoreDict import ScoreDict


def train(model, balance, max_iter=None, max_depth=None,
          num_estimators=None, warm_start=None,
          ignored_feats=set()):
    """
    Trains the model
    :param model:
    :param balance:
    :param max_iter:
    :param max_depth:
    :param num_estimators:
    :param warm_start:
    :param ignored_feats:
    :return:
    """
    global log, train_file, meta_dict, model_file

    log.tic('info', "Loading training data")
    x, y, ids = \
        data_util.load_very_sparse_feats(train_file,
                                         meta_dict,
                                         ignored_feats)

    log.toc('info')

    log.tic('info', "Training")
    class_weight = None
    if balance:
        class_weight = 'balanced'
    #endif
    learner = None
    if model == 'svm':
        learner = SVC(probability=True, class_weight=class_weight, max_iter=max_iter)
    elif model == 'logistic':
        learner = LogisticRegression(class_weight=class_weight, max_iter=max_iter, n_jobs=-1)
    elif model == "decision_tree":
        learner = DecisionTreeClassifier(max_depth=max_depth, class_weight=class_weight)
    elif model == 'random_forest':
        learner = RandomForestClassifier(n_estimators=num_estimators, max_depth=max_depth, n_jobs=-1,
                                         warm_start=warm_start, class_weight=class_weight)
    #endif
    learner.fit(x, y)
    log.toc('info')

    log.info("Saving model")
    with open(model_file, 'wb') as pickle_file:
        cPickle.dump(learner, pickle_file)
#enddef


def evaluate(lemma_file=None, hyp_file=None,
             ignored_feats=set(), save_scores=True):
    """
    Evaluates the model against the eval data
    :param lemma_file:
    :param hyp_file:
    :param ignored_feats:
    :param save_scores:
    :return:
    """
    global log, eval_file, model_file, scores_file

    log.info("Loading model from file")
    learner = cPickle.load(open(model_file, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = \
        data_util.load_very_sparse_feats(eval_file, meta_dict,
                                         ignored_feats)

    lemma_dict = dict()
    lemmas = set()
    if lemma_file is not None:
        log.info("Loading mention lemmas")
        with open(lemma_file, 'r') as f:
            for line in f:
                parts = line.replace('"', '').strip().split(",")
                lemma_dict[parts[0]] = parts[1]
                lemmas.add(parts[1])
            #endfor
            f.close()
        #endwith
    #endif

    hypernyms = set()
    if hyp_file is not None:
        log.info("Loading mention hypernyms")
        with open(hyp_file, 'r') as f:
            id_hyp_dict = json.load(f)
        for hyps in id_hyp_dict.values():
            if isinstance(hyps, list):
                for h in hyps:
                    hypernyms.add(h)
            else:
                hypernyms.add(hyps)
            #endif
        #endfor
    #endif

    log.info("Evaluating")
    lemma_scores = dict()
    for l in lemmas:
        lemma_scores[l] = ScoreDict()
    hyp_scores = dict()
    for h in hypernyms:
        hyp_scores[h] = ScoreDict()

    y_pred_eval = learner.predict_proba(x_eval)
    scores = ScoreDict()
    pred_scores = dict()
    for idx in range(len(y_pred_eval)):
        id = ids_eval[idx]
        pred_scores[id] = y_pred_eval[idx]

        pred = 0 if pred_scores[id][0] > pred_scores[id][1] else 1
        scores.increment(y_eval[idx], pred)
    #endfor

    log.info("---Confusion matrix---")
    scores.print_confusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.get_score(label).to_string() + " - %d (%.2f%%)" % \
              (scores.get_gold_count(label), scores.get_gold_percent(label))
    log.info(None, "Accuracy: %.2f%%", scores.get_accuracy())

    if save_scores:
        log.info("Writing scores to " + scores_file)
        with open(scores_file, 'w') as f:
            for id in pred_scores.keys():
                score_line = list()
                score_line.append(id)
                for s in pred_scores[id]:
                    score = s
                    if score == 0:
                        score = np.nextafter(0, 1)
                    score = str(np.log(score))
                    score_line.append(score)
                #endfor
                f.write(','.join(score_line) + '\n')
            f.close()
        #endwith
    #endif
#enddef


log = Logger(lvl='debug', delay=45)

models = ['svm', 'logistic', 'decision_tree', 'random_forest']
parser = ArgumentParser("ImageCaptionLearn_py: Nonvisual Mention Classifier")
parser.add_argument("--model", choices=models, default='logistic', help="train opt; Model to train")
parser.add_argument("--max_iter", type=int, default=100, help="train opt; Max SVM/logistic iterations")
parser.add_argument("--max_tree_depth", type=int, default=None, help="train opt; Max decision tree depth")
parser.add_argument("--balance", action='store_true',
                    help="train_opt; Whether to use class weights inversely proportional to the data distro")
parser.add_argument("--num_estimators", type=int, default=100, help="train_opt; Number of trees for random_forest")
parser.add_argument("--warm", action='store_true', help='train_opt; Whether to use warm start')
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
parser.add_argument("--model_file", type=str, required=True,
                    help="Model file to save/load")
args = parser.parse_args()
arg_dict = vars(args)
util.dump_args(arg_dict, log)

# Construct data files from the root directory and filename
data_dir = arg_dict['data_dir'] + "/"
data_root = arg_dict['data_root']
eval_data_root = arg_dict['eval_data_root']
train_file = data_dir + "feats/" + data_root + "_nonvis_classifier.feats"
eval_file = data_dir + "feats/" + eval_data_root + "_nonvis_classifier.feats"
scores_file = data_dir + "scores/" + eval_data_root + "_nonvis_classifier.scores"
meta_file = data_dir + "feats/" + data_root + "_nonvis_classifier_meta.json"
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
model_type = arg_dict['model']
max_iter = arg_dict['max_iter']
max_tree_depth = arg_dict['max_tree_depth']
balance = arg_dict['balance']
num_estimators = arg_dict['num_estimators']
warm_start = arg_dict['warm']

if arg_dict['train']:
    train(model_type, balance, max_iter, max_tree_depth, num_estimators, warm_start)
if arg_dict['predict']:
    evaluate(None, None, set(), True)