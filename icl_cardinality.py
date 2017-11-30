import cPickle
import json
from argparse import ArgumentParser
from os.path import abspath, expanduser

from sklearn.linear_model import LogisticRegression
from scipy import stats
from utils.Logger import Logger
from utils import core as util
from utils import data as data_util
from utils.ScoreDict import ScoreDict


def train(max_iter, balance=False, warm_start=None, ignored_feats=set()):
    """
    Trains the cardinality classifier as a multinomial logistic regression
    model with max_iter iterations; optional parameters enable balanced class
    weights, warm start, and the ability to ignore features
    :param max_iter:
    :param balance:
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

    learner = LogisticRegression(class_weight=class_weight,
                                 solver='lbfgs',
                                 max_iter=max_iter,
                                 multi_class='multinomial',
                                 n_jobs=-1, warm_start=warm_start)
    #learner = mord.OrdinalRidge(max_iter=max_iter)

    learner.fit(x, y)
    log.toc('info')

    log.info("Saving model")
    with open(model_file, 'wb') as pickle_file:
        cPickle.dump(learner, pickle_file)
#enddef


def evaluate(ignored_feats=set()):
    """
    Evaluates the model, optionally ignoring features and saving
    predicted class scores
    :param ignored_feats:
    :return:
    """
    global log, eval_file, model_file, scores_file

    log.info("Loading model from file")
    learner = cPickle.load(open(model_file, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = \
        data_util.load_very_sparse_feats(eval_file, meta_dict,
                                         ignored_feats)

    log.info("Evaluating")
    y_pred_probs = learner.predict_log_proba(x_eval)
    scores = ScoreDict()
    for i in range(len(y_eval)):
        y_pred = 0
        max_prob = -float('inf')
        for j in range(len(y_pred_probs[i])):
            if y_pred_probs[i][j] > max_prob:
                max_prob = y_pred_probs[i][j]
                y_pred = j
            #endif
        #endfor

        scores.increment(y_eval[i], y_pred)
    #endfor

    log.info("---Confusion matrix---")
    scores.print_confusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.get_score(label).to_latex_string() + " & %.2f\\%%\\\\" % \
                                                                            (scores.get_gold_percent(label))
    kurtoses = list()
    for log_proba in y_pred_probs:
        kurtoses.append(stats.kurtosis(log_proba, axis=0, fisher=True, bias=True))
    log.info(None, "Accuracy: %.2f%%; Kurtoses: %.2f",
             scores.get_accuracy(), sum(kurtoses) / len(kurtoses))

    log.info("Writing probabilities to file")
    with open(scores_file, 'w') as f:
        for i in range(len(ids_eval)):
            line = list()
            line.append(ids_eval[i])
            for j in range(len(y_pred_probs[i])):
                line.append(str(y_pred_probs[i][j]))
            f.write(','.join(line) + '\n')
        #endfor
    #endwith
#enddef


log = Logger(lvl='debug', delay=45)

#Parse arguments
parser = ArgumentParser("ImageCaptionLearn_py: Box Cardinality Classifier")
parser.add_argument("--max_iter", type=int, default=100, help="train opt; Max SVM/logistic iterations")
parser.add_argument("--balance", action='store_true',
                    help="train_opt; Whether to use class weights inversely proportional to the data distro")
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
train_file = data_dir + "feats/" + data_root + "_card_classifier.feats"
eval_file = data_dir + "feats/" + eval_data_root + "_card_classifier.feats"
scores_file = data_dir + "scores/" + eval_data_root + "_card_classifier.scores"
meta_file = data_dir + "feats/" + data_root + "_card_classifier_meta.json"
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

if arg_dict['train']:
    train(max_iter, balance, warm_start, set())
if arg_dict['predict']:
    evaluate(set())
