from sklearn.linear_model import LogisticRegression
from argparse import ArgumentParser
from os.path import abspath, expanduser
import cPickle
import json
from scipy import stats

from LogUtil import LogUtil
from ScoreDict import ScoreDict
import icl_util as util

"""
Trains the cardinality classifier as a multinomial logistic regression
model with max_iter iterations; optional parameters enable balanced class
weights, warm start, and the ability to ignore features
"""
def train(max_iter, balance=False, warm_start=None, ignored_feats=set()):
    global log, train_file, meta_dict, model_file

    log.tic('info', "Loading training data")
    x, y, ids = util.load_feats_data(train_file, meta_dict, ignored_feats, log)
    log.toc('info')

    log.tic('info', "Training")
    class_weight = None
    if balance:
        class_weight = 'balanced'
    #endif

    learner = LogisticRegression(class_weight=class_weight, solver='lbfgs',
              max_iter=max_iter, multi_class='multinomial', n_jobs=-1, warm_start=warm_start)
    learner.fit(x, y)
    log.toc('info')

    log.info("Saving model")
    with open(model_file, 'wb') as pickle_file:
        cPickle.dump(learner, pickle_file)
#enddef

"""
Evaluates the model, optionally ignoring features and saving
predicted class scores
"""
def evaluate(ignored_feats=set(), save_scores=True):
    global log, eval_file, model_file, scores_file

    log.info("Loading model from file")
    learner = cPickle.load(open(model_file, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = util.load_feats_data(eval_file, meta_dict, ignored_feats, log)

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
    scores.printConfusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.getScore(label).toLatexString() + " & %.2f\\%%\\\\" % \
              (scores.getGoldPercent(label))
    kurtoses = list()
    for log_proba in y_pred_probs:
        kurtoses.append(stats.kurtosis(log_proba, axis=0, fisher=True, bias=True))
    log.info(None, "Accuracy: %.2f%%; Kurtoses: %.2f",
             scores.getAccuracy(), sum(kurtoses) / len(kurtoses))

    if save_scores:
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
    #endif
#enddef


log = LogUtil(lvl='debug', delay=45)

#Parse arguments
parser = ArgumentParser("ImageCaptionLearn_py: Box Cardinality Classifier")
parser.add_argument("--train", action='store_true', help="Trains and saves a model")
parser.add_argument("--eval", action='store_true', help="Evaluates the saved model")
parser.add_argument("--ablation", type=str, help="Performs ablation, removing the specified " +
                                                 "pipe-separated features (or 'all', for all features)")
parser.add_argument("--max_iter", type=int, default=100, help="train opt; Max SVM/logistic iterations")
parser.add_argument("--balance", action='store_true',
                    help="train_opt; Whether to use class weights inversely proportional to the data distro")
parser.add_argument("--warm", action='store_true', help='train_opt; Whether to use warm start')
parser.add_argument("--save_scores", action='store_true', help='eval opt; Saves scores in eval_file ' +
                                                               '(.scores instead of .feats)')
parser.add_argument("--train_file", type=str, help="Train feature file")
parser.add_argument("--eval_file", type=str, help="Eval feature file")
parser.add_argument("--model_file", type=str, help="File to save learned model")
parser.add_argument("--meta_file", type=str, help="Meta feature file (typically associated with train file")
args = parser.parse_args()
arg_dict = vars(args)
util.dump_args(arg_dict, log)

'''
train_file = abspath(expanduser("~/source/data/feats/card_20170214.feats"))
eval_file = abspath(expanduser("~/source/data/feats/card_test_20170214.feats"))
scores_file = eval_file.replace(".feats", ".scores")
model_file = "models/box_card.model"
meta_file = abspath(expanduser("~/source/data/feats/card_train_20170214_meta.json"))
'''

train_file = abspath(expanduser(arg_dict['train_file']))
eval_file = abspath(expanduser(arg_dict['eval_file']))
scores_file = eval_file.replace(".feats", ".scores")
model_file = abspath(expanduser(arg_dict['model_file']))
meta_file = abspath(expanduser(arg_dict['meta_file']))
meta_dict = json.load(open(meta_file, 'r'))
max_iter = arg_dict['max_iter']
balance = arg_dict['balance']
warm_start = arg_dict['warm']

# Ablation testing found hypernym box doesn't work very well
fixed_ignored = {"hypernym_bow"}

evaluate(fixed_ignored)

if arg_dict['train']:
    train(max_iter, balance, warm_start, fixed_ignored)
if arg_dict['eval']:
    evaluate(fixed_ignored)

#Ablate
ablation_str = arg_dict['ablation']
if ablation_str is not None:
    ablation_feats = set()
    if ablation_str == 'all':
        for feat in meta_dict.keys():
            if feat != "max_idx":
                ablation_feats.add(feat)
    else:
        for feat in ablation_str.split("|"):
            if feat in meta_dict.keys():
                ablation_feats.add(feat)
            else:
                log.error(None, "Specified unknown ablation feature '%s'; ignoring", feat)
    #endif

    if ablation_str == 'all':
        log.info('Baseline (ignoring no features)')
        train(max_iter, balance, warm_start, set())
        evaluate(set(), False)
    #endif

    log.info("Running ablation over the following features")
    print "|".join(ablation_feats)

    for feat in ablation_feats:
        log.info(None, "---- Removing feature %s ----", feat)
        ignored = set(); ignored.add(feat)
        train(max_iter, balance, warm_start, ignored)
        evaluate(ignored, False)
    #endfor
#endif

