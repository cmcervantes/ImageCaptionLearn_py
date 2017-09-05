from sklearn.linear_model import LogisticRegression
from argparse import ArgumentParser
from os.path import abspath, expanduser
import cPickle
import json
from scipy import stats
import mord

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
    x, y, ids = util.load_sparse_feats(train_file, meta_dict, ignored_feats, log)
    log.toc('info')

    log.tic('info', "Training")
    class_weight = None
    if balance:
        class_weight = 'balanced'
    #endif

    '''
    learner = LogisticRegression(class_weight=class_weight, solver='lbfgs',
              max_iter=max_iter, multi_class='multinomial', n_jobs=-1, warm_start=warm_start)
    '''
    learner = mord.OrdinalRidge(max_iter=max_iter)

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
def evaluate(ignored_feats=set()):
    global log, eval_file, model_file, scores_file

    log.info("Loading model from file")
    learner = cPickle.load(open(model_file, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = util.load_sparse_feats(eval_file, meta_dict, ignored_feats, log)

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







log = LogUtil(lvl='debug', delay=45)

#Parse arguments
parser = ArgumentParser("ImageCaptionLearn_py: Box Cardinality Classifier")
parser.add_argument("--max_iter", type=int, default=100, help="train opt; Max SVM/logistic iterations")
parser.add_argument("--balance", action='store_true',
                    help="train_opt; Whether to use class weights inversely proportional to the data distro")
parser.add_argument("--warm", action='store_true', help='train_opt; Whether to use warm start')
parser.add_argument("--train_file", type=str, help="Train feature file")
parser.add_argument("--eval_file", type=str, help="Eval feature file")
parser.add_argument("--model_file", type=str, help="File to save learned model")
parser.add_argument("--meta_file", type=str, help="Meta feature file (typically associated with train file")
parser.add_argument("--ablation_file", type=str, help="Performs ablation, using the groupings specified "
                                                      "in the given ablation config file ")
args = parser.parse_args()
arg_dict = vars(args)
util.dump_args(arg_dict, log)


# Parse our file paths
train_file = arg_dict['train_file']
if train_file is not None:
    train_file = abspath(expanduser(train_file))
eval_file = arg_dict['eval_file']
scores_file = None
if eval_file is not None:
    eval_file = abspath(expanduser(eval_file))
    scores_file = eval_file.replace(".feats", ".scores")
meta_file = arg_dict['meta_file']
meta_dict = None
if meta_file is not None:
    meta_file = abspath(expanduser(meta_file))
    meta_dict = json.load(open(meta_file, 'r'))
model_file = arg_dict['model_file']
if model_file is not None:
    model_file = abspath(expanduser(model_file))
ablation_file = arg_dict['ablation_file']
ablation_groups = None
if ablation_file is not None:
    ablation_file = abspath(expanduser(ablation_file))
    ablation_groups = util.load_ablation_file(ablation_file)

# Parse the other args
max_iter = arg_dict['max_iter']
balance = arg_dict['balance']
warm_start = arg_dict['warm']

# Ensure we don't have an invalid collection of options
if train_file is not None and (meta_file is None or model_file is None):
    log.critical("Specified train_file without meta or model files; exiting")
    parser.print_usage()
    quit()
if eval_file is not None and model_file is None:
    log.critical("Specified eval_file without model_file; exiting")
    parser.print_usage()
    quit()
if ablation_file is not None and (train_file is None or eval_file is None or meta_file is None):
    log.critical("Specified ablation_file without train, eval, or meta file; exiting")
    parser.print_usage()
    quit()
if train_file is None and eval_file is None:
    log.critical("Did not specify train or eval file; exiting")
    parser.print_usage()
    quit()


# If an ablation file was given priority goes to that operation
if ablation_file is not None:
    log.info("Running ablation testing")

    log.info("---------- Baseline ----------")
    train(max_iter, balance, warm_start, set())
    evaluate(set())

    for ablation_feats in ablation_groups:
        ablation_feats_str = "|".join(ablation_feats)
        log.info(None, "---------- Removing %s ----------", ablation_feats_str)
        train(max_iter, balance, warm_start, ablation_feats)
        evaluate(ablation_feats)
    #endfor
else:
    if train_file is not None:
        train(max_iter, balance, warm_start, set())
    if eval_file is not None:
        evaluate(set())
#endif
