from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from argparse import ArgumentParser
from os.path import abspath, expanduser, isfile
import cPickle

from LogUtil import LogUtil
from ScoreDict import ScoreDict
import icl_util as util

_NONVIS_IDX_LIMIT = 10364 # actual max idx: 10363


def train(trainFile, modelFile, model, balance_train, max_iter=None, max_depth=None):
    global log, _NONVIS_IDX_LIMIT

    log.tic('info', "Loading training data")
    X_train, Y_train, IDs_train = util.load_feats_data(trainFile, _NONVIS_IDX_LIMIT, log)
    log.toc('info')

    log.tic('info', "Training")
    class_weight = None
    if balance_train:
        class_weight = 'balanced'
    #endif
    learner = None
    if model == 'svm':
        learner = SVC(probability=True, class_weight=class_weight, verbose=1, max_iter=max_iter)
    elif model == 'logistic':
        learner = LogisticRegression(class_weight=class_weight, max_iter=max_iter, verbose=1, n_jobs=-1)
    elif model == "decision_tree":
        learner = DecisionTreeClassifier(max_depth=max_depth, class_weight=class_weight)
    #endif
    learner.fit(X_train, Y_train)
    log.toc('info')

    log.info("Saving model")
    with open(modelFile, 'wb') as pickle_file:
        cPickle.dump(learner, pickle_file)
#enddef

"""
Evaluates the model (in model file) against the
eval data
"""
def evaluate(eval_file, model_file):
    global log, _NONVIS_IDX_LIMIT

    log.info("Loading model from file")
    learner = cPickle.load(open(model_file, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = util.load_feats_data(eval_file, _NONVIS_IDX_LIMIT, log)

    log.info("Evaluating")
    y_pred_eval = learner.predict_proba(x_eval)
    scores = ScoreDict()
    pred_scores = dict()
    for idx in range(len(y_pred_eval)):
        predictions = y_pred_eval[idx]
        pred = None
        max_prob = 0.0
        for label in range(len(predictions)):
            if predictions[label] > max_prob:
                pred = label
                max_prob = predictions[label]
            #endif
        #endfor
        scores.increment(y_eval[idx], pred)
        if pred == 0:
            max_prob *= -1
        #endif
        pred_scores[ids_eval[idx]] = max_prob
    #endfor

    log.info("---Confusion matrix---")
    scores.printConfusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.getScore(label).toString() + " - %d (%.2f%%)" % \
                                                                      (scores.getGoldCount(label), scores.getGoldPercent(label))
    #endfor

    log.info("Writing scores to nonvis.scores")
    f = open('nonvis.scores', 'w')
    for id in pred_scores.keys():
        f.write(",".join((id, str(pred_scores[id]))))
    #endfor
    f.close()
#enddef



def init():
    global log

    models = ['svm', 'logistic', 'decision_tree']

    #parse args
    parser = ArgumentParser("ImageCaptionLearn_py: Nonvisual Mention Classifier")
    parser.add_argument('--log_lvl', choices=LogUtil.get_logging_lvls(), default='debug',
                        help="log opt; Default logger level")
    parser.add_argument("--log_delay", type=int, default=30,
                        help="log opt; log status delay")
    parser.add_argument("--train", type=str, help="train opt; Trains with the specified train data")
    parser.add_argument("--model", choices=models, help="train opt; Model to train")
    parser.add_argument("--max_iter", type=int, default=100, help="train opt; Max SVM/logistic iterations")
    parser.add_argument("--max_tree_depth", type=int, default=None, help="train opt; Max decision tree depth")
    parser.add_argument("--balance_train", action='store_true',
                        help="train_opt; Whether to use class weights inversely proportional to the data distro")
    parser.add_argument("--model_file", type=str,
                        help="train/eval opt; Specifies a model file to save / load, depending on task")
    parser.add_argument("--eval", type=str, help="eval opt; Evaluates with the specified eval file")
    args = parser.parse_args()
    arg_dict = vars(args)

    # Set up up the logger
    log_lvl = arg_dict['log_lvl']
    log_delay = arg_dict['log_delay']
    log = LogUtil(lvl=log_lvl, delay=log_delay)
    util.dump_args(arg_dict, log)

    # Train / eval
    train_file = arg_dict['train']
    if train_file is not None:
        train_file = abspath(expanduser(train_file))
    model_file = arg_dict['model']
    if model_file is not None:
        model_file = abspath(expanduser(model_file))
    eval_file = arg_dict['eval']
    if eval_file is not None:
        eval_file = abspath(expanduser(eval_file))

    validArgSet = False
    if model_file is not None:
        if (train_file is not None and isfile(train_file)) or \
                (eval_file is not None and isfile(eval_file)):
            validArgSet = True
        #endif
    #endif

    if not validArgSet:
        log.error("Must specify a model file and either a train file or an eval file")
        parser.print_usage()
        quit()
    #endif

    if train_file is not None:
        train(train_file, model_file, arg_dict['model'], arg_dict['balance_train'],
              arg_dict['max_iter'], arg_dict['max_tree_depth'])
    if eval_file is not None:
        evaluate(eval_file, model_file)
    #endif
#enddef

init()