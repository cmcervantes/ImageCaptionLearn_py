import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import cPickle
from os.path import abspath, expanduser, isfile
from argparse import ArgumentParser
from ScoreDict import ScoreDict
from LogUtil import LogUtil
import icl_util as util

_NONVIS_FILE = "/home/ccervan2/source/PanOpt/flickr30kEntities_v2_nonvis_dev.feats"
_PAIRWISE_IDX_LIMIT = 1050823 # actual max idx: 1050822
solvers = ['lbfgs', 'newton-cg', 'sag']
multiclass_modes = ['multinomial', 'ovr']

"""
Trains a multiclass classifier on the given trainFile, using the
specified opts
"""
def train(trainFile, modelFile, solver, max_iter, balance_train, norm, warm_start, multiclass_mode):
    global log, _PAIRWISE_IDX_LIMIT

    log.tic('info', "Loading training data")
    X_train, Y_train, IDs_train = util.load_feats_data(trainFile, _PAIRWISE_IDX_LIMIT, log)
    if norm is not None:
        normalize(X_train, norm=norm, copy=False)
    log.toc('info')

    log.tic('info', "Training")
    class_weight = None
    if balance_train:
        class_weight = 'balanced'
    #endif
    logistic = LogisticRegression(class_weight=class_weight, solver=solver,
                 max_iter=max_iter, multi_class=multiclass_mode, n_jobs=-1, verbose=1,
                 warm_start=warm_start)
    logistic.fit(X_train, Y_train)
    log.toc('info')

    log.info("Saving model")
    with open(modelFile, 'wb') as pickle_file:
        cPickle.dump(logistic, pickle_file)
#enddef

"""
Evaluates the model (in model file) against the
eval data
"""
def evaluate(evalFile, modelFile, norm):
    global log, _PAIRWISE_IDX_LIMIT

    log.info("Loading model from file")
    logistic = cPickle.load(open(modelFile, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = util.load_feats_data(evalFile, _PAIRWISE_IDX_LIMIT, log)
    if norm is not None:
        normalize(x_eval, norm=norm, copy=False)

    log.info("Loading gold nonvis data from file")
    x_nonvis, y_nonvis, ids_nonvis = util.load_feats_data(_NONVIS_FILE, 10364, log)
    ids_nonvis_gold = set()
    for i in range(len(ids_nonvis)):
        if y_nonvis[i] == 1:
            ids_nonvis_gold.add(ids_nonvis[i])
        #endif
    #endfor

    log.info("Evaluating")
    Y_pred_dev = logistic.predict(x_eval)
    scores = ScoreDict()
    mistake_dict = dict()
    for i in range(len(y_eval)):
        # pairwise ids are in the form
        # doc:<img_id>;caption_1:<cap_idx>;mention_1:<mention_idx>;caption_2:<cap_idx>;mention_2:<mention_idx>
        # and nonvis ids are in the form
        # <img_id>#<cap_idx>;mention:<mention_idx>
        # So we need to do some processing
        id = ids_eval[i]
        id_parts = id.split(";")
        img_id = id_parts[0].replace("doc:", "")
        id_1 = img_id + "#" + id_parts[1].replace("caption_1:", "") + ";" + id_parts[2].replace("_1", "")
        id_2 = img_id + "#" + id_parts[3].replace("caption_2:", "") + ";" + id_parts[4].replace("_2", "")

        #now ignore any pair in which a nonvisual appears (as these do not have identity relations)
        if id_1 not in ids_nonvis_gold and id_2 not in ids_nonvis_gold:
            scores.increment(y_eval[i], Y_pred_dev[i])
            if y_eval[i] != Y_pred_dev[i]:
                label_pair = (y_eval[i], Y_pred_dev[i])
                if label_pair not in mistake_dict.keys():
                    mistake_dict[label_pair] = list()
                mistake_dict[label_pair].append((id_1, id_2))
            #endif
        #endif
    #endfor

    log.info("Writing mistakes in pairwise_mistakes.csv")
    with open('pairwise_mistakes.csv', 'w') as f:
        f.write("Gold,Pred,Mention_1,Mention_2")
        for label_pair in mistake_dict.keys():
            for id_pair in mistake_dict[label_pair]:
                f.write("%d,%d,%s,%s\n" % (label_pair[0], label_pair[1], id_pair[0], id_pair[1]))
            #endfor
        #endfor
        f.close()
    #endwith

    log.info("---Confusion matrix---")
    scores.printConfusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.getScore(label).toString() + " - %d (%.2f%%)" % \
                (scores.getGoldCount(label), scores.getGoldPercent(label))
    #endfor
#enddef

def tune(trainFile, evalFile):
    log.tic('info', 'Loading data')
    X_train, Y_train, IDs_train = util.load_feats_data(trainFile, _PAIRWISE_IDX_LIMIT, log)
    x_eval, y_eval, ids_eval = util.load_feats_data(evalFile, _PAIRWISE_IDX_LIMIT, log)
    x_nonvis, y_nonvis, ids_nonvis = util.load_feats_data(_NONVIS_FILE, 10364, log)
    ids_nonvis_gold = set()
    for i in range(len(ids_nonvis)):
        if y_nonvis[i] == 1:
            ids_nonvis_gold.add(ids_nonvis[i])
            #endif
    #endfor
    log.toc('info')

    for slvr in solvers:
        for mcc_mode in multiclass_modes:
            for balance in [True, False]:
                for warm in [True, False]:
                    if slvr == 'sag' and mcc_mode=='multinomial':
                        continue # sag is only usable with ovr
                    log.info(None, "----slvr:%s; mode:%s; balance:%s; warm:%s----",
                             slvr, mcc_mode, str(balance), str(warm))

                    log.tic('info', "Training")
                    class_weight = None
                    if balance:
                        class_weight = 'balanced'
                    #endif
                    logistic = LogisticRegression(class_weight=class_weight, solver=slvr,
                                                  max_iter=1000, multi_class=mcc_mode, n_jobs=-1, verbose=1,
                                                  warm_start=warm)
                    logistic.fit(X_train, Y_train)
                    log.toc('info')

                    log.info("Evaluating")
                    y_pred_eval = logistic.predict(x_eval)
                    scores = ScoreDict()
                    for i in range(len(y_eval)):
                        # pairwise ids are in the form
                        # doc:<img_id>;caption_1:<cap_idx>;mention_1:<mention_idx>;caption_2:<cap_idx>;mention_2:<mention_idx>
                        # and nonvis ids are in the form
                        # <img_id>#<cap_idx>;mention:<mention_idx>
                        # So we need to do some processing
                        id = ids_eval[i]
                        id_parts = id.split(";")
                        img_id = id_parts[0].replace("doc:", "")
                        id_1 = img_id + "#" + id_parts[1].replace("caption_1:", "") + ";" + id_parts[2].replace("_1", "")
                        id_2 = img_id + "#" + id_parts[3].replace("caption_2:", "") + ";" + id_parts[4].replace("_2", "")

                        #now ignore any pair in which a nonvisual appears (as these do not have identity relations)
                        if id_1 not in ids_nonvis_gold and id_2 not in ids_nonvis_gold:
                            scores.increment(y_eval[i], y_pred_eval[i])
                    #endfor

                    log.info("---Confusion matrix---")
                    scores.printConfusion()

                    log.info("---Scores---")
                    for label in scores.keys:
                        print str(label) + "\t" + scores.getScore(label).toString() + " - %d (%.2f%%)" % \
                                  (scores.getGoldCount(label), scores.getGoldPercent(label))
                    #endfor
                #endfor
            #endfor
        #endfor
    #endfor
#enddef

def init():
    global log

    #parse args
    parser = ArgumentParser("ImageCaptionLearn_py: Pairwise Mention Identity Classifier")
    parser.add_argument('--log_lvl', choices=LogUtil.get_logging_lvls(), default='debug',
                        help="log opt; Default logger level")
    parser.add_argument("--log_delay", type=int, default=30,
                        help="log opt; log status delay")
    parser.add_argument("--norm", type=str, help="preproc opt; Specify data normalization")
    parser.add_argument("--train", type=str, help="train opt; Trains with the specified train data")
    parser.add_argument("--solver", choices=solvers, default=solvers[0],
                        help="train opt; Multiclass solver to use")
    parser.add_argument("--warm", action='store_true', help="train opt; Uses previous solution as init")
    parser.add_argument("--max_iter", type=int, default=100, help="train opt; Specifies the max iterations")
    parser.add_argument("--balance_train", action='store_true',
                        help="train_opt; Whether to use class weights inversely proportional to the data distro")
    parser.add_argument("--mcc_mode", choices=multiclass_modes, default=multiclass_modes[0],
                        help="train opt; multiclass mode")
    parser.add_argument("--model", type=str,
                        help="train/eval opt; Specifies a model file to save / load, depending on task")
    parser.add_argument("--eval", type=str, help="eval opt; Evaluates with the specified eval file")
    args = parser.parse_args()
    arg_dict = vars(args)

    # Set up up the logger
    log = LogUtil(lvl=arg_dict['log_lvl'], delay=arg_dict['log_delay'])
    util.dump_args(arg_dict, log)

    # Train / eval
    trainFile = arg_dict['train']
    if trainFile is not None:
        trainFile = abspath(expanduser(trainFile))
    modelFile = arg_dict['model']
    if modelFile is not None:
        modelFile = abspath(expanduser(modelFile))
    evalFile = arg_dict['eval']
    if evalFile is not None:
        evalFile = abspath(expanduser(evalFile))

    validArgSet = False
    if modelFile is not None:
        if (trainFile is not None and isfile(trainFile)) or \
           (evalFile is not None and isfile(evalFile)):
            validArgSet = True
        #endif
    #endif

    if not validArgSet:
        log.error("Must specify a model file and either a train file or an eval file")
        parser.print_usage()
        quit()
    #endif

    #tune(trainFile, evalFile)
    #quit()

    norm = arg_dict['norm']
    if trainFile is not None:
        train(trainFile, modelFile, arg_dict['solver'], arg_dict['max_iter'],
              arg_dict['balance_train'], norm, arg_dict['warm'], arg_dict['mcc_mode'])
    if evalFile is not None:
        evaluate(evalFile, modelFile, norm)
    #endif
#enddef

init()
