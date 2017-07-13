from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import cPickle
from os.path import abspath, expanduser
from argparse import ArgumentParser
from ScoreDict import ScoreDict
from LogUtil import LogUtil
import icl_util as util
import json

"""
Trains the relation model as a multinomial logistic regression model
"""
def train(solver, max_iter, balance, norm, warm_start, multiclass_mode, ignored_feats=set()):
    global log, meta_dict, train_file, model_file

    log.tic('info', "Loading training data")
    x, y, ids = util.load_feats_data(train_file, meta_dict, ignored_feats, log)
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

"""
Saves predicted scores to file
"""
def save_scores(ignored_feats=set()):
    global log, eval_file, model_file, scores_file, meta_dict

    log.info("Loading model from file")
    learner = cPickle.load(open(model_file, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = \
        util.load_feats_data(eval_file, meta_dict,
                             ignored_feats, log)

    log.info("Predicting scores")
    y_pred_probs = learner.predict_log_proba(x_eval)

    log.info("Writing probabilities to " + scores_file)
    with open(scores_file, 'w') as f:
        for i in range(len(ids_eval)):
            line = list()
            line.append(ids_eval[i])
            for j in range(len(y_pred_probs[i])):
                line.append(str(y_pred_probs[i][j]))
            f.write(','.join(line) + '\n')
        f.close()
    #endwith
#enddef

"""
Evaluates the saved model against the eval file
"""
def evaluate(norm, ignored_feats=set()):
    global log, eval_file, model_file, nonvis_file, scores_file, meta_dict

    log.info("Loading model from file")
    logistic = cPickle.load(open(model_file, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = \
        util.load_feats_data(eval_file, meta_dict,
                             ignored_feats, log)
    if norm is not None:
        normalize(x_eval, norm=norm, copy=False)
    #endif

    log.info("Loading gold nonvis data from file")
    ids_nonvis_gold = set()
    if nonvis_file is not None:
        ids_nonvis_gold = load_nonvis_ids()

    log.info("Evaluating")
    y_pred_probs = logistic.predict_log_proba(x_eval)
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
            y_pred = 0
            max_prob = -float('inf')
            for j in range(len(y_pred_probs[i])):
                if y_pred_probs[i][j] > max_prob:
                    max_prob = y_pred_probs[i][j]
                    y_pred = j

            scores.increment(y_eval[i], y_pred)
            if y_eval[i] != y_pred:
                label_pair = (y_eval[i], y_pred)
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

    if scores_file is not None:
        log.info("Writing probabilities to file")
        f = open(scores_file, 'w')
        for i in range(len(ids_eval)):
            line = list()
            line.append(ids_eval[i])
            for j in range(len(y_pred_probs[i])):
                line.append(str(y_pred_probs[i][j]))
            f.write(','.join(line) + '\n')
        #endfor
    #endif

    print "Acc: " + str(scores.getAccuracy()) + "%"
#enddef

"""
Loads nonvisual IDs
"""
def load_nonvis_ids():
    global nonvis_file
    ids_nonvis_gold = set()
    with open(nonvis_file, 'r') as f:
        f.seek(0)
        for line in f:
            commentSplit = line.split(" # ")
            vectorSplit = commentSplit[0].strip().split(" ")
            if int(float(vectorSplit[0])) == 1:
                ids_nonvis_gold.add(commentSplit[1].strip())
        #endfor
        f.close()
    #endwith
    return ids_nonvis_gold
#enddef


"""
Tunes the parameters over the model
"""
def tune():
    global train_file, eval_file, meta_dict

    log.tic('info', 'Loading data')
    x_train, y_train, ids_train = util.load_feats_data(train_file, meta_dict['max_idx'], log)
    x_eval, y_eval, ids_eval = util.load_feats_data(eval_file, meta_dict['max_idx'], log)
    ids_nonvis_gold = load_nonvis_ids()
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
                    logistic.fit(x_train, y_train)
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

# At one time I had more arguments, but in the end it's much
# easier not to specify all this on the command line
log = LogUtil(lvl='debug', delay=45)

# Parse what arguments remain
solvers = ['lbfgs', 'newton-cg', 'sag']
multiclass_modes = ['multinomial', 'ovr']
parser = ArgumentParser("ImageCaptionLearn_py: Pairwise Relation Classifier")
parser.add_argument("--norm", type=str, help="preproc opt; Specify data normalization")
parser.add_argument("--train", action='store_true', help="Trains and saves a model")
parser.add_argument("--eval", action='store_true', help="Evaluates using a saved model")
parser.add_argument("--ablation", type=str, help="Performs ablation, removing the specified " +
                    "pipe-separated features (or 'all', for all features)")
parser.add_argument("--solver", choices=solvers, default=solvers[0],
                    help="train opt; Multiclass solver to use")
parser.add_argument("--warm", action='store_true', help="train opt; Uses previous solution as init")
parser.add_argument("--max_iter", type=int, default=100, help="train opt; Specifies the max iterations")
parser.add_argument("--balance", action='store_true',
                    help="train_opt; Whether to use class weights inversely proportional to the data distro")
parser.add_argument("--mcc_mode", choices=multiclass_modes, default=multiclass_modes[0],
                    help="train opt; multiclass mode")
parser.add_argument("--train_file", type=str, help="train feats file")
parser.add_argument("--eval_file", type=str, help="eval feats file")
parser.add_argument("--meta_file", type=str, help="meta feature file (typically associated with train file)")
parser.add_argument("--model_file", type=str, help="saves model to file")
parser.add_argument("--nonvis_file", type=str, help="Retrieves nonvis labels from file; excludes from eval")
args = parser.parse_args()
arg_dict = vars(args)
util.dump_args(arg_dict, log)
train_file = None
if arg_dict['train_file'] is not None:
    train_file = abspath(expanduser(arg_dict['train_file']))
eval_file = None
if arg_dict['eval_file'] is not None:
    eval_file = abspath(expanduser(arg_dict['eval_file']))
meta_file = abspath(expanduser(arg_dict['meta_file']))
model_file = abspath(expanduser(arg_dict['model_file']))
scores_file = eval_file.replace(".feats", ".scores")
nonvis_file = None
meta_nonvis_file = None
if arg_dict['nonvis_file'] is not None:
    nonvis_file = abspath(expanduser(arg_dict['nonvis_file']))
    meta_nonvis_file = nonvis_file.replace(".feats", "_meta.json")
#endif

# load the meta dict
meta_dict = json.load(open(meta_file, 'r'))

log.info("Ignoring hypernym_bow")
fixed_ignored = {"hypernym_bow"}

# Train
if arg_dict['train']:
    train(arg_dict['solver'], arg_dict['max_iter'],
          arg_dict['balance'], arg_dict['norm'],
          arg_dict['warm'], arg_dict['mcc_mode'],
          fixed_ignored)
    save_scores(fixed_ignored)
# Evaluate
if arg_dict['eval']:
    evaluate(arg_dict['norm'], fixed_ignored)
    save_scores(fixed_ignored)

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
        log.info('Baseline (no ignored features)')
        train(arg_dict['solver'], arg_dict['max_iter'], arg_dict['balance'],
              arg_dict['norm'], arg_dict['warm'], arg_dict['mcc_mode'])
        evaluate(arg_dict['norm'])
    #endif

    log.info("Running ablation over the following features")
    print "|".join(ablation_feats)

    for feat in ablation_feats:
        log.info(None, "---- Removing feature %s ----", feat)
        ignored = set(); ignored.add(feat)
        train(arg_dict['solver'], arg_dict['max_iter'], arg_dict['balance'],
              arg_dict['norm'], arg_dict['warm'], arg_dict['mcc_mode'], ignored)
        evaluate(arg_dict['norm'], ignored)
    #endfor
#endif