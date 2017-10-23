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
from utils.LogUtil import LogUtil

from utils import icl_util as util, icl_data_util
from utils.ScoreDict import ScoreDict

#from icl_affinity import load_cca_data

"""
Trains the model
"""
def train(model, balance, max_iter=None, max_depth=None,
          num_estimators=None, warm_start=None, ignored_feats=set()):
    global log, train_file, meta_dict, model_file

    log.tic('info', "Loading training data")
    x, y, ids = icl_data_util.load_sparse_feats(train_file, meta_dict, ignored_feats, log)

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

"""
Evaluates the model against the eval data
"""
def evaluate(lemma_file=None, hyp_file=None, ignored_feats=set(), save_scores=True):
    global log, eval_file, model_file, scores_file

    log.info("Loading model from file")
    learner = cPickle.load(open(model_file, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = icl_data_util.load_sparse_feats(eval_file, meta_dict, ignored_feats, log)

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
        pred_scores[id] = max_prob

        if lemma_file is not None and hyp_file is not None:
            lemma = lemma_dict[id]
            lemma_scores[lemma].increment(y_eval[idx], pred)
            for hyps in id_hyp_dict[id]:
                if isinstance(hyps, list):
                    for h in hyps:
                        hyp_scores[h].increment(y_eval[idx], pred)
                else:
                    hyp_scores[hyps].increment(y_eval[idx], pred)
            #endfor
        #endif
    #endfor

    if lemma_file is not None:
        with open('nonvisual_scores_lemma.csv', 'w') as f:
            f.write("lemma,gold_freq,gold_vis,gold_nonvis,pred_vis,pred_nonvis,vis_r,vis_p,vis_f1,"+\
                    "nonvis_r,nonvis_p,nonvis_f1,accuracy,correct_count\n")
            for l in lemma_scores.keys():
                s = lemma_scores[l]
                totalCount = s.get_gold_count(0) + s.get_gold_count(1)
                if totalCount > 0:
                    line = l + "," + str(totalCount) + ","
                    line += str(s.get_gold_count(0)) + "," + str(s.get_gold_count(1)) + ","
                    line += str(s.get_pred_count(0)) + "," + str(s.get_pred_count(1)) + ","
                    score_vis = s.get_score(0)
                    score_nonvis = s.get_score(1)
                    line += str(score_vis.r) + "," + str(score_vis.p) + "," + str(score_vis.f1) + ","
                    line += str(score_nonvis.r) + "," + str(score_nonvis.p) + "," + str(score_nonvis.f1) + ","
                    line += str(s.get_accuracy() / 100.0) + "," + str(s.get_correct_count())
                    line += "\n"
                    f.write(line)
            f.close()
    if hyp_file is not None:
        with open('nonvisual_scores_hyp.csv', 'w') as f:
            f.write("hypernym,gold_freq,gold_vis,gold_nonvis,pred_vis,pred_nonvis,vis_r,vis_p,vis_f1,"+ \
                    "nonvis_r,nonvis_p,nonvis_f1,accuracy,correct_count\n")
            for h in hyp_scores.keys():
                s = hyp_scores[h]
                totalCount = s.get_gold_count(0) + s.get_gold_count(1)
                if totalCount > 0:
                    line = h + "," + str(totalCount) + ","
                    line += str(s.get_gold_count(0)) + "," + str(s.get_gold_count(1)) + ","
                    line += str(s.get_pred_count(0)) + "," + str(s.get_pred_count(1)) + ","
                    score_vis = s.get_score(0)
                    score_nonvis = s.get_score(1)
                    line += str(score_vis.r) + "," + str(score_vis.p) + "," + str(score_vis.f1) + ","
                    line += str(score_nonvis.r) + "," + str(score_nonvis.p) + "," + str(score_nonvis.f1) + ","
                    line += str(s.get_accuracy() / 100.0) + "," + str(s.get_correct_count())
                    line += "\n"
                    f.write(line)
            f.close()

    log.info("---Confusion matrix---")
    scores.print_confusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.get_score(label).toString() + " - %d (%.2f%%)" % \
                                                                       (scores.get_gold_count(label), scores.get_gold_percent(label))
    log.info(None, "Accuracy: %.2f%%", scores.get_accuracy())

    if save_scores:
        log.info("Writing scores to " + scores_file)
        with open(scores_file, 'w') as f:
            for id in pred_scores.keys():
                f.write(','.join((id, str(pred_scores[id]))) + '\n')
            f.close()
        #endwith
    #endif
#enddef


"""

"""
def nonvis_with_grounding(max_iter):
    global eval_file, meta_dict, log

    ### Affinity ###
    fit_id_file = abspath(expanduser("~/source/data/feats/cca_ids_dev.txt"))
    fit_scores_file = abspath(expanduser("~/source/data/feats/cca_scores_dev.txt"))
    fit_label_file = abspath(expanduser("~/source/data/feats/cca_labels_dev.txt"))
    fit_type_file = abspath(expanduser("~/source/data/feats/cca_types_dev.csv"))
    eval_id_file = fit_id_file
    eval_scores_file = fit_scores_file
    eval_label_file = fit_label_file
    eval_type_file = fit_type_file
    type_id_dict_eval, type_x_dict_eval, type_y_dict_eval = \
        load_cca_data(eval_id_file, eval_scores_file, eval_label_file, eval_type_file)

    log.info("Getting affinity scores")
    id_affinity_scores = dict()
    for type in type_x_dict_eval.keys():
        x = np.array(type_x_dict_eval[type]).reshape((-1,1))
        y = np.array(type_y_dict_eval[type])
        learner = cPickle.load(open('models/cca_affinity_' + type + ".model", 'r'))
        y_pred_probs = learner.predict_proba(np.array(x))
        ids = type_id_dict_eval[type]
        for i in range(0, len(y)):
            id = ids[i].split("|")[0]
            if id not in id_affinity_scores.keys():
                id_affinity_scores[id] = list()
            id_affinity_scores[id].append(y_pred_probs[i][1])
        #endfor
    #endfor

    ### Nonvis stuff ###
    log.info("Getting nonvis feats")
    x_nonvis, y_nonvis, ids_nonvis = icl_data_util.load_sparse_feats(eval_file, meta_dict, set(), log)
    label_dict_nonvis = dict()
    for i in range(0, len(ids_nonvis)):
        label_dict_nonvis[ids_nonvis[i]] = y_nonvis[i]
    #endfor

    # Each data point is now the max, min, average affinity
    ids = list()
    y_vals = list()
    x_cols = list()
    x_rows = list()
    x_vals = list()

    row_idx = 0
    log.info("Reshaping data")
    for id in id_affinity_scores.keys():
        ids.append(id)

        y_vals.append(label_dict_nonvis[id])

        min_affinity = min(id_affinity_scores[id])
        max_affinity = max(id_affinity_scores[id])
        mean_affinity = np.mean(id_affinity_scores[id])

        x_rows.append(row_idx)
        x_cols.append(1)
        x_vals.append(min_affinity)
        x_rows.append(row_idx)
        x_cols.append(2)
        x_vals.append(max_affinity)
        x_rows.append(row_idx)
        x_cols.append(3)
        x_vals.append(mean_affinity)

        row_idx += 1
    #endfor

    x = sparse.csc_matrix((np.array(x_vals), (x_rows, x_cols)),
                          shape=(row_idx, 4))
    y = np.array(y_vals)

    log.tic('info', "Training")
    learner = LogisticRegression(max_iter=max_iter, n_jobs=-1)
    learner.fit(x, y)
    log.toc('info')

    model_file = "models/nonvis_with_grounding.model"
    log.info("Saving model as " + model_file)
    with open(model_file, 'wb') as pickle_file:
        cPickle.dump(learner, pickle_file)

    x_eval = x
    ids_eval = ids
    y_eval = y
    y_pred_eval = learner.predict_proba(x_eval)
    scores = ScoreDict()
    pred_scores = dict()
    for idx in range(len(y_pred_eval)):
        id = ids_eval[idx]
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
        pred_scores[id] = max_prob
    #endfor

    log.info("---Confusion matrix---")
    scores.print_confusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.get_score(label).toString() + " - %d (%.2f%%)" % \
                                                                       (scores.get_gold_count(label), scores.get_gold_percent(label))
    log.info(None, "Accuracy: %.2f%%", scores.get_accuracy())
#enddef

log = LogUtil(lvl='debug', delay=45)

models = ['svm', 'logistic', 'decision_tree', 'random_forest']
parser = ArgumentParser("ImageCaptionLearn_py: Nonvisual Mention Classifier")
parser.add_argument("--model", choices=models, default='logistic', help="train opt; Model to train")
parser.add_argument("--max_iter", type=int, default=100, help="train opt; Max SVM/logistic iterations")
parser.add_argument("--max_tree_depth", type=int, default=None, help="train opt; Max decision tree depth")
parser.add_argument("--balance", action='store_true',
                    help="train_opt; Whether to use class weights inversely proportional to the data distro")
parser.add_argument("--num_estimators", type=int, default=100, help="train_opt; Number of trees for random_forest")
parser.add_argument("--warm", action='store_true', help='train_opt; Whether to use warm start')
parser.add_argument("--train_file", type=str, help="train feats file")
parser.add_argument("--eval_file", type=str, help="eval feats file")
parser.add_argument("--meta_file", type=str, help="meta feature file (typically associated with train file)")
parser.add_argument("--model_file", type=str, help="saves model to file")
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
    ablation_groups = icl_data_util.load_ablation_file(ablation_file)

# Parse the other args
model_type = arg_dict['model']
max_iter = arg_dict['max_iter']
max_tree_depth = arg_dict['max_tree_depth']
balance = arg_dict['balance']
num_estimators = arg_dict['num_estimators']
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
    train(model_type, balance, max_iter, max_tree_depth, num_estimators, warm_start)
    evaluate(None, None, set(), False)

    for ablation_feats in ablation_groups:
        ablation_feats_str = "|".join(ablation_feats)
        log.info(None, "---------- Removing %s ----------", ablation_feats_str)
        train(model_type, balance, max_iter, max_tree_depth, num_estimators, warm_start, ablation_feats)
        evaluate(None, None, ablation_feats, False)
    #endfor
else:
    if train_file is not None:
        train(model_type, balance, max_iter, max_tree_depth, num_estimators, warm_start)
    if eval_file is not None:
        evaluate(None, None, set(), True)
#endif

