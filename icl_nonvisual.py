from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
from os.path import abspath, expanduser, isfile
import random as r
import cPickle
import json
import numpy as np
from scipy import sparse

from LogUtil import LogUtil
from ScoreDict import ScoreDict
import icl_util as util
#from icl_affinity import load_cca_data

"""
Trains the model
"""
def train(model, balance, max_iter=None, max_depth=None,
          num_estimators=None, warm_start=None, ignored_feats=set()):
    global log, train_file, meta_dict, model_file

    log.tic('info', "Loading training data")
    x, y, ids = util.load_feats_data(train_file, meta_dict, ignored_feats, log)
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
    x_eval, y_eval, ids_eval = util.load_feats_data(eval_file, meta_dict, ignored_feats, log)

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
                totalCount = s.getGoldCount(0) + s.getGoldCount(1)
                if totalCount > 0:
                    line = l + "," + str(totalCount) + ","
                    line += str(s.getGoldCount(0)) + "," + str(s.getGoldCount(1)) + ","
                    line += str(s.getPredCount(0)) + "," + str(s.getPredCount(1)) + ","
                    score_vis = s.getScore(0)
                    score_nonvis = s.getScore(1)
                    line += str(score_vis.r) + "," + str(score_vis.p) + "," + str(score_vis.f1) + ","
                    line += str(score_nonvis.r) + "," + str(score_nonvis.p) + "," + str(score_nonvis.f1) + ","
                    line += str(s.getAccuracy() / 100.0) + "," + str(s.getCorrectCount())
                    line += "\n"
                    f.write(line)
            f.close()
    if hyp_file is not None:
        with open('nonvisual_scores_hyp.csv', 'w') as f:
            f.write("hypernym,gold_freq,gold_vis,gold_nonvis,pred_vis,pred_nonvis,vis_r,vis_p,vis_f1,"+ \
                    "nonvis_r,nonvis_p,nonvis_f1,accuracy,correct_count\n")
            for h in hyp_scores.keys():
                s = hyp_scores[h]
                totalCount = s.getGoldCount(0) + s.getGoldCount(1)
                if totalCount > 0:
                    line = h + "," + str(totalCount) + ","
                    line += str(s.getGoldCount(0)) + "," + str(s.getGoldCount(1)) + ","
                    line += str(s.getPredCount(0)) + "," + str(s.getPredCount(1)) + ","
                    score_vis = s.getScore(0)
                    score_nonvis = s.getScore(1)
                    line += str(score_vis.r) + "," + str(score_vis.p) + "," + str(score_vis.f1) + ","
                    line += str(score_nonvis.r) + "," + str(score_nonvis.p) + "," + str(score_nonvis.f1) + ","
                    line += str(s.getAccuracy() / 100.0) + "," + str(s.getCorrectCount())
                    line += "\n"
                    f.write(line)
            f.close()

    log.info("---Confusion matrix---")
    scores.printConfusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.getScore(label).toString() + " - %d (%.2f%%)" % \
              (scores.getGoldCount(label), scores.getGoldPercent(label))
    log.info(None, "Accuracy: %.2f%%", scores.getAccuracy())

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
    x_nonvis, y_nonvis, ids_nonvis = util.load_feats_data(eval_file, meta_dict, set(), log)
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
    scores.printConfusion()

    log.info("---Scores---")
    for label in scores.keys:
        print str(label) + "\t" + scores.getScore(label).toString() + " - %d (%.2f%%)" % \
                                                                      (scores.getGoldCount(label), scores.getGoldPercent(label))
    log.info(None, "Accuracy: %.2f%%", scores.getAccuracy())
#enddef

log = LogUtil(lvl='debug', delay=45)
train_file = "~/source/data/feats/nonvis_20170215.feats"
train_file = abspath(expanduser(train_file))
eval_file = "~/source/data/feats/nonvis_test_20170215.feats"
eval_file = abspath(expanduser(eval_file))
scores_file = eval_file.replace(".feats", ".scores")
model_file = "models/nonvis.model"
meta_file = train_file.replace(".feats", "_meta.json")

models = ['svm', 'logistic', 'decision_tree', 'random_forest']
parser = ArgumentParser("ImageCaptionLearn_py: Nonvisual Mention Classifier")
parser.add_argument("--model", choices=models, help="train opt; Model to train")
parser.add_argument("--train", action='store_true', help="Trains and saves a model")
parser.add_argument("--eval", action='store_true', help="Evaluates using a saved model")
parser.add_argument("--ablation", type=str, help="Performs ablation, removing the specified " +
                                                 "pipe-separated features (or 'all', for all features)")
parser.add_argument("--max_iter", type=int, default=100, help="train opt; Max SVM/logistic iterations")
parser.add_argument("--max_tree_depth", type=int, default=None, help="train opt; Max decision tree depth")
parser.add_argument("--balance", action='store_true',
                    help="train_opt; Whether to use class weights inversely proportional to the data distro")
parser.add_argument("--num_estimators", type=int, default=100, help="train_opt; Number of trees for random_forest")
parser.add_argument("--warm", action='store_true', help='train_opt; Whether to use warm start')
args = parser.parse_args()
arg_dict = vars(args)
util.dump_args(arg_dict, log)

meta_dict = json.load(open(meta_file, 'r'))

log.info("Ignoring hypernym_bow")
fixed_ignored = {"hypernym_bow"}

#nonvis_with_grounding(arg_dict['max_iter'])
#quit()

#Train
if arg_dict['train']:
    train(arg_dict['model'], arg_dict['balance'], arg_dict['max_iter'],
          arg_dict['max_tree_depth'], arg_dict['num_estimators'], arg_dict['warm'],
          fixed_ignored)
#Evaluate
if arg_dict['eval']:
    evaluate(None, None, fixed_ignored)
    #lemma_file = abspath(expanduser('~/source/ImageCaptionLearn_py/ex_lemma_20161229.csv'))
    #hyp_file = abspath(expanduser('~/source/ImageCaptionLearn_py/id_hyp_dict.json'))
    #evaluate(lemma_file, hyp_file, fixed_ignored)
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
        train(arg_dict['model'], arg_dict['balance'], arg_dict['max_iter'],
              arg_dict['max_tree_depth'], arg_dict['num_estimators'], arg_dict['warm'])
        lemma_file = abspath(expanduser('~/source/ImageCaptionLearn_py/ex_lemma_20161229.csv'))
        hyp_file = abspath(expanduser('~/source/ImageCaptionLearn_py/id_hyp_dict.json'))
        evaluate(lemma_file, hyp_file, set(), False)
    #endif

    log.info("Running ablation over the following features")
    print "|".join(ablation_feats)

    for feat in ablation_feats:
        log.info(None, "---- Removing feature %s ----", feat)
        ignored = set(); ignored.add(feat)
        train(arg_dict['model'], arg_dict['balance'], arg_dict['max_iter'],
              arg_dict['max_tree_depth'], arg_dict['num_estimators'], arg_dict['warm'], ignored)
        lemma_file = abspath(expanduser('~/source/ImageCaptionLearn_py/ex_lemma_20161229.csv'))
        hyp_file = abspath(expanduser('~/source/ImageCaptionLearn_py/id_hyp_dict.json'))
        evaluate(lemma_file, hyp_file, ignored, False)
    #endfor
#endif