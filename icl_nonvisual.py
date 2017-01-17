from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
from os.path import abspath, expanduser, isfile
import random as r
import cPickle
import json

from LogUtil import LogUtil
from ScoreDict import ScoreDict
import icl_util as util


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
Evaluates the model (in model file) against the
eval data
"""
def evaluate(lemma_file=None, hyp_file=None, ignored_feats=set()):
    global log, eval_file, model_file, scores_file

    log.info("Loading model from file")
    learner = cPickle.load(open(model_file, 'r'))

    log.info("Loading eval data")
    x_eval, y_eval, ids_eval = util.load_feats_data(eval_file, meta_dict, ignored_feats, log)

    log.info("Loading mention lemmas")
    lemma_dict = dict()
    lemmas = set()
    with open(lemma_file, 'r') as f:
        for line in f:
            parts = line.replace('"', '').strip().split(",")
            lemma_dict[parts[0]] = parts[1]
            lemmas.add(parts[1])
        #endfor
        f.close()
    #endwith

    log.info("Loading mention hypernyms")
    with open(hyp_file, 'r') as f:
        id_hyp_dict = json.load(f)
    hypernyms = set()
    for hyps in id_hyp_dict.values():
        if isinstance(hyps, list):
            for h in hyps:
                hypernyms.add(h)
        else:
            hypernyms.add(hyps)

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

        lemma = lemma_dict[id]
        lemma_scores[lemma].increment(y_eval[idx], pred)
        for hyps in id_hyp_dict[id]:
            if isinstance(hyps, list):
                for h in hyps:
                    hyp_scores[h].increment(y_eval[idx], pred)
            else:
                hyp_scores[hyps].increment(y_eval[idx], pred)
    #endfor

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
    #endfor

    log.info("Writing scores to " + scores_file)
    with open(scores_file, 'w') as f:
        for id in pred_scores.keys():
            f.write(','.join((id, str(pred_scores[id]))) + '\n')
        f.close()
    #endwith
#enddef

def generate_meta_features(orig_feats, model_root, num_estimators):
    global log, _NONVIS_IDX_LIMIT

    log.tic('info', "Loading data")
    x, y, ids = util.load_feats_data(orig_feats, _NONVIS_IDX_LIMIT, log)
    log.toc('info')

    # each entry is a feature vector in the same order as the IDs
    meta_fv = list()
    for i in range(0, len(ids)):
        meta_fv.append(list())

    log.info("Generating feature vectors")
    for i in range(0, num_estimators):
        # load the model for this step
        learner = cPickle.load(open(model_root + "_" + str(i) + ".model", 'r'))
        y_pred = learner.predict(x)
        for j in range(0, len(ids)):
            meta_fv[j].append(y_pred[j])
        #endfor
    #endfor

    log.info("writing meta feature file")
    new_feats = orig_feats.replace(".feats", "_meta.feats")
    with open(new_feats, 'w') as f:
        for i in range(0, len(ids)):
            fv = meta_fv[i]
            y_gold = y[i]
            id = ids[i]
            s = str(y_gold) + " "
            for j in range(0, len(fv)):
                s += str(j+1) + ":" + str(fv[j]) + " "
            s += "# " + id + "\n"
            f.write(s)
        #endfor
        f.close()
    #endwith
#enddef

def train_meta_models(train_file, model_root, num_estimators, max_depth):
    global log, _NONVIS_IDX_LIMIT

    log.tic('info', "Loading data")
    x_train, y_train, ids_train = util.load_feats_data(train_file, _NONVIS_IDX_LIMIT, log)
    indices = range(0, len(ids_train)-1)
    log.toc('info')

    for i in range(0, num_estimators):
        log.log_status('info', None, "Training model %d", i)
        # get a random 10% of the data
        indices_r = r.sample(indices, int(len(indices) * 0.1))
        x_train_r = list(); y_train_r = list(); ids_train_r = list()
        for idx in indices_r:
            x_train_r = x_train[idx]
            y_train_r = y_train[idx]
            ids_train_r = ids_train[idx]
        #endfor

        # train a decision tree using this partition
        learner = DecisionTreeClassifier(max_depth=max_depth)
        learner.fit(x_train_r, y_train_r)

        # save this model
        with open(model_root + "_" + str(i) + ".model", 'wb') as pickle_file:
            cPickle.dump(learner, pickle_file)
        #endwith
    #endfor
#enddef


log = LogUtil(lvl='debug', delay=45)
train_file = "~/source/data/feats/flickr30kEntities_v2_nonvis_train.feats"
train_file = abspath(expanduser(train_file))
eval_file = "~/source/data/feats/flickr30kEntities_v2_nonvis_dev.feats"
eval_file = abspath(expanduser(eval_file))
scores_file = eval_file.replace(".feats", ".scores")
model_file = "nonvis.model"
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


#Train
if arg_dict['train']:
    train(arg_dict['model'], arg_dict['balance'], arg_dict['max_iter'],
          arg_dict['max_tree_depth'], arg_dict['num_estimators'], arg_dict['warm'])
#Evaluate
if arg_dict['eval']:
    lemma_file = abspath(expanduser('~/source/ImageCaptionLearn_py/ex_lemma_20161229.csv'))
    hyp_file = abspath(expanduser('~/source/ImageCaptionLearn_py/id_hyp_dict.json'))
    evaluate(lemma_file, hyp_file)
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
        evaluate(lemma_file, hyp_file)
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
        evaluate(lemma_file, hyp_file, ignored)
    #endfor
#endif


""" --- single mods ---
unbounded decision trees
---Confusion matrix---
   | 0             1
0  | 48103 (99.7%) 1760 (58.6%)
1  | 134 (0.3%)    1243 (41.4%)
---Scores---
0       P:  96.47% | R:  99.72% | F1:  98.07% - 48237 (94.14%)
1       P:  90.27% | R:  41.39% | F1:  56.76% - 3003 (5.86%)

logistic; max iter 1000
16:43:23 (INFO): ---Confusion matrix---
   | 0             1
0  | 48112 (99.7%) 1805 (60.1%)
1  | 125 (0.3%)    1198 (39.9%)
16:43:23 (INFO): ---Scores---
0	P:  96.38% | R:  99.74% | F1:  98.03% - 48237 (94.14%)
1	P:  90.55% | R:  39.89% | F1:  55.39% - 3003 (5.86%)
"""

""" -- numerical and other mods ---
logistic 1000

12:06:34 (INFO): ---Confusion matrix---
   | 0             1
0  | 48641 (99.6%) 1817 (60.3%)
1  | 182 (0.4%)    1195 (39.7%)
12:06:34 (INFO): ---Scores---
0	P:  96.40% | R:  99.63% | F1:  97.99% - 48823 (94.19%)
1	P:  86.78% | R:  39.67% | F1:  54.45% - 3012 (5.81%)

"""
