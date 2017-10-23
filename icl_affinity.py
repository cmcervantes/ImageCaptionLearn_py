import cPickle
from argparse import ArgumentParser
from os.path import abspath, expanduser, isfile

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import icl_util as util
from utils.LogUtil import LogUtil

"""
Loads cca data from file as an (ids, x, y) tuple
"""
def load_cca_data(id_file, scores_file, label_file, type_file):
    # load the data into dictionaries indexed by ID
    ids = list()
    with open(id_file, 'r') as f:
        for line in f:
            ids.append(line.strip())
        f.close()
    #endwith
    id_score_dict = dict()
    with open(scores_file, 'r') as f:
        i = 0
        for line in f:
            id_score_dict[ids[i]] = float(line.strip())
            i += 1
        f.close()
    #endwith
    id_label_dict = dict()
    with open(label_file, 'r') as f:
        i = 0
        for line in f:
            id_label_dict[ids[i]] = int(line.strip())
            i += 1
        f.close()
    #endwith
    type_dict = dict()
    with open(type_file, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            type_dict[parts[0]] = parts[1]
        f.close()
    #endwith

    # partition the data by lexical type
    type_x_dict = dict()
    type_y_dict = dict()
    type_id_dict = dict()
    for id in id_score_dict.keys():
        score = id_score_dict[id]
        label = id_label_dict[id]
        mention_id = id.split("|")[0]
        types = type_dict[mention_id].split("/")
        for type in types:
            if type not in type_x_dict.keys():
                type_x_dict[type] = list()
                type_y_dict[type] = list()
                type_id_dict[type] = list()
            type_x_dict[type].append(score)
            type_y_dict[type].append(label)
            type_id_dict[type].append(id)
        #endfor
    #endfor

    return type_id_dict, type_x_dict, type_y_dict
#enddef


"""
Loads the nonvisual mention IDs from the nonvis file
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

log = LogUtil(lvl='debug', delay=45)

# Parse args
parser = ArgumentParser("ImageCaptionLearn_py: Affinity Classifier")
parser.add_argument("--train", action='store_true', help="Trains and saves a model")
parser.add_argument("--eval", action='store_true', help="Evaluates using a saved model")
parser.add_argument("--max_iter", type=int, default=100, help="train opt; Max SVM/logistic iterations")
parser.add_argument("--data_root", type=str, default="~/data/tacl201708/", help="Data directory root (assumes scores, "
                                                                                "feats, and cca subdirs)")
parser.add_argument("--lex_types", type=str, choices=['30k', 'coco'], default='30k', help="Lexical types to use")
parser.add_argument("--fit_prefix", type=str, default="flickr30k_dev", help="Prefix for fit files")
parser.add_argument("--eval_prefix", type=str, default="flickr30k_dev", help="Prefix for evaluation files")
parser.add_argument("--cca_model_type", type=str, default="30k", help="CCA model type (internal file name part)")
parser.add_argument("--model_root", type=str, default="~/models/tacl201708/", help="Model directory root")

args = parser.parse_args()
arg_dict = vars(args)
util.dump_args(arg_dict, log)

data_root = arg_dict['data_root']
fit_prefix = arg_dict['fit_prefix']
eval_prefix = arg_dict['eval_prefix']
cca_model_type = arg_dict['cca_model_type']
model_root = arg_dict['model_root']
lexical_types = arg_dict['lex_types']

# Set up all the files
fit_id_file = abspath(expanduser(data_root + "cca/" + fit_prefix + "_id.txt"))
fit_label_file = abspath(expanduser(data_root + "cca/" + fit_prefix + "_label.txt"))
fit_type_file = abspath(expanduser(data_root + "cca/" + fit_prefix + "_type_" + lexical_types + ".csv"))
fit_scores_file = abspath(expanduser(data_root + "scores/" + fit_prefix +
                                     "_" + cca_model_type + "_ccaScores.csv"))
eval_id_file = abspath(expanduser(data_root + "cca/" + eval_prefix + "_id.txt"))
eval_label_file = abspath(expanduser(data_root + "cca/" + eval_prefix + "_label.txt"))
eval_type_file = abspath(expanduser(data_root + "cca/" + eval_prefix + "_type_" + lexical_types + ".csv"))
eval_scores_file = abspath(expanduser(data_root + "scores/" + eval_prefix +
                                      "_" + cca_model_type + "_ccaScores.csv"))
scores_file = abspath(expanduser(data_root + "scores/" + eval_prefix +
                                 "_" + cca_model_type + "Model_" + lexical_types +
                                 "Types_affinity.scores"))
nonvis_file = abspath(expanduser(data_root + "feats/" + eval_prefix + "_nonvis.feats"))


if arg_dict['train']:
    type_id_dict, type_x_dict, type_y_dict = \
        load_cca_data(fit_id_file, fit_scores_file, fit_label_file, fit_type_file)

    # learn a separate curve for each lexical type
    for type in type_x_dict.keys():
        log.info('Training ' + type)
        x = np.array(type_x_dict[type]).reshape((-1,1))
        y = np.array(type_y_dict[type])
        learner = LogisticRegression(max_iter=arg_dict['max_iter'], n_jobs=-1)
        learner.fit(x, y)
        model_file = abspath(expanduser(model_root + "affinity_" + cca_model_type.replace("Model", "") + \
                     "_" + type + ".model"))
        with open(model_file, 'wb') as pickle_file:
            cPickle.dump(learner, pickle_file)
        #endwith
    #endfor
#endif

if arg_dict['eval']:
    type_id_dict_eval, type_x_dict_eval, type_y_dict_eval = \
        load_cca_data(eval_id_file, eval_scores_file, eval_label_file, eval_type_file)

    # save the scores in a single file
    log.info('Saving scores to ' + scores_file)
    with open(scores_file, 'w') as f:
        for type in type_x_dict_eval.keys():
            x = np.array(type_x_dict_eval[type]).reshape((-1,1))
            y = np.array(type_y_dict_eval[type])
            model_file = abspath(expanduser(model_root + "affinity_" + cca_model_type.replace("Model", "") + \
                         "_" + type + ".model"))
            if not isfile(model_file):
                model_file = abspath(expanduser(model_root + "affinity_" +
                                                cca_model_type.replace("Model", "") +
                                                "_other.model"))
            #endif

            learner = cPickle.load(open(model_file, 'r'))
            y_pred_probs = learner.predict_log_proba(np.array(x))
            ids = type_id_dict_eval[type]
            for i in range(0, len(y)):
                line = list()
                line.append(ids[i])
                for j in range(len(y_pred_probs[i])):
                    line.append(str(y_pred_probs[i][j]))
                f.write(','.join(line) + '\n')
                #endfor
        #endfor
        f.close()
    #endwith
#endif





