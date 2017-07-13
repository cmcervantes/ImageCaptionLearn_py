from os.path import abspath, expanduser
from sklearn.linear_model import LogisticRegression
import numpy as np
import cPickle
from argparse import ArgumentParser
from LogUtil import LogUtil
import icl_util as util

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
#file_root = '~/source/data/feats/'
#fit_id_file = abspath(expanduser(file_root + "cca_ids_dev.txt"))
file_root = "~/data/tacl201708/"
fit_prefix = "flickr30k_dev"
eval_prefix = "mscoco_dev"
fit_id_file = abspath(expanduser(file_root + fit_prefix + "_id.txt"))
fit_label_file = abspath(expanduser(file_root + fit_prefix + "_label.txt"))
fit_type_file = abspath(expanduser(file_root + fit_prefix + "_type.csv"))
fit_scores_file = abspath(expanduser(file_root + fit_prefix + "_coco30kModel_ccaScores.csv"))
eval_id_file = abspath(expanduser(file_root + eval_prefix + "_id.txt"))
eval_label_file = abspath(expanduser(file_root + eval_prefix + "_label.txt"))
eval_type_file = abspath(expanduser(file_root + eval_prefix + "_type.csv"))
eval_scores_file = abspath(expanduser(file_root + eval_prefix + "_coco30kModel_ccaScores.csv"))
scores_file = abspath(expanduser(file_root + eval_prefix + "_coco30kModel_affinity.scores"))

nonvis_file = "~/data/tacl201708/feats/" + eval_prefix + "_nonvis.feats"
nonvis_file = abspath(expanduser(nonvis_file))

# load the data from the files
type_id_dict, type_x_dict, type_y_dict = \
    load_cca_data(fit_id_file, fit_scores_file, fit_label_file, fit_type_file)
type_id_dict_eval, type_x_dict_eval, type_y_dict_eval = \
    load_cca_data(eval_id_file, eval_scores_file, eval_label_file, eval_type_file)

# Parse args
parser = ArgumentParser("ImageCaptionLearn_py: Affinity Classifier")
parser.add_argument("--max_iter", type=int, default=100,
                    help="train opt; Specifies the max iterations")
args = parser.parse_args()
arg_dict = vars(args)
util.dump_args(arg_dict, log)

# learn a separate curve for each lexical type
for type in type_x_dict.keys():
    log.info('Training ' + type)
    x = np.array(type_x_dict[type]).reshape((-1,1))
    y = np.array(type_y_dict[type])
    learner = LogisticRegression(max_iter=arg_dict['max_iter'], n_jobs=-1)
    learner.fit(x, y)
    with open('models/cca_affinity_' + type + "_flickr30k.model", 'wb') as pickle_file:
        cPickle.dump(learner, pickle_file)
    #endwith
#endfor

# save the scores in a single file
log.info('Saving scores')
with open(scores_file, 'w') as f:
    for type in type_x_dict_eval.keys():
        x = np.array(type_x_dict_eval[type]).reshape((-1,1))
        y = np.array(type_y_dict_eval[type])
        learner = cPickle.load(open('models/cca_affinity_' + type +
                                    "_flickr30k.model", 'r'))
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