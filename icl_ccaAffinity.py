from os.path import abspath, expanduser
from sklearn.linear_model import LogisticRegression
import numpy as np
import cPickle
from ScoreDict import ScoreDict

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
    for id in id_score_dict.keys():
        score = id_score_dict[id]
        label = id_label_dict[id]
        mention_id = id.split("|")[0]
        types = type_dict[mention_id].split("/")
        for type in types:
            if type not in type_x_dict.keys():
                type_x_dict[type] = list()
                type_y_dict[type] = list()
            type_x_dict[type].append(score)
            type_y_dict[type].append(label)
        #endfor
    #endfor

    return type_x_dict, type_y_dict
#enddef

fit_id_file = abspath(expanduser("~/source/data/feats/cca_ids_dev.txt"))
fit_scores_file = abspath(expanduser("~/source/data/feats/cca_scores_dev.txt"))
fit_label_file = abspath(expanduser("~/source/data/feats/cca_labels_dev.txt"))
fit_type_file = abspath(expanduser("~/source/data/feats/cca_types_dev.csv"))
eval_id_file = fit_id_file
eval_scores_file = fit_scores_file
eval_label_file = fit_label_file
eval_type_file = fit_type_file

# load the data from the files
type_x_dict, type_y_dict = \
    load_cca_data(fit_id_file, fit_scores_file, fit_label_file, fit_type_file)

# learn a separate curve for each lexical type
max_iter = 1000
for type in type_x_dict.keys():
    x = np.array(type_x_dict[type]).reshape((-1,1))
    y = np.array(type_y_dict[type])
    learner = LogisticRegression(max_iter=max_iter, verbose=1, n_jobs=-1)
    learner.fit(x, y)
    with open('cca_affinity_' + type + ".model", 'wb') as pickle_file:
        cPickle.dump(learner, pickle_file)
    #endwith
#endfor

# evaluate
type_x_dict_eval, type_y_dict_eval = \
    load_cca_data(eval_id_file, eval_scores_file, eval_label_file, eval_type_file)
totalScore = ScoreDict()
for type in type_x_dict_eval.keys():
    x = np.array(type_x_dict_eval[type]).reshape((-1,1))
    y = np.array(type_y_dict_eval[type])
    learner = cPickle.load(open('cca_affinity_' + type + ".model", 'r'))
    y_hat = learner.predict(np.array(x))
    s = ScoreDict()
    for i in range(0, len(y)):
        s.increment(y[i], y_hat[i])
        totalScore.increment(y[i], y_hat[i])
    print type
    print "0 & " + s.getScore(0).toLatexString()
    print "1 & " + s.getScore(1).toLatexString()
#endfor
print "Total"
print "0 & " + totalScore.getScore(0).toLatexString()
print "1 & " + totalScore.getScore(1).toLatexString()

print "Accuracy: " + str(totalScore.getAccuracy())



"""
bodyparts
0 &  87.91\% &  97.69\% &  92.54\% \\
1 &  73.27\% &  32.03\% &  44.57\% \\
animals
0 &  87.19\% &  95.10\% &  90.98\% \\
1 &  76.98\% &  53.97\% &  63.45\% \\
people
0 &  85.98\% &  94.81\% &  90.18\% \\
1 &  68.83\% &  42.57\% &  52.60\% \\
instruments
0 &  86.05\% &  98.29\% &  91.76\% \\
1 &  77.39\% &  26.81\% &  39.82\% \\
vehicles
0 &  93.18\% &  97.80\% &  95.43\% \\
1 &  78.27\% &  52.56\% &  62.89\% \\
scene
0 &  94.95\% &  98.73\% &  96.80\% \\
1 &  70.10\% &  36.23\% &  47.77\% \\
colors
0 &  84.24\% &  99.92\% &  91.41\% \\
1 &  62.50\% &   0.69\% &   1.36\% \\
other
0 &  89.07\% &  98.57\% &  93.58\% \\
1 &  67.75\% &  19.85\% &  30.71\% \\
clothing
0 &  89.64\% &  97.74\% &  93.52\% \\
1 &  76.56\% &  39.55\% &  52.16\% \\
Total
0 &  89.19\% &  97.24\% &  93.04\% \\
1 &  70.66\% &  36.04\% &  47.73\% \\
"""



