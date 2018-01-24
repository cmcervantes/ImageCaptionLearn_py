import json
import numpy as np
from os import listdir
from os.path import isfile
import random as rand
from utils import string as str_util
from utils import data as data_util
from utils.Word2Vec import Word2Vec

__author__ = 'ccervantes'

# Global vars
__w2v = None
__WORD_2_VEC_PATH = '/shared/projects/word2vec/GoogleNews-vectors-negative300.bin.gz'
__GLOVE_PATH = '/home/ccervan2/data/tacl201711/coco30k_train_glove.vec'
__GLOVE_DICT = None
__WORD_EMBEDDING_WIDTH = 300
__BOX_EMBEDDING_WIDTH = 4096


def init_w2v():
    """
    Initializes this utility's word2vec module
    """
    global __w2v, __WORD_2_VEC_PATH
    __w2v = Word2Vec(__WORD_2_VEC_PATH)
#enddef


def init_glove():
    """
    Initializes this utility's glove module
    :return:
    """
    global __GLOVE_DICT, __GLOVE_PATH
    __GLOVE_DICT = dict()
    with open(__GLOVE_PATH, 'r') as f:
        for line in f.readlines():
            line_parts = line.split(" ")
            word = line_parts[0]
            vector = list()
            for i in range(1, len(line_parts)):
                vector.append(float(line_parts[i]))
            __GLOVE_DICT[word] = vector
        #endfor
    #endwith
#enddef


def get_glove_matrix(sentence):
    """
    Returns the glove matrix for the given sentence
    :param sentence:
    :return:
    """
    global __GLOVE_DICT, __GLOVE_PATH

    # Initialize our matrix
    glove_matrix = np.empty(shape=(len(sentence), 300))

    # iterate through each word in the sentence,
    # replacing unknown words with a constant random [-1,1]
    # vector
    unk = np.random.uniform(-1, 1, 300)
    glove_keys = set(__GLOVE_DICT.keys())
    for i in range(0, len(sentence)):
        word = sentence[i]
        if word in glove_keys:
            glove_matrix[i] = __GLOVE_DICT[word]
        else:
            glove_matrix[i] = unk
    #endfor
    return glove_matrix
#enddef


def load_sentences(sentence_file, embedding_type='w2v'):
    """
    Reads the given sentence file and maps sentence IDs to
    word2vec or glove matrices

    :param sentence_file:   File containing captions and IDs
    :param embedding_type: Type of embeddings to use (w2v, glove)
    :return:
    """
    global __w2v, __WORD_EMBEDDING_WIDTH
    data_dict = dict()

    # Load the sentence file, which we assume is
    # in the format
    #   <img_id>#<cap_idx>    <caption_less_punc>
    data_dict['sentences'] = dict()
    if sentence_file is not None:
        with open(sentence_file, 'r') as f:
            for line in f.readlines():
                id_split = line.split("\t")
                sentence = id_split[1].split(" ")
                vector = None
                if embedding_type == 'w2v':
                    vector = __w2v.get_w2v_matrix(sentence)
                elif embedding_type == 'glove':
                    vector = get_glove_matrix(sentence)
                #endif
                data_dict['sentences'][id_split[0].strip()] = vector
            #endfor
        #endwith
    #endif

    # Get the maximum sentence length (max seq length)
    data_dict['max_seq_len'] = -1
    for sentence_matrix in data_dict['sentences'].values():
        if len(sentence_matrix) > data_dict['max_seq_len']:
            data_dict['max_seq_len'] = len(sentence_matrix)
    #endfor

    # We have a set number of embedding widths
    data_dict['word_embedding_width'] = __WORD_EMBEDDING_WIDTH

    return data_dict
#enddef


def load_mentions(mention_idx_file, task, feats_file, feats_meta_file, n_classes):
    """
    Reads the mention index file, mapping mention indices (either by the pair
    or individually) with indices, labels, and feature vectors

    :param mention_idx_file: File containing mention pair or mention IDs and index tuples
    :param task: {rel_intra, rel_cross, nonvis, card, affinity}
    :param feats_file: File containing engineered (sparse) features
    :param feats_meta_file: File associating engineered feature indices with
                            human-readable names
    :param n_classes: Number of classes
    :return: Dictionary storing the aforementioned dictionaries
    """
    data_dict = dict()

    # Load the mention index file, which we assume is in the format
    #   <m_id>      <m1_start>,<m1_end>                         <label>
    # or
    #   <pair_id>   <m1_start>,<m1_end>,<m2_start>,<m2_end>     <label>
    # depending whether we're operating on mentions or mention pairs
    data_dict['caption_ids'] = dict()
    data_dict['mention_indices'] = dict()
    data_dict['labels'] = dict()
    if mention_idx_file is not None:
        with open(mention_idx_file, 'r') as f:
            for line in f.readlines():
                # Parse the ID to get the caption(s) from it
                line_split = line.strip().split("\t")
                id = line_split[0].strip()

                if "rel" in task:
                    # If we're dealing with mention pairs, associate
                    # each with a tuple of their caption IDs
                    id_dict = str_util.kv_str_to_dict(id)
                    cap_1 = id_dict['doc'] + "#" + id_dict['caption_1']
                    cap_2 = id_dict['doc'] + "#" + id_dict['caption_2']
                    data_dict['caption_ids'][id] = (cap_1, cap_2)
                elif task == "nonvis" or task == "card" or task == "affinity":
                    # If we're dealing with mentions, associate each
                    # with their originating caption ID
                    data_dict['caption_ids'][id] = id.split(";")[0]
                #endif

                # Parse the indices as integers and store the lists
                indices = list()
                for i in line_split[1].strip().split(","):
                    indices.append(int(i.strip()))
                data_dict["mention_indices"][id] = indices

                # the label is a one-hot, where the entry in the
                # file is the appropriate index; ignore affinity
                # labels, since these are populated elsewhere
                if task != "affinity":
                    label = np.zeros([n_classes])
                    label[int(line_split[2].strip())] = 1.0
                    data_dict['labels'][id] = label
                #endif
            #endfor
        #endwith
    #endif

    # Add the feature files to the data dict
    if feats_file is not None and feats_meta_file is not None:
        meta_dict = json.load(open(feats_meta_file, 'r'))
        data_dict['n_mention_feats'] = meta_dict['max_idx'] + 1
        X, _, IDs = data_util.load_sparse_feats(feats_file, meta_dict)
        data_dict['mention_features'] = dict()
        for i in range(0, len(IDs)):
            data_dict['mention_features'][IDs[i]] = X[i]
    #endif

    return data_dict
#enddef


def load_mpe_data(mpe_idx_file):
    data_dict = dict()

    # Load the MPE index file, which we assume is in the format
    #   <doc_id>    <label>
    data_dict['doc_ids'] = list()
    data_dict['caption_ids'] = dict()
    data_dict['labels'] = dict()
    if(mpe_idx_file is not None):
        with open(mpe_idx_file, 'r') as f:
            for line in f.readlines():
                line_split = line.strip().split("\t")
                id = line_split[0].strip()
                data_dict['caption_ids'][id] = [None, None, None, None, None]
                for i in range(0, 5):
                    data_dict['caption_ids'][id][i] = id + "#" + str(i)
                data_dict['labels'][id] = line_split[1].strip()
                data_dict['doc_ids'].append(id)
            #endfor
        #endwith
    #endif

    return data_dict
#enddef


def load_boxes(mention_box_label_file, box_dir, box_category_file=None):
    """
    Reads the box index file, mapping mention/box indices with
    labels and loads all box features from the box_dir
    :param box_dir: Directory where box feature files are located
    :param mention_box_label_file: File containing box/mention affinity labels
    :return: Dictionary storing the aforementioned dictionaries
    """
    global __BOX_EMBEDDING_WIDTH
    data_dict = dict()

    # Retrieve the one-hot mention/box labels from the file
    data_dict['labels'] = dict()
    with open(mention_box_label_file, 'r') as f:
        for line in f.readlines():
            line_split = line.strip().split("\t")
            label_vec = np.zeros([2])
            label_vec[int(line_split[1].strip())] = 1.0
            data_dict['labels'][line_split[0].strip()] = label_vec
        #endfor
    #endwith

    # If a box-category file has been specified,
    # load its categories
    data_dict['box_categories'] = dict()
    data_dict['n_box_feats'] = None
    if box_category_file is not None:
        with open(box_category_file, 'r') as f:
            for line in f.readlines():
                line_split = line.strip().split("\t")
                vec = list()
                for i in line_split[1].split(","):
                    vec.append(float(i))
                data_dict['n_box_feats'] = max(data_dict['n_box_feats'], len(vec))
                data_dict['box_categories'][line_split[0].strip()] = np.array(vec)
            #endfor
        #endwith
    #endif

    # Store the box directory in the data dict so other methods
    # can pull it from this and we don't have to keep passing it around
    data_dict['box_dir'] = box_dir

    # We have a set number of embedding widths
    data_dict['box_embedding_width'] = __BOX_EMBEDDING_WIDTH

    return data_dict
#enddef


def build_model_filename(arg_dict, task):
    """
    Builds the neural network model file name,
    given an argument dict; any None values are
    omitted from the resulting name
    :param arg_dict: Argument dictionary
    :param task: {rel_intra, rel_cross, nonvis, card, affinity}
    :return: Model file name
    """

    # Set up the root of the model file
    if 'data_root' in arg_dict.keys():
        model_file = arg_dict['data_root']
    else:
        model_file = arg_dict['data'] + "_" + arg_dict['split']
    if 'rel_type' in arg_dict and arg_dict['rel_type'] is not None:
        model_file += "_" + task.replace("_", "_" + arg_dict['rel_type'] + "_")
    else:
        model_file += "_" + task

    # Add the encoding scheme
    if 'encoding_scheme' in arg_dict:
        enc_scheme = arg_dict['encoding_scheme']
        if enc_scheme == "first_last_sentence":
            model_file += "_fls"
        elif enc_scheme == "first_last_mention":
            model_file += "_flm"
    #endif

    # Add the standard items that should be present across tasks, if specified
    model_file += "_" + arg_dict['activation'] + "_" + \
                  "epch" + str(int(arg_dict['epochs'])) + "_" + \
                  "lrn" + str(arg_dict['learn_rate']) + "_" + \
                  "btch" + str(int(arg_dict['batch_size'])) + "_" + \
                  "drp" + str(int(arg_dict['lstm_input_dropout'] * 100)) + \
                  str(int(arg_dict['dropout'] * 100)) + "_" + \
                  "lstm" + str(int(arg_dict['lstm_hidden_width'])) + "_" + \
                  "hdn" + str(int(arg_dict['start_hidden_width'])) + "-" + \
                  str(int(arg_dict['hidden_depth'])) + "_" + \
                  "admEps" + str(arg_dict['adam_epsilon'])

    # Add the optional items
    if arg_dict['clip_norm'] is not None:
        model_file += "_clip" + str(arg_dict['clip_norm'])
    if arg_dict['data_norm']:
        model_file += "_dataNorm"
    if arg_dict['weighted_classes']:
        model_file += "_weighted"
    if 'early_stopping' in arg_dict.keys():
        if arg_dict['early_stopping']:
            model_file += "_early"

    return model_file + ".model"
#enddef


def load_relation_labels(filename):
    """
    Parses the given relation label file, assumed to be in the format
        m_i_id m_j_id label
    and returns a mapping of pair ID tuples to pairwise labels
    :param filename: Relation label file
    :return: gold_label_dict
    """
    gold_label_dict = dict()
    with open(filename, 'r') as f:
        for line in f.readlines():
            line_split = line.split(" ")
            gold_label_dict[(line_split[0].strip(), line_split[1].strip())] = line_split[2].strip()
        #endfor
    #endwith
    return gold_label_dict
#enddef


def load_batch(ids, data_dict, task, n_classes):
    """
    Loads a batch of data, given a list of IDs,
    a data dictionary, the task we're retrieving
    a batch for, the number of classes, and the
    width of our word embeddings
    :param ids: List of mentions, mention pairs,
                or mention/box pairs
    :param data_dict: Dictionary of all data
                      dictionaries (for sentences, etc)
    :param task: {rel_intra, rel_cross, nonvis,
                  card, affinity}
    :param n_classes: Number of classes, for the task
    :return: Batch tensors
    """

    # Load the batch tensors and size
    batch_tensors = dict()
    batch_size = len(ids)
    n_seq = batch_size
    if task == "rel_intra":
        n_seq = batch_size
    elif task == "rel_cross":
        n_seq = 2 * batch_size

    # Populate our sentence tensor and sequence length array
    batch_tensors['sentences'] = np.zeros([n_seq, data_dict['max_seq_len'],
                                           data_dict['word_embedding_width']])
    batch_tensors['seq_lengths'] = np.zeros([n_seq])
    batch_idx = 0
    for i in range(0, batch_size):
        id = ids[i]
        if task == "affinity":
            id = id.split("|")[0]

        # set this sentence idx to the sentence embedding,
        # implicitly padding the end of the sequence with 0s
        # If this is cross-relation prediction, recall that we're
        # sending two sentences to the lstm for each item
        sentence_ids = list()
        if task == "rel_cross":
            for s_id in data_dict['caption_ids'][id]:
                sentence_ids.append(s_id)
        elif task == "rel_intra":
            sentence_ids.append(data_dict['caption_ids'][id][0])
        elif task == "nonvis" or task == "card" or task == "affinity":
            sentence_ids.append(data_dict['caption_ids'][id])

        for sentence_id in sentence_ids:
            sentence_matrix = data_dict['sentences'][sentence_id]
            for j in range(0, len(sentence_matrix)):
                batch_tensors['sentences'][batch_idx][j] = sentence_matrix[j]
            batch_tensors['seq_lengths'][batch_idx] = len(sentence_matrix)
            batch_idx += 1
        #endfor
    #endfor

    # Load all indices into each batch, and let downstream tasks
    # worry about how to use those indices with respect to encoding
    batch_tensors['labels'] = np.zeros([batch_size, n_classes])
    matrix_names = ['first_i_bw', 'first_i_fw', 'last_i_fw',
                    'last_i_bw', 'sent_last_i_fw',
                    'sent_first_i_bw', 'first_j_bw',
                    'last_j_fw', 'first_j_fw', 'last_j_bw',
                    'sent_last_j_fw', 'sent_first_j_bw']
    for name in matrix_names:
        batch_tensors[name] = np.zeros([batch_size, 3])

    # Add the feature matrices
    if "rel" in task:
        n_features = len(data_dict['mention_features'][ids[0]])
        batch_tensors['ij_feats'] = np.zeros([batch_size, n_features])
    elif task == "affinity":
        m_id, b_id = ids[0].split("|")
        batch_tensors['m_feats'] = np.zeros([batch_size,
                                             len(data_dict['mention_features'][m_id])])
        batch_tensors['box_embeddings'] = np.zeros([batch_size,
                                                    data_dict['box_embedding_width']])
        if data_dict['box_categories']:
            batch_tensors['b_feats'] = np.zeros([batch_size, data_dict['n_box_feats']])
    else:
        n_features = len(data_dict['mention_features'][ids[0]])
        batch_tensors['m_feats'] = np.zeros([batch_size, n_features])
    #endif

    # We're going to keep a dictionary of ids -> box features
    # for the current image we're looking at
    box_embedding_dict = dict()
    loaded_box_ids = set()

    # Iterate through the IDs, loading our data
    for i in range(0, batch_size):
        label_id = ids[i]
        m_id = label_id
        b_id = None
        if task == 'affinity':
            id_split = label_id.split("|")
            m_id = id_split[0]
            b_id = id_split[1]
        #endif

        batch_tensors['labels'][i] = data_dict['labels'][label_id]

        # Relation indices are first_i, last_i, first_j, last_j
        # Other indices are first_i, last_i
        first_j = None
        last_j = None
        if 'rel' in task:
            first_i, last_i, first_j, last_j = data_dict['mention_indices'][m_id]
        else:
            first_i, last_i = data_dict['mention_indices'][m_id]

        # Retrieve the sentence indices, depending on the task
        if task == 'rel_cross':
            sent_i = i * 2
            sent_j = i * 2 + 1
        else:
            sent_i = i
            sent_j = i
        #endif

        # Mention indices
        batch_tensors['first_i_fw'][i] = np.array([0, sent_i, first_i])
        batch_tensors['first_i_bw'][i] = np.array([1, sent_i, first_i])
        batch_tensors['last_i_fw'][i] = np.array([0, sent_i, last_i])
        batch_tensors['last_i_bw'][i] = np.array([1, sent_i, last_i])
        if first_j is not None:
            batch_tensors['first_j_fw'][i] = np.array([0, sent_j, first_j])
            batch_tensors['first_j_bw'][i] = np.array([1, sent_j, first_j])
        if last_j is not None:
            batch_tensors['last_j_fw'][i] = np.array([0, sent_j, last_j])
            batch_tensors['last_j_bw'][i] = np.array([1, sent_j, last_j])

        # Sentence indices
        batch_tensors['sent_last_i_fw'][i] = \
            np.array([0, sent_i, batch_tensors['seq_lengths'][sent_i] - 1])
        batch_tensors['sent_first_i_bw'][i] = np.array([1, sent_i, 0])
        batch_tensors['sent_last_j_fw'][i] = \
            np.array([0, sent_j, batch_tensors['seq_lengths'][sent_j] - 1])
        batch_tensors['sent_first_j_bw'][i] = np.array([1, sent_j, 0])

        # Add the mention feats
        if "rel" in task:
            batch_tensors['ij_feats'][i] = data_dict['mention_features'][m_id]
        else:
            batch_tensors['m_feats'][i] = data_dict['mention_features'][m_id]

        # Load box features for this ID; instead of loading all
        # box features into memory -- which is unrealistic --
        # we're going to open files as we need them
        if task == 'affinity':
            # Add features, if we have any to add
            # UPDATE: apparently empty dictionaries evaluate to false in python
            if data_dict['box_categories']:
                if b_id in data_dict['box_categories'].keys():
                    batch_tensors['b_feats'][i] = data_dict['box_categories'][b_id]

            # If this box isn't in the set of loaded boxes, we need to open
            # the appropriate file and load all boxes for that image;
            # This is naturally horribly inefficient if we randomly shuffle boxes,
            # but we're relying on the affinity classifiers to order mention/box
            # pairs by image, that way we open a new file at most
            #   max_n_boxes / batch_size + 2
            # times
            if b_id not in loaded_box_ids:
                img_id = b_id.split(";")[0]
                box_features, _, box_ids = \
                    data_util.load_sparse_feats(data_dict['box_dir'] + "/" +
                                                img_id.replace('.jpg', '.feats'),
                                                None, None,
                                                data_dict['box_embedding_width'])
                for j in range(0, len(box_ids)):
                    box_embedding_dict[box_ids[j]] = box_features[j]
                loaded_box_ids = set(box_ids)
            #endif
            batch_tensors['box_embeddings'][i] = box_embedding_dict[b_id]
    #endfor]

    return batch_tensors
#enddef


def load_batch_mpe(ids, data_dict, n_classes):
    """
    Loads a batch for MPE data

    :param ids:
    :param data_dict:
    :return:
    """

    # Load the batch tensors and size
    batch_tensors = dict()
    batch_size = len(ids)
    n_seq_premise = 4 * batch_size
    n_seq_hyp = batch_size

    # Populate our sentence tensors and sequence length arrays
    batch_tensors['premises'] = np.zeros([n_seq_premise, data_dict['max_seq_len'],
                                          data_dict['word_embedding_width']])
    batch_tensors['seq_lengths_premise'] = np.zeros([n_seq_premise])
    batch_tensors['hypotheses'] = np.zeros([n_seq_hyp, data_dict['max_seq_len'],
                                            data_dict['word_embedding_width']])
    batch_tensors['seq_lengths_hyp'] = np.zeros([n_seq_hyp])
    batch_tensors['DEBUG_sentences'] = np.zeros([n_seq_premise + n_seq_hyp, data_dict['max_seq_len'],
                                                 data_dict['word_embedding_width']])
    batch_tensors['DEBUG_seq_lengths'] = np.zeros([n_seq_premise + n_seq_hyp])
    batch_tensors['labels'] = np.zeros([batch_size, n_classes])
    premise_idx = 0
    DEBUG_idx = 0
    for i in range(0, batch_size):
        id = ids[i]
        batch_tensors['labels'][i] = data_dict['labels'][id]

        # Each ID has five captions, the last of which is the hypothesis
        premise_ids = list()
        for i in range(0, 4):
            premise_ids.append(data_dict['caption_ids'][id][i])
        rand.shuffle(premise_ids)
        hyp_id = data_dict['caption_ids'][id][4]

        # Package the premises and hypothesis into tensors
        for premise_id in premise_ids:
            sentence_matrix = data_dict['sentences'][premise_id]
            for w_idx in range(0, len(sentence_matrix)):
                batch_tensors['premises'][premise_idx][w_idx] = sentence_matrix[w_idx]
                batch_tensors['DEBUG_sentences'][DEBUG_idx][w_idx] = sentence_matrix[w_idx]
            batch_tensors['seq_lengths_premise'][premise_idx] = len(sentence_matrix)
            batch_tensors['DEBUG_seq_lengths'][DEBUG_idx] = len(sentence_matrix)
            premise_idx += 1
            DEBUG_idx += 1
        #endfor
        hyp_matrix = data_dict['sentences'][hyp_id]
        for w_idx in range(0, len(hyp_matrix)):
            batch_tensors['hypotheses'][i][w_idx] = hyp_matrix[w_idx]
            batch_tensors['DEBUG_sentences'][DEBUG_idx][w_idx] = hyp_matrix[w_idx]
        batch_tensors['seq_lengths_hyp'][i] = len(hyp_matrix)
        batch_tensors['DEBUG_seq_lengths'][DEBUG_idx] = len(hyp_matrix)
        DEBUG_idx += 1
    #endfor

    # Each item is the set of first/last words in each premise sentence
    # and the hypothesis
    for i in range(0, 4):
        batch_tensors['premise_' + str(i) + '_first_bw'] = np.zeros([batch_size, 3])
        batch_tensors['premise_' + str(i) + '_last_fw'] = np.zeros([batch_size, 3])
        batch_tensors['DEBUG_premise_' + str(i) + '_first_bw'] = np.zeros([batch_size, 3])
        batch_tensors['DEBUG_premise_' + str(i) + '_last_fw'] = np.zeros([batch_size, 3])
    #endfor
    batch_tensors['hyp_first_bw'] = np.zeros([batch_size, 3])
    batch_tensors['hyp_last_fw'] = np.zeros([batch_size, 3])
    batch_tensors['DEBUG_hyp_first_bw'] = np.zeros([batch_size, 3])
    batch_tensors['DEBUG_hyp_last_fw'] = np.zeros([batch_size, 3])
    for i in range(0, batch_size):
        premise_indices = range(i*4, (i*4)+4)
        DEBUG_indices = range(i*5, (i*5)+5)
        for j in range(0, 4):
            batch_tensors['premise_' + str(j) + '_first_bw'][i] = \
                np.array([1, premise_indices[j], 0])
            batch_tensors['premise_' + str(j) + '_last_fw'][i] = \
                np.array([0, premise_indices[j], batch_tensors['seq_lengths_premise'][premise_indices[j]]-1])
            batch_tensors['DEBUG_premise_' + str(j) + '_first_bw'][i] = \
                np.array([1, DEBUG_indices[j], 0])
            batch_tensors['DEBUG_premise_' + str(j) + '_last_fw'][i] = \
                np.array([0, DEBUG_indices[j], batch_tensors['DEBUG_seq_lengths'][DEBUG_indices[j]]-1])
        #endfor
        batch_tensors['hyp_first_bw'][i] = np.array([1, i, 0])
        batch_tensors['hyp_last_fw'][i] = np.array([0, i, batch_tensors['seq_lengths_hyp'][i]-1])
        batch_tensors['DEBUG_hyp_first_bw'][i] = np.array([1, DEBUG_indices[4], 0])
        batch_tensors['DEBUG_hyp_last_fw'][i] = np.array([0, DEBUG_indices[4],
                                                          batch_tensors['DEBUG_seq_lengths'][DEBUG_indices[4]]-1])
    #endfor

    return batch_tensors
#enddef

