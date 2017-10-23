import json
import numpy as np
from utils import string as str_util
from utils import data as data_util
from utils.Word2Vec import Word2Vec

__author__ = 'ccervantes'

# Global vars
__w2v = None
__WORD_2_VEC_PATH = '/shared/projects/word2vec/GoogleNews-vectors-negative300.bin.gz'
__GLOVE_PATH = '/home/ccervan2/data/tacl201711/coco30k_train_glove.vec'
__GLOVE_DICT = None


def init_w2v():
    """
    Initializes this utility's word2vec module
    """
    global __w2v, __WORD_2_VEC_PATH
    __w2v = Word2Vec(__WORD_2_VEC_PATH)
#enddef


def init_glove():
    """

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
    global __w2v
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
    return data_dict
#enddef


def load_mention_pair_data(mention_idx_file, feats_file=None, feats_meta_file=None):
    """
    Reads the mention pair index file, mapping mention pair IDs to
    first/last word indices, sentence IDs, labels, and
    normalization vectors (for averages, where appropriate).
    Optionally reads a feature and feature meta file and adds these
    vectors to the data dict

    :param mention_idx_file: File containing mention pair IDs and index tuples
    :param feats_file: File containing engineered (sparse) features
    :param feats_meta_file: File associating engineered feature indices with
                            human-readable names
    :return Dictionary storing the aforementioned dictionaries
    """
    data_dict = dict()

    # Load the mention index file, which we assume is in the format
    #   <pair_id>     <m1_start>,<m1_end>,<m2_start>,<m2_end>   <label>
    data_dict['mention_pair_cap_ids'] = dict()
    data_dict['mention_pair_indices'] = dict()
    data_dict['mention_pair_norm_vecs'] = dict()
    data_dict['mention_pair_labels'] = dict()
    if mention_idx_file is not None:
        with open(mention_idx_file, 'r') as f:
            for line in f.readlines():
                # Parse the ID to get the caption(s) from it
                id_split = line.split("\t")
                pair_id = id_split[0].strip()

                # Associate each mention pair with a tuple of its caption IDs
                id_dict = str_util.kv_str_to_dict(pair_id)
                cap_1 = id_dict['doc'] + "#" + id_dict['caption_1']
                cap_2 = id_dict['doc'] + "#" + id_dict['caption_2']
                data_dict['mention_pair_cap_ids'][pair_id] = (cap_1, cap_2)

                # Parse the mention pair indices as actual integers
                # and store those lists
                indices_str = id_split[1].strip().split(",")
                indices = list()
                for i in indices_str:
                    indices.append(int(i))
                data_dict['mention_pair_indices'][pair_id] = indices

                '''
                # Create the normalization vectors for each mention, associating
                # each with 2*n_max_seq 0s, except for indices corresponding
                # to the mention (twice, because we need both forward and backward);
                # Since we're averaging the forward and backward outputs of a mention,
                # we want the norm value to be 1 / 2|m|
                # NOTE: If we're running into memory problems, we should have one
                # of these arrays per unique mention, not one per appearance in a mention
                # pair
                norm_vec_i = np.zeros(2*data_dict['max_seq_len'])
                norm_vec_j = np.zeros(2*data_dict['max_seq_len'])
                norm_i = 1 / (2 * (1 + indices[1] - indices[0]))
                norm_j = 1 / (2 * (1 + indices[3] - indices[2]))
                for idx in range(indices[0], indices[1]+1):
                    norm_vec_i[idx] = norm_i
                    norm_vec_i[data_dict['max_seq_len'] + idx] = norm_i
                for idx in range(indices[2], indices[3]+1):
                    norm_vec_j[idx] = norm_j
                    norm_vec_j[data_dict['max_seq_len'] + idx] = norm_j
                #endfor
                data_dict['mention_pair_norm_vecs'][pair_id] = (norm_vec_i, norm_vec_j)
                '''

                # Represent the label as a one-hot (and since this is for
                # relations, we know this should be 4-dimensional
                label = np.zeros([4])
                label[int(id_split[2].strip())] = 1.0
                data_dict['mention_pair_labels'][pair_id] = label
            #endfor
        #endwith
    #endif

    # If feature files have been provided, add those to the data dict too
    if feats_file is not None and feats_meta_file is not None:
        meta_dict = None
        if feats_meta_file is not None:
            meta_dict = json.load(open(feats_meta_file, 'r'))
            data_dict['max_feat_idx'] = meta_dict['max_idx']
        X, _, IDs = data_util.load_sparse_feats(feats_file, meta_dict)
        data_dict['mention_pair_feats'] = dict()
        for i in range(0, len(IDs)):
            data_dict['mention_pair_feats'][IDs[i]] = X[i]
    #endif
    return data_dict
#enddef


def load_mention_data(mention_idx_file, n_classes,
                      feats_file=None, feats_meta_file=None):
    """
    Reads the mention index file, mapping mention IDs to
    first/last word indices, sentence IDs, labels, and
    normalization vectors (for averages, where appropriate).
    Optionally reads a feature and feature meta file and adds these
    vectors to the data dict

    :param mention_idx_file: File containing mention IDs and index tuples
    :param n_classes: Number of classes for this task
    :param feats_file: File containing engineered (sparse) features
    :param feats_meta_file: File associating engineered feature indices with
                            human-readable names
    :return Dictionary storing the aforementioned dictionaries
    """
    data_dict = dict()

    # Load the mention index file, which we assume is in the format
    #   <m_id>     <m1_start>,<m1_end>   <label>
    data_dict['mention_cap_ids'] = dict()
    data_dict['mention_indices'] = dict()
    data_dict['mention_norm_vecs'] = dict()
    data_dict['mention_labels'] = dict()
    if mention_idx_file is not None:
        with open(mention_idx_file, 'r') as f:
            for line in f.readlines():
                # Parse the line into the ID/indices/label pieces
                tab_split = line.split("\t")
                m_id = tab_split[0].strip()
                idx_split = tab_split[1].split(",")
                label_val = int(tab_split[2])

                # Parse the caption ID from the mention ID
                cap_id = m_id.split(";")[0]

                # Store the association between this mention
                # and its originating caption
                data_dict['mention_cap_ids'][m_id] = cap_id

                # Store the indices as an integer list
                indices = list()
                for i in idx_split:
                    indices.append(int(i))
                data_dict['mention_indices'][m_id] = indices

                '''
                # Create the normalization vectors for each mention, associating
                # each with 2*n_max_seq 0s, except for indices corresponding
                # to the mention (twice, because we need both forward and backward);
                # Since we're averaging the forward and backward outputs of a mention,
                # we want the norm value to be 1 / 2|m|
                # NOTE: If we're running into memory problems, we should have one
                # of these arrays per unique mention, not one per appearance in a mention
                # pair
                norm_vec = np.zeros(2 * data_dict['max_seq_len'])
                norm = 1 / (2 * (1 + indices[1] - indices[0]))
                for idx in range(indices[0], indices[1]+1):
                    norm_vec[idx] = norm
                    norm_vec[data_dict['max_seq_len'] + idx] = norm
                data_dict['mention_norm_vecs'][m_id] = norm_vec
                '''

                # Store the label as a one-hot for the binary label
                label_vec = np.zeros([n_classes])
                label_vec[label_val] = 1.0
                data_dict['mention_labels'][m_id] = label_vec
            #endfor
        #endwith
    #endif

    # If feature files have been provided, add those to the data dict too
    if feats_file is not None and feats_meta_file is not None:
        meta_dict = None
        if feats_meta_file is not None:
            meta_dict = json.load(open(feats_meta_file, 'r'))
            data_dict['max_feat_idx'] = meta_dict['max_idx']
        X, _, IDs = data_util.load_sparse_feats(feats_file, meta_dict)
        data_dict['mention_feats'] = dict()
        for i in range(0, len(IDs)):
            data_dict['mention_feats'][IDs[i]] = X[i]
    #endif
    return data_dict
#enddef


def build_model_file_name(arg_dict, model_type):
    """
    Builds the neural network model file name,
    given an argument dict; any None values are
    omitted from the resulting name
    :param arg_dict:
    :param model_type: relation_lstm, nonvis_lstm, relation_ffw, etc.
    :return:
    """

    # Set up the root of the model file
    model_file = arg_dict['data_root']
    if 'rel_type' in arg_dict and arg_dict['rel_type'] is not None:
        model_file += "_" + model_type.replace("_", "_" + arg_dict['rel_type'] + "_")
    else:
        model_file += "_" + model_type

    # Add the pair encoding scheme, if this is a relation model
    if 'pair_enc_scheme' in arg_dict:
        pair_enc_scheme = arg_dict['pair_enc_scheme']
        if pair_enc_scheme is not None:
            if pair_enc_scheme == 'first_avg_last':
                model_file += "_fal"
            elif pair_enc_scheme == 'first_last_sentence':
                model_file += "_fls"
        #endif
    #endif

    # Add the standard items that should be present across tasks, if specified
    model_file += "_" + arg_dict['activation'] + "_" + \
                  "epch" + str(int(arg_dict['epochs'])) + "_" + \
                  "lrn" + str(arg_dict['learn_rate']) + "_" + \
                  "btch" + str(int(arg_dict['batch_size'])) + "_" + \
                  "drp" + str(int(arg_dict['input_keep_prob'] * 100)) + \
                  str(int(arg_dict['other_keep_prob'] * 100)) + "_" + \
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


def load_batch_first_last_sentence_mention(mentions, data_dict,
                                           n_classes, n_embedding_feats=300):
    """
    Loads a batch of data, given a list of mentions
    and the data dictionary. This version of load_batch
    returns the necessary index matrices for producing a
        sent_first sent_last m_first m_last
    mention representation via lstm output tensor transformations

    :param mentions: List of mention pair IDs
    :param data_dict: Dictionary of all data dictionaries (for sentences, etc)
    :param n_classes: number of possible labels
    :param n_embedding_feats: size of word embeddings (typically 300)
    :return: dictionary of batch tensors with aforementioned keys
    """
    batch_tensors = dict()
    batch_size = len(mentions)
    n_seq = batch_size

    # Populate our sentence tensor and sequence length array
    batch_tensors['sentences'] = np.zeros([n_seq, data_dict['max_seq_len'], n_embedding_feats])
    batch_tensors['seq_lengths'] = np.zeros([n_seq])
    for i in range(0, batch_size):
        # set this sentence idx to the sentence embedding,
        # implicitly padding the end of the sequence with 0s
        sentence_id = data_dict['mention_cap_ids'][mentions[i]]
        sentence_matrix = data_dict['sentences'][sentence_id]
        for j in range(0, len(sentence_matrix)):
            batch_tensors['sentences'][i][j] = sentence_matrix[j]
        batch_tensors['seq_lengths'][i] = len(sentence_matrix)
    #endfor

    # We need four matrices of size [batch_size, 3] corresponding to the
    # first word of the mention, last word of the mention, the first word
    # in the sentence, and the last word in the sentence
    batch_tensors['first_indices'] = np.zeros([batch_size, 3])
    batch_tensors['last_indices'] = np.zeros([batch_size, 3])
    batch_tensors['sent_first_indices'] = np.zeros([batch_size, 3])
    batch_tensors['sent_last_indices'] = np.zeros([batch_size, 3])

    # We also need to account for mention features
    batch_tensors['mention_feats'] = np.zeros([batch_size, data_dict['max_feat_idx']+1])

    # Iterate through batch_size mentions, storing the appropriate
    # vectors into the tensors
    batch_tensors['labels'] = np.zeros([batch_size, n_classes])
    for i in range(0, batch_size):
        m_id = mentions[i]

        # Add this pair's label to the label batch
        batch_tensors['labels'][i] = data_dict['mention_labels'][m_id]

        # get this mention pair's word indices and caption indices
        first, last = data_dict['mention_indices'][m_id]

        # In this context, we know that the referred-to sentence index
        # is the batch index, so we can populate indices in the batch tensors
        batch_tensors['first_indices'][i] = np.array((1, i, first))
        batch_tensors['last_indices'][i] = np.array((0, i, last))
        batch_tensors['sent_first_indices'][i] = np.array((1, i, 0))
        batch_tensors['sent_last_indices'][i] = \
            np.array((0, i, batch_tensors['seq_lengths'][i] - 1))

        # Feature arrays, if specified; we don't have to
        # add anything to the tensors, because they're already 0s
        if m_id in data_dict['mention_feats']:
            batch_tensors['mention_feats'][i] = data_dict['mention_feats'][m_id]
    #endfor
    return batch_tensors
#enddef
