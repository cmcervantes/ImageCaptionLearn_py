from gensim.models import KeyedVectors as gs_kv
import numpy as np

__author__ = "ccervantes"


class Word2Vec:
    """
    Word2VecUtil serves as a wrapper for
    word2vec loading and retrieval
    """

    def __init__(self, w2v_bin_file):
        """
        Initializes the word2vec util, loading
        the specified binary file
        :param w2v_bin_file - pretrained word2vec binary file
        """
        self._w2v_model = gs_kv.load_word2vec_format(w2v_bin_file, binary=True)
    #enddef

    def get_w2v_matrix(self, sentence):
        """
        Given a sentence, returns a word2vec matrix

        :param sentence: N-dimensional list of words
        :return:         an Nx300 dimensional w2v matrix
        """

        # Initialize our matrix
        w2v_matrix = np.empty(shape=(len(sentence), 300))

        # iterate throuch each word in the sentence,
        # replacing unknown words with UNK
        for i in range(0, len(sentence)):
            word = sentence[i]
            if word not in self._w2v_model:
                word = "UNK"
            w2v_matrix[i] = self._w2v_model[word]
        return w2v_matrix
    #enddef
#endclass

