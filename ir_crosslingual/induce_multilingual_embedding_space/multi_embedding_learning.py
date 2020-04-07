from ir_crosslingual.induce_multilingual_embedding_space import mono_embedding_loading as mono
from ir_crosslingual.induce_multilingual_embedding_space import subspace_creation as sub
import numpy as np
from numpy.linalg import svd
import os

LEARNING_METHODS = {'procrustes'}  # todo: implement further methods


def learn_projection_matrix(s_vecs, t_vecs, train_expert_dict, s_word2id: dict = None, t_word2id: dict = None,
                            method: str = 'procrustes', n_max: int = 50000):
    """
    Learns projection matrix W that maps source language monolingual embedding into multilingual word embedding space.
    :param s_vecs: path of fastText source monolingual embedding text file or array of word embedding
    :param t_vecs: path of fastText target monolingual embedding text file or array of word embedding
                   (same type as s_vecs required)
    :param train_expert_dict: path of external expert training translation dictionary
    :param s_word2id: word/id dictionary, needed if only word vector embeddings are specified in s_vecs
    :param t_word2id: word/id dictionary, needed if only word vector embeddings are specified in t_vecs
    :param method: method to solve the learning problem
    :param n_max: maximum number of most frequent words that are loaded in monolingual word embeddings
    :return: projection matrix W
    """
    if method not in LEARNING_METHODS:
        raise ValueError("Method must be one of {}.".format(LEARNING_METHODS))
    if isinstance(s_vecs, np.ndarray) and s_word2id is None:
        raise TypeError("word2id dictionaries have to be specified if embeddings are given as numpy arrays.")

    if isinstance(s_vecs, str) and os.path.isfile(s_vecs):
        l1_emb, l1_id2word, l1_word2id = mono.load_monolingual_embedding(s_vecs, n_max)
        l2_emb, l2_id2word, l2_word2id = mono.load_monolingual_embedding(t_vecs, n_max)
        d_index, d_word = sub.extract_seed_dictionary(train_expert_dict, l1_word2id, l2_word2id)
        s_emb, t_emb = sub.align_monolingual_subspaces(l1_emb, l2_emb, d_index)
    else:
        d_index, d_word = sub.extract_seed_dictionary(train_expert_dict, s_word2id, t_word2id)
        s_emb, t_emb = sub.align_monolingual_subspaces(s_vecs, t_vecs, d_index)

    if method == 'procrustes':
        U, s, Vt = svd(np.transpose(s_emb) @ t_emb)
        W = U @ Vt
        return W
