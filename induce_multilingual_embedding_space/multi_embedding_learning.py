from mono_embedding_loading import load_monolingual_embedding
from subspace_creation \
    import create_translation_dictionary, extract_seed_dictionary, align_monolingual_subspaces
import numpy as np
from numpy.linalg import svd

LEARNING_METHODS = {'procrustes'}  # todo: implement further methods


def learn_projection_matrix(s_path: str, t_path: str, train_expert_dict, method: str = 'procrustes', n_max: int = 50000):
    """
    Learns projection matrix W that maps source language monolingual embedding into multilingual word embedding space.
    :param s_path: path of fastText source monolingual embedding text file
    :param t_path: path of fastText target monolingual embedding text file
    :param train_expert_dict: path of external expert training translation dictionary
    :param method: method to solve the learning problem
    :param n_max: maximum number of most frequent words that are loaded in monolingual word embeddings
    :return: projection matrix W
    """
    if method not in LEARNING_METHODS:
        raise ValueError("Method must be one of {}.".format(LEARNING_METHODS))

    l1_emb, l1_id2word, l1_word2id = load_monolingual_embedding(s_path, n_max)
    l2_emb, l2_id2word, l2_word2id = load_monolingual_embedding(t_path, n_max)
    d_index, d_word = extract_seed_dictionary(train_expert_dict, l1_word2id, l2_word2id)

    s_emb, t_emb = align_monolingual_subspaces(l1_emb, l2_emb, d_index)

    if method == 'procrustes':
        U, s, Vt = svd(np.transpose(s_emb) @ t_emb)
        W = U @ Vt
        return W
