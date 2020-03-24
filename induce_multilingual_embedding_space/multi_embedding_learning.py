from mono_embedding_loading import load_monolingual_embedding
from subspace_creation \
    import create_translation_dictionary, extract_seed_dictionary, align_monolingual_subspaces
import numpy as np
from numpy.linalg import svd

LEARNING_METHODS = {'procrustes'}  # todo: implement further methods


def projection_learning(s_path: str, t_path: str, trans_dict, method: str = 'procrustes', n_max: int = 50000):
    if method not in LEARNING_METHODS:
        raise ValueError("Method must be one of {}.".format(LEARNING_METHODS))

    X_L1, id2word_L1, word2id_L1 = load_monolingual_embedding(s_path, n_max)
    X_L2, id2word_L2, word2id_L2 = load_monolingual_embedding(t_path, n_max)
    D_index, D_word = extract_seed_dictionary(trans_dict, word2id_L1, word2id_L2)

    X_S, X_T = align_monolingual_subspaces(X_L1, X_L2, D_index)

    if method == 'procrustes':
        U, s, Vt = svd(np.transpose(X_S) @ X_T)
        W = U @ Vt
        return W



