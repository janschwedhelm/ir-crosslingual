from ir_crosslingual.induce_multilingual_embedding_space import mono_embedding_loading as mono
from ir_crosslingual.induce_multilingual_embedding_space import subspace_creation as sub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

SIMILARITY_MEASURES = {'cosine'}  # todo: implement further similarity measures, e.g. CSLS


def evaluate_multilingual_embedding(s_emb, t_emb, proj_matrix: np.ndarray, test_expert_dict, s_word2id: dict = None,
                                    t_word2id: dict = None, s_nmax: int = 50000, t_nmax: int = 200000,
                                    measure: str = 'cosine', k: int = 1):
    """
    Evaluate induced multilingual word embedding using an independent test expert dictionary.
    :param s_emb: path of fastText source monolingual embedding text file or array of word embedding
    :param t_emb: path of fastText target monolingual embedding text file or array of word embedding
                   (same type as s_vecs required)
    :param proj_matrix: learned projection matrix
    :param test_expert_dict: path of external expert test translation dictionary
    :param s_word2id: word/id dictionary, needed if only word vector embeddings are specified in s_vecs
    :param t_word2id: word/id dictionary, needed if only word vector embeddings are specified in t_vecs
    :param s_nmax: maximum number of most frequent source words that are loaded in monolingual word embeddings
    :param t_nmax: maximum number of most frequent target words that are loaded in monolingual word embeddings
    :param measure: similarity measure that is used to compute similarity between vectors
    :param k: k used for precision@k, which accounts for the fraction of pairs for which the correct translation of
              the source words is in the k-th nearest neighbors (with respect to measure)
    :return: precision@k, top k translations for each word
    """
    if measure not in SIMILARITY_MEASURES:
        raise ValueError("Similarity measure must be one of {}.".format(SIMILARITY_MEASURES))

    if isinstance(s_emb, str) and os.path.isfile(s_emb):
        s_emb, s_id2word, s_word2id = mono.load_monolingual_embedding(s_emb, s_nmax)
        t_emb, t_id2word, t_word2id = mono.load_monolingual_embedding(t_emb, t_nmax)
        true_translations_indices, true_translations_words = sub.extract_seed_dictionary(test_expert_dict,
                                                                                     s_word2id=s_word2id,
                                                                                     t_word2id=t_word2id)
    elif isinstance(s_emb, np.ndarray) and s_word2id is not None:
        s_id2word = {v: k for k, v in s_word2id.items()}
        t_id2word = {v: k for k, v in t_word2id.items()}
        true_translations_indices, true_translations_words = sub.extract_seed_dictionary(test_expert_dict,
                                                                                     s_word2id=s_word2id,
                                                                                     t_word2id=t_word2id)
    else:
        raise TypeError("Invalid input: word2id dictionaries have to be specified "
                        "if embeddings are given as numpy arrays.")

    s_test_indices = list({tup[0] for tup in true_translations_indices})
    print("Aims to find correct translations between {} source words "
          "and {} target words.".format(len(s_test_indices), len(list({tup[1] for tup in true_translations_indices}))))
    if len(s_test_indices) == 0:
        raise ZeroDivisionError("Cannot evaluate since no word pair of test dictionary is covered by "
                                "monolingual vocabularies.")

    if measure == 'cosine':
        topk_translations = {s_id2word[i]: [t_id2word[word_id]
                                            for word_id in cosine_similarity((s_emb[i] @ proj_matrix)
                                                                             .reshape(1, -1), t_emb)[0]
                                                               .argsort()[-k:][::-1]] for i in s_test_indices}

    correct_translations = sum([bool(set(topk_translations[s_id2word[index]])
                                     .intersection(set([tup[1] for tup in true_translations_words
                                                        if tup[0] == s_id2word[index]]))) for index in s_test_indices])

    return correct_translations/len(s_test_indices), topk_translations
