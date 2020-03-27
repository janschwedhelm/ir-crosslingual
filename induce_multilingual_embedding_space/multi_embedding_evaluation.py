from mono_embedding_loading import load_monolingual_embedding
from subspace_creation \
    import create_translation_dictionary, extract_seed_dictionary, align_monolingual_subspaces
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_MEASURES = {'cosine'}  # todo: implement further similarity measures, e.g. CSLS


def evaluate_multilingual_embedding(s_path: str, t_path: str, proj_matrix: np.ndarray, test_expert_dict,
                                    s_nmax: int = 50000, t_nmax: int = 200000, measure: str = 'cosine', k: int = 1):
    """
    Evaluate induced multilingual word embedding using an independent test expert dictionary.
    :param s_path: path of fastText source monolingual embedding text file
    :param t_path: path of fastText target monolingual embedding text file
    :param proj_matrix: learned projection matrix
    :param test_expert_dict: path of external expert test translation dictionary
    :param s_nmax: maximum number of most frequent source words that are loaded in monolingual word embeddings
    :param t_nmax: maximum number of most frequent target words that are loaded in monolingual word embeddings
    :param measure: similarity measure that is used to compute similarity between vectors
    :param k: k used for precision@k, which accounts for the fraction of pairs for which the correct translation of
              the source words is in the k-th nearest neighbors (with respect to measure)
    :return: precision@k, top k translations for each word
    """
    if measure not in SIMILARITY_MEASURES:
        raise ValueError("Similarity measure must be one of {}.".format(SIMILARITY_MEASURES))

    s_emb, s_id2word, s_word2id = load_monolingual_embedding(s_path, s_nmax)
    t_emb, t_id2word, t_word2id = load_monolingual_embedding(t_path, t_nmax)
    true_translations_indices, true_translations_words = extract_seed_dictionary(test_expert_dict, s_word2id=s_word2id,
                                                                                 t_word2id=t_word2id)

    s_test_indices = list({tup[0] for tup in true_translations_indices})

    if measure == 'cosine':
        topk_translations = {s_id2word[i]: [t_id2word[word_id]
                                            for word_id in cosine_similarity((s_emb[i] @ proj_matrix)
                                                                             .reshape(1, -1), t_emb)[0]
                                                               .argsort()[-k:][::-1]] for i in s_test_indices}

    correct_translations = sum([bool(set(topk_translations[s_id2word[index]])
                                     .intersection(set([tup[1] for tup in true_translations_words
                                                        if tup[0] == s_id2word[index]]))) for index in s_test_indices])

    return correct_translations/len(s_test_indices), topk_translations
