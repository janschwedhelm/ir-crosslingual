import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


# Updated functions
def cos_sim(src_vec, trg_vec, single_source=False):
    """
    Computes cosine similarity between the embeddings of two sentences
    :param src_vec: Sentence embedding of the source sentence
    :param trg_vec: Sentence embedding of the target sentence
    :param single_source: True if only one source sentence is compared to multiple target sentences
    (can be used when ranking target sentences for a single source sentence, e.g. in the WebApp)
    :return: Cosine similarity of the source and the target sentence embedding
    """
    if single_source:
        return cosine_similarity(src_vec[0].reshape(1, -1), np.vstack(trg_vec))[0]
    else:
        return cosine_similarity(src_vec.reshape(1, -1), trg_vec.reshape(1, -1))


# TODO: Old functions that need to adapted to new structure
def euclidean_dist(src_vec, trg_vec):
    """
    Computes euclidean distance between the embeddings of two supervised_classification
    :param src_vec: Sentence embedding of the source sentence
    :param trg_vec: Sentence embedding of the target sentence
    :return: Euclidean distance of the source and the target sentence embedding
    """
    return distance.euclidean(src_vec, trg_vec)


def jenson_shannon_dist(src_vec, trg_vec):
    """
    Computes jenson shannon distance between the embeddings of two supervised_classification
    :param src_vec: Sentence embedding of the source sentence
    :param trg_vec: Sentence embedding of the target sentence
    :return: Jenson-shannon distance of the source and the target sentence embedding
    """
    # TODO: Not working yet, currently returns inf for all vectors in the dataframe
    return distance.jensenshannon(src_vec, trg_vec)


def wasserstein_dist(src_vec, trg_vec):
    """
    Computes wasserstein distance between the embeddings of two supervised_classification
    :param src_vec: Sentence embedding of the source sentence
    :param trg_vec: Sentence embedding of the target sentence
    :return: Wasserstein distance of the source and the target sentence embedding
    """
    return wasserstein_distance(src_vec, trg_vec)


def most_similar(initial_word: str, words2compare: list):
    """
    Identifies the most similar word from a list of words in comparison the the given initial word
    :param initial_word: Embeddings of the initial word that has to be compared to all of the words from words2compare
    (given in source language)
    :param words2compare: List of word embeddings of the words that have to be compared to the initial_word (given in target language)
    :return: Tuple consisting of the embedding of the most similar word from words2compare and its cosine similarity to the initial_word
    """
    # TODO: So far very inefficient and time consuming
    # IDEA: Maybe save similarity values that have been computed before?
    max_sim = ('', 0.0)
    for word in words2compare:
        tmp_sim = cos_sim(initial_word, word)
        if tmp_sim > max_sim[1]:
            max_sim = (word, tmp_sim)
    return max_sim


def ngas(src_sen: list, trg_sen: list):
    """
    Computes the non-symmetric greedy association similarity.
    I.e., the sum of all most similar alignments between the source and the target sentence,
    normalized by the length of the source sentence
    :param src_sen: List of word embeddings of all words in the source sentence
    :param trg_sen: List of word embeddings of all words in the target sentence
    :return: Non-symmetric greedy association similarity of the source and the target sentence
    """
    similarities_sum = 0.0
    for word in src_sen:
        similarities_sum += most_similar(word, trg_sen)[1]
    try:
        return similarities_sum / len(src_sen)
    except ZeroDivisionError:
        return -1


def gas(src_sen: list, trg_sen: list):
    """
    Computes the symmetric greedy association similarity.
    I.e., the sum of both non-symmetric association similarities (both directions), divided by 2
    :param src_sen: List of word embeddings of all words in the source sentence
    :param trg_sen: List of word embeddings of all words in the target sentence
    :return: Symmetric greedy association similarity of the source and the target sentence
    """
    return 0.5*(ngas(src_sen, trg_sen) + ngas(trg_sen, src_sen))


# Dictionary of all vector_based features that can be extracted
# alongside the corresponding function that needs to be executed for the given feature
# Structure: {'feature_name': [function to be called, column on which the function needs to be performed]}
FEATURES = {
    'cosine_similarity': [cos_sim, 'embedding']
}
