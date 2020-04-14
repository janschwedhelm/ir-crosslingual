from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


def cosine_sim(s_vec, t_vec):
    """
    Computes cosine similarity between the embeddings of two sentences
    :param s_vec: Sentence embedding of the source sentence
    :param t_vec: Sentence embedding of the target sentence
    :return: Cosine similarity of the source and the target sentence embedding
    """
    cos_sim = cosine_similarity(s_vec.reshape(1,-1), t_vec.reshape(1,-1))
    return cos_sim[0][0]


def euclidean_dist(s_vec, t_vec):
    """
    Computes euclidean distance between the embeddings of two sentences
    :param s_vec: Sentence embedding of the source sentence
    :param t_vec: Sentence embedding of the target sentence
    :return: Euclidean distance of the source and the target sentence embedding
    """
    return distance.euclidean(s_vec, t_vec)


def jenson_shannon_dist(s_vec, t_vec):
    """
    Computes jenson shannon distance between the embeddings of two sentences
    :param s_vec: Sentence embedding of the source sentence
    :param t_vec: Sentence embedding of the target sentence
    :return: Jenson-shannon distance of the source and the target sentence embedding
    """
    # TODO: Not working yet, currently returns inf for all vectors in the dataframe
    return distance.jensenshannon(s_vec, t_vec)


def wasserstein_dist(s_vec, t_vec):
    """
    Computes wasserstein distance between the embeddings of two sentences
    :param s_vec: Sentence embedding of the source sentence
    :param t_vec: Sentence embedding of the target sentence
    :return: Wasserstein distance of the source and the target sentence embedding
    """
    return wasserstein_distance(s_vec, t_vec)


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
        tmp_sim = cosine_sim(initial_word, word)
        if tmp_sim > max_sim[1]:
            max_sim = (word, tmp_sim)
    return max_sim


def ngas(s_sen: list, t_sen: list):
    """
    Computes the non-symmetric greedy association similarity.
    I.e., the sum of all most similar alignments between the source and the target sentence,
    normalized by the length of the source sentence
    :param s_sen: List of word embeddings of all words in the source sentence
    :param t_sen: List of word embeddings of all words in the target sentence
    :return: Non-symmetric greedy association similarity of the source and the target sentence
    """
    similarities_sum = 0.0
    for word in s_sen:
        similarities_sum += most_similar(word, t_sen)[1]
    return similarities_sum / len(s_sen)


def gas(s_sen: list, t_sen: list):
    """
    Computes the symmetric greedy association similarity.
    I.e., the sum of both non-symmetric association similarities (both directions), divided by 2
    :param s_sen: List of word embeddings of all words in the source sentence
    :param t_sen: List of word embeddings of all words in the target sentence
    :return: Symmetric greedy association similarity of the source and the target sentence
    """
    return 0.5*(ngas(s_sen, t_sen) + ngas(t_sen, s_sen))
