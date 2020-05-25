import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


# Updated functions
def euclidean_dist(src_vec, trg_vec, single_source=False):
    """
    Computes euclidean distance between the embeddings of two supervised_classification
    :param src_vec: Sentence embedding of the source sentence
    :param trg_vec: Sentence embedding of the target sentence
    :param single_source: True if only one source sentence is compared to multiple target sentences
    (can be used when ranking target sentences for a single source sentence, e.g. in the WebApp)
    :return: Euclidean distance of the source and the target sentence embedding
    """
    return distance.euclidean(src_vec, trg_vec)


def cos_sim(src_vec, trg_vec, single_source=False):
    """
    Computes cosine similarity between the embeddings of two sentences
    :param src_vec: Sentence embedding of the source sentence
    :param trg_vec: Sentence embedding of the target sentence
    :param single_source: True if only one source sentence is compared to multiple target sentences
    (can be used when ranking target sentences for a single source sentence, e.g. in the WebApp)
    :return: Cosine similarity of the source and the target sentence embedding
    """
    return cosine_similarity(src_vec.reshape(1, -1), trg_vec.reshape(1, -1))[0][0]


# Dictionary of all vector_based features that can be extracted
# alongside the corresponding function that needs to be executed for the given feature
# Structure: {'feature_name': [function to be called, column on which the function needs to be performed]}
FEATURES = {
    'euclidean_distance': [euclidean_dist, 'embedding_aligned', 'embedding'],
    'cosine_similarity': [cos_sim, 'embedding_aligned', 'embedding']
}
