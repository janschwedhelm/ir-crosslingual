import io
import numpy as np


def load_monolingual_embedding(path: str, n_max: int = 50000):
    """
    Load fastText Wiki text model in Python.
    :param path: path of fastText monolingual embedding text file
    :param n_max: maximum number of most frequent words that are loaded
    :return: 300-dim embedding vectors of respective words, mappings of indices and words (i.e. Python dictionaries)
    """
    vectors = []
    word2id = {}
    with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as file:
        next(file)
        for index, line in enumerate(file):
            word, vec = line.rstrip().split(' ', 1)
            vec = np.fromstring(vec, sep=' ')
            vectors.append(vec)
            word2id[word] = index
            if len(word2id) == n_max:
                break
    embeddings = np.vstack(vectors)
    id2word = {v: k for k, v in word2id.items()}
    return embeddings, id2word, word2id
