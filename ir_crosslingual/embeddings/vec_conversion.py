import io, os
import numpy as np
import pickle

from ir_crosslingual.utils import paths


def load_vec(language: str, n_max: int = 50000):
    vectors = []
    path = paths.monolingual_embedding_vec_paths[language]
    word2id = dict()
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
    return embeddings, word2id


def save_files(language: str, n_max: int = 50000):
    if language not in paths.monolingual_embedding_paths.keys():
        raise KeyError('There\'s no monolingual embedding path for this language yet.')
    embeddings, word2id = load_vec(language, n_max)
    path = paths.monolingual_embedding_paths[language]
    if not os.path.exists(path):
        os.makedirs(path)
    # Save embeddings
    np.save('{}embeddings'.format(path), embeddings)
    print('Embeddings saved for {}'.format(language))
    # Save word2id dictionary
    with open('{}word2id.pkl'.format(path), 'wb+') as f:
        pickle.dump(word2id, f, pickle.HIGHEST_PROTOCOL)
    print('word2id dictionary saved for {}'.format(language))


if __name__ == '__main__':
    for language in 'en de fr'.split():
        save_files(language)
