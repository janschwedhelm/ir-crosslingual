import numpy as np
from googletrans import Translator
import io
import os


def create_translation_dictionary(s_words: list, s_lan: str, t_lan: str, n_max: int = 20000,
                                  write_to_path: str = None):
    """
    Create translation dictionary via Google Translate API.
    :param s_words: list of words to be translated
    :param s_lan: source language
    :param t_lan: target language
    :param n_max: maximum number of words to be translated
    :param write_to_path: if not None, translation dictionary is written to specified path
    :return: dictionary with source words as keys and respective translations as values
    """
    # translator API has to be reinitialized on every iteration in order to bypass Google API's request limit
    translated_dict = {word_src: Translator().translate(word_src, src=s_lan, dest=t_lan).text.lower()
                       for index, word_src in enumerate(s_words) if index < n_max}
    if write_to_path:
        with io.open(write_to_path, 'w') as file:
            for k, v in translated_dict.items():
                file.write("{} {}\n".format(k, v))
            file.close()
    return translated_dict


def extract_seed_dictionary(trans_dict, s_word2id: dict, t_word2id: dict):
    """
    Extract seed dictionary from external expert dictionary according to vocabularies included in monolingual embedding
    spaces.
    :param trans_dict: external expert dictionary (either text file or Python dictionary)
    :param s_word2id: source dictionary of words and indices as returned from load_monolingual_embedding
    :param t_word2id: target dictionary of indices and words as returned from load_monolingual_embedding
    :return: index/word pairs of resulting seed dictionary
    """
    index_pairs = []
    word_pairs = []
    misfit = 0
    misfit_s = 0
    misfit_t = 0

    if isinstance(trans_dict, str) and os.path.isfile(trans_dict):
        with io.open(trans_dict, 'r', encoding='utf-8') as file:
            for index, word_pair in enumerate(file):
                s_word, t_word = word_pair.rstrip().split()
                if s_word in s_word2id and t_word in t_word2id:
                    index_pairs.append((s_word2id[s_word], t_word2id[t_word]))
                    word_pairs.append((s_word, t_word))
                else:
                    misfit += 1
                    misfit_s += int(s_word not in s_word2id)
                    misfit_t += int(t_word not in t_word2id)
            print('Found {} valid translation pairs.\n'
                  '{} other pairs contained at least one unknown word ({} in source language, {} in target language).'
                  .format(len(word_pairs), misfit, misfit_s, misfit_t))
            return index_pairs, word_pairs

    elif isinstance(trans_dict, dict):
        for s_word, t_word in trans_dict.items():
            if s_word in s_word2id and t_word in t_word2id:
                index_pairs.append((s_word2id[s_word], t_word2id[t_word]))
                word_pairs.append((s_word, t_word))
            else:
                misfit += 1
                misfit_s += int(s_word not in s_word2id)
                misfit_t += int(t_word not in t_word2id)
        print('Found {} valid translation pairs.\n'
              '{} other pairs contained at least one unknown word ({} in source language, {} in target language).'
              .format(len(word_pairs), misfit, misfit_s, misfit_t))
        return index_pairs, word_pairs

    else:
        print('Invalid translation dictionary type. Text file or Python dictionary is required.')
        return False


def align_monolingual_subspaces(s_embedding: np.ndarray, t_embedding: np.ndarray, seed_dictionary: list):
    """
    Create aligned monolingual subspaces from seed dictionary.
    :param s_embedding: monolingual source embedding as returned from load_monolingual_embedding
    :param t_embedding: monolingual target embedding as returned from load_monolingual_embedding
    :param seed_dictionary: index pairs of seed dictionary as returned from extract_seed_dictionary
    :return: aligned source and target subspaces
    """
    s_subspace = s_embedding[[tuples[0] for tuples in seed_dictionary]]
    t_subspace = t_embedding[[tuples[1] for tuples in seed_dictionary]]
    print("Resulting subspace dimension: {}".format(s_subspace.shape))
    return s_subspace, t_subspace

