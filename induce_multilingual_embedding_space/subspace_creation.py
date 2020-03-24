from googletrans import Translator
import io
import os


def create_translation_dictionary(src_words: list, src_lan: str, tgt_lan: str, n_max: int = 20000,
                                  write_to_path: str = None):
    """
    Create translation dictionary via Google Translate API.
    :param src_words: list of words to be translated
    :param src_lan: source language
    :param tgt_lan: target language
    :param n_max: maximum number of words to be translated
    :param write_to_path: if not None, translation dictionary is written to specified path
    :return: dictionary with source words as keys and respective translations as values
    """
    # translator API has to be reinitialized on every iteration in order to bypass Google API's request limit
    translated_dict = {word_src: Translator().translate(word_src, src=src_lan, dest=tgt_lan).text.lower()
                       for index, word_src in enumerate(src_words) if index < n_max}
    if write_to_path:
        with io.open(write_to_path, 'w') as file:
            for k, v in translated_dict.items():
                file.write("{} {}\n".format(k, v))
            file.close()
    return translated_dict


def extract_seed_dictionary(trans_dict, src_word2id: dict, tgt_word2id: dict):
    """
    Extract seed dictionary from external expert dictionary according to vocabularies included in monolingual embedding
    spaces.
    :param trans_dict: external expert dictionary (either text file or Python dictionary)
    :param src_word2id: source dictionary of words and indices as returned from load_monolingual_embedding
    :param tgt_word2id: target dictionary of indices and words as returned from load_monolingual_embedding
    :return: index/word pairs of resulting seed dictionary
    """
    index_pairs = []
    word_pairs = []
    misfit = 0
    misfit_src = 0
    misfit_tgt = 0

    if isinstance(trans_dict, str) and os.path.isfile(trans_dict):
        with io.open(trans_dict, 'r', encoding='utf-8') as file:
            for index, word_pair in enumerate(file):
                src_word, tgt_word = word_pair.rstrip().split()
                if src_word in src_word2id and tgt_word in tgt_word2id:
                    index_pairs.append((src_word2id[src_word], tgt_word2id[tgt_word]))
                    word_pairs.append((src_word, tgt_word))
                else:
                    misfit += 1
                    misfit_src += int(src_word not in src_word2id)
                    misfit_tgt += int(tgt_word not in tgt_word2id)
            print('Found {} valid translation pairs.\n'
                  '{} other pairs contained at least one unknown word ({} in source language, {} in target language).'
                  .format(len(word_pairs), misfit, misfit_src, misfit_tgt))
            return index_pairs, word_pairs

    elif isinstance(trans_dict, dict):
        for src_word, tgt_word in trans_dict.items():
            if src_word in src_word2id and tgt_word in tgt_word2id:
                index_pairs.append((src_word2id[src_word], tgt_word2id[tgt_word]))
                word_pairs.append((src_word, tgt_word))
            else:
                misfit += 1
                misfit_src += int(src_word not in src_word2id)
                misfit_tgt += int(tgt_word not in tgt_word2id)
        print('Found {} valid translation pairs.\n'
              '{} other pairs contained at least one unknown word ({} in source language, {} in target language).'
              .format(len(word_pairs), misfit, misfit_src, misfit_tgt))
        return index_pairs, word_pairs

    else:
        print('Invalid translation dictionary type. Text file or Python dictionary is required.')
        return False
