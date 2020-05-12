import io
import os
import pickle
import numpy as np
from numpy.linalg import svd
from ir_crosslingual.utils import paths

class WordEmbeddings:

    # Class attributes
    all_embeddings = dict()
    projection_matrices = dict()
    seed_words = dict()
    seed_ids = dict()

    seed_dicts = dict()

    N_MAX = 50000 # Having n_max as a class attribute means that for each language
    # we always load the same number of most frequent words.
    # Alternatively, this could also be passed to load_embeddings as a parameter
    LEARNING_METHODS = ['procrustes']  # todo: implement further methods

    def __init__(self, language):
        # Initiates new instance for given language. I.e., one instance for each language
        self.language = language
        self.embeddings = None
        self.word2id = dict()
        self.id2word = dict()
        self.aligned_subspace = dict()

        WordEmbeddings.all_embeddings[language] = self

    def load_embeddings(self):
        """
        Load monolingual word embeddings for the given language and save them in self.embeddings.
        Also store self.word2id and self.id2word dictionaries
        :return:
        """
        # Load embeddings
        self.embeddings = np.load('{}embeddings.npy'.format(paths.monolingual_embedding_paths[self.language]))
        # Load word2id dictionary
        with open('{}word2id.pkl'.format(paths.monolingual_embedding_paths[self.language]), 'rb') as f:
            self.word2id = pickle.load(f)
        # Create id2word dictionary
        self.id2word = {v: k for k, v in self.word2id.items()}

    def align_monolingual_embeddings(self, languages: str, source: bool):
        """
        Align monolingual embeddings of self.language where self.language is either the source language
        if source == True and self.language is the target language if source == False.
        Aligned embedding is then stored for self.language in self.aligned_subspace[other_language]
        :param languages: Language pair to align the embedding space. Given in 'src-trg' format
        :param source: If True, self.language is the source language.
        If False, self.language is the target language
        """
        seed_dict = WordEmbeddings.get_seed_ids(src_lang=languages[:2], trg_lang=languages[-2:])
        self.aligned_subspace[languages[-2:] if source else languages[:2]] = self.embeddings[[tuples[not source] for tuples in seed_dict]]
        print("---- INFO: Resulting subspace dimension: {}".format(self.aligned_subspace[languages[-2:] if source else languages[:2]].shape))

    @classmethod
    def get_embeddings(cls, language: str):
        try:
            return cls.all_embeddings[language]
        except KeyError:
            print('---- ERROR: Embeddings for language {} do not exist yet.'.format(language))
            return -1

    @classmethod
    def get_projection_matrix(cls, src_lang: str, trg_lang: str):
        languages = '{}-{}'.format(src_lang, trg_lang)
        try:
            return cls.projection_matrices[languages]
        except KeyError:
            print('---- ERROR: Projection matrix for language pair {} does not exist yet.'.format(languages))
            return -1

    @classmethod
    def get_seed_words(cls, src_lang: str, trg_lang: str):
        languages = '{}-{}'.format(src_lang, trg_lang)
        try:
            return cls.seed_words[languages]
        except KeyError:
            print('---- ERROR: Seed words dictionary for language pair {} does not exist yet.'.format(languages))
            return -1

    @classmethod
    def get_seed_ids(cls, src_lang: str, trg_lang: str):
        languages = '{}-{}'.format(src_lang, trg_lang)
        try:
            return cls.seed_ids[languages]
        except KeyError:
            print('---- ERROR: Seed IDs dictionary for language pair {} does not exist yet.'.format(languages))
            return -1

    @classmethod
    def set_seed_dictionary(cls, src_lang: str, trg_lang: str):
        """
        Set seed dictionary for a given source/target language pair.
        Seed dictionary is then stored in WordEmbeddings.seed_words[language_pair]
        and WordEmbeddings.seed_ids[language_pair], for language_pair in in 'src-trg' format
        :param src_lang: Source language of the seed dictionary (short form, e.g. 'de')
        :param trg_lang: Target language of the seed dictionary (short form, e.g. 'de')
        """
        # TODO: Check that languages are in list

        source = cls.get_embeddings(src_lang)
        target = cls.get_embeddings(trg_lang)
        languages = '{}-{}'.format(src_lang, trg_lang)

        expert_dict = paths.expert_dictionaries[languages]
        index_pairs = []
        word_pairs = []
        misfit = 0
        misfit_s = 0
        misfit_t = 0

        if isinstance(expert_dict, str) and os.path.isfile(expert_dict):
            with io.open(expert_dict, 'r', encoding='utf-8') as file:
                for index, word_pair in enumerate(file):
                    s_word, t_word = word_pair.rstrip().split()
                    if s_word in source.word2id and t_word in target.word2id:
                        index_pairs.append((source.word2id[s_word], target.word2id[t_word]))
                        word_pairs.append((s_word, t_word))
                    else:
                        misfit += 1
                        misfit_s += int(s_word not in source.word2id)
                        misfit_t += int(t_word not in target.word2id)
                print('---- INFO: Found {} valid translation pairs in expert dictionary.\n'
                      '---- INFO: {} other pairs contained at least one unknown word ({} in source language, {} in target language).'
                      .format(len(word_pairs), misfit, misfit_s, misfit_t))
                # return index_pairs, word_pairs
                cls.seed_words[languages] = word_pairs
                cls.seed_ids[languages] = index_pairs
                print(f'---- DONE: Seed dictionary extracted for the languages: {languages}')

                # Create seed dictionary from cls.seed_words
                cls.seed_dicts[languages] = dict()
                for s in cls.seed_words[languages]:
                    try:
                        cls.seed_dicts[languages][s[0]].append(s[1])
                    except KeyError:
                        cls.seed_dicts[languages][s[0]] = [s[1]]

        elif isinstance(expert_dict, dict):
            for s_word, t_word in expert_dict.items():
                if s_word in source.word2id and t_word in target.word2id:
                    index_pairs.append((source.word2id[s_word], target.word2id[t_word]))
                    word_pairs.append((s_word, t_word))
                else:
                    misfit += 1
                    misfit_s += int(s_word not in source.word2id)
                    misfit_t += int(t_word not in target.word2id)
            print('---- INFO:Found {} valid translation pairs.\n'
                  '---- INFO: {} other pairs contained at least one unknown word ({} in source language, {} in target language).'
                  .format(len(word_pairs), misfit, misfit_s, misfit_t))
            # return index_pairs, word_pairs
            cls.seed_words[languages] = word_pairs
            cls.seed_ids[languages] = index_pairs
            print(f'---- DONE: Seed dictionary extracted for the languages: {languages}')

        else:
            print(expert_dict)
            print('Invalid translation dictionary type. Text file or Python dictionary is required.')
            return -1

    @classmethod
    def learn_projection_matrix(cls, src_lang: str, trg_lang: str, method: str = 'procrustes', extract_seed: bool = True):
        """
        Learn projection matrices for a language pair in both directions.
        Projection matrices are then stored in WordEmbeddings.projection_matrices['src_lang-trg_lang']
        and WordEmbeddings.projection_matrices['trg_lang-src_lang']
        :param src_lang: Source language (short form, e.g. 'de')
        :param trg_lang: Target language (short form, e.g. 'de')
        :param method: Learning method
        :param extract_seed: Boolean variable to indicate whether the seed dictionary has be set or has been set yet.
        If True, set seed dictionary in this function.
        If False, seed dictionary has already been set yet and is stored in WordEmbeddings.seed_words and
        WordEmbeddings.seed_ids already
        :return: Return projection matrices in both directions
        """
        for s_lang, t_lang in zip([src_lang, trg_lang], [trg_lang, src_lang]):
            # TODO: Always check that languages are in list
            print('---- INFO: Learn projection matrix for {}-{}'.format(s_lang, t_lang))
            source = cls.get_embeddings(s_lang)
            target = cls.get_embeddings(t_lang)
            languages = '{}-{}'.format(s_lang, t_lang)

            if extract_seed:
                WordEmbeddings.set_seed_dictionary(src_lang=s_lang, trg_lang=t_lang)

            if (source.embeddings is None) or (target.embeddings is None):
                print('---- ERROR: Monolingual word embeddings for source and target languages have to be loaded first.')
                return -1

            if method not in cls.LEARNING_METHODS:
                raise ValueError("---- ERROR: Method must be one of {}.".format(cls.LEARNING_METHODS))
            if isinstance(source.embeddings, np.ndarray) and source.word2id is None:
                raise TypeError("---- ERROR: word2id dictionaries have to be specified if "
                                "embeddings are given as numpy arrays.")
            if isinstance(target.embeddings, np.ndarray) and target.word2id is None:
                raise TypeError("---- ERROR: word2id dictionaries have to be specified if "
                                "embeddings are given as numpy arrays.")

            # Align subspaces
            source.align_monolingual_embeddings(languages=languages, source=True)
            target.align_monolingual_embeddings(languages=languages, source=False)

            if method == 'procrustes':
                U, s, Vt = svd(np.transpose(source.aligned_subspace[t_lang]) @ target.aligned_subspace[s_lang])
                W = U @ Vt
                cls.projection_matrices[languages] = W
            print(f'---- DONE: Projection matrix from {s_lang} to {t_lang}')
        return cls.projection_matrices['{}-{}'.format(src_lang, trg_lang)],\
            cls.projection_matrices['{}-{}'.format(trg_lang, src_lang)]

    # TODO: Add evaluation of multilingual embedding spaces


if __name__ == '__main__':
    german = WordEmbeddings('de')
    german.load_embeddings()

    english = WordEmbeddings('en')
    english.load_embeddings()

    WordEmbeddings.set_seed_dictionary(src_lang='en', trg_lang='de')
    # print('\n', WordEmbeddings.get_seed_words('en', 'de')) # [:100] does not work if -1 is returned (in error case)
    print('\n', WordEmbeddings.seed_words['en-de'][:100])

    WordEmbeddings.learn_projection_matrix(src_lang='en', trg_lang='de')
    # print(WordEmbeddings.get_projection_matrix('en', 'de')) # .shape does not work if -1 is returned (in error case)
    print(WordEmbeddings.projection_matrices['en-de'].shape)