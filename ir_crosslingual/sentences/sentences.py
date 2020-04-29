import io
import re
import math
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter

from ir_crosslingual.features import text_based
from ir_crosslingual.features import vector_based
from ir_crosslingual.embeddings.embeddings import WordEmbeddings
from ir_crosslingual.utils import strings


# Class for aligned translations for a given language pair
class Sentences:

    all_language_pairs = dict()

    AGGREGATION_METHODS = {'average', 'tf_idf'}

    def __init__(self,src_words: WordEmbeddings, trg_words: WordEmbeddings):
        # Initialize attributes indicating the languages of this translation pair
        self.src_lang = src_words.language
        self.trg_lang = trg_words.language
        self.languages = [self.src_lang, self.trg_lang]

        self.word_embeddings = {self.src_lang: src_words, self.trg_lang: trg_words}
        self.projection_matrix = WordEmbeddings.get_projection_matrix(src_lang=self.src_lang, trg_lang=self.trg_lang)

        self.sentences = dict()  # Initialized in load_sentences
        self.sentences_preprocessed = dict()  # Initialized in preprocess_sentences
        self.sentence_embeddings = dict()  # Initialized in transform_into_sentence_vectors
        # self.id2sentence = dict()  # Initialized in transform_into_sentence_vectors
        self.words_found = dict()  # Initialized in transform_into_sentence_vectors
        # self.words_found_embeddings = dict()  # Initialized in transform_into_sentence_vectors, needed for gas feature
        self.invalid_sentences = set()  # Initialized in transform_into_sentence_vectors

        # Default values for preprocessing settings and word embeddings
        self.to_lower = True
        self.remove_stopwords = True
        self.remove_punctuation = False
        self.agg_method = 'average'

        self.n_max = 500000
        self.single_source = bool

        self.prepared_features = list()
        self.src_prepared_features = list()
        self.trg_prepared_features = list()
        self.features_dict = dict()

        self.data = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()

        # Sentences.all_language_pairs['{}-{}'.format(self.src_lang, self.trg_lang)] = self

    def get_word_embeddings(self, source: bool = True):
        return self.word_embeddings[not source] if source else self.word_embeddings[not source]

    def load_sentences(self, language: str):
        """
        Loads sentences from Europarl text file.
        :param language: Language abbreviation for which to load Europarl sentences (short form, e.g. 'de')
        :return: list of Europarl sentences, striped
        """
        sentences = []
        try:
            path = strings.sentence_dictionaries['{}-{}'.format(self.src_lang, self.trg_lang)]
        except KeyError:
            print('No Europarl data available for this language')
        with io.open('{}.{}'.format(path, language), 'r', encoding='utf-8', newline='\n', errors='ignore') as file:
            for index, line in enumerate(file):
                if not line.strip() == '':
                    sentences.append(line.strip())
                if len(sentences) == self.n_max:
                    break
        return [" ".join(sen.split()) for sen in sentences]

    def preprocess_sentences(self):
        """
        Preprocesses sentences according to specified steps in self.load_data.
        Tokenized sentences, further preproccessed according to specified preprocessing steps are then stored in
        self.preprocessed_sentences[src_lang] and self.preprocessed_sentences[trg_lang]
        """
        for language in self.languages:
            if self.to_lower:
                self.sentences_preprocessed[language] = [sen.strip().lower() for sen in self.sentences[language]]

            if self.remove_punctuation:
                self.sentences_preprocessed[language] = [sen.translate(str.maketrans('', '', string.punctuation))
                                                         for sen in self.sentences_preprocessed[language]]

            self.sentences_preprocessed[language] = [re.findall(r"\w+|[^\w\s]", sen, re.UNICODE)
                                                     for sen in self.sentences_preprocessed[language]]

            if self.remove_stopwords:
                stops = set(stopwords.words(strings.languages[language]))
                self.sentences_preprocessed[language] = [[word for word in tokens if word not in stops]
                                                         for tokens in self.sentences_preprocessed[language]]

    # TODO: Adapt this function to new sentence structure
    @staticmethod
    def tf_idf(sentence_tokens: list):
        """
        Computes TF-IDF scores of term-sentence pairs.
        :param sentence_tokens: list of tokenized supervised_classification
        :return: list of TF-IDF scores of term-sentence pairs, arranged as dictionaries
        """
        df = dict(Counter(token for sen in sentence_tokens for token in set(sen)))
        idf = {k: math.log10(len(sentence_tokens) / v) for k, v in df.items()}
        tf = [{k: (1 + math.log10(v)) / (1 + math.log10(np.max(list(dic.values())))) for k, v in dic.items()} for dic in
              [dict(Counter(sen)) for sen in sentence_tokens]]
        tf_idf = [{k: v * idf[k] for k, v in dic.items()} for dic in tf]
        return tf_idf

    def transform_src_embedding_space(self):
        """
        Transforms source sentence embeddings to shared embedding space
        """
        self.sentence_embeddings[self.src_lang] = self.sentence_embeddings[self.src_lang] @ self.projection_matrix
        print('Embedding space of source language transformed according to projection matrix')

    def delete_invalid_sentences(self):
        """
        Delete sentences of invalid indices in all relevant variables
        """
        for idx in list(self.invalid_sentences):
            for language in self.languages:
                del self.sentences[language][idx]
                del self.sentences_preprocessed[language][idx]
                self.sentence_embeddings[language] = np.delete(self.sentence_embeddings[language], idx, axis=0)
                del self.words_found[language][idx]

    def transform_into_sentence_vectors(self):
        """
        Transform preprocessed sentences into sentence vectors.
        Transformed sentence embeddings are then stored in self.sentence_embeddings[src_lang] and
        self.sentence_embeddings[trg_lang]
        """
        for language in self.languages:
            word_embeddings = self.word_embeddings[language].embeddings
            sentences = self.sentences_preprocessed[language]
            word2id = self.word_embeddings[language].word2id

            words_found = [[word2id[word] for word in sen if word in word2id.keys()] for sen in sentences]

            self.invalid_sentences.update({i for i in range(len(sentences))
                                           if len(words_found[i]) == 0})

            if self.invalid_sentences:
                for i in self.invalid_sentences:
                    print("Could not find a term of the sentence '{}' in word embedding vocabulary and thus, "
                      "could not calculate the respective embedding vector.".format(self.sentences[language][i]))

            if self.agg_method == 'average':
                self.sentence_embeddings[language] = [sum(word_embeddings[words_found[i]]) / len(words_found[i])
                                                      if i not in self.invalid_sentences
                                                      else [0]*300 for i in range(len(sentences))]
                print('Sentences embeddings extracted in {}'.format(language))

            # TODO: Adapt tfidf version to new structure
            if self.agg_method == 'tf_idf':
                self.sentence_embeddings[language] = []
                if len(self.sentences[language]) == 1:
                    raise ZeroDivisionError(
                        "TF-IDF scores cannot be computed since number of supervised_classification equals 1. "
                        "Use 'average' instead.")
                tf_idf_scores = Sentences.tf_idf(self.sentences[language])
                for i, tokens in enumerate(self.sentences[language]):
                    if i not in self.invalid_sentences[language]:
                        vec = np.zeros((1, 300))
                        for token in tokens:
                            if token in self.word_embeddings[language].word2id.keys():
                                vec += tf_idf_scores[i][token] * self.word_embeddings[language].embeddings[self.word_embeddings[language].word2id[token]]
                        self.sentence_embeddings[language].append(vec / sum([v for k, v in tf_idf_scores[i].items() if k in self.word_embeddings[language].word2id.keys()]))
                    else:
                        self.sentence_embeddings[language].append(np.zeros(300))
            self.words_found[language] = words_found
        self.delete_invalid_sentences()
        self.transform_src_embedding_space()

    def prepare_features(self, features):
        """
        Prepare text-based features for each sentence individually
        :param features: List of feature names to prepare.
        If features = 'all', all features specified in text_based.py are prepared
        :return: self.data -> DataFrame that contains each source and target sentence in raw,
        preprocessed and embedded form alongside the prepared features for each sentence
        Return value not necessary -> self.data can also be accessed directly on the instance of this class
        """
        if features == 'all':
            self.prepared_features = text_based.PREPARED_FEATURES
        else:
            self.prepared_features = dict((name,function) for name, function in text_based.PREPARED_FEATURES.items() if name in features)
        if self.single_source:
            for name, function in self.prepared_features.items():
                print('Start preparation of feature {} in src sentence'.format(name))
                self.data['src_{}'.format(name)] = function[0](self.data['src_{}'.format(function[1])][0], **function[2])
            for name, function in self.prepared_features.items():
                print('Start preparation of feature {} in trg sentences'.format(name))
                self.data['trg_{}'.format(name)] = self.data.apply(
                    lambda row: function[0](row['trg_{}'.format(function[1])], **function[2]), axis=1)
        else:
            for name, function in self.prepared_features.items():
                print('Start preparation of feature {}'.format(name))
                for e in ['src', 'trg']:
                    self.data['{}_{}'.format(e, name)] = self.data.apply(
                        lambda row: function[0](row['{}_{}'.format(e, function[1])], **function[2]), axis=1)
        return self.data

    def load_data(self, src_sentences=None, trg_sentences=None, single_source: bool = False, n_max: int = 5000,
                  to_lower = True, remove_stopwords: bool = True, remove_punctuation: bool = False,
                  agg_method: str = 'average', features=None):
        """
        :param src_sentences: Single source sentence in string format.
        If None, Europarl sentences for the source language are loaded
        :param trg_sentences: List of target sentences or single target sentence in string format.
        If None, Europarl sentences for the target language are loaded
        :param single_source: Boolean variable indicating whether a single source sentence is considered
        or a list of different source sentences
        If True, single source sentence is considered
        If False, list of different source sentences is considered
        :param remove_stopwords: If True, stopwords are removed
        :param remove_punctuation: If True, punctuation marks are removed
        :param n_max: Number of maximum lines to be read (only required when loading Europarl data)
        :param to_lower: If true, set all characters in all sentences to lower case
        :param agg_method: Aggregation method for transforming list of word vectors into sentence vectors
        :param features: List of feature names to prepare.
        If features = 'all', all features specified in text_based.py are prepared
        :return: self.data -> DataFrame that contains each source and target sentence in raw,
        preprocessed and embedded form and, if specified so, alongside the prepared features for each sentence
        Return value not necessary -> self.data can also be accessed directly on the instance of this class
        Return -1, if no valid source sentence left after preprocessing
        Return -2, if no valid target sentence left after preprocessing
        """
        self.single_source = single_source
        self.n_max = n_max
        self.to_lower = to_lower
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        if agg_method not in Sentences.AGGREGATION_METHODS:
            raise ValueError("Method must be one of {}.".format(Sentences.AGGREGATION_METHODS))
        else:
            self.agg_method = agg_method
        self.sentences[self.trg_lang] = self.load_sentences(self.trg_lang) if trg_sentences is None \
            else [trg_sentences] if isinstance(trg_sentences, str) else trg_sentences
        print('Target sentences loaded')
        self.sentences[self.src_lang] = self.load_sentences(self.src_lang) if src_sentences is None \
            else [src_sentences] * len(self.sentences[self.trg_lang])
        print('Source sentences loaded')
        self.preprocess_sentences()
        print('Sentences preprocessed')
        self.transform_into_sentence_vectors()
        print('Sentences transformed')
        if len(self.sentences_preprocessed[self.src_lang]) == 0:
            print('No valid source sentence left after preprocessing steps')
            return -1
        if len(self.sentences_preprocessed[self.trg_lang]) == 0:
            print('No valid target sentence left after preprocessing steps')
            return -2
        self.data['src_sentence'] = self.sentences[self.src_lang]
        self.data['trg_sentence'] = self.sentences[self.trg_lang]
        self.data['src_preprocessed'] = self.sentences_preprocessed[self.src_lang]
        self.data['trg_preprocessed'] = self.sentences_preprocessed[self.trg_lang]
        self.data['src_embedding'] = list(self.sentence_embeddings[self.src_lang])
        self.data['trg_embedding'] = list(self.sentence_embeddings[self.trg_lang])
        if features is not None:
            self.prepare_features(features=features)
        return self.data

    # TODO: Adapt function to have less source sentences than target sentences in the test data set
    def create_datasets(self, n_train: int = 4000, n_test: int = 1000, frac_pos: float = 0.5):
        """
        Create train and test dataset based on given number of training and test instances
        and a given fraction of positive samples in the training and test dataset
        :param n_train: Number of instances in the training dataset
        :param n_test: Number of instances in the test dataset
        :param frac_pos: Fraction of positive samples in the training and test dataset
        :return: self.train_data, self.test_data -> DataFrames containing training and test datasets.
        Return value not necessary
        -> self.train_data and self.test_data can also be accessed directly on the instance of this class
        """
        df = self.data

        df_train = df[:n_train]
        df_test = df[-n_test:]

        self.src_prepared_features = ['src_{}'.format(feature) for feature in ['sentence', 'preprocessed', 'embedding']] \
                                     + ['src_{}'.format(feature) for feature in self.prepared_features]
        self.trg_prepared_features = ['trg_{}'.format(feature) for feature in ['sentence', 'preprocessed', 'embedding']]\
                                     + ['trg_{}'.format(feature) for feature in self.prepared_features]

        res_df = []

        for i, data in enumerate([(n_train, df_train), (n_test, df_test)]):
            n_pos = math.ceil(data[0] * frac_pos)
            df_pos = data[1][:n_pos]
            df_pos.loc[:, 'translation'] = 1
            df_neg = data[1][self.src_prepared_features][n_pos:data[0]]
            neg_indices = [np.random.choice(data[1].drop(index=i, axis=0).index, 1)[0] for i in df_neg.index]
            for feature in self.trg_prepared_features:
                df_neg[feature] = list(data[1][feature].loc[neg_indices])
            df_neg.loc[:, 'translation'] = 0
            res_df.append(df_pos.append(df_neg, ignore_index=True))

        self.train_data, self.test_data = tuple(res_df)
        return self.train_data, self.test_data

    def set_features_dict(self, features_dict):
        """
        Set instance variable self.features_dict to the dictionary that is passed as an argument
        :param features_dict: Dictionary of features, having 'text_based' and 'vectorbased' as keys
        """
        if features_dict == 'all':
            self.features_dict['text_based'] = text_based.FEATURES
            self.features_dict['vector_based'] = vector_based.FEATURES
        else:
            self.features_dict['text_based'] = dict((name, function)
                                                    for name, function in text_based.FEATURES.items()
                                                    if name in features_dict['text_based'])
            self.features_dict['vector_based'] = dict((name, function)
                                                      for name, function in vector_based.FEATURES.items()
                                                      if name in features_dict['vector_based'])

    def extraction(self, data: pd.DataFrame, evaluation=False):
        """
        Actual extraction of features on the given dataset
        :param data: Dataset to extract features for
        :param evaluation: If True, the dataset with the extracted features is passed as a return value and thus,
        this function can be used from outside for a given dataset. I.e., can then be used for evaluating the ranking
        of sentences
        :return: Dataset with all extracted features, only if evaluation == True
        """
        for name, function in self.features_dict['text_based'].items():
            data[name] = function[0](data['src_{}'.format(function[1])],
                                     data['trg_{}'.format(function[1])],
                                     self.single_source)
            # Optional: Drop individual columns with this feature
            data.drop(columns=['src_{}'.format(function[1]), 'trg_{}'.format(function[1])], inplace=True)

        for name, function in self.features_dict['vector_based'].items():
            data[name] = function[0](data['src_{}'.format(function[1])],
                                     data['trg_{}'.format(function[1])],
                                     self.single_source)
        if evaluation:
            return data

    def extract_features(self, features_dict, data='all'):
        """
        Sets self.features_dict to the given dictionary of features, having 'text_based' and 'vectorbased' as keys,
        and triggers feature extraction based on a given list of features (both text_based and vector_based)
        :param features_dict: Dictionary containing lists of text_based and vector_based features, respectively
        :param data: Variable indicating the dataset to extract features for.
        If 'train_test, features are extracted on self.train_data and self.test_data.
        If 'train', features are extracted on self.train_data only.
        If 'test', features are extracted on self.test_data only.
        else, features are extracted on self.data.
        :return: If data == 'train_test, return self.train_data and self.test_data.
        If data == 'train', return self.train_data only.
        If data == 'test', return self.test_data only.
        else, return self.data.
        Return value not necessary
        -> self.data, self.train_data and self.test_data can also be accessed directly on the instance of this class
        """
        # self.set_features_dict(self, features_dict=features_dict)
        if features_dict == 'all':
            self.features_dict['text_based'] = text_based.FEATURES
            self.features_dict['vector_based'] = vector_based.FEATURES
        else:
            self.features_dict['text_based'] = dict((name, function)
                                                    for name, function in text_based.FEATURES.items()
                                                    if name in features_dict['text_based'])
            self.features_dict['vector_based'] = dict((name, function)
                                                      for name, function in vector_based.FEATURES.items()
                                                      if name in features_dict['vector_based'])

        if data == 'train_test':
            for data in [self.train_data, self.test_data]:
                self.extraction(data=data)
            return self.train_data, self.test_data
        elif data == 'train':
            self.extraction(data=self.train_data)
            return self.train_data
        elif data == 'test':
            self.extraction(data=self.test_data)
            return self.test_data
        else:
            self.extraction(data=self.data)
            return self.data


if __name__ == '__main__':
    """
    Test section
    """
    sens = Sentences(None, None)
