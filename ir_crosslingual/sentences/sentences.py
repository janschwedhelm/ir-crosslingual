import io, re, math, string, random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from ir_crosslingual.features import text_based
from ir_crosslingual.features import vector_based
from ir_crosslingual.embeddings.embeddings import WordEmbeddings
from ir_crosslingual.utils import paths


# Class for aligned translations for a given language pair
class Sentences:

    all_language_pairs = dict()

    AGGREGATION_METHODS = {'average', 'tf_idf'}

    def __init__(self, src_words: WordEmbeddings, trg_words: WordEmbeddings):
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
        self.words_found_embeddings = dict()  # Initialized in transform_into_sentence_vectors, needed for gas feature
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

        self.vectorizer = dict()

        self.data = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.test_collection = pd.DataFrame()

        self.dim_emb = 300

        # Sentences.all_language_pairs['{}-{}'.format(self.src_lang, self.trg_lang)] = self

    @classmethod
    def load_from_file(cls, train_data: str, test_collection: str, file_format: str = 'pkl',
                       src_language: str = 'en', trg_language: str = 'de'):
        source = WordEmbeddings(src_language)
        source.load_embeddings()

        target = WordEmbeddings(trg_language)
        target.load_embeddings()

        W_st, W_ts = WordEmbeddings.learn_projection_matrix(src_lang=src_language, trg_lang=trg_language)

        sens = Sentences(source, target)

        if file_format == 'pkl':
            sens.train_data = pd.read_pickle(train_data)
            print(f'---- INFO: File loaded containing training data')
            sens.test_collection = pd.read_pickle(test_collection)
            print(f'---- INFO: File loaded containing test collection')
        elif file_format == 'csv':
            sens.train_data = pd.read_csv(train_data)
            sens.test_collection = pd.read_csv(test_collection)

        sens.prepared_features = set([feature[4:] for feature in list(sens.train_data.columns)
                                      if feature[4:] in list(text_based.PREPARED_FEATURES.keys())])

        try:
            sens.features_dict['text_based'] = [feature for feature in list(sens.train_data.columns)
                                                if feature in list(text_based.FEATURES.keys())]
        except KeyError:
            print(f'---- INFO: No text-based features have been extracted in this file')
            pass

        try:
            sens.features_dict['vector_based'] = [feature for feature in list(sens.train_data.columns)
                                                  if feature in list(vector_based.FEATURES.keys())]
        except KeyError:
            print(f'---- INFO: No vector-based features have been extracted in this file')
            pass

        try:
            sens.features_dict['vector_elements'] = [f'src_embedding_{i}' for i in range(300)
                                                  if f'src_embedding_{i}' in list(sens.train_data.columns)] + \
                                                 [f'trg_embedding_{i}' for i in range(300)
                                                  if f'trg_embedding_{i}' in list(sens.train_data.columns)]
        except KeyError:
            print(f'---- INFO: No vector elements as features have been extracted in this file')
        print(f'---- DONE: All files loaded and features extracted')
        return sens, sens.train_data.copy(), sens.test_collection.copy(), \
               [feature for values in sens.features_dict.values() for feature in values]

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
            path = paths.sentence_dictionaries['{}-{}'.format(self.src_lang, self.trg_lang)]
        except KeyError:
            print('---- ERROR: No Europarl data available for this language')
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
                stops = set(stopwords.words(paths.languages[language]))
                self.sentences_preprocessed[language] = [[word for word in tokens if word not in stops]
                                                         for tokens in self.sentences_preprocessed[language]]

    def fit_tfidf_vectorizer(self, n_train: int):
        vectorizer = dict()
        for language in self.languages:
            vectorizer[language] = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
            vectorizer[language].fit(self.sentences_preprocessed[language][:n_train])
        return vectorizer

    @staticmethod
    def tf(sentence: list):
        """
        Compute term-frequency values for each unique term in a given (tokenized) sentence
        :param sentence: Sentence to compute tf values for, given as a list of tokens
        :return: term-frequency values of the given sentence
        """
        sen_counter = dict(Counter(sen for sen in sentence))
        tf_values = {k: (1 + math.log10(v)) / (1 + math.log10(np.max(list(sen_counter.values())))) for k, v in
                     sen_counter.items()}
        return tf_values

    def tf_idf(self, sentence: list, language: str):
        """
        Computes TF-IDF scores of all terms in a given sentence.
        :param sentence: Sentence to compute tf-idf weights for, given as a list of tokens
        :param language: language of the sentence
        :return: list of TF-IDF scores of term-sentence pairs, arranged as dictionaries
        """
        tf_values = self.tf(sentence)
        tf_idf_values = {k: v * self.vectorizer[language].idf_[self.vectorizer[language].vocabulary_[k]]
                         for k, v in tf_values.items() if k in self.vectorizer[language].vocabulary_.keys()}
        return tf_idf_values

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
                del self.words_found_embeddings[language][idx]

    def get_found_word_embeddings(self):
        print(f'---- INFO: Shape of source word embeddings')
        src_embeddings = self.word_embeddings[self.src_lang].embeddings @ self.projection_matrix
        trg_embeddings = self.word_embeddings[self.trg_lang].embeddings.copy()
        self.words_found_embeddings[self.src_lang] = [[src_embeddings[word] for word in sentence]
                                                     for sentence in self.words_found[self.src_lang]]
        self.words_found_embeddings[self.trg_lang] = [[trg_embeddings[word] for word in sentence]
                                                      for sentence in self.words_found[self.trg_lang]]

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
            # words_found_embeddings = [[word_embeddings[word] for word in sentence] for sentence in words_found]

            self.invalid_sentences.update({i for i in range(len(sentences))
                                           if len(words_found[i]) == 0})

            if self.invalid_sentences:
                for i in self.invalid_sentences:
                    print(f'---- ERROR: Sentence embedding failed in {language} on ID {i}: {self.sentences[language][i]}')

            if self.agg_method == 'average':
                self.sentence_embeddings[language] = [sum(word_embeddings[words_found[i]]) / len(words_found[i])
                                                      if i not in self.invalid_sentences
                                                      else [0]*300 for i in range(len(sentences))]
                print(f'---- INFO: Sentences embeddings extracted in {language}')

            elif self.agg_method == 'tf_idf':
                self.sentence_embeddings[language] = []
                for i, tokens in enumerate(sentences):
                    if i not in self.invalid_sentences:
                        tf_idf_scores = {word2id[k]: v for k, v in self.tf_idf(tokens, language=language).items()
                                         if k in word2id.keys()}
                        vec = np.zeros((1, 300))
                        if (i % 10000) == 0:
                            print(f'---- INFO: Starting sentence vector aggregation for index {i}, language {language}')
                        for word_idx, tf_idf_score in tf_idf_scores.items():
                            vec += tf_idf_score * word_embeddings[word_idx]
                        self.sentence_embeddings[language].append(vec / sum(tf_idf_scores.values()))
                    else:
                        self.sentence_embeddings[language].append([0]*300)
                self.sentence_embeddings[language] = np.vstack(self.sentence_embeddings[language])

            self.words_found[language] = words_found
        self.get_found_word_embeddings()
        print(f'---- INFO: Extracted word embeddings of found words')
        self.delete_invalid_sentences()

    def prepare_features(self, features):
        """
        Prepare text-based features for each sentence individually
        :param features: List of feature names to prepare.
        If features = 'all', all features specified in text_based.py are prepared
        :return: self.data -> DataFrame that contains each source and target sentence in raw,
        preprocessed and embedded form alongside the prepared features for each sentence
        Return value not necessary -> self.data can also be accessed directly on the instance of this class
        """
        self.prepared_features = dict((name, function) for name, function in text_based.PREPARED_FEATURES.items() if name in features)

        if self.single_source:
            for name, function in self.prepared_features.items():
                print(f'---- INFO: Start preparation of text-based feature {name} in src sentence')
                if name == 'translated_words':
                    self.data['src_translated_words'] = function[0](self.data['src_{}'.format(function[1])],
                                                                    WordEmbeddings.seed_dicts['{}-{}'.format(self.src_lang, self.trg_lang)])
                else:
                    self.data['src_{}'.format(name)] = function[0](self.data['src_{}'.format(function[1])][0], **function[2])
            for name, function in self.prepared_features.items():
                print(f'---- INFO: Start preparation of text-based feature {name} in trg sentences')
                if name == 'translated_words':
                    self.data['trg_translated_words'] = function[0](self.data['trg_{}'.format(function[1])],
                                                                    WordEmbeddings.seed_dicts[
                                                                        '{}-{}'.format(self.trg_lang, self.src_lang)])
                else:
                    self.data['trg_{}'.format(name)] = self.data.apply(
                        lambda row: function[0](row['trg_{}'.format(function[1])], **function[2]), axis=1)
        else:
            for name, function in self.prepared_features.items():
                print(f'---- INFO: Start preparation of text-based feature {name}')
                if name == 'translated_words':
                    self.data['src_translated_words'] = function[0](self.data['src_{}'.format(function[1])],
                                                                    WordEmbeddings.seed_dicts[
                                                                        '{}-{}'.format(self.src_lang, self.trg_lang)])
                    self.data['trg_translated_words'] = function[0](self.data['trg_{}'.format(function[1])],
                                                                    WordEmbeddings.seed_dicts[
                                                                        '{}-{}'.format(self.trg_lang, self.src_lang)])
                else:
                    for e in ['src', 'trg']:
                        self.data['{}_{}'.format(e, name)] = self.data.apply(
                            lambda row: function[0](row['{}_{}'.format(e, function[1])], **function[2]), axis=1)
        return self.data

    def reduce_dim(self, data, new_dim, use_ppa=True, threshold=8):
        # 1. PPA #1
        # PCA to get Top Components
        self.dim_emb = new_dim

        def ppa(data, N, D):
            pca = PCA(n_components=N)
            data = data - np.mean(data)
            _ = pca.fit_transform(data)
            U = pca.components_

            z = []

            # Removing Projections on Top Components
            for v in data:
                for u in U[0:D]:
                    v = v - np.dot(u.transpose(), v) * u
                z.append(v)
            return np.asarray(z)

        X = np.vstack(data)

        if use_ppa:
            X = ppa(X, X.shape[1], threshold)

        # 2. PCA
        # PCA Dim Reduction
        pca = PCA(n_components=new_dim)
        X = X - np.mean(X)
        X = pca.fit_transform(X)

        # 3. PPA #2
        if use_ppa:
            X = ppa(X, new_dim, threshold)

        return pd.Series(X.tolist())

    def load_data(self, src_sentences=None, trg_sentences=None, single_source: bool = False, n_max: int = 5000,
                  to_lower: bool = True, remove_stopwords: bool = True, remove_punctuation: bool = False,
                  agg_method: str = 'average', features=None, vectorizer=None):
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
        :param vectorizer: Contains previously trained tfidf vectorizers for source and target language.
        If int, take this amount of data instances for training a new tfidf vectorizer (in this case, this number
        should be the same as the number of training instances)
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
        if (agg_method == 'tf_idf') and (vectorizer is None):
            raise ValueError('Vectorizer must be specified '
                             'when choosing tf-idf as aggregation method for sentence embeddings')
        self.sentences[self.trg_lang] = self.load_sentences(self.trg_lang) if trg_sentences is None \
            else [trg_sentences] if isinstance(trg_sentences, str) else trg_sentences
        print('---- DONE: Target sentences loaded')
        self.sentences[self.src_lang] = self.load_sentences(self.src_lang) if src_sentences is None \
            else [src_sentences] * len(self.sentences[self.trg_lang])
        print('---- DONE: Source sentences loaded')
        self.preprocess_sentences()
        print('---- DONE: Sentences preprocessed')
        self.vectorizer = self.fit_tfidf_vectorizer(vectorizer) if isinstance(vectorizer, int) else vectorizer
        self.transform_into_sentence_vectors()
        print('---- DONE: Sentences transformed')
        self.data['src_embedding'] = list(self.sentence_embeddings[self.src_lang])
        self.data['src_embedding_aligned'] = list(self.sentence_embeddings[self.src_lang] @ self.projection_matrix)
        print('---- INFO: Embedding space of source language transformed according to projection matrix')
        self.data['trg_embedding'] = list(self.sentence_embeddings[self.trg_lang])
        if len(self.sentences_preprocessed[self.src_lang]) == 0:
            print('---- ERROR: No valid source sentence left after preprocessing steps')
            return -1
        if len(self.sentences_preprocessed[self.trg_lang]) == 0:
            print('---- ERROR: No valid target sentence left after preprocessing steps')
            return -2
        self.data['src_sentence'] = self.sentences[self.src_lang]
        self.data['trg_sentence'] = self.sentences[self.trg_lang]
        self.data['src_preprocessed'] = self.sentences_preprocessed[self.src_lang]
        self.data['trg_preprocessed'] = self.sentences_preprocessed[self.trg_lang]
        self.data['src_words'] = [[word for word in sen if word.isalpha()]
                                  for sen in self.sentences_preprocessed[self.src_lang]]
        print('---- DONE: Source words extracted')
        self.data['trg_words'] = [[word for word in sen if word.isalpha()]
                                  for sen in self.sentences_preprocessed[self.trg_lang]]
        print('---- DONE: Target words extracted')
        self.data['src_words_found_embedding'] = list(self.words_found_embeddings[self.src_lang])
        self.data['trg_words_found_embedding'] = list(self.words_found_embeddings[self.trg_lang])
        print(f'---- INFO: Embeddings of found words added as a column')
        if features is not None:
            self.prepare_features(features=features)
        print('---- DONE: All features prepared')
        self.data.drop_duplicates(['src_sentence', 'trg_sentence'], inplace=True)
        print('---- DONE: Dropped duplicates and created full dataset')
        print(f'---- INFO: Length of dataset after preprocessing and duplicate handling: {len(self.data)}')
        return self.data

    def build_separate_prepared_features_list(self):
        self.src_prepared_features = ['src_{}'.format(feature)
                                      for feature in
                                      ['sentence', 'preprocessed', 'embedding', 'embedding_aligned', 'words', 'words_found_embedding']] \
                                      + ['src_{}'.format(feature) for feature in self.prepared_features]

        self.trg_prepared_features = ['trg_{}'.format(feature)
                                      for feature in ['sentence', 'preprocessed', 'embedding', 'words', 'words_found_embedding']] \
                                      + ['trg_{}'.format(feature) for feature in self.prepared_features]

    def create_train_set(self, n_train: int, frac_pos: float):
        df = self.data
        df_train = df[:n_train]

        if len(self.src_prepared_features) == 0:
            self.build_separate_prepared_features_list()

        n_pos = math.ceil(n_train * frac_pos)
        df_pos = df_train[:n_pos]
        df_pos['translation'] = 1
        print('---- INFO: Translation dataframe created')
        df_neg = df_train[self.src_prepared_features][n_pos:n_train]
        print('---- INFO: Non-translation dataframe created')
        neg_indices = list(df_neg.index)
        random.shuffle(neg_indices)
        print('---- INFO: Determined and shuffled non-translation indices')
        for feature in self.trg_prepared_features:
            df_neg[feature] = list(df_train[feature].loc[neg_indices])
            print(f'---- INFO: Feature column {feature} appended')
        print('---- INFO: All features appended')
        df_neg['translation'] = df_neg['trg_sentence'] == df_train.iloc[n_pos:n_train]['trg_sentence']
        print('---- INFO: Added non-translation indicator')

        self.train_data = df_pos.append(df_neg, ignore_index=True)
        print('---- DONE: Training dataset created')

        return self.train_data

    def create_test_collection(self, n_queries: int, n_docs: int):
        df_test = self.data[-n_docs:]

        if len(self.src_prepared_features) == 0:
            self.build_separate_prepared_features_list()

        df_queries = df_test[:n_queries][self.src_prepared_features]
        print('---- INFO: Preliminary queries dataframe created')
        df_docs = df_test[self.trg_prepared_features]
        print('---- INFO: Preliminary documentes dataframe created')

        cart_prod = pd.merge(df_queries.assign(key=0), df_docs.assign(key=0), on='key').drop('key', axis=1)
        print('---- INFO: Merged queries and documents dataframe')

        self.test_collection = pd.merge(cart_prod, df_test, how='left', on=['src_sentence', 'trg_sentence'],
                                        indicator='translation')
        print('---- INFO: Merged with test dataframe')

        self.test_collection['translation'] = np.where(self.test_collection.translation == 'both', 1, 0)
        print('---- INFO: Added translation indicator')

        self.test_collection.rename(columns={col: col.split('_x')[0] for col in self.test_collection.columns
                                        if col.endswith('_x')}, inplace=True)

        self.test_collection.drop(columns=[col for col in self.test_collection.columns if col.endswith('_y')],
                                  inplace=True)
        print('---- DONE: Test collection created')
        return self.test_collection

    def set_features_dict(self, features_dict):
        """
        Set instance variable self.features_dict to the dictionary that is passed as an argument
        :param features_dict: Dictionary of features, having 'text_based' and 'vector_based' as keys
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

    def extraction(self, data: pd.DataFrame, evaluation=False, drop_prepared=True):
        """
        Actual extraction of features on the given dataset
        :param data: Dataset to extract features for
        :param evaluation: If True, the dataset with the extracted features is passed as a return value and thus,
        this function can be used from outside for a given dataset. I.e., can then be used for evaluating the ranking
        of sentences
        :param drop_prepared: If true, drop columns with prepared features
        :return: Dataset with all extracted features, only if evaluation == True
        """
        try:
            for name, function in self.features_dict['text_based'].items():
                print(f'---- INFO: Started extraction of text-based feature {name}')
                if 'occ' in name:
                    data[name] = function[0](data['src_{}'.format(function[1])],
                                             data['trg_{}'.format(function[1])],
                                             self.single_source)

                elif 'translated' in name:
                    data[name] = data.apply(lambda row: function[0](row['src_{}'.format(function[1][0])],
                                                                    row['trg_{}'.format(function[1][0])],
                                                                    row['src_{}'.format(function[1][1])],
                                                                    row['trg_{}'.format(function[1][1])]
                                                                    ), axis=1)
                elif name[:3] == 'abs':
                    data[name] = abs(data[f'src_{function[1]}'] - data[f'trg_{function[1]}'])
                elif name[:3] == 'rel':
                    data[name] = data[f'src_{function[1]}'] - data[f'trg_{function[1]}']
                elif name[:4] == 'norm':
                    data[name] = function[0](data['src_{}'.format(function[1])],
                                             data['trg_{}'.format(function[1])],
                                             self.single_source)
        except KeyError as key:
            print(f'---- ERROR: Key error occurred in text-based features for key {key}')
            pass

        if drop_prepared:
            data.drop(columns=['src_{}'.format(feature) for feature in self.prepared_features]
                      + ['trg_{}'.format(feature) for feature in self.prepared_features], inplace=True)

        try:
            for name, function in self.features_dict['vector_based'].items():
                print(f'---- INFO: Started extraction of vector-based feature {name}')
                data[name] = data.apply(lambda row: function[0](row['src_{}'.format(function[1])],
                                                                row['trg_{}'.format(function[2])],
                                                                self.single_source), axis=1)
        except KeyError:
            pass

        try:
            if int(len(self.features_dict['vector_elements'])/2) == 300:
                print('---- INFO: Started extracting vector elements')
                self.dim_emb = 300
                for prefix in ['src', 'trg']:
                    data[['{}_embedding_{}'.format(prefix, i) for i in range(self.dim_emb)]] \
                        = pd.DataFrame(self.data['{}_embedding'.format(prefix)].tolist())
            elif self.features_dict['vector_elements']:
                print('---- INFO: Started extracting vector elements')
                self.dim_emb = int(len(self.features_dict['vector_elements'])/2)
                for prefix in ['src', 'trg']:
                    data[['{}_embedding_{}'.format(prefix, i) for i in range(self.dim_emb)]] \
                        = pd.DataFrame(self.reduce_dim(self.data['{}_embedding'.format(prefix)], self.dim_emb).tolist())

        except KeyError:
            pass

        if evaluation:
            return data

    def extract_features(self, features_dict, data='all', drop_prepared=True):
        """
        Sets self.features_dict to the given dictionary of features, having 'text_based' and 'vectorbased' as keys,
        and triggers feature extraction based on a given list of features (both text_based and vector_based)
        :param features_dict: Dictionary containing lists of text_based and vector_based features, respectively
        :param data: Variable indicating the dataset to extract features for.
        If 'train_test, features are extracted on self.train_data and self.test_data.
        If 'train', features are extracted on self.train_data only.
        If 'test', features are extracted on self.test_data only.
        else, features are extracted on self.data.
        :param drop_prepared: If true, drop columns with prepared features
        :return: If data == 'train_test, return self.train_data and self.test_data.
        If data == 'train', return self.train_data only.
        If data == 'test', return self.test_data only.
        else, return self.data.
        Return value not necessary
        -> self.data, self.train_data and self.test_data can also be accessed directly on the instance of this class
        """
        # self.set_features_dict(self, features_dict=features_dict)
        try:
            self.features_dict['text_based'] = dict((name, function)
                                                    for name, function in text_based.FEATURES.items()
                                                    if name in features_dict['text_based'])
        except KeyError:
            print('---- INFO: No text-based features specified')
            pass

        try:
            self.features_dict['vector_based'] = dict((name, function)
                                                      for name, function in vector_based.FEATURES.items()
                                                      if name in features_dict['vector_based'])
        except KeyError:
            print('---- INFO: No vector-based features specified')
            pass

        try:
            self.features_dict['vector_elements'] = features_dict['vector_elements']
        except KeyError:
            print('---- INFO: No vector elements as features specified')
            pass

        if data == 'train_test':
            self.extraction(data=self.train_data, drop_prepared=drop_prepared)
            print('---- DONE: All given features extracted on train dataset\n-----------------------')
            self.extraction(data=self.test_collection, drop_prepared=drop_prepared)
            print('---- DONE: All given features extracted on test collection')
            return self.train_data, self.test_collection
        elif data == 'train':
            self.extraction(data=self.train_data, drop_prepared=drop_prepared)
            return self.train_data
        elif data == 'test':
            self.extraction(data=self.test_collection, drop_prepared=drop_prepared)
            return self.test_collection
        else:
            self.extraction(data=self.data, drop_prepared=drop_prepared)
            return self.data


if __name__ == '__main__':
    """
    Test section
    """
    sens = Sentences(None, None)
