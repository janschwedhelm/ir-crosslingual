import io
import numpy as np
import re
from nltk.corpus import stopwords
import string
from collections import Counter
import math

AGGREGATION_METHODS = {'average', 'tf_idf'}
LANGUAGES = {'arabic', 'azerbaijani', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'greek', 'hungarian',
             'indonesian', 'italian', 'kazakh', 'nepali', 'norwegian', 'portuguese', 'romanian', 'russian', 'slovene',
             'spanish', 'swedish', 'tajik', 'turkish'}


def load_sentences(path: str, n_max: int = 500000):
    """
    Loads sentences from Europarl text file.
    :param path: path of Europarl text file
    :param n_max: number of maximum lines to be read
    :return: list of sentences, lower case and striped
    """
    sentences = []
    with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as file:
        for index, line in enumerate(file):
            if not line.strip().lower() == '':
                sentences.append(line.strip().lower())
            if len(sentences) == n_max:
                break
    return [" ".join(sen.lower().split()) for sen in sentences]


def preprocess_sentences(sentences: list, language: str, to_lower: bool = True,
                         remove_stopwords: bool = True, remove_punctuation: bool = False):
    """
    Preprocesses sentences according to specified steps.
    :param sentences: list of sentence strings
    :param language: language of sentence strings
    :param to_lower: if True, sentences are converted to lower case
    :param remove_stopwords: if True, stopwords are removed
    :param remove_punctuation: if True, punctuation is removed
    :return: list of tokenized sentences, further processed according to specified preprocessing steps
    """
    if language not in LANGUAGES:
        raise ValueError("Language must be one of {}.".format(LANGUAGES))
    if to_lower:
        sentences = [sen.strip().lower() for sen in sentences]
    if remove_punctuation:
        sentences = [sen.translate(str.maketrans('', '', string.punctuation)) for sen in sentences]

    sentences_preprocessed = [re.findall(r"\w+|[^\w\s]", sen, re.UNICODE) for sen in sentences]

    if remove_stopwords:
        stops = set(stopwords.words(language))
        sentences_preprocessed = [[word for word in tokens if word not in stops] for tokens in sentences_preprocessed]
    return sentences_preprocessed


def tf_idf(sentence_tokens: list):
    """
    Computes TF-IDF scores of term-sentence pairs.
    :param sentence_tokens: list of tokenized sentences
    :return: list of TF-IDF scores of term-sentence pairs, arranged as dictionaries
    """
    df = dict(Counter(token for sen in sentence_tokens for token in set(sen)))
    idf = {k: math.log10(len(sentence_tokens)/v) for k, v in df.items()}
    tf = [{k: (1 + math.log10(v)) / (1 + math.log10(np.max(list(dic.values())))) for k, v in dic.items()} for dic in
          [dict(Counter(sen)) for sen in sentence_tokens]]
    tf_idf = [{k: v * idf[k] for k, v in dic.items()} for dic in tf]
    return tf_idf


def transform_into_sentence_vectors(sentences: list, language: str, word_emb: np.ndarray, word2id: dict,
                                    agg_method: str = 'average', ignore_stopwords=True, ignore_punctuation=False):
    """
    Transforms sentences into sentence embedding vectors.
    :param sentences: list of sentences to be transformed
    :param language: language of sentences
    :param word_emb: word embeddings of specified language
    :param word2id: word/id dictionary to map words and respective word embeddings
    :param agg_method: aggregation method
    :param ignore_stopwords: if True, stopwords are ignored when calculating sentence embedding vectors
    :param ignore_punctuation: if True, punctuation is ignored when calculating sentence embedding vectors
    :return: 300-dim sentence embeddings vectors
    """
    if agg_method not in AGGREGATION_METHODS:
        raise ValueError("Method must be one of {}.".format(AGGREGATION_METHODS))

    if not isinstance(sentences, list):
        sentences = [sentences]
    sentences_preprocessed = preprocess_sentences(sentences, language, ignore_stopwords, ignore_punctuation)

    words_found = [[word2id[word] for word in sen if word in word2id.keys()] for sen in sentences_preprocessed]
    invalid_sentences = {i for i in range(len(sentences)) if len(words_found[i]) == 0}

    if invalid_sentences:
        for i in invalid_sentences:
            print("Could not find a term of the sentence '{}' in word embedding vocabulary and thus, "
                  "could not calculate the respective embedding vector.".format(sentences[i]))

    if agg_method == 'average':
        sen_emb = [sum(word_emb[words_found[i]]) / len(words_found[i])
                   for i in range(len(sentences)) if i not in invalid_sentences]

    if agg_method == 'tf_idf':
        if len(sentences) == 1:
            raise ZeroDivisionError("TF-IDF scores cannot be computed since number of sentences equals 1. "
                                    "Use 'average' instead.")
        tf_idf_scores = tf_idf(sentences_preprocessed)
        sen_emb = []
        for i, tokens in enumerate(sentences_preprocessed):
            if i not in invalid_sentences:
                vec = np.zeros((1,300))
                for token in tokens:
                    if token in word2id.keys():
                        vec += tf_idf_scores[i][token] * word_emb[word2id[token]]
                sen_emb.append(vec / sum([v for k, v in tf_idf_scores[i].items() if k in word2id.keys()]))

    id2sentence = {i: sen for i, sen in enumerate([sentences[j] for j in range(len(sentences))
                                                   if j not in invalid_sentences])}

    return np.vstack(sen_emb), id2sentence
