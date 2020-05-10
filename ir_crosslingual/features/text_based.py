import string
import nltk
from statistics import mean
from collections import Counter

POS_TAGS = {'noun': 'NN NNS NNP NNPS'.split(),
            'verb': 'VB VBD VBG VBN VBP VBZ'.split(),
            'adverb': 'RB RBR RBS RP'.split(),
            'adjective': 'JJ JJR JJS'.split(),
            'wh': 'WDT WP WP$ WRB'.split(),
            'pronoun': ' PDT POS PRP PRP$'.split()
            }


# Preparation of features
def occ_punctuation(sentence: list, punctuation: str):
    """
    Checks whether a given punctuation mark occurs in a given sentence
    :param sentence: List of sentence tokens in which to search for the given punctuation marks
    :param punctuation: Punctuation mark to search for in sen
    :return: True, if the sentence contains the punctuation mark.
    False, if it doesn't
    """
    return punctuation in sentence


def count_tokens(sentence: list, punctuation=True):
    """
    Counts the absolute number of to words or punctuation marks, respectively.
    :param sentence: List of sentence tokens in which to search for words or punctuation marks
    :param punctuation: If True, count number of punctuation marks. Else, count number of words
    :return: Absolute number of words or punctuation marks, respectively
    """
    try:
        return sum([1 for word in sentence if word in string.punctuation]) if punctuation \
            else sum([1 for word in sentence if word not in string.punctuation])
    except TypeError:
        print('Parameter needs to be in string format')
        return -1


def count_nltk_tags(sentence: list, word_group: str = 'noun'):
    """
    Counts the absolute number of a given word group in a list of sentences
    :param sentence: List of sentence tokens in which to count the occurrence of the given word group
    :param word_group: Word group to search for in sen
    :return: Absolute number of the occurrence of the word group
    """
    # TODO: Can be made more efficient by loading POS TAGS only once
    #  and then returning the number of all requested word groups all together instead of loading POS TAGS
    #  for each word_group individually
    if word_group not in POS_TAGS:
        raise ValueError('POS-TAG must be one of {}.'.format(POS_TAGS))
    pos_tags = POS_TAGS[word_group]
    try:
        _, tags = zip(*nltk.pos_tag(sentence))
        return abs(sum(value for key, value in dict(Counter(tags)).items() if key in pos_tags))
    except ValueError:
        return 0


def translate_words(sentence: list, seed_dict: dict):
    """
    Create list with all possible translated words for the given sentence
    :param sentence: List of sentence tokens to translate (list of lists)
    :param seed_dict: Seed dictionary to use for translation
    :return: List of lists with translated words
    """
    def flatten(sentence_list):
        return [[item for sublist in sentence_words for item in sublist]
                                     for sentence_words in sentence_list]
    return flatten([[seed_dict[word] for word in src if word in seed_dict.keys()] for src in sentence])


# Extraction of features
def abs_difference(src_sen, trg_sen, single_source):
    """
    Counts the absolute difference of occurrences in a given source and target sentence
    :param src_sen: Number of occurrences in the source sentence
    :param trg_sen: Number of occurrences in the target sentence
    :param single_source: True if only one source sentence is compared to multiple target sentences
    (can be used when ranking target sentences for a single source sentence, e.g. in the WebApp)
    :return: Absolute difference in the number of occurrences between the source and the target sentence
    """
    if single_source:
        return [abs(src_sen[0] - trg) for trg in trg_sen]
    else:
        return [abs(src_sen[i] - trg_sen[i]) for i in range(len(src_sen))]


def rel_difference(src_sen, trg_sen, single_source):
    """
    Counts the relative difference of occurrences in a given source and target sentence
    :param src_sen: Number of occurrences in the source sentence
    :param trg_sen: Number of occurrences in the target sentence
    :param single_source: True if only one source sentence is compared to multiple target sentences
    (can be used when ranking target sentences for a single source sentence, e.g. in the WebApp)
    :return: Relative difference in the number of occurrences between the source and the target sentence
    """
    if single_source:
        return [src_sen[0] - trg for trg in trg_sen]
    else:
        return [src_sen[i] - trg_sen[i] for i in range(len(src_sen))]


def norm_difference(src_sen, trg_sen, single_source):
    """
    Counts the normalized, relative difference of occurrences in a given source and target sentence
    :param src_sen: Number of occurrences in the source sentence
    :param trg_sen: Number of occurrences in the target sentence
    :param single_source: True if only one source sentence is compared to multiple target sentences
    (can be used when ranking target sentences for a single source sentence, e.g. in the WebApp)
    :return: Normalized difference in the number of occurrences between the source and the target sentence
    """
    # TODO: If mean(src, trg) = 0, return 0 or -1?
    if single_source:
        return [(src_sen[0] - trg) / mean([src_sen[0], trg])
                if mean([src_sen[0], trg]) > 0 else 0
                for trg in trg_sen]
    else:
        return [(src_sen[i] - trg_sen[i]) / mean([src_sen[i], trg_sen[i]])
                if mean([src_sen[i], trg_sen[i]]) > 0 else 0
                for i in range(len(src_sen))]


def equal_occurrence(src_sen, trg_sen, single_source):
    """
    Check the difference of binary occurrence of a punctuation mark between a source and a target sentence
    :param src_sen: Binary occurrence of a punctuation mark in the source sentence
    :param trg_sen: Binary occurrence of a punctuation mark in the target sentence
    :param single_source: True if only one source sentence is compared to multiple target sentences
    (can be used when ranking target sentences for a single source sentence, e.g. in the WebApp)
    :return: 2, if both sentences contain the punctuation mark.
    1, if none of the sentences contains the punctuation mark.
    0, if only one of the sentences contains the punctuation mark.
    """
    if single_source:
        return [2 if src_sen[0] == trg == 1 else int(src_sen[0] == trg) for trg in trg_sen]
    else:
        return [2 if src_sen[i] == trg_sen[i] == 1 else int(src_sen[i] == trg_sen[i]) for i in range(len(src_sen))]


def equal_words_ratio(src_words, trg_words, src_translated, trg_translated):
    """
    Compute jaccard similarity between a source and a target sentence
    :param src_words: List of sentence tokens in the source language
    :param trg_words: List of sentence tokens in the target language
    :param src_translated: List of translated source words in the target language
    :param trg_translated: List of translated target words in the source language
    :return:
    """
    try:
        return mean([
            len(set(src_words).intersection(set(trg_translated))) / len(set(src_words)),
            len(set(trg_words).intersection(set(src_translated))) / len(set(trg_words))
        ])
    except ZeroDivisionError:
        return 0


# Dictionary of prepared text_based features that can be extracted on a single sentence
# alongside the corresponding function that needs to be executed for te given feature
# Structure:
# {'feature_name': [function to be called which has to be extended by 'src' and 'trg' for the actual dataframe,
# column on which the function needs to be performed
# {'argument_name_to_pass': 'argument_value_to_pass'}]}
PREPARED_FEATURES = {
    'num_words': [count_tokens, 'preprocessed', {'punctuation': False}],
    'num_punctuation': [count_tokens, 'preprocessed', {'punctuation': True}],
    'occ_question_mark': [occ_punctuation, 'preprocessed', {'punctuation': '?'}],
    'occ_exclamation_mark': [occ_punctuation, 'preprocessed', {'punctuation': '!'}],
    'translated_words': [translate_words, 'words']
}

for word_group in POS_TAGS:
    PREPARED_FEATURES['num_{}'.format(word_group)] = [count_nltk_tags, 'preprocessed', {'word_group': word_group}]


# Dictionary of all text_based features that can be extracted on two sentences to compare
# alongside the corresponding function that needs to be executed for the given feature
# Structure: {'feature_name': [function to be called, column on which the function needs to be performed
# which has to be extended by 'src' and 'trg' for the actual dataframe]}
FEATURES = {
    'diff_num_words': [abs_difference, 'num_words'],
    'diff_num_punctuation': [abs_difference, 'num_punctuation'],
    'diff_occ_question_mark': [equal_occurrence, 'occ_question_mark'],
    'diff_occ_exclamation_mark': [equal_occurrence, 'occ_exclamation_mark'],
    'diff_num_noun': [abs_difference, 'num_noun'],
    'diff_num_verb': [abs_difference, 'num_verb'],
    'diff_num_adverb': [abs_difference, 'num_adverb'],
    'diff_num_adjective': [abs_difference, 'num_adjective'],
    'diff_num_wh': [abs_difference, 'num_wh'],
    'diff_num_pronoun': [abs_difference, 'num_pronoun'],

    'rel_diff_num_words': [rel_difference, 'num_words'],
    'rel_diff_num_punctuation': [rel_difference, 'num_punctuation'],
    'rel_diff_num_noun': [rel_difference, 'num_noun'],
    'rel_diff_num_verb': [rel_difference, 'num_verb'],
    'rel_diff_num_adverb': [rel_difference, 'num_adverb'],
    'rel_diff_num_adjective': [rel_difference, 'num_adjective'],
    'rel_diff_num_wh': [rel_difference, 'num_wh'],
    'rel_diff_num_pronoun': [rel_difference, 'num_pronoun'],

    'norm_diff_num_words': [norm_difference, 'num_words'],
    'norm_diff_num_punctuation': [norm_difference, 'num_punctuation'],
    'norm_diff_num_noun': [norm_difference, 'num_noun'],
    'norm_diff_num_verb': [norm_difference, 'num_verb'],
    'norm_diff_num_adverb': [norm_difference, 'num_adverb'],
    'norm_diff_num_adjective': [norm_difference, 'num_adjective'],
    'norm_diff_num_wh': [norm_difference, 'num_wh'],
    'norm_diff_num_pronoun': [norm_difference, 'num_pronoun'],

    'norm_diff_translated_words': [equal_words_ratio, ['words', 'translated_words']]
}

if __name__ == '__main__':
    """
    Test section
    """
    b = count_tokens('deshalb möchte ich sie nochmals ersuchen , dafür sorge zu tragen'.split(), punctuation=False)
    print(b)
