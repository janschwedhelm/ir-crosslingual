import string
import nltk
from collections import Counter

POS_TAGS = {'noun': 'NN NNS NNP NNPS'.split(),
            'verb': 'VB VBD VBG VBN VBP VBZ'.split(),
            'adverb': 'RB RBR RBS RP'.split(),
            'adjective': 'JJ JJR JJS'.split(),
            'wh': 'WDT WP WP$ WRB'.split(),
            'pronoun': ' PDT POS PRP PRP$'.split()
            }


def equal_occurrence_punctuation(s_sen: list, t_sen: list, punctuation: str):
    """
    Compares the occurrence of a given punctuation mark in the source and target sentence
    :param s_sen: Source sentence given as a list of words
    :param t_sen: Target sentence given as a list of words
    :param punctuation: Punctuation mark after which the comparison shall be performed
    :return: Returns 2 if both sentences contain the given punctuation mark,
                returns 1 if only on the sentences contains the given punctuation mark,
                returns 0 if none of the sentences contains the given punctuation mark
    """
    try:
        return (punctuation in s_sen) + (punctuation in t_sen)
        # return 1 if (punctuation in s_sen) == (punctuation in t_sen) else 0
    except TypeError:
        print('Parameters need to be in string format')
        return -1


def count_tokens(sentence: list, punctuation=True):
    """
    Counts the absolute number of to words or punctuation marks, respectively.
    :param sentence: Sentence in which to search for words or punctuation marks
    :param punctuation: If True, count number of punctuation marks. Else, count number of words
    :return: Absolute number of words or punctuation marks, respectively
    """
    try:
        return sum([1 for word in sentence if word in string.punctuation]) if punctuation \
            else sum([1 for word in sentence if word not in string.punctuation])
    except TypeError:
        print('Parameter needs to be in string format')
        return -1


def difference_count_tokens(s_sen: list, t_sen: list, punctuation=True):

    """
    Compute the difference in the absolute number of words or punctuation marks, respectively,
    between the source and the target sentence
    :param s_sen: Source sentence given as a list of words
    :param t_sen: Target sentence given as a list of words
    :param punctuation: If True, compute the difference in the absolute number of punctuation marks.
    Else, compute the difference in the absolute number of words
    :return: Absolute difference in the number of words or punctuation marks, respectively,
    between the source and the target sentence
    """
    return abs(count_tokens(s_sen, punctuation) - count_tokens(t_sen, punctuation))


def difference_count_nltk_tags(s_sen: list, t_sen: list, word_group: str = 'noun'):
    """
    Compute the difference in the absolute number of occurrences of a given group of pos tags
    between the source and the target sentence
    :param s_sen: Source sentence given as a list of words
    :param t_sen: Target sentence given as a list of words
    :param word_group: Word group after which the comparison shall be performed
    :return: The absolute difference in the number of occurrences of the given word group
    between the source and the target sentence
    """
    if word_group not in POS_TAGS:
        raise ValueError('POS-TAG must be one of {}.'.format(POS_TAGS))
    pos_tags = POS_TAGS[word_group]
    _, s_tags = zip(*nltk.pos_tag(s_sen))
    s_count = sum(value for key, value in dict(Counter(s_tags)).items() if key in pos_tags)
    _, t_tags = zip(*nltk.pos_tag(t_sen))
    t_count = sum(value for key, value in dict(Counter(t_tags)).items() if key in pos_tags)
    return abs(s_count - t_count)


if __name__ == '__main__':
    """
    Test section
    """
    s = 'i would therefore once more ask you to ensure that we get a dutch channel as well .'.split()
    t = 'deshalb möchte ich sie nochmals ersuchen , dafür sorge zu tragen , ' \
        'daß auch ein niederländischer sender eingespeist wird .'.split()
    print(equal_occurrence_punctuation(s, t, '?'))
