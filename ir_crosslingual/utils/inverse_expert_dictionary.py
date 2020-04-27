import io
import os
import sys

from ir_crosslingual.utils import strings


def inverse(origin_languages: str):
    for data in ['', '.0-5000', '.5000-6500']:
        word_pairs = []
        with io.open('{}expert_dictionaries/{}/MUSE_{}{}.txt'.format(strings.data_path,
                                                                     origin_languages,
                                                                     origin_languages, data),
                     'r', encoding='utf-8') as file:
            for index, word_pair in enumerate(file):
                s_word, t_word = word_pair.rstrip().split()
                word_pairs.append((t_word, s_word))

        new_src_lang = origin_languages[-2:]
        new_trg_lang = origin_languages[:2]

        if not os.path.exists('{}expert_dictionaries/{}-{}'.format(strings.data_path, new_src_lang, new_trg_lang)):
            os.makedirs('{}expert_dictionaries/{}-{}'.format(strings.data_path, new_src_lang, new_trg_lang))

        with open('{}expert_dictionaries/{}-{}/MUSE_{}-{}{}.txt'.format(strings.data_path,
                                                                      new_src_lang, new_trg_lang,
                                                                      new_src_lang, new_trg_lang, data), 'w+') as fp:
            fp.write('\n'.join('{} {}'.format(x[0], x[1]) for x in word_pairs))


if __name__ == '__main__':
    inverse(sys.argv[1])
