data_path = '../../data/'

model_path = '../main/models/'

monolingual_embedding_vec_paths = {
    'de': '{}fastText_mon_emb/wiki.de.vec'.format(data_path),
    'en': '{}fastText_mon_emb/wiki.en.vec'.format(data_path),
    'fr': '{}fastText_mon_emb/wiki.fr.vec'.format(data_path)
}

monolingual_embedding_paths = {
    'de': '{}fastText_mon_emb/de/'.format(data_path),
    'en': '{}fastText_mon_emb/en/'.format(data_path),
    'fr': '{}fastText_mon_emb/fr/'.format(data_path)
}

expert_dictionaries = {
    'de-en': '{}expert_dictionaries/de-en/MUSE_de-en.0-5000.txt'.format(data_path),
    'en-de': '{}expert_dictionaries/en-de/MUSE_en-de.0-5000.txt'.format(data_path),
    'de-fr': '{}expert_dictionaries/de-fr/MUSE_de-fr.0-5000.txt'.format(data_path),
    'fr-de': '{}expert_dictionaries/fr-de/MUSE_fr-de.0-5000.txt'.format(data_path),
    'fr-en': '{}expert_dictionaries/fr-en/MUSE_fr-en.0-5000.txt'.format(data_path),
    'en-fr': '{}expert_dictionaries/en-fr/MUSE_en-fr.0-5000.txt'.format(data_path)
}

test_expert_dictionaries = {
    'de-en': '{}expert_dictionaries/de-en/MUSE_de-en.5000-6500.txt'.format(data_path),
    'en-de': '{}expert_dictionaries/en-de/MUSE_en-de.5000-6500.txt'.format(data_path),
    'de-fr': '{}expert_dictionaries/de-fr/MUSE_de-fr.5000-6500.txt'.format(data_path),
    'fr-de': '{}expert_dictionaries/fr-de/MUSE_fr-de.5000-6500.txt'.format(data_path),
    'fr-en': '{}expert_dictionaries/fr-en/MUSE_fr-en.5000-6500.txt'.format(data_path),
    'en-fr': '{}expert_dictionaries/en-fr/MUSE_en-fr.5000-6500.txt'.format(data_path)
}

sentence_dictionaries = {
    'de-en': '{}europarl_datasets/de-en/Europarl.de-en'.format(data_path),
    'en-de': '{}europarl_datasets/de-en/Europarl.de-en'.format(data_path),
    'de-fr': '{}europarl_datasets/de-fr/Europarl.de-fr'.format(data_path),
    'fr-de': '{}europarl_datasets/de-fr/Europarl.de-fr'.format(data_path),
    'en-fr': '{}europarl_datasets/en-fr/Europarl.en-fr'.format(data_path),
    'fr-en': '{}europarl_datasets/en-fr/Europarl.en-fr'.format(data_path)
}
languages = {
    'de': 'german',
    'en': 'english',
    'fr': 'french'
}

languages_inversed = {v: k for k, v in languages.items()}
