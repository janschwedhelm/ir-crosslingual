data_path = '../../data/'

monolingual_embedding_paths = {
    'de': '{}fastText_mon_emb/wiki.de.vec'.format(data_path),
    'en': '{}fastText_mon_emb/wiki.en.vec'.format(data_path),
    'fr': ''
}

expert_dictionaries = {
    'de-en': '{}expert_dictionaries/de-en/MUSE_de-en.0-5000.txt'.format(data_path),
    'en-de': '{}expert_dictionaries/en-de/MUSE_en-de.0-5000.txt'.format(data_path)
}

sentence_dictionaries = {
    'de-en': '{}europarl_datasets/de-en/Europarl.de-en'.format(data_path),
    'en-de': '{}europarl_datasets/de-en/Europarl.de-en'.format(data_path),
    'de-fr': '{}europarl_datasets/de-fr/Europarl.de-fr'.format(data_path),
    'fr-de': '{}europarl_datasets/de-fr/Europarl.de-fr'.format(data_path)
}

languages = {
    'de': 'german',
    'en': 'english',
    'fr': 'french'
}
