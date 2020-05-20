data_path = '../../data/'

model_path = '../main/models/'

monolingual_embedding_vec_paths = {
    'de': '{}fastText_mon_emb/wiki.de.vec'.format(data_path),
    'en': '{}fastText_mon_emb/wiki.en.vec'.format(data_path),
    'fr': '{}fastText_mon_emb/wiki.fr.vec'.format(data_path),
    'fi': '{}fastText_mon_emb/wiki.fi.vec'.format(data_path)
}

monolingual_embedding_paths = {
    'de': '{}fastText_mon_emb/de/'.format(data_path),
    'en': '{}fastText_mon_emb/en/'.format(data_path),
    'fr': '{}fastText_mon_emb/fr/'.format(data_path),
    'fi': '{}fastText_mon_emb/fi/'.format(data_path)
}

expert_dictionaries = {
    'de-en': '{}expert_dictionaries/de-en/MUSE_de-en.0-5000.txt'.format(data_path),
    'en-de': '{}expert_dictionaries/en-de/MUSE_en-de.0-5000.txt'.format(data_path),
    'de-fr': '{}expert_dictionaries/de-fr/MUSE_de-fr.0-5000.txt'.format(data_path),
    'fr-de': '{}expert_dictionaries/fr-de/MUSE_fr-de.0-5000.txt'.format(data_path),
    'fr-en': '{}expert_dictionaries/fr-en/MUSE_fr-en.0-5000.txt'.format(data_path),
    'en-fr': '{}expert_dictionaries/en-fr/MUSE_en-fr.0-5000.txt'.format(data_path),
    'en-fi': '{}expert_dictionaries/en-fi/MUSE_en-fi.0-5000.txt'.format(data_path),
    'fi-en': '{}expert_dictionaries/fi-en/MUSE_fi-en.0-5000.txt'.format(data_path)
}

sentence_dictionaries = {
    'de-en': '{}europarl_datasets/de-en/Europarl.de-en'.format(data_path),
    'en-de': '{}europarl_datasets/de-en/Europarl.de-en'.format(data_path),
    'de-fr': '{}europarl_datasets/de-fr/Europarl.de-fr'.format(data_path),
    'fr-de': '{}europarl_datasets/de-fr/Europarl.de-fr'.format(data_path),
    'en-fr': '{}europarl_datasets/en-fr/Europarl.en-fr'.format(data_path),
    'fr-en': '{}europarl_datasets/en-fr/Europarl.en-fr'.format(data_path),
    'en-fi': '{}europarl_datasets/en-fi/Europarl.en-fi'.format(data_path),
    'fi-en': '{}europarl_datasets/en-fi/Europarl.en-fi'.format(data_path)
}

extracted_data = {
    'en-de': '{}extracted_data/global/en-de/'.format(data_path),
    'de-en': '{}extracted_data/global/de-en/'.format(data_path),
    'en-fr': '{}extracted_data/global/en-fr/'.format(data_path),
    'fr-en': '{}extracted_data/global/fr-en/'.format(data_path),
    'en-fi': '{}extracted_data/global/en-fi/'.format(data_path),
    'fi-en': '{}extracted_data/global/fi-en/'.format(data_path)
}

languages = {
    'de': 'german',
    'en': 'english',
    'fr': 'french',
    'fi': 'finnish'
}

languages_inversed = {v: k for k, v in languages.items()}
