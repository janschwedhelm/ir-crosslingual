import pandas as pd
from ir_crosslingual.sentences.sentences import Sentences


def vec2features(sens: Sentences):
    unique_queries = sens.test_collection.drop_duplicates('src_sentence')
    print('---- INFO: Unique queries extracted')
    unique_documents = sens.test_collection.drop_duplicates('trg_sentence')
    print('---- INFO: Unique documents extracted')
    dim = 10
    for prefix in ['src', 'trg']:
        print('---- INFO: Started extraction for {} language.'.format(prefix))
        sens.train_data[['{}_embedding_pca_{}'.format(prefix, i) for i in range(dim)]] = \
            pd.DataFrame(sens.reduce_dim(sens.train_data['{}_embedding'.format(prefix)], dim, use_ppa=False).tolist())
        print('---- INFO: {}_embedding_pca elements extracted for train data.'.format(prefix))
        sens.train_data[['{}_embedding_{}'.format(prefix, i) for i in range(300)]] = pd.DataFrame(
            sens.train_data['{}_embedding'.format(prefix)].tolist())
        print('---- INFO: {}_embedding elements extracted for train data.'.format(prefix))
        if prefix == 'src':
            unique_queries[['{}_embedding_pca_{}'.format(prefix, i) for i in range(dim)]] = pd.DataFrame(
                sens.reduce_dim(unique_queries['{}_embedding'.format(prefix)], dim, use_ppa=False).tolist())
            print('---- INFO: {}_embedding_pca elements extracted for unique queries.'.format(prefix))
            unique_queries[['{}_embedding_{}'.format(prefix, i) for i in range(300)]] = pd.DataFrame(
                unique_queries['{}_embedding'.format(prefix)].tolist())
            print('---- INFO: {}_embedding elements extracted for unique queries.'.format(prefix))

            merge_features = ['{}_embedding_pca_{}'.format(prefix, i) for i in range(dim)] \
                             + ['{}_embedding_{}'.format(prefix, i) for i in range(300)] + ['{}_sentence'.format(prefix)]
            unique_queries = unique_queries[merge_features]
            sens.test_collection = pd.merge(left=sens.test_collection, right=unique_queries,
                                            on='{}_sentence'.format(prefix), how='left')
            print('---- INFO: Unique queries merged to test collection')
        else:
            unique_documents[['{}_embedding_pca_{}'.format(prefix, i) for i in range(dim)]] = pd.DataFrame(
                sens.reduce_dim(unique_documents['{}_embedding'.format(prefix)], dim, use_ppa=False).tolist())
            print('---- INFO: {}_embedding_pca elements extracted for unique documents.'.format(prefix))
            unique_documents[['{}_embedding_{}'.format(prefix, i) for i in range(300)]] = pd.DataFrame(
                unique_documents['{}_embedding'.format(prefix)].tolist())
            print('---- INFO: {}_embedding elements extracted for unique documents.'.format(prefix))
            merge_features = ['{}_embedding_pca_{}'.format(prefix, i) for i in range(dim)] \
                             + ['{}_embedding_{}'.format(prefix, i) for i in range(300)] + [
                                 '{}_sentence'.format(prefix)]
            unique_documents = unique_documents[merge_features]
            sens.test_collection = pd.merge(left=sens.test_collection, right=unique_documents,
                                            on='{}_sentence'.format(prefix), how='left')
            print('---- INFO: Unique documents merged to test collection')
    sens.features_dict['vector_elements'] = ['src_embedding_{}'.format(i) for i in range(300)] \
                                            + ['src_embedding_{}'.format(i) for i in range(300)]
    sens.features_dict['vector_elements_pca'] = ['src_embedding_pca_{}'.format(i) for i in range(dim)] \
                                            + ['src_embedding_pca_{}'.format(i) for i in range(dim)]
    print('---- DONE: Extracted all vector elements and merged to test collection')
    return sens
