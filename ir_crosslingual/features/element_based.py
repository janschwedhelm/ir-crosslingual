import pandas as pd
import numpy as np
from ir_crosslingual.sentences.sentences import Sentences


def vec2features(sens: Sentences, pca, mean_scaler, train=True):
    unique_queries = sens.test_collection.drop_duplicates('src_sentence', ignore_index=True)
    print('---- INFO: Unique queries extracted')
    unique_documents = sens.test_collection.drop_duplicates('trg_sentence', ignore_index=True)
    print('---- INFO: Unique documents extracted')
    for prefix in ['src', 'trg']:
        if train:
            print('---- INFO: Started extraction for {} language.'.format(prefix))
            X = np.vstack(sens.train_data['{}_embedding'.format(prefix)])
            sens.train_data[['{}_embedding_pca_{}'.format(prefix, i) for i in range(10)]] = \
                pd.DataFrame(pca['{}'.format(prefix)].transform(mean_scaler['{}'.format(prefix)].transform(X)).tolist())
            print('---- INFO: {}_embedding_pca elements extracted for train data.'.format(prefix))
        if prefix == 'src':
            X = np.vstack(unique_queries['{}_embedding'.format(prefix)])
            unique_queries[['{}_embedding_pca_{}'.format(prefix, i) for i in range(10)]] = \
                pd.DataFrame(pca['{}'.format(prefix)].transform(mean_scaler['{}'.format(prefix)].transform(X)).tolist())
            print('---- INFO: {}_embedding_pca elements extracted for unique queries.'.format(prefix))
            merge_features = ['{}_embedding_pca_{}'.format(prefix, i) for i in range(10)] \
                             + ['{}_sentence'.format(prefix)]
            sens.test_collection = pd.merge(left=sens.test_collection, right=unique_queries[merge_features],
                                            on='{}_sentence'.format(prefix), how='left')
            print('---- INFO: Unique queries merged to test collection')
        else:
            X = np.vstack(unique_documents['{}_embedding'.format(prefix)])
            unique_documents[['{}_embedding_pca_{}'.format(prefix, i) for i in range(10)]] = \
                pd.DataFrame(pca['{}'.format(prefix)].transform(mean_scaler['{}'.format(prefix)].transform(X)).tolist())
            print('---- INFO: {}_embedding_pca elements extracted for unique documents.'.format(prefix))

            merge_features = ['{}_embedding_pca_{}'.format(prefix, i) for i in range(10)] \
                             + ['{}_sentence'.format(prefix)]
            sens.test_collection = pd.merge(left=sens.test_collection, right=unique_documents[merge_features],
                                            on='{}_sentence'.format(prefix), how='left')
            print('---- INFO: Unique documents merged to test collection')
    sens.features_dict['vector_elements_pca'] = ['src_embedding_pca_{}'.format(i) for i in range(10)] \
                                                + ['trg_embedding_pca_{}'.format(i) for i in range(10)]
    print('---- DONE: Extracted all vector elements and merged to test collection')
    return sens

