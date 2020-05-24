import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import importlib
import datetime

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from ir_crosslingual.supervised_classification import sup_model
from ir_crosslingual.utils import paths
from ir_crosslingual.embeddings import embeddings
from ir_crosslingual.sentences import sentences

train_file_avg = f'{paths.data_path}extracted_data/global/training_data_avg.pkl'
test_file_avg = f'{paths.data_path}extracted_data/global/test_collection_avg.pkl'
sens_avg, train_data_avg, test_collection_avg, features_avg = sentences.Sentences.load_from_file(train_file_avg, test_file_avg)

print('Data loaded.')

dim = 20
for prefix in ['src', 'trg']:
    print('Started extraction for {} language.'.format(prefix))
    train_data_avg[['{}_embedding_pca_{}'.format(prefix, i) for i in range(dim)]] = pd.DataFrame(sens_avg.reduce_dim(train_data_avg['{}_embedding'.format(prefix)], dim, use_ppa=False).tolist())
    print('{}_embedding_pca elements extracted for train data.'.format(prefix))
    test_collection_avg[['{}_embedding_pca_{}'.format(prefix, i) for i in range(dim)]] = pd.DataFrame(sens_avg.reduce_dim(test_collection_avg['{}_embedding'.format(prefix)], dim, use_ppa=False).tolist())
    print('{}_embedding_pca elements extracted for test collection.'.format(prefix))
    train_data_avg[['{}_embedding_{}'.format(prefix, i) for i in range(300)]] = pd.DataFrame(train_data_avg['{}_embedding'.format(prefix)].tolist())
    print('{}_embedding elements extracted for train data.'.format(prefix))
    test_collection_avg[['{}_embedding_{}'.format(prefix, i) for i in range(300)]] = pd.DataFrame(test_collection_avg['{}_embedding'.format(prefix)].tolist())
    print('{}_embedding elements extracted for test collection.'.format(prefix))

path = f'{paths.data_path}extracted_data/global'
train_data.to_pickle(f'{path}/training_data_avg_final.pkl')
test_collection.to_pickle(f'{path}/test_collection_avg_final.pkl')
