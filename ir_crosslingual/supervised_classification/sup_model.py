import os
import json
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

from ir_crosslingual.sentences.sentences import Sentences
from ir_crosslingual.features import text_based
from ir_crosslingual.features import vector_based
from ir_crosslingual.utils import paths


class SupModel:
    def __init__(self):
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None

    @staticmethod
    def save_model(name: str, model, prepared_features: list, features_dict: dict, info: str = None):
        """
        Save a pretrained model to file, alongside files containing the prepared and the actual features that have been
        used to train the model and an optional text file containing further information about the training process
        :param name: Name of the model that shall be stored. Will be the name of the folder containing all model files
        :param model: Pretrained model object to be saved
        :param prepared_features: List of features that have been prepared when training this model
        :param features_dict: Dictionary of text_based and vector_based features that have been prepared
        when training this model
        :param info: Optional: Further information regarding the training process. E.g., number of training examples
        :return: -1 if a folder with the given name already exists
        """
        if not os.path.exists('{}{}'.format(paths.model_path, name)):
            os.makedirs('{}{}'.format(paths.model_path, name))
        else:
            print('A folder with this name already exists. Please choose a different one.')
            return -1

        # Save model to file
        joblib.dump(model, '{}{}/model.pkl'.format(paths.model_path, name))

        # Save list of prepared features to file
        with open('{}{}/prepared_features.txt'.format(paths.model_path, name), 'w') as f:
            f.write('\n'.join(prepared_features))

        # Save dict of actual features
        with open('{}{}/features.json'.format(paths.model_path, name), 'w') as f:
            json.dump(features_dict, f)

        # Save info if given
        if info is not None:
            with open('{}{}/info.txt'.format(paths.model_path, name), 'w') as f:
                f.write(info)

    @staticmethod
    def load_model(name: str):
        """
        Load a model with the given name, alongside files containing the prepared and the actual features that have been
        used to train the model
        :param name: Name of the model to be loaded
        :return: Pretrained model object, list of features to prepare and dictionary containing text_based
        and vector_based features
        """
        if not os.path.exists('{}{}'.format(paths.model_path, name)):
            raise FileNotFoundError('A model with this name does not exist yet.')

        # Load model
        model = joblib.load(open('{}{}/model.pkl'.format(paths.model_path, name), 'rb'))

        # Load list of prepared features from file
        with open('{}{}/prepared_features.txt'.format(paths.model_path, name)) as f:
            prepared_features = f.read().splitlines()

        # Load dict of actual features
        with open('{}{}/features.json'.format(paths.model_path, name)) as handle:
            features = json.loads(handle.read())

        return model, prepared_features, features

    def evaluate_boolean(self, model, sentences: Sentences, features=None):
        data = sentences.test_collection.copy()
        if features is None:
            features_dict = sentences.features_dict
            features = [feature for values in features_dict.values() for feature in values]

        preds = model.predict(data[features])

        self.accuracy = accuracy_score(data['translation'], preds)
        self.precision = precision_score(data['translation'], preds)
        self.recall = recall_score(data['translation'], preds)
        self.f1 = f1_score(data['translation'], preds)

        return self

    @staticmethod
    def compute_map(model, sentences: Sentences, features=None):
        print('---- INFO: Start computing the MAP')
        data = sentences.test_collection.copy()
        if features is None:
            features_dict = sentences.features_dict
            features = [feature for values in features_dict.values() for feature in values]

        pred_probas = model.predict_proba(data[features])[:, 1]
        print('---- INFO: Probabilities predicted')
        data['trans_proba'] = pred_probas

        eval_rank = pd.DataFrame()
        eval_rank[['query', 'true_translation']] = data[data['translation'] == 1][
            ['src_sentence', 'trg_sentence']]
        print('---- INFO: Dataframe with evaluation ranking created')
        eval_rank['ranking'] = eval_rank.apply(lambda row: list(
            data[data['src_sentence'] == row['query']].sort_values('trans_proba', ascending=False)[
                'trg_sentence']), axis=1)
        print('---- INFO: Probabilities sorted for each query')
        eval_rank['rank_true'] = eval_rank.apply(lambda row: row['ranking'].index(row['true_translation']) + 1, axis=1)
        print('---- INFO: Index of ranking of true translation retrieved')

        return sum([1 / rank for rank in eval_rank['rank_true']]) / len(eval_rank)

    @staticmethod
    def rank_trg_sentences(model, sentences: Sentences, features, single_source: bool = False, evaluation: bool = True):
        """
        Rank target sentences for each given source sentence according to their predicted probability
        :param model: Pretrained model object
        :param sentences: Sentences object containing the data to perform ranking on
        :param single_source: Boolean variable indicating whether a single source sentence is considered or a list of
        different source sentences
        :param evaluation: If True, this method is used for evaluation and thus, ranking is performed only on
        sentences.test_data.
        If False, ranking in this method is performed on sentences.data
        :return: List of ranked sentences and list of ranked probabilities
        """

        data = sentences.test_collection.copy() if evaluation else sentences.data.copy()

        def predict(prediction_data):
            predictions = model.predict_proba(prediction_data[features])
            ranked_indices = predictions[:, 1].argsort()[::-1]
            top_sen = list(prediction_data['trg_sentence'].iloc[ranked_indices])
            top_prob = predictions[:, 1][ranked_indices]
            return [top_sen, top_prob]

        if single_source:
            return predict(data)
