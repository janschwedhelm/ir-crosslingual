import numpy as np

from ir_crosslingual.features.vector_based import cos_sim


class UnsupModel:

    def __init__(self):
        print()

    @staticmethod
    def predict_proba(data):
        features = list(data.columns)
        if 'cosine_similarity' in features:
            return np.asarray(list(zip([0]*len(data), list(data['cosine_similarity']))))
        else:
            return np.asarray(list(zip([0]*len(data),
                                       [cos_sim(data[features[0]][i], data[features[1]][i])
                                        for i in range(len(data))])))
