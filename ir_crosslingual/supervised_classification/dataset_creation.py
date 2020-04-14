import math
import os
import pandas as pd
import numpy as np
from ir_crosslingual.load_tmx_sentence import tmx_dataframe


def create_datasets(europarl_dataset: str, n_train: int, frac_pos: float, n_test: int):
    if isinstance(europarl_dataset, str) and os.path.isfile(europarl_dataset):
        _, df = tmx_dataframe(europarl_dataset)
    else:
        df = europarl_dataset
    n_train_pos = math.ceil(n_train*frac_pos)
    df_train_pos = df[:n_train_pos]
    df_train_pos.loc[:, 'translation'] = 1
    multiple = math.ceil(n_train/n_train_pos)
    df_train_neg = pd.concat([df_train_pos[['source_sentence']]] * multiple, ignore_index=True)[:n_train-n_train_pos]
    df_train_neg['target_sentence'] = np.random.choice(df[n_train_pos:-n_test]['target_sentence'],
                                                       n_train-n_train_pos)
    df_train_neg.loc[:, 'translation'] = 0
    df_train = df_train_pos.append(df_train_neg, ignore_index=True)
    df_test = df[-n_test:].reset_index(drop=True)
    return df_train, df_test
