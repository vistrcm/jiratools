import os
import pickle

import pandas as pd

import preprocessor


def maybe_process(store_file, force=False):
    if force or not os.path.exists(store_file):
        data = preprocessor.process_dir("dump/issues/")
        with open(store_file, 'wb') as data_file:
            pickle.dump(data, data_file)
    else:
        with open(store_file, 'rb') as data_file:
            data = pickle.load(data_file)
        return data


def vocabularies(df):
    user_vocabulary = pd.concat([df["assignee"], df["reporter"]]).unique()
    assignee_vocabulary = df["assignee"].unique()
    return user_vocabulary, assignee_vocabulary
