import os
import pickle

import pandas as pd

import preprocessor


def maybe_process(store_file, dump_dir="dump/issues/", force=False):
    if force or not os.path.exists(store_file):
        data = preprocessor.process_dir(dump_dir)
        df = pd.DataFrame(data)
        df = extend_df(df)
        with open(store_file, 'wb') as data_file:
            pickle.dump(df, data_file)
    else:
        with open(store_file, 'rb') as data_file:
            df = pickle.load(data_file)
        return df


def extend_df(df):
    print("extending DF")
    df = df.fillna("Unknown")
    df["summary_clean"] = df["summary"].map(lambda x: " ".join(preprocessor.process_text(x)))
    df["description_clean"] = df["description"].map(lambda x: " ".join(preprocessor.process_text(x)))
    del df["summary"]
    del df["description"]
    print("extending DF done")
    return df


def vocabularies(df):
    user_vocabulary = pd.concat([df["assignee"], df["reporter"]]).unique()
    assignee_vocabulary = df["assignee"].unique()
    return user_vocabulary, assignee_vocabulary
