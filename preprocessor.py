import json
import os
import pickle
from collections import Counter
import pandas as pd
import numpy as np

DUMP_DIR = "dump"


def get_key(issue, filed):
    structure = issue["fields"][filed]
    if not structure:
        return "unknown"

    if structure:
        return structure["key"]
    return None


def get_most_active(issue, stop_list=None):
    """Find most active user in comments. Return assignee if no comments"""
    if not stop_list:
        stop_list = []

    comments = issue["fields"]["comment"]["comments"]
    if not comments:  # no comments in the issue
        return get_key(issue, "assignee")

    authors = [comment["author"]["key"] for comment in comments]
    # remove stop words from authors
    cleaned_authors = [author for author in authors if author not in stop_list]

    # list may became empty
    if not cleaned_authors:  # no comments in the issue
        return get_key(issue, "assignee")

    counter = Counter(cleaned_authors)
    return counter.most_common(1)[0][0]


def process_issue(issue, verbose=False):
    if verbose:
        print("processing {}".format(issue["key"]))

    most_active = get_most_active(issue, stop_list=["jiralinuxcli", "sneelaudhuri", "noc"])
    cleaned = {
        "id": issue["id"],
        "key": issue["key"],
        "assignee": get_key(issue, "assignee").lower(),
        "most_active": most_active.lower(),
        "status": issue["fields"]["status"]["name"],
        "reporter": get_key(issue, "reporter"),
        # decode - hack to avoid strange symbols
        "description": issue["fields"]["description"].encode('unicode-escape').decode('utf-8'),
        "summary": issue["fields"]["summary"].encode('unicode-escape').decode('utf-8'),  # hack to avoid strange symbols
        # "comment": issue["fields"]["comment"],
    }
    return cleaned


def get_text(issue):
    return "\n\n".join([issue["fields"]["summary"], issue["fields"]["description"]])


def process_dir(directory):
    files = os.listdir(directory)
    issues = []
    for json_file in files:
        with open(os.path.join(directory, json_file), 'r') as json_data:
            issue = json.load(json_data)
            issues.append(process_issue(issue))
    return issues


def maybe_process(store_file, dump_dir="dump/issues/", force=False):
    if force or not os.path.exists(store_file):
        print("processing json dump")
        data = process_dir(dump_dir)
        print("preparing dataframe")
        df = pd.DataFrame(data)
        print("extending dataframe")
        df = extend_df(df)
        print("saving store_file: {}".format(store_file))
        with open(store_file, 'wb') as data_file:
            pickle.dump(df, data_file)
    else:
        with open(store_file, 'rb') as data_file:
            df = pickle.load(data_file)
    return df


def split_data(df, seed=1, limit=0.8):
    # Now, split the data into two parts -- training and evaluation.
    np.random.seed(seed=seed)  # makes result reproducible
    msk = np.random.rand(len(df)) < limit
    traindf = df[msk]
    evaldf = df[~msk]
    return traindf, evaldf


def extend_df(df, verbose=False):
    if verbose:
        print("extending DF")
    df = df.set_index('id')
    df = df.fillna("unknown")
    del df["summary"]
    del df["description"]
    if verbose:
        print("extending DF done")
    return df


def vocabularies(df):
    user_vocabulary = pd.concat([df["assignee"], df["reporter"], df["most_active"]]).unique()
    assignee_vocabulary = df["assignee"].unique()
    most_active_vocabulary = df["most_active"].unique()
    return user_vocabulary, assignee_vocabulary, most_active_vocabulary


def prepare_CSVs(df):
    print("preparing CVS files")
    train_df, eval_df = split_data(df)
    train_df.to_csv(os.path.join(DUMP_DIR, 'train.csv'))
    eval_df.to_csv(os.path.join(DUMP_DIR, 'eval.csv'))
    df.to_csv(os.path.join(DUMP_DIR, 'all.csv'))


def main():
    df = maybe_process(os.path.join(DUMP_DIR, "data.pkl"))
    prepare_CSVs(df)


if __name__ == "__main__":
    main()
