import csv
import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

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

    stop_list.append("undefined")  # extend stop_list with 'undefined'

    comments = issue["fields"]["comment"]["comments"]
    if not comments:  # no comments in the issue
        return get_key(issue, "assignee")

    authors = [comment.get("author", {"key": "undefined"})["key"] for comment in comments]
    # remove stop words from authors
    cleaned_authors = [author for author in authors if author not in stop_list]

    # list may became empty
    if not cleaned_authors:  # no comments in the issue
        return get_key(issue, "assignee")

    counter = Counter(cleaned_authors)
    return counter.most_common(1)[0][0]


def get_comment_text(issue):
    comments = issue["fields"]["comment"]["comments"]
    return "\n".join([comment["body"] for comment in comments])


def process_issue(issue, verbose=False):
    if verbose:
        print("processing {}".format(issue["key"]))

    most_active = get_most_active(issue, stop_list=["jiralinuxcli", "sneelaudhuri", "noc"])

    summary = issue["fields"]["summary"]

    # sometime description is None for some reason.
    description = issue["fields"].get("description")
    if description is None:
        description = ""

    comments_text = get_comment_text(issue)

    text_raw = " ".join([summary, description, comments_text])
    # cleanup text
    text = ' '.join(text_raw.splitlines())

    cleaned = {
        "id": issue["id"],
        "key": issue["key"],
        "assignee": get_key(issue, "assignee").lower(),
        "most_active": most_active.lower(),
        "status": issue["fields"]["status"]["name"],
        "reporter": get_key(issue, "reporter"),
        # decode - hack to avoid strange symbols
        "description": description.encode('unicode-escape').decode('utf-8'),
        "summary": summary.encode('unicode-escape').decode('utf-8'),  # hack to avoid strange symbols
        "comments_text": comments_text,
        # text from summary, description and comments
        "text": text,
    }
    return cleaned


def get_text(issue):
    return "\n\n".join([issue["fields"]["summary"], issue["fields"]["description"]])


def process_dir(directory):
    issues = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            if not name.lower().endswith(".json"):
                continue
            with open(os.path.join(root, name), 'r') as json_data:
                issue = json.load(json_data)
            issues.append(process_issue(issue))
    return issues


def split_df(df, seed=1, limit=0.8):
    # TODO: fix split. Make it balanced.
    # Now, split the data into two parts -- training and evaluation.
    np.random.seed(seed=seed)  # makes result reproducible
    msk = np.random.rand(len(df)) < limit
    df["is_valid"] = ~msk
    # make sure all labels from valid presented in training set
    # if not, flip 'is_valid' flag to make it training
    train_targets = df[~df["is_valid"]]["most_active"].unique()
    for element in df[df["is_valid"]]["most_active"].unique():
        if element not in train_targets:
            df.loc[df["most_active"] == element, "is_valid"] = False
    return df


def maybe_process(store_file, dump_dir="dump/", force=False):
    if force or not os.path.exists(store_file):
        print("processing json dump")
        data = process_dir(dump_dir)
        print("preparing dataframe")
        df = pd.DataFrame(data)
        print("extending dataframe")
        df = extend_df(df)
        print("splitting dataset")
        df = split_df(df)
        print("saving store_file: {}".format(store_file))
        with open(store_file, 'wb') as data_file:
            pickle.dump(df, data_file)
    else:
        print(f"loading saved data from {store_file}")
        with open(store_file, 'rb') as data_file:
            df = pickle.load(data_file)
    return df


def extend_df(df, verbose=False):
    if verbose:
        print("extending DF")
    df = df.set_index('id')
    df = df.fillna("unknown")
    if verbose:
        print("extending DF done")
    return df


def vocabularies(df):
    user_vocabulary = pd.concat([df["assignee"], df["reporter"], df["most_active"]]).unique()
    assignee_vocabulary = df["assignee"].unique()
    most_active_vocabulary = df["most_active"].unique()
    return user_vocabulary, assignee_vocabulary, most_active_vocabulary


def prepare_csv(df):
    lang_model_data = df[['label', 'text', 'is_valid']]
    classifier_data = df[['label', 'summary', 'description', 'is_valid']]
    print("preparing CVS files")
    lang_model_data.to_csv(os.path.join(DUMP_DIR, 'texts.csv'), index=False, quoting=csv.QUOTE_ALL)
    classifier_data.to_csv(os.path.join(DUMP_DIR, 'classifier.csv'), index=False, quoting=csv.QUOTE_ALL)


def main():
    df = maybe_process(os.path.join(DUMP_DIR, "data.pkl"))
    df = df.rename(columns={'most_active': 'label'})
    prepare_csv(df)


if __name__ == "__main__":
    main()
