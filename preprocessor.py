import json
import os
import pickle
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

DUMP_DIR = "dump"


def get_key(issue, filed):
    structure = issue["fields"][filed]
    if not structure:
        return "unknown"

    if structure:
        return structure["key"]
    return None


def get_most_active(issue):
    """Find most active user in comments. Return assignee if no comments"""
    comments = issue["fields"]["comment"]["comments"]
    if not comments:
        return get_key(issue, "assignee")

    counter = Counter([comment["author"]["key"] for comment in comments])
    return counter.most_common(1)[0][0]


def process_issue(issue):
    print("processing {}".format(issue["key"]))
    most_active = get_most_active(issue)
    cleaned = {
        "id": issue["id"],
        "key": issue["key"],
        "assignee": get_key(issue, "assignee").lower(),
        "most_active": most_active.lower(),
        "status": issue["fields"]["status"]["name"],
        "reporter": get_key(issue, "reporter"),
        "description": issue["fields"]["description"].encode('unicode-escape').decode('utf-8'),  # hack to avoid strange symbols
        "summary": issue["fields"]["summary"].encode('unicode-escape').decode('utf-8'),  # hack to avoid strange symbols
        # "comment": issue["fields"]["comment"],
    }
    return cleaned


def print_processed(issue):
    print("key: {}".format(issue["key"]))
    print("summary:\n{}\n".format(issue["summary"]))
    print("new summary:\n{}\n".format(" ".join(
        process_text(issue["summary"]))
    ))
    print("description:\n{}\n".format(issue["description"]))
    print("new description:\n{}\n".format(" ".join(
        process_text(issue["description"])))
    )
    print("-" * 80)


def get_text(issue):
    return "\n\n".join([issue["fields"]["summary"], issue["fields"]["description"]])


def main():
    process_dir(DUMP_DIR)


def process_dir(directory):
    files = os.listdir(directory)
    issues = []
    for file in files:
        with open(os.path.join(directory, file), 'r') as json_data:
            issue = json.load(json_data)
            issues.append(process_issue(issue))
    return issues


def get_megatext(out_file="megatext.txt"):
    megatext = ""
    files = os.listdir(DUMP_DIR)
    for file in files:
        with open(os.path.join(DUMP_DIR, file), 'r') as json_data:
            issue = json.load(json_data)
            megatext += get_text(issue)
    with open(out_file, "w") as out:
        out.write(megatext)
    print("{} created".format(out_file))


def keep_token(tk):
    if len(tk) < 2:
        return False
    if not tk.isalnum():
        return False
    if tk.isdigit():
        return False
    if tk in stopwords.words('english'):
        return False
    return True


def process_text(text):
    text = text.lower()

    no_code = text.replace("{code}", "")
    no_under = no_code.replace("_", " ")

    tokens = nltk.wordpunct_tokenize(no_under)
    clean_tokens = [t for t in tokens if keep_token(t)]

    # lemmanization
    lemmatizer = nltk.WordNetLemmatizer()
    lemmanized = map(lemmatizer.lemmatize, clean_tokens)

    return lemmanized


def process_megatext(in_file="megatext.txt"):
    with open(in_file, 'r') as inf:
        data = inf.read()
        out = process_text(data)

    freq = nltk.FreqDist(out)
    for key, val in freq.items():
        print("{}: {}".format(val, key))

    # print(out)


def maybe_process(store_file, dump_dir="dump/issues/", force=False):
    if force or not os.path.exists(store_file):
        data = process_dir(dump_dir)
        df = pd.DataFrame(data)
        df = extend_df(df)
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


def extend_df(df):
    print("extending DF")
    df = df.set_index('id')
    df = df.fillna("Unknown")
    df["summary_clean"] = df["summary"].map(lambda x: " ".join(process_text(x)))
    df["description_clean"] = df["description"].map(lambda x: " ".join(process_text(x)))
    del df["summary"]
    del df["description"]
    print("extending DF done")
    return df


def vocabularies(df):
    user_vocabulary = pd.concat([df["assignee"], df["reporter"], df["most_active"]]).unique()
    assignee_vocabulary = df["assignee"].unique()
    most_active_vocabulary = df["most_active"].unique()
    return user_vocabulary, assignee_vocabulary, most_active_vocabulary


def prepare_csvs():
    df = maybe_process(os.path.join(DUMP_DIR, "data.pkl"))
    train_df, eval_df = split_data(df)
    train_df.to_csv(os.path.join(DUMP_DIR, 'train.csv'))
    eval_df.to_csv(os.path.join(DUMP_DIR, 'eval.csv'))
    df.to_csv(os.path.join(DUMP_DIR, 'all.csv'))


if __name__ == "__main__":
    # main()
    prepare_csvs()
    # process_megatext()
