import json
import os
from collections import Counter

import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

DUMP_DIR = "dump"


def get_key(issue, filed):
    structure = issue["fields"][filed]
    if structure:
        return structure["key"]
    return None


def get_most_active(issue):
    """Find most active user in comments. Return assignee if no comments"""
    comments = issue["comment"]["comments"]
    if not comments:
        return issue["assignee"]

    counter = Counter([comment["author"]["key"] for comment in comments])
    return counter.most_common(1)[0][1]


def process_issue(issue):
    most_active = get_most_active(issue)
    cleaned = {
        "id": issue["id"],
        "key": issue["key"],
        "assignee": get_key(issue, "assignee"),
        "most_active": most_active,
        "status": issue["fields"]["status"]["name"],
        "reporter": get_key(issue, "reporter"),
        "description": issue["fields"]["description"],
        "summary": issue["fields"]["summary"],
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
    if not tk.isalnum():
        return False
    if tk.isdigit():
        return False
    if tk in stopwords.words('english'):
        return False
    if len(tk) < 2:
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


if __name__ == "__main__":
    main()
    # process_megatext()
