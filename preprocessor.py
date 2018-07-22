import json
import os
from pprint import pprint

import nltk
from nltk.corpus import stopwords

DUMP_DIR = "dump"


def process_issue(issue):
    print("processing {}".format(issue["key"]))

    cleaned = {
        "id": issue["id"],
        "key": issue["key"],
        "assignee": issue["fields"]["assignee"]["key"],
        "status": issue["fields"]["status"]["name"],
        "reporter": issue["fields"]["reporter"]["key"],
        "description": issue["fields"]["description"],
        "summary": issue["fields"]["summary"],
        # "comment": issue["fields"]["comment"],
    }

    pprint(cleaned)


def get_text(issue):
    return "\n\n".join([issue["fields"]["summary"], issue["fields"]["description"]])


def main():
    files = os.listdir(DUMP_DIR)
    for file in files:
        with open(os.path.join(DUMP_DIR, file), 'r') as json_data:
            issue = json.load(json_data)
            process_issue(issue)


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
    # print("processing text")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    text = text.lower()

    no_code = text.replace("{code}", "")

    tokens = nltk.wordpunct_tokenize(no_code)
    clean_tokens = [t for t in tokens if keep_token(t)]

    # lemmanization
    lemmatizer = nltk.WordNetLemmatizer()
    lemmanized = map(lemmatizer.lemmatize, clean_tokens)

    return lemmanized


def process_megatext(in_file="megatext.txt"):
    with open(in_file, 'r') as inf:
        data = inf.read()
        limit_line = len(data) // 10
        # out = process_text(data[:limit_line])
        out = process_text(data)

    freq = nltk.FreqDist(out)
    for key, val in freq.items():
        print("{}: {}".format(val, key))

    # print(out)


if __name__ == "__main__":
    # main()
    process_megatext()
