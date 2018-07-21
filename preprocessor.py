import json
import os
from pprint import pprint

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


def main():
    files = os.listdir(DUMP_DIR)
    for file in files:
        with open(os.path.join(DUMP_DIR, file), 'r') as json_data:
            issue = json.load(json_data)
            process_issue(issue)


if __name__ == "__main__":
    main()
