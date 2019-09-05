"""Jira crawler"""
import argparse
import json
import os

import requests


def str2bool(value):
    """convert string value to bool"""
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """parse CLI args"""
    parser = argparse.ArgumentParser(description='Dump jira tickets to files')
    parser.add_argument('url', type=str, help='jira api url')
    parser.add_argument('jql', type=str, help='search query.')
    parser.add_argument('--user',
                        type=str, default=os.getenv('JIRA_USER', "user"),
                        help='Username to connect to jira. Can be set via environment variable JIRA_USER.'
                        )
    parser.add_argument('--password',
                        type=str, default=os.getenv('JIRA_PASSWD', "user"),
                        help='Password to connect to jira. Can be set via environment variable JIRA_USER.'
                        )
    parser.add_argument('--dst', default="dump", help="destination to store files")
    parser.add_argument("--verify_ssl", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="verify ssl for jira connection")
    return parser.parse_args()


def search(api_url, session, jql):
    """do a paginated search"""
    search_url = "{}/search".format(api_url)
    issues = []
    count = 0

    while True:
        payload = {'jql': jql, 'startAt': count}
        resp = session.get(search_url, params=payload)
        resp.raise_for_status()
        data = resp.json()
        tmp_issues = data["issues"]
        retrieved = len(tmp_issues)
        if retrieved <= 0:
            break
        issues.extend(tmp_issues)
        count += retrieved
    return issues


def dump(session, issues, dst):
    """dump issues to the `dst` directory"""
    # create dst directory
    os.makedirs(dst, exist_ok=True)

    amount = len(issues)
    count = 0
    for elem in issues:
        key = elem["key"]
        try:
            issue = get_issue(elem, session)
        except requests.exceptions.HTTPError as http_ex:
            if http_ex.response == requests.codes.internal_server_error:
                print(f"WARNING. got 500 while getting issue {key}. Skipping. Exception: {http_ex}")
                continue
            print(f"ERROR. exception retrieving issue {key}: {http_ex}")
            raise RuntimeError(f"Error getting issue {key}. Details: {http_ex}")

        outfile = "{dst}/{name}.json".format(dst=dst, name=key)
        print("saving {} to {}".format(key, outfile))
        with open(outfile, 'w') as outfile:
            json.dump(issue, outfile)
        count += 1
        print("stored {}/{} issues".format(count, amount))


def get_issue(elem, session):
    print("retrieving {}".format(elem["key"]))
    resp = session.get(elem["self"])
    resp.raise_for_status()
    issue = resp.json()
    return issue


def main():
    """execute search and dump"""
    args = parse_args()

    jira_session = requests.Session()
    jira_session.auth = (args.user, args.password)
    jira_session.verify = args.verify_ssl

    print("searching for issues")
    issues = search(args.url, jira_session, args.jql)
    print("found {} issues".format(len(issues)))

    dump(jira_session, issues, args.dst)


if __name__ == "__main__":
    main()
