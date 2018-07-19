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
    parser.add_argument('user', type=str, help='username to connect to jira')
    parser.add_argument('password', type=str, help='password to connect to jira')
    parser.add_argument('jql', type=str, help='search query')
    parser.add_argument('--dst', default="dump", help="destination to store files")
    parser.add_argument('--no_verify_ssl', type=bool, help="do not verify ssl for jira connection")
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
    print("SV: {}".format(len(issues)))
    return issues


def dump(session, issues, dst):
    """dump issues to the `dst` directory"""
    # create dst directory
    os.makedirs(dst, exist_ok=True)

    amount = len(issues)
    count = 0
    for elem in issues:
        key = elem["key"]
        print("retrieving {}".format(key))
        resp = session.get(elem["self"])
        resp.raise_for_status()
        issue = resp.json()
        outfile = "{dst}/{name}.json".format(dst=dst, name=key)
        print("saving {} to {}".format(key, outfile))
        with open(outfile, 'w') as outfile:
            json.dump(issue, outfile)
        count += 1
        print("stored {}/{} issues".format(count, amount))


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
