# jiratools

Set of tools to work with JIRA.

## crawler

crawler.py search JIRA tickets defined by pattern and save it as json files in specified folder.

```bash
$ python crawler.py -h
usage: crawler.py [-h] [--user USER] [--password PASSWORD] [--dst DST]
                  [--verify_ssl [VERIFY_SSL]]
                  url jql

Dump jira tickets to files

positional arguments:
  url                   jira api url
  jql                   search query.

optional arguments:
  -h, --help            show this help message and exit
  --user USER           Username to connect to jira. Can be set via
                        environment variable JIRA_USER.
  --password PASSWORD   Password to connect to jira. Can be set via
                        environment variable JIRA_USER.
  --dst DST             destination to store files
  --verify_ssl [VERIFY_SSL]
                        verify ssl for jira connection
```

To start in docker run

```bash
$ docker run -it -v ${PWD}/dump:/dump \
    vistrcm/jiratoolscrawler \
        https://jira/rest/api/2 user "superpass" "assignee = currentUser()" --dst /dump
```
