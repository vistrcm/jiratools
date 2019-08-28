# jiratools

Set of tools to work with JIRA.

## crawler

crawler.py search JIRA tickets defined by pattern and save it as json files in specified folder.

```bash
$ python crawler.py -h
  usage: crawler.py [-h] [--dst DST] [--no_verify_ssl NO_VERIFY_SSL]
                    [--verify_ssl [VERIFY_SSL]]
                    url user password jql
  
  Dump JIRA tickets to files
  
  positional arguments:
    url                   JIRA api url
    user                  username to connect to JIRA
    password              password to connect to JIRA
    jql                   search query
  
  optional arguments:
    -h, --help            show this help message and exit
    --dst DST             destination to store files
    --no_verify_ssl NO_VERIFY_SSL
                          do not verify ssl for JIRA connection
    --verify_ssl [VERIFY_SSL]
                          verify ssl for JIRA connection
```

To start in docker run

```bash
$ docker run -it -v ${PWD}/dump:/dump \
    vistrcm/jiratoolscrawler \
        https://jira/rest/api/2 user "superpass" "assignee = currentUser()" --dst /dump
```
