# jiratools
Set of tools to work with jira.

# crawler
crawler.py search jira tickets defined by pattern and save it as json files in specified folder.

```bash
$ python crawler.py -h
  usage: crawler.py [-h] [--dst DST] [--no_verify_ssl NO_VERIFY_SSL]
                    [--verify_ssl [VERIFY_SSL]]
                    url user password jql
  
  Dump jira tickets to files
  
  positional arguments:
    url                   jira api url
    user                  username to connect to jira
    password              password to connect to jira
    jql                   search query
  
  optional arguments:
    -h, --help            show this help message and exit
    --dst DST             destination to store files
    --no_verify_ssl NO_VERIFY_SSL
                          do not verify ssl for jira connection
    --verify_ssl [VERIFY_SSL]
                          verify ssl for jira connection
``` 
