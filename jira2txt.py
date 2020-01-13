import os

import hashlib

from preprocessor import process_dir

def md5(text: str):
    h = hashlib.new("md5")
    h.update(text.encode("utf8"))
    return h.hexdigest()

def main():
    src_dir = "dump/"
    dst_dir = "jira_texts"

    os.makedirs(dst_dir, exist_ok=True)

    data = process_dir(src_dir)

    for issue in data:
        text = issue["text"]
        digest = md5(text)

        dst = os.path.join(dst_dir, digest + ".txt")
        with open(dst, "w", encoding="utf8") as dst_file:
            dst_file.write(text)


if __name__ == '__main__':
    main()